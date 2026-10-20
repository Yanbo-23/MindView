# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Shared CPU-only validation for MindView SFT records."""

import re
from pathlib import PurePosixPath

SCHEMA_VERSION = "mindview_bagel_single_shot_v1"
PREPROCESS_ID = "center_square_then_bagel_resize_v1"
SUPPORTED_SHOTS = (3, 6, 10)

_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "shot",
        "record_uid",
        "dataset",
        "source_family",
        "task_type",
        "scene_id",
        "source_sample_id",
        "anchor_frame_id",
        "prompt",
        "contexts",
        "target_frame_id",
        "target_image",
        "preprocess_id",
    }
)
_ANCHOR_FAMILIES = frozenset(("cam_loc", "rel_loc", "re_facing"))
_NO_ANCHOR_FAMILIES = frozenset(("obj_loc", "view_loc"))
_EXPECTED_TASK_TYPES = {
    "cam_loc": frozenset(("translation", "rotation", "composition")),
    "rel_loc": frozenset(
        ("anchor_guided_relative_repositioning", "relative_repositioning")
    ),
    "obj_loc": frozenset(("obj_loc", "object_localization", "view_grounding")),
    "re_facing": frozenset(
        ("loc-re-facing", "re_facing", "target_facing_view_generation")
    ),
    "view_loc": frozenset(("view_localization", "view_loc")),
}
_UID_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class MindViewDataError(RuntimeError):
    """Raised when a MindView snapshot or referenced image is invalid."""


class MindViewSchema:
    strict_schema = True
    expected_schema_version = SCHEMA_VERSION
    expected_preprocess_id = PREPROCESS_ID

    @staticmethod
    def _require_int(value, name):
        if not isinstance(value, int) or isinstance(value, bool):
            raise MindViewDataError(f"{name} must be an integer")
        return value

    @staticmethod
    def _require_string(row, field, location):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            raise MindViewDataError(f"{field} must be a non-empty string: {location}")
        return value

    @staticmethod
    def _canonical_relpath(value, field, location):
        if not isinstance(value, str) or not value or value != value.strip():
            raise MindViewDataError(
                f"{field} must be a non-empty relative path: {location}"
            )
        if "\\" in value or "\x00" in value:
            raise MindViewDataError(f"unsafe {field}={value!r}: {location}")
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != value
            or not path.name
        ):
            raise MindViewDataError(f"unsafe {field}={value!r}: {location}")
        return path

    @classmethod
    def _validate_layout(cls, dataset, scene_id, relpath, field, location):
        path = cls._canonical_relpath(relpath, field, location)
        parts = path.parts
        if dataset == "scannet":
            valid = len(parts) == 3 and parts[0] == scene_id and parts[1] == "color"
        elif dataset in {"scannetpp", "arkitscenes"}:
            valid = (
                len(parts) == 4
                and parts[0] == "color"
                and parts[1] in {"train", "val", "test"}
                and parts[2] == scene_id
            )
        else:
            valid = False
        if not valid:
            raise MindViewDataError(
                f"{field} does not match the {dataset} scene layout: "
                f"{relpath!r}: {location}"
            )
        return path

    @staticmethod
    def _frame_id_from_path(dataset, scene_id, path, field, location):
        frame_id = path.stem
        if dataset == "arkitscenes":
            prefix = f"{scene_id}_"
            if not frame_id.startswith(prefix) or len(frame_id) == len(prefix):
                raise MindViewDataError(
                    f"{field} has no ARKitScenes scene prefix: "
                    f"{path.as_posix()!r}: {location}"
                )
            frame_id = frame_id[len(prefix) :]
        return frame_id

    def _validate_row(self, row, jsonl_path, line_no):
        location = f"{jsonl_path}:{line_no}"
        if not isinstance(row, dict):
            raise MindViewDataError(f"JSONL row must be an object: {location}")
        if self.strict_schema and frozenset(row) != _RECORD_FIELDS:
            missing = sorted(_RECORD_FIELDS - frozenset(row))
            extra = sorted(frozenset(row) - _RECORD_FIELDS)
            raise MindViewDataError(
                f"unexpected row fields at {location}: missing={missing}, extra={extra}"
            )
        missing = sorted(_RECORD_FIELDS - frozenset(row))
        if missing:
            raise MindViewDataError(f"missing fields {missing}: {location}")
        if row["schema_version"] != self.expected_schema_version:
            raise MindViewDataError(
                f"schema_version mismatch: {row['schema_version']!r}: {location}"
            )
        if row["preprocess_id"] != self.expected_preprocess_id:
            raise MindViewDataError(
                f"preprocess_id mismatch: {row['preprocess_id']!r}: {location}"
            )

        shot = self._require_int(row["shot"], "shot")
        if shot not in SUPPORTED_SHOTS:
            raise MindViewDataError(f"unsupported shot={shot}: {location}")

        uid = self._require_string(row, "record_uid", location)
        if _UID_PATTERN.fullmatch(uid) is None:
            raise MindViewDataError(f"invalid record_uid={uid!r}: {location}")
        dataset = self._require_string(row, "dataset", location)
        if dataset not in self.image_roots:
            raise MindViewDataError(
                f"no image root configured for dataset={dataset!r}: {location}"
            )
        source_family = self._require_string(row, "source_family", location)
        task_type = self._require_string(row, "task_type", location)
        if source_family not in _EXPECTED_TASK_TYPES:
            raise MindViewDataError(
                f"unsupported source_family={source_family!r}: {location}"
            )
        if task_type not in _EXPECTED_TASK_TYPES[source_family]:
            raise MindViewDataError(
                f"unexpected task_type={task_type!r} for {source_family}: {location}"
            )
        scene_id = self._require_string(row, "scene_id", location)
        self._require_string(row, "source_sample_id", location)
        self._require_string(row, "prompt", location)
        target_frame_id = self._require_string(row, "target_frame_id", location)

        anchor_frame_id = row["anchor_frame_id"]
        if source_family in _ANCHOR_FAMILIES:
            if not isinstance(anchor_frame_id, str) or not anchor_frame_id:
                raise MindViewDataError(
                    f"anchor task has no anchor_frame_id: {location}"
                )
        elif source_family in _NO_ANCHOR_FAMILIES and anchor_frame_id is not None:
            raise MindViewDataError(
                f"no-anchor task has anchor_frame_id={anchor_frame_id!r}: {location}"
            )

        contexts = row["contexts"]
        expected_context_keys = {str(shot)}
        if not isinstance(contexts, dict) or set(contexts) != expected_context_keys:
            raise MindViewDataError(
                f"contexts must contain exactly {sorted(expected_context_keys)}: {location}"
            )

        key = str(shot)
        values = contexts[key]
        if not isinstance(values, list) or len(values) != shot:
            raise MindViewDataError(
                f"contexts[{key!r}] must contain exactly {shot} paths: {location}"
            )
        if len(set(values)) != shot:
            raise MindViewDataError(
                f"contexts[{key!r}] contains duplicate paths: {location}"
            )
        paths = [
            self._validate_layout(
                dataset, scene_id, value, f"contexts.{key}", location
            )
            for value in values
        ]
        first_frame_id = self._frame_id_from_path(
            dataset, scene_id, paths[0], f"contexts.{key}", location
        )
        if source_family in _ANCHOR_FAMILIES and first_frame_id != anchor_frame_id:
            raise MindViewDataError(
                f"anchor is not first in contexts[{key!r}]: {location}"
            )

        target_image = row["target_image"]
        target_path = self._validate_layout(
            dataset, scene_id, target_image, "target_image", location
        )
        path_target_frame_id = self._frame_id_from_path(
            dataset, scene_id, target_path, "target_image", location
        )
        if path_target_frame_id != target_frame_id:
            raise MindViewDataError(
                f"target image/frame ID mismatch: {target_image!r}, "
                f"{target_frame_id!r}: {location}"
            )
        if target_image in values:
            raise MindViewDataError(f"target image leaks into contexts: {location}")

