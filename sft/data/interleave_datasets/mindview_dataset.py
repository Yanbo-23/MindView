# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""MindView multi-reference image-generation dataset for BAGEL SFT."""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .interleave_t2i_dataset import InterleavedBaseIterableDataset
from ..data_utils import pil_img2rgb


from ..mindview_schema import (
    MindViewDataError, MindViewSchema, SCHEMA_VERSION, PREPROCESS_ID,
    SUPPORTED_SHOTS,
)


@dataclass(frozen=True)
class _SnapshotRecord:
    row: dict[str, Any]
    jsonl_path: Path
    line_no: int

    @property
    def location(self) -> str:
        return f"{self.jsonl_path}:{self.line_no}"


class MindViewIterableDataset(InterleavedBaseIterableDataset, MindViewSchema):
    """Read MindView JSONL rows as ``[context images, prompt, target image]``.

    Every context image contributes both a clean VAE term and a ViT term.  The
    prompt is conditioning only, and the final image is the sole MSE target.
    This is the same sequence construction recommended by the BAGEL
    maintainers for OmniGen-style multi-image generation data.
    """

    def __init__(
        self,
        dataset_name,
        transform,
        tokenizer,
        vit_transform,
        data_dir_list,
        jsonl_path_list,
        num_used_data,
        image_roots,
        expected_schema_version=SCHEMA_VERSION,
        expected_preprocess_id=PREPROCESS_ID,
        expected_manifest_sha256=None,
        strict_schema=True,
        enable_condition_dropout=True,
        shuffle_seed=0,
        local_rank=0,
        world_size=1,
        num_workers=8,
        data_status=None,
    ):
        super().__init__(dataset_name, local_rank, world_size, num_workers)

        self._validate_runtime_args(local_rank, world_size, num_workers)
        self.transform = transform
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.expected_schema_version = expected_schema_version
        self.expected_preprocess_id = expected_preprocess_id
        self.strict_schema = self._require_bool(strict_schema, "strict_schema")
        self.enable_condition_dropout = self._require_bool(
            enable_condition_dropout, "enable_condition_dropout"
        )
        self.shuffle_seed = self._require_int(shuffle_seed, "shuffle_seed")
        self.image_roots = self._prepare_image_roots(image_roots)

        if not hasattr(self.transform, "stride"):
            raise MindViewDataError("transform must expose a stride attribute")
        if not hasattr(self.vit_transform, "stride"):
            raise MindViewDataError("vit_transform must expose a stride attribute")
        if not callable(getattr(self.tokenizer, "encode", None)):
            raise MindViewDataError("tokenizer must expose an encode(text) method")

        self._records = self._load_snapshot_records(
            jsonl_path_list=jsonl_path_list,
            data_dir_list=data_dir_list,
            num_used_data=num_used_data,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        total_consumers = self.world_size * self.num_workers
        if len(self._records) < total_consumers:
            raise MindViewDataError(
                "MindView has fewer rows than rank x workers: "
                f"rows={len(self._records)}, world_size={self.world_size}, "
                f"num_workers={self.num_workers}"
            )

        self.data_paths = list(self._records)
        self.epoch_seed = 0
        self.set_epoch(0)

    @staticmethod
    def _require_bool(value, name):
        if not isinstance(value, bool):
            raise MindViewDataError(f"{name} must be a boolean")
        return value

    @staticmethod
    def _require_int(value, name):
        if not isinstance(value, int) or isinstance(value, bool):
            raise MindViewDataError(f"{name} must be an integer")
        return value

    @classmethod
    def _validate_runtime_args(cls, local_rank, world_size, num_workers):
        local_rank = cls._require_int(local_rank, "local_rank")
        world_size = cls._require_int(world_size, "world_size")
        num_workers = cls._require_int(num_workers, "num_workers")
        if world_size <= 0 or num_workers <= 0:
            raise MindViewDataError("world_size and num_workers must be positive")
        if not 0 <= local_rank < world_size:
            raise MindViewDataError(
                f"local_rank={local_rank} is outside world_size={world_size}"
            )

    @staticmethod
    def _prepare_image_roots(image_roots):
        if not isinstance(image_roots, Mapping) or not image_roots:
            raise MindViewDataError("image_roots must be a non-empty mapping")

        roots = {}
        for dataset, raw_root in image_roots.items():
            if not isinstance(dataset, str) or not dataset:
                raise MindViewDataError("image_roots keys must be non-empty strings")
            try:
                expanded = os.path.expandvars(os.path.expanduser(os.fspath(raw_root)))
            except TypeError as exc:
                raise MindViewDataError(
                    f"image root for {dataset!r} is not path-like"
                ) from exc
            root = Path(expanded).resolve(strict=False)
            if not root.is_dir():
                raise MindViewDataError(
                    f"image root for {dataset!r} is missing: {root}"
                )
            roots[dataset] = root.resolve(strict=True)
        return roots

    @staticmethod
    def _normalize_hashes(expected_manifest_sha256, count):
        if expected_manifest_sha256 is None:
            return [None] * count
        if isinstance(expected_manifest_sha256, str):
            values = [expected_manifest_sha256]
        elif isinstance(expected_manifest_sha256, Sequence):
            values = list(expected_manifest_sha256)
        else:
            raise MindViewDataError(
                "expected_manifest_sha256 must be a string or sequence"
            )
        if len(values) != count:
            raise MindViewDataError(
                "expected_manifest_sha256 must match jsonl_path_list length"
            )
        for value in values:
            if not isinstance(value, str) or _UID_PATTERN.fullmatch(value) is None:
                raise MindViewDataError(
                    "each expected manifest SHA256 must be 64 lowercase hex characters"
                )
        return values

    @classmethod
    def _normalize_num_used_data(cls, num_used_data, count):
        if not isinstance(num_used_data, Sequence) or isinstance(
            num_used_data, (str, bytes)
        ):
            raise MindViewDataError("num_used_data must be a sequence")
        if len(num_used_data) != count:
            raise MindViewDataError("num_used_data must match jsonl_path_list length")
        limits = []
        for value in num_used_data:
            value = cls._require_int(value, "num_used_data item")
            if value == 0 or value < -1:
                raise MindViewDataError("num_used_data items must be -1 or positive")
            limits.append(value)
        return limits

    def _load_snapshot_records(
        self,
        jsonl_path_list,
        data_dir_list,
        num_used_data,
        expected_manifest_sha256,
    ):
        if not isinstance(jsonl_path_list, Sequence) or isinstance(
            jsonl_path_list, (str, bytes)
        ):
            raise MindViewDataError("jsonl_path_list must be a sequence")
        jsonl_paths = [
            Path(os.path.expandvars(os.path.expanduser(os.fspath(path))))
            for path in jsonl_path_list
        ]
        if not jsonl_paths:
            raise MindViewDataError("jsonl_path_list cannot be empty")
        if not isinstance(data_dir_list, Sequence) or isinstance(
            data_dir_list, (str, bytes)
        ):
            raise MindViewDataError("data_dir_list must be a sequence")
        if len(data_dir_list) != len(jsonl_paths):
            raise MindViewDataError("data_dir_list must match jsonl_path_list length")
        for raw_dir in data_dir_list:
            data_dir = Path(
                os.path.expandvars(os.path.expanduser(os.fspath(raw_dir)))
            )
            if not data_dir.is_dir():
                raise MindViewDataError(f"configured data_dir is missing: {data_dir}")

        limits = self._normalize_num_used_data(num_used_data, len(jsonl_paths))
        expected_hashes = self._normalize_hashes(
            expected_manifest_sha256, len(jsonl_paths)
        )
        seen_uids = set()
        seen_lineages = set()
        records = []

        for jsonl_path, limit, expected_hash in zip(
            jsonl_paths, limits, expected_hashes
        ):
            jsonl_path = jsonl_path.resolve(strict=False)
            if not jsonl_path.is_file():
                raise MindViewDataError(f"snapshot JSONL is missing: {jsonl_path}")

            digest = hashlib.sha256()
            file_records = []
            try:
                with jsonl_path.open("rb") as handle:
                    for line_no, raw_line in enumerate(handle, start=1):
                        digest.update(raw_line)
                        if not raw_line.strip():
                            raise MindViewDataError(
                                f"blank JSONL line: {jsonl_path}:{line_no}"
                            )
                        try:
                            row = json.loads(raw_line)
                        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                            raise MindViewDataError(
                                f"invalid JSONL: {jsonl_path}:{line_no}: {exc}"
                            ) from exc
                        self._validate_row(row, jsonl_path, line_no)

                        uid = row["record_uid"]
                        lineage = (
                            row["dataset"],
                            row["source_family"],
                            row["source_sample_id"],
                            row["shot"],
                        )
                        if uid in seen_uids:
                            raise MindViewDataError(
                                f"duplicate record_uid={uid}: {jsonl_path}:{line_no}"
                            )
                        if lineage in seen_lineages:
                            raise MindViewDataError(
                                f"duplicate source lineage={lineage}: "
                                f"{jsonl_path}:{line_no}"
                            )
                        seen_uids.add(uid)
                        seen_lineages.add(lineage)
                        if limit == -1 or len(file_records) < limit:
                            file_records.append(
                                _SnapshotRecord(dict(row), jsonl_path, line_no)
                            )
            except OSError as exc:
                raise MindViewDataError(
                    f"cannot read snapshot JSONL {jsonl_path}: {exc}"
                ) from exc

            if limit != -1 and len(file_records) != limit:
                raise MindViewDataError(
                    f"snapshot has fewer rows than num_used_data for {jsonl_path}: "
                    f"requested={limit}, available={len(file_records)}"
                )
            actual_hash = digest.hexdigest()
            if expected_hash is not None and actual_hash != expected_hash:
                raise MindViewDataError(
                    f"snapshot SHA256 mismatch for {jsonl_path}: "
                    f"expected={expected_hash}, actual={actual_hash}"
                )
            records.extend(file_records)

        if not records:
            raise MindViewDataError("MindView snapshot contains no selected records")
        return records






    def set_epoch(self, seed=42):
        """Create one deterministic global order; rank/worker slicing happens later."""
        seed = self._require_int(seed, "epoch seed")
        self.epoch_seed = seed
        ordered = sorted(self._records, key=lambda record: record.row["record_uid"])
        random.Random(self.shuffle_seed + seed).shuffle(ordered)
        self.data_paths = ordered

    def _records_for_worker(self, worker_id, workers_per_rank):
        worker_id = self._require_int(worker_id, "worker_id")
        workers_per_rank = self._require_int(workers_per_rank, "workers_per_rank")
        if workers_per_rank <= 0 or not 0 <= worker_id < workers_per_rank:
            raise MindViewDataError(
                f"invalid worker topology: worker_id={worker_id}, "
                f"workers_per_rank={workers_per_rank}"
            )
        global_worker_id = self.local_rank * workers_per_rank + worker_id
        global_worker_count = self.world_size * workers_per_rank
        records = self.data_paths[global_worker_id::global_worker_count]
        if not records:
            raise MindViewDataError(
                "MindView worker shard is empty: "
                f"rank={self.local_rank}, worker={worker_id}, "
                f"world_size={self.world_size}, workers_per_rank={workers_per_rank}, "
                f"rows={len(self.data_paths)}"
            )
        return records

    def get_data_paths_per_worker(self):
        info = torch.utils.data.get_worker_info()
        if info is None:
            worker_id = 0
            workers_per_rank = 1
        else:
            worker_id = info.id
            workers_per_rank = info.num_workers
        return self._records_for_worker(worker_id, workers_per_rank), worker_id

    def _resume_position(self, worker_id, shard_size):
        if self.data_status is None:
            return 0, 0
        if not isinstance(self.data_status, Mapping):
            raise MindViewDataError("MindView data_status must be a worker mapping")
        status = self.data_status.get(worker_id, self.data_status.get(str(worker_id)))
        if status is None:
            return 0, 0

        if isinstance(status, int) and not isinstance(status, bool):
            cycle, cursor = 0, status
        elif (
            isinstance(status, Sequence)
            and not isinstance(status, (str, bytes))
            and len(status) == 2
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in status
            )
        ):
            cycle, cursor = status
        else:
            raise MindViewDataError(
                f"invalid MindView data_status for worker {worker_id}: {status!r}"
            )
        if cycle < 0 or cursor < 0:
            raise MindViewDataError(
                f"negative MindView data_status for worker {worker_id}: {status!r}"
            )

        cursor += 1
        cycle += cursor // shard_size
        cursor %= shard_size
        return cycle, cursor

    def _resolve_image_path(self, row, relpath, location):
        relative = self._canonical_relpath(relpath, "image path", location)
        root = self.image_roots[row["dataset"]]
        candidate = (root / Path(*relative.parts)).resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise MindViewDataError(
                f"image path escapes dataset root: {relpath!r}: {location}"
            ) from exc
        if not candidate.is_file():
            raise MindViewDataError(
                f"missing image for record_uid={row['record_uid']}: "
                f"{candidate}: {location}"
            )
        return candidate

    @staticmethod
    def _center_square(image):
        width, height = image.size
        side = min(width, height)
        if side <= 0:
            raise MindViewDataError(f"image has invalid dimensions: {image.size}")
        left = (width - side) // 2
        top = (height - side) // 2
        return image.crop((left, top, left + side, top + side))

    def _load_image(self, row, relpath, location):
        path = self._resolve_image_path(row, relpath, location)
        try:
            with Image.open(path) as image:
                image.load()
                image = pil_img2rgb(image)
                image = self._center_square(image)
                image.load()
                return image
        except MindViewDataError:
            raise
        except Exception as exc:
            raise MindViewDataError(
                f"cannot decode image for record_uid={row['record_uid']}: "
                f"{path}: {location}: {exc}"
            ) from exc

    def parse_row(self, row, location="<MindView row>"):
        """Convert one already validated snapshot row into BAGEL's raw sample."""
        shot = row["shot"]
        if shot not in SUPPORTED_SHOTS:
            raise MindViewDataError(f"unsupported shot={shot}: {location}")

        data = self._init_data()
        for relpath in row["contexts"][str(shot)]:
            data = self._add_image(
                data,
                self._load_image(row, relpath, location),
                need_loss=False,
                need_vae=True,
                need_vit=True,
                enable_cfg=self.enable_condition_dropout,
            )

        data = self._add_text(
            data,
            row["prompt"],
            need_loss=False,
            enable_cfg=self.enable_condition_dropout,
        )
        if not data["text_ids_list"][-1]:
            raise MindViewDataError(
                f"tokenizer produced no prompt tokens for record_uid={row['record_uid']}: "
                f"{location}"
            )

        data = self._add_image(
            data,
            self._load_image(row, row["target_image"], location),
            need_loss=True,
            need_vae=False,
            need_vit=False,
            enable_cfg=False,
        )

        expected_types = [item for _ in range(shot) for item in ("vae_image", "vit_image")]
        expected_types.extend(("text", "vae_image"))
        actual_types = [item["type"] for item in data["sequence_plan"]]
        losses = [item["loss"] for item in data["sequence_plan"]]
        if actual_types != expected_types or losses != [0] * (len(losses) - 1) + [1]:
            raise MindViewDataError(
                f"internal MindView sequence-plan error for record_uid={row['record_uid']}"
            )
        if len(data["image_tensor_list"]) != 2 * shot + 1:
            raise MindViewDataError(
                f"internal MindView image alignment error for record_uid={row['record_uid']}"
            )
        return data

    def __iter__(self):
        records, worker_id = self.get_data_paths_per_worker()
        cycle, row_start_id = self._resume_position(worker_id, len(records))
        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at cycle#{cycle}, local_row#{row_start_id}"
        )

        while True:
            for local_row_idx in range(row_start_id, len(records)):
                record = records[local_row_idx]
                data = self.parse_row(
                    record.row,
                    location=record.location,
                )
                data["data_indexes"] = {
                    "data_indexes": [cycle, local_row_idx],
                    "worker_id": worker_id,
                    "dataset_name": self.dataset_name,
                    "record_uid": record.row["record_uid"],
                    "shot": record.row["shot"],
                }
                yield data

            cycle += 1
            row_start_id = 0
            print(
                f"{self.dataset_name} repeat cycle#{cycle} in "
                f"rank-{self.local_rank} worker-{worker_id}"
            )
