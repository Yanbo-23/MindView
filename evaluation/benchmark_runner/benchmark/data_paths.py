#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset-aware image path and C3/C6/C10 protocol helpers."""

from pathlib import Path
from typing import Any


SUPPORTED_DATASETS = ("scannet", "scannetpp", "arkitscenes")
SUPPORTED_SHOTS = (3, 6, 10)

_DATASET_ALIASES = {
    "scannet": "scannet",
    "scannetpp": "scannetpp",
    "arkitscenes": "arkitscenes",
}


def canonical_dataset_name(value: Any) -> str:
    """Return the stable dataset id stored in outputs and result manifests."""
    key = str(value or "").strip().lower().replace("_", "").replace("-", "")
    dataset = _DATASET_ALIASES.get(key)
    if dataset is None:
        raise ValueError(
            f"Unsupported or missing dataset={value!r}; "
            f"expected one of {SUPPORTED_DATASETS}"
        )
    return dataset


def validate_shot(value: Any) -> int:
    try:
        shot = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid shot={value!r}; expected one of {SUPPORTED_SHOTS}") from exc
    if shot not in SUPPORTED_SHOTS:
        raise ValueError(f"Unsupported shot={shot}; expected one of {SUPPORTED_SHOTS}")
    return shot


def _dataset_root(args: Any, dataset: str) -> Path:
    attr = {
        "scannet": "scannet_root",
        "scannetpp": "scannetpp_root",
        "arkitscenes": "arkitscenes_root",
    }[dataset]
    value = getattr(args, attr, "")

    # Backward-compatible alias for old ScanNet-only invocations.
    if dataset == "scannet" and not value:
        value = getattr(args, "scan_root", "")
    if not value:
        raise ValueError(f"--{attr} is required for dataset={dataset}")
    return Path(value)


def _frame_stem(frame_id: Any) -> str:
    """Normalize a frame id while accepting manifests that include an extension."""
    name = Path(str(frame_id)).name
    suffix = Path(name).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return Path(name).stem
    return name


def image_path(
    args: Any,
    dataset: Any,
    scene_id: Any,
    frame_id: Any,
) -> str:
    """Resolve one RGB frame using the storage convention of its dataset."""
    dataset = canonical_dataset_name(dataset)
    root = _dataset_root(args, dataset)
    scene = str(scene_id)
    frame = _frame_stem(frame_id)

    if dataset == "scannet":
        path = root / scene / "color" / f"{frame}.jpg"
    elif dataset == "scannetpp":
        path = root / "color" / str(args.data_split) / scene / f"{frame}.jpg"
    else:
        file_stem = frame if frame.startswith(f"{scene}_") else f"{scene}_{frame}"
        path = root / "color" / str(args.data_split) / scene / f"{file_stem}.png"
    return str(path)
