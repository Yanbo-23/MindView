#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset-aware source-frame paths used by metric fallback discovery."""

from pathlib import Path
from typing import Any


SUPPORTED_DATASETS = ("scannet", "scannetpp", "arkitscenes")
SUPPORTED_SHOTS = (3, 6, 10)

_DATASET_ALIASES = {
    "scannet": "scannet",
    "scannetpp": "scannetpp",
    "arkitscenes": "arkitscenes",
}


def canonical_dataset_name(value: Any, default: str = "scannet") -> str:
    """Normalize dataset ids; missing ids mean legacy ScanNet-only output."""
    key = str(value or default).strip().lower().replace("_", "").replace("-", "")
    dataset = _DATASET_ALIASES.get(key)
    if dataset is None:
        raise ValueError(
            f"Unsupported dataset={value!r}; expected one of {SUPPORTED_DATASETS}"
        )
    return dataset


def _frame_stem(frame_id: Any) -> str:
    name = Path(str(frame_id)).name
    if Path(name).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
        return Path(name).stem
    return name


def image_path(
    dataset: Any,
    scene_id: Any,
    frame_id: Any,
    *,
    scannet_root: str = "",
    scannetpp_root: str = "",
    arkitscenes_root: str = "",
    data_split: str = "val",
) -> str:
    dataset = canonical_dataset_name(dataset)
    roots = {
        "scannet": scannet_root,
        "scannetpp": scannetpp_root,
        "arkitscenes": arkitscenes_root,
    }
    root = roots[dataset]
    if not root:
        return ""

    root_path = Path(root)
    scene = str(scene_id)
    frame = _frame_stem(frame_id)
    if dataset == "scannet":
        path = root_path / scene / "color" / f"{frame}.jpg"
    elif dataset == "scannetpp":
        path = root_path / "color" / data_split / scene / f"{frame}.jpg"
    else:
        file_stem = frame if frame.startswith(f"{scene}_") else f"{scene}_{frame}"
        path = root_path / "color" / data_split / scene / f"{file_stem}.png"
    return str(path)
