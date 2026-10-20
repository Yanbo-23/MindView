# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

from .interleave_datasets import MindViewIterableDataset

_snapshot = os.environ.get("TRAINING_JSONL", os.environ.get("MINDVIEW_SNAPSHOT", ""))
DATASET_REGISTRY = {"mindview_sft": MindViewIterableDataset}
DATASET_INFO = {
    "mindview_sft": {
        "mindview_shot6_20k": {
            "data_dir": str(Path(_snapshot).expanduser().parent) if _snapshot else "",
            "jsonl_path": _snapshot,
            "num_total_samples": int(os.environ.get("SFT_SNAPSHOT_ROWS", "20000")),
        },
    },
}
