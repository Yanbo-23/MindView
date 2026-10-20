#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Strict, content-addressed resume support for metric evaluation.

This module only decides whether an existing per-sample result is reusable.  It
does not alter metric inputs, preprocessing, inference, scores, or aggregation.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

from . import clip_dino, qwen_vl, vggt_geometry
from .common import EvalPair, pair_to_record, read_json, save_json


RESUME_SCHEMA_VERSION = "metric-content-sha256-v1"
RESUME_CACHE_NAME = "resume_records.json"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: str | Path) -> str:
    file_path = Path(path)
    before = file_path.stat()
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = file_path.stat()
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in stable_fields):
        raise RuntimeError(f"Metric input changed while hashing: {file_path}")
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _hf_cached_revision(model_name: str) -> str | None:
    if not model_name or "/" not in model_name:
        return None
    hub_root = os.environ.get("HF_HUB_CACHE")
    if hub_root:
        hub_path = Path(hub_root).expanduser()
    else:
        hf_home = Path(
            os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
        ).expanduser()
        hub_path = hf_home / "hub"
    ref_path = hub_path / f"models--{model_name.replace('/', '--')}" / "refs" / "main"
    try:
        revision = ref_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return revision or None


def _source_digest(paths: Iterable[Path]) -> str:
    entries = []
    for path in sorted({p.resolve() for p in paths if p.is_file()}, key=str):
        entries.append((path.name, _file_sha256(path)))
    return _canonical_sha256(entries)


def _tree_source_digest(root: str | Path) -> str | None:
    root_path = Path(root).expanduser()
    package_path = root_path / "vggt"
    if not package_path.is_dir():
        return None
    entries = [
        (str(path.relative_to(package_path)), _file_sha256(path))
        for path in sorted(package_path.rglob("*.py"))
        if path.is_file()
    ]
    return _canonical_sha256(entries)


def _runtime_fingerprint() -> Dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    runtime: Dict[str, Any] = {
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
        "pillow": _package_version("Pillow"),
        "numpy": _package_version("numpy"),
        "cuda_runtime": torch.version.cuda,
        "device_type": "cuda" if cuda_available else "cpu",
    }
    if cuda_available:
        runtime["cuda_capability"] = list(torch.cuda.get_device_capability())
        runtime["cuda_device_name"] = torch.cuda.get_device_name()
    return runtime


def _metric_config_payload(args, metric_name: str) -> Dict[str, Any]:
    module = {
        clip_dino.METRIC_NAME: clip_dino,
        qwen_vl.METRIC_NAME: qwen_vl,
        vggt_geometry.METRIC_NAME: vggt_geometry,
    }[metric_name]
    common_source = Path(__file__).with_name("common.py")
    payload: Dict[str, Any] = {
        "resume_schema_version": RESUME_SCHEMA_VERSION,
        "metric": metric_name,
        "image_size": int(args.image_size),
        "implementation_sha256": _source_digest(
            [Path(module.__file__), common_source, Path(__file__)]
        ),
        "runtime": _runtime_fingerprint(),
    }

    if metric_name == clip_dino.METRIC_NAME:
        payload.update(
            {
                "clip_model_name": args.clip_model_name,
                "clip_model_revision": _hf_cached_revision(args.clip_model_name),
                "dino_model_name": args.dino_model_name,
                "dino_model_revision": _hf_cached_revision(args.dino_model_name),
            }
        )
    elif metric_name == qwen_vl.METRIC_NAME:
        payload.update(
            {
                "qwen_model_name": args.qwen_model_name,
                "qwen_model_revision": _hf_cached_revision(args.qwen_model_name),
                "judge_version": qwen_vl.JUDGE_VERSION,
                "fidelity_prompt_sha256": _canonical_sha256(
                    qwen_vl.build_fidelity_prompt()
                ),
                "quality_prompt_sha256": _canonical_sha256(
                    qwen_vl.build_quality_prompt()
                ),
                "max_new_tokens": int(args.max_new_tokens),
                "use_flash_attn": bool(args.use_flash_attn),
                "generation": {
                    "do_sample": False,
                    "temperature": None,
                    "enable_thinking_for_qwen3_5": False,
                },
            }
        )
    elif metric_name == vggt_geometry.METRIC_NAME:
        common_reference = str(
            getattr(args, "vggt_common_reference_generation_results", "")
            or getattr(args, "vggt_canonical_json", "") or ""
        ).strip()
        payload.update(
            {
                "vggt_model_name": args.vggt_model_name,
                "vggt_model_revision": _hf_cached_revision(args.vggt_model_name),
                "vggt_source_sha256": _tree_source_digest(args.vggt_repo_path),
                "rotation_error_cap_deg": vggt_geometry.ROT_ERROR_CAP_DEG,
                "translation_error_cap_m": vggt_geometry.TRANS_ERROR_CAP_M,
                "effective_pose_dtype": (
                    "bfloat16"
                    if torch.cuda.is_available()
                    and torch.cuda.get_device_capability()[0] >= 8
                    else "float16"
                ),
                "do_vis": bool(args.do_vis),
                "vis_dir": str(args.vis_dir) if args.do_vis else None,
                "common_reference_sha256": (
                    _file_sha256(common_reference) if common_reference else None
                ),
            }
        )
    return payload


def _identity_payload(pair: EvalPair, metric_name: str) -> Dict[str, Any]:
    return {
        "metric": metric_name,
        "model": pair.model,
        "run_name": pair.run_name,
        "dataset": pair.dataset,
        "shot": int(pair.shot),
        "task_type": pair.task_type,
        "sample_id": pair.sample_id,
        "scene_id": pair.scene_id,
        "gt_frame_id": pair.gt_frame_id,
        "anchor_frame_id": pair.anchor_frame_id,
    }


def _input_files(pair: EvalPair, args, metric_name: str) -> List[Tuple[str, str]]:
    files: List[Tuple[str, str]] = [
        ("ground_truth", pair.gt_path),
        ("prediction", pair.pred_path),
    ]
    if metric_name != vggt_geometry.METRIC_NAME:
        return files

    files = [
        (f"context[{index}]", path)
        for index, path in enumerate(pair.context_source_paths)
    ] + files
    gt_source = vggt_geometry._gt_source_path(pair, args)
    files.append(("ground_truth_source_rgb", gt_source))
    real_rgb_paths = list(pair.context_source_paths) + [gt_source]
    files.extend(
        (f"metric_depth[{index}]", str(vggt_geometry._depth_path_from_rgb(path)))
        for index, path in enumerate(real_rgb_paths)
    )
    return files


def sign_pairs(args, metric_name: str, pairs: Sequence[EvalPair]) -> None:
    config_sha256 = _canonical_sha256(_metric_config_payload(args, metric_name))
    for pair in pairs:
        identity = _identity_payload(pair, metric_name)
        input_entries = [
            {"role": role, "sha256": _file_sha256(path)}
            for role, path in _input_files(pair, args, metric_name)
        ]
        pair.resume_schema_version = RESUME_SCHEMA_VERSION
        pair.eval_key = _canonical_sha256(identity)
        pair.input_sha256 = _canonical_sha256(input_entries)
        pair.metric_config_sha256 = config_sha256
        pair.evaluation_sha256 = _canonical_sha256(
            {
                "resume_schema_version": RESUME_SCHEMA_VERSION,
                "metric": metric_name,
                "eval_key": pair.eval_key,
                "input_sha256": pair.input_sha256,
                "metric_config_sha256": pair.metric_config_sha256,
            }
        )


def _is_finite_metric_row(row: Dict[str, Any], metric_keys: Sequence[str]) -> bool:
    if row.get("metric_success") is not True:
        return False
    for key in metric_keys:
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if not math.isfinite(float(value)):
            return False
    return True


def _valid_signed_row(
    row: Dict[str, Any], metric_name: str, metric_keys: Sequence[str]
) -> bool:
    if not _is_finite_metric_row(row, metric_keys):
        return False
    required = (
        "eval_key",
        "input_sha256",
        "metric_config_sha256",
        "evaluation_sha256",
    )
    if row.get("resume_schema_version") != RESUME_SCHEMA_VERSION:
        return False
    if any(not isinstance(row.get(key), str) or not row[key] for key in required):
        return False
    expected = _canonical_sha256(
        {
            "resume_schema_version": RESUME_SCHEMA_VERSION,
            "metric": metric_name,
            "eval_key": row["eval_key"],
            "input_sha256": row["input_sha256"],
            "metric_config_sha256": row["metric_config_sha256"],
        }
    )
    return row["evaluation_sha256"] == expected


def _read_json_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        payload = read_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def _read_jsonl_rows(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    rows: List[Dict[str, Any]] = []
    invalid = 0
    if not path.is_file():
        return rows, invalid
    with path.open("rb") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            try:
                line = raw_line.decode("utf-8")
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                invalid += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                invalid += 1
    return rows, invalid


def _deduplicate_valid_rows(
    rows: Iterable[Dict[str, Any]],
    metric_name: str,
    metric_keys: Sequence[str],
) -> Tuple[Dict[str, Dict[str, Any]], int, int]:
    selected: Dict[str, Dict[str, Any]] = {}
    conflicts = set()
    invalid = 0
    for row in rows:
        if not _valid_signed_row(row, metric_name, metric_keys):
            invalid += 1
            continue
        key = row["evaluation_sha256"]
        if key in conflicts:
            continue
        previous = selected.get(key)
        if previous is None:
            selected[key] = row
            continue
        if any(previous.get(metric) != row.get(metric) for metric in metric_keys):
            conflicts.add(key)
            selected.pop(key, None)
    return selected, invalid, len(conflicts)


def prepare_resume_cache(
    metric_dir: str | Path,
    metric_name: str,
    metric_keys: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    """Snapshot reusable rows atomically, then clear stale shard assignments."""
    metric_path = Path(metric_dir)
    shard_paths = sorted((metric_path / "shards").glob("rank_*.jsonl"))
    rows: List[Dict[str, Any]] = []
    rows.extend(_read_json_rows(metric_path / RESUME_CACHE_NAME))
    rows.extend(_read_json_rows(metric_path / "results.json"))
    invalid_jsonl = 0
    for shard_path in shard_paths:
        shard_rows, invalid = _read_jsonl_rows(shard_path)
        rows.extend(shard_rows)
        invalid_jsonl += invalid

    selected, invalid_rows, conflicts = _deduplicate_valid_rows(
        rows, metric_name, metric_keys
    )
    ordered = sorted(selected.values(), key=lambda row: row["evaluation_sha256"])
    metric_path.mkdir(parents=True, exist_ok=True)
    save_json(metric_path / RESUME_CACHE_NAME, ordered)

    # The cache is safely on disk before old rank files are removed.  Current
    # workers rebuild their shards for the current world size and selection.
    for shard_path in shard_paths:
        shard_path.unlink()

    print(
        f"[resume] metric={metric_name}, reusable={len(ordered)}, "
        f"ignored_unsigned_or_failed={invalid_rows}, "
        f"conflicts={conflicts}, ignored_incomplete_jsonl={invalid_jsonl}"
    )
    return selected


def load_resume_cache(
    metric_dir: str | Path,
    metric_name: str,
    metric_keys: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    rows = _read_json_rows(Path(metric_dir) / RESUME_CACHE_NAME)
    selected, _, _ = _deduplicate_valid_rows(rows, metric_name, metric_keys)
    return selected


def partition_pairs(
    args,
    metric_name: str,
    metric_keys: Sequence[str],
    pairs: Sequence[EvalPair],
) -> Tuple[List[Dict[str, Any]], List[EvalPair]]:
    """Sign current pairs and split them into exact cache hits and pending work."""
    sign_pairs(args, metric_name, pairs)
    if not getattr(args, "skip_completed", False):
        return [], list(pairs)

    cached = load_resume_cache(
        Path(args.metrics_root) / metric_name, metric_name, metric_keys
    )
    reused: List[Dict[str, Any]] = []
    pending: List[EvalPair] = []
    for pair in pairs:
        previous = cached.get(pair.evaluation_sha256 or "")
        if previous is None or previous.get("eval_key") != pair.eval_key:
            pending.append(pair)
            continue

        # Keep metric values/diagnostics, but refresh every identity and path
        # field from the current discovery result so no snapshot path goes stale.
        row = dict(previous)
        row.update(pair_to_record(pair))
        row.pop("error", None)
        if metric_name == vggt_geometry.METRIC_NAME:
            row.update(
                {
                    "num_contexts": len(pair.context_source_paths),
                    "original_num_contexts": pair.raw_record.get(
                        "vggt_original_num_contexts", len(pair.context_source_paths)
                    ),
                    "context_source_shot": pair.raw_record.get(
                        "vggt_context_source_shot", pair.shot
                    ),
                }
            )
            if pair.raw_record.get("vggt_common_reference_active"):
                row.update(
                    {
                        "common_reference_active": True,
                        "common_reference_generation_results": pair.raw_record.get(
                            "vggt_common_reference_generation_results"
                        ),
                        "common_reference_sha256": pair.raw_record.get(
                            "vggt_common_reference_sha256"
                        ),
                        "common_reference_sample_id": pair.raw_record.get(
                            "vggt_common_reference_sample_id"
                        ),
                        "common_reference_shot": pair.raw_record.get(
                            "vggt_common_reference_shot"
                        ),
                    }
                )
        reused.append(row)
    return reused, pending
