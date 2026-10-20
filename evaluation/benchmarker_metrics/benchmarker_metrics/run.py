#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .common import (
    EvalPair,
    append_jsonl,
    discover_eval_pairs,
    launch_metric_workers,
    merge_shards,
    MetricRunLockedError,
    metric_run_lock,
    parse_int_list,
    safe_rmtree,
    save_json,
    write_metric_outputs,
)
from .resume import partition_pairs, prepare_resume_cache
from .data_paths import canonical_dataset_name, image_path
from . import clip_dino, qwen_vl, vggt_geometry

METRIC_MODULES = {
    clip_dino.METRIC_NAME: clip_dino,
    qwen_vl.METRIC_NAME: qwen_vl,
    vggt_geometry.METRIC_NAME: vggt_geometry,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run benchmark metrics over benchmark_runner outputs.")
    p.add_argument("--metric", required=True, choices=sorted(METRIC_MODULES))
    p.add_argument("--benchmark_output_root", required=True, help="Output root produced by run_benchmark.sh, e.g. benchmark_runner/outputs")
    p.add_argument("--scannet_root", default="", help="ScanNet source-frame root")
    p.add_argument("--scannetpp_root", default="", help="ScanNet++ processed-data root")
    p.add_argument("--arkitscenes_root", default="", help="ARKitScenes processed-data root")
    p.add_argument("--data_split", default="val", help="Split used by ScanNet++ and ARKitScenes")
    p.add_argument("--scan_root", default="", help=argparse.SUPPRESS)
    p.add_argument("--metrics_root", default=str(Path(__file__).resolve().parents[3] / "outputs/metrics"))
    p.add_argument("--models", default="all", help="all or comma-separated model names")
    p.add_argument("--runs", default="all", help="all or comma-separated run names")
    p.add_argument("--shots", default="3,6,10", help="all or comma-separated shots, e.g. 3,6,10")
    p.add_argument("--tasks", default="all", help="all or comma-separated task_type names")
    p.add_argument("--datasets", default="all", help="all or comma-separated dataset names")
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--gpus", default="", help="e.g. 0,1,2,3; in Slurm pass CUDA_VISIBLE_DEVICES")
    p.add_argument("--rank", type=int, default=-1)
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--skip_completed",
        action="store_true",
        help=(
            "Reuse only successful rows whose input bytes and metric "
            "configuration signatures exactly match the current evaluation."
        ),
    )
    p.add_argument("--save_pairs", action="store_true", help="Save discovered eval pairs before sharding.")

    # Add metric-specific args. Argparse allows all args even when metric differs.
    clip_dino.add_args(p)
    qwen_vl.add_args(p)
    vggt_geometry.add_args(p)
    return p


def _get_metric_info(metric: str):
    module = METRIC_MODULES[metric]
    return module, module.METRIC_NAME, module.METRIC_KEYS


def _canonical_anchor_frame_id(
    anchor_frame_id: Any,
    scene_id: Any,
    dataset: str,
) -> str:
    """Normalize equivalent anchor encodings used by benchmark artifacts.

    Some generated outputs serialize a missing anchor as the literal string
    ``"None"`` while reference artifacts use an empty value.  ARKitScenes
    artifacts also exist with either ``<scene_id>_<frame_id>`` or the bare
    ``<frame_id>``.  These are representation differences, not different
    semantic samples, so normalize them only for common-reference matching.
    """
    if anchor_frame_id is None:
        return ""

    anchor = str(anchor_frame_id).strip()
    if not anchor or anchor.lower() in {"none", "null"}:
        return ""

    scene = "" if scene_id is None else str(scene_id).strip()
    scene_prefix = f"{scene}_" if scene else ""
    if (
        dataset == "arkitscenes"
        and scene_prefix
        and anchor.startswith(scene_prefix)
    ):
        return anchor[len(scene_prefix):]
    return anchor


def _semantic_key_from_record(record: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    dataset = canonical_dataset_name(record.get("dataset"))
    scene_id = str(record.get("scene_id") or "")
    return (
        dataset,
        str(record.get("task_type") or "default"),
        scene_id,
        str(record.get("gt_frame_id") or ""),
        _canonical_anchor_frame_id(record.get("anchor_frame_id"), scene_id, dataset),
    )


def _semantic_key_from_pair(pair: EvalPair) -> Tuple[str, str, str, str, str]:
    dataset = canonical_dataset_name(pair.dataset)
    scene_id = str(pair.scene_id or "")
    return (
        dataset,
        str(pair.task_type or "default"),
        scene_id,
        str(pair.gt_frame_id or ""),
        _canonical_anchor_frame_id(pair.anchor_frame_id, scene_id, dataset),
    )


def _nested(record: Dict[str, Any], *keys: str):
    value: Any = record
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _resolve_reference_path(raw_path: Any, source_path: Path, label: str) -> str:
    if not raw_path:
        raise ValueError(f"VGGT common reference is missing {label}")
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = source_path.parent / path
    return str(path.absolute())


def _load_vggt_common_references(source: str) -> Tuple[Dict[tuple, Dict[str, Any]], str, str]:
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(
            f"VGGT common reference generation_results.json not found: {source_path}"
        )

    payload = source_path.read_bytes()
    try:
        records = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid VGGT common reference JSON: {source_path}") from exc
    if not isinstance(records, list):
        raise ValueError("VGGT common reference JSON must contain a list of records")

    references: Dict[tuple, Dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("VGGT common reference contains a non-object record")
        try:
            shot = int(record.get("shot", -1))
        except (TypeError, ValueError):
            continue
        if shot != 10:
            continue
        if record.get("status") not in (None, "success"):
            raise ValueError(
                "VGGT common reference contains an unsuccessful shot-10 record: "
                f"{record.get('sample_id')}"
            )

        key = _semantic_key_from_record(record)
        if key in references:
            raise ValueError(f"Duplicate VGGT common reference semantic key: {key}")

        context_paths = (
            _nested(record, "contexts", "source_paths")
            or _nested(record, "input", "image_paths")
            or []
        )
        if not isinstance(context_paths, list) or len(context_paths) != 10:
            raise ValueError(
                "Each VGGT common reference must contain exactly 10 individual "
                f"context paths; key={key}, count={len(context_paths) if isinstance(context_paths, list) else 'invalid'}"
            )
        resolved_contexts = []
        for index, path in enumerate(context_paths):
            label = f"context[{index}] for {key}"
            resolved = _resolve_reference_path(path, source_path, label)
            resolved_contexts.append(resolved)
        if len(set(resolved_contexts)) != 10:
            raise ValueError(f"VGGT common reference contexts must be unique: {key}")

        gt_label = f"GT image for {key}"
        gt_path = _resolve_reference_path(
            record.get("gt_image") or _nested(record, "ground_truth", "image_path"),
            source_path,
            gt_label,
        )
        gt_source_label = f"GT source image for {key}"
        gt_source_path = _resolve_reference_path(
            _nested(record, "ground_truth", "source_path"),
            source_path,
            gt_source_label,
        )

        context_vis_path = None
        context_vis_candidates = (
            record.get("context_image"),
            _nested(record, "contexts", "image_path"),
            record.get("context_images"),
            _nested(record, "contexts", "image_paths"),
        )
        for candidate in context_vis_candidates:
            if isinstance(candidate, list):
                candidate = candidate[0] if candidate else None
            if not candidate:
                continue
            candidate_path = Path(str(candidate)).expanduser()
            if not candidate_path.is_absolute():
                candidate_path = source_path.parent / candidate_path
            candidate_path = candidate_path.absolute()
            if candidate_path.is_file():
                context_vis_path = str(candidate_path)
                break

        ground_truth = copy.deepcopy(record.get("ground_truth") or {})
        ground_truth["image_path"] = gt_path
        ground_truth["source_path"] = gt_source_path
        references[key] = {
            "context_source_paths": resolved_contexts,
            "gt_path": gt_path,
            "ground_truth": ground_truth,
            "context_vis_path": context_vis_path,
            "sample_id": str(record.get("sample_id") or ""),
            "shot": shot,
        }

    if not references:
        raise ValueError(
            f"VGGT common reference contains no successful shot-10 records: {source_path}"
        )
    return references, str(source_path), hashlib.sha256(payload).hexdigest()


def _load_vggt_canonical_references(args) -> Tuple[Dict[tuple, Dict[str, Any]], str, str]:
    """Resolve frozen C10 inputs independently of generated baseline artifacts."""
    source = Path(args.vggt_canonical_json).expanduser().resolve()
    payload = source.read_bytes()
    references = {}
    for line in payload.splitlines():
        record = json.loads(line)
        if int(record.get("shot", -1)) != 10:
            continue
        contexts = record.get("contexts_frame_id", [])
        if len(contexts) != 10 or len(set(map(str, contexts))) != 10:
            raise ValueError(f"Canonical VGGT C10 contexts must contain ten unique frames: {record.get('sample_id')}")
        key = _semantic_key_from_record(record)
        if key in references:
            raise ValueError(f"Duplicate canonical VGGT semantic key: {key}")
        roots = dict(scannet_root=args.scannet_root or args.scan_root,
                     scannetpp_root=args.scannetpp_root,
                     arkitscenes_root=args.arkitscenes_root, data_split=args.data_split)
        paths = [image_path(record["dataset"], record["scene_id"], frame, **roots) for frame in contexts]
        gt = image_path(record["dataset"], record["scene_id"], record["gt_frame_id"], **roots)
        if not gt or any(not path for path in paths):
            raise ValueError(f"Configure source dataset roots for canonical VGGT inputs: {key}")
        references[key] = dict(context_source_paths=paths, gt_path=gt,
                               ground_truth={"image_path": gt, "source_path": gt},
                               context_vis_path=None, sample_id=record["sample_id"], shot=10)
    if not references:
        raise ValueError(f"No C10 references in canonical benchmark: {source}")
    return references, str(source), hashlib.sha256(payload).hexdigest()


def _apply_vggt_common_reference(pairs: List[EvalPair], args) -> None:
    source = str(
        getattr(args, "vggt_common_reference_generation_results", "") or ""
    ).strip()
    canonical = str(getattr(args, "vggt_canonical_json", "") or "").strip()
    if source and canonical:
        raise ValueError("Use only one VGGT common reference source")
    if not source and not canonical:
        return

    references, source_path, source_sha256 = (
        _load_vggt_canonical_references(args) if canonical
        else _load_vggt_common_references(source)
    )
    missing = sorted({_semantic_key_from_pair(pair) for pair in pairs} - set(references))
    if missing:
        preview = ", ".join(repr(key) for key in missing[:3])
        raise ValueError(
            f"VGGT common reference is missing {len(missing)} selected semantic keys; "
            f"first: {preview}"
        )

    for pair in pairs:
        reference = references[_semantic_key_from_pair(pair)]
        pair.context_source_paths = list(reference["context_source_paths"])
        # Canonical JSONL supplies frame identities, not a replacement encoding
        # of the saved GT. Retain the evaluated run's identical GT preprocessing.
        if not canonical:
            pair.gt_path = reference["gt_path"]
        if reference["context_vis_path"]:
            pair.context_vis_path = reference["context_vis_path"]

        # Metric depth calibration reads the source RGB from raw_record, so this
        # must follow the same common GT. Prediction fields are intentionally
        # untouched: only reference inputs may change in this opt-in mode.
        pair.raw_record["ground_truth"] = copy.deepcopy(reference["ground_truth"])
        pair.raw_record["ground_truth"]["image_path"] = pair.gt_path
        pair.raw_record["gt_image"] = pair.gt_path
        pair.raw_record["vggt_context_source_shot"] = reference["shot"]
        pair.raw_record["vggt_common_reference_active"] = True
        pair.raw_record["vggt_common_reference_generation_results"] = source_path
        pair.raw_record["vggt_common_reference_sha256"] = source_sha256
        pair.raw_record["vggt_common_reference_sample_id"] = reference["sample_id"]
        pair.raw_record["vggt_common_reference_shot"] = reference["shot"]

    print(
        "[vggt-common-reference] "
        f"pairs={len(pairs)}, references={len(references)}, "
        f"source={source_path}, sha256={source_sha256}"
    )


def _discover_for_metric(args, require_contexts: bool):
    # VGGT must use identical evaluation contexts for C3/C6/C10. Discover all
    # shots first so even a C3-only run can borrow the matching maximum-shot
    # contexts; requested-shot filtering is applied afterwards.
    pairs = discover_eval_pairs(
        output_root=args.benchmark_output_root,
        models=args.models,
        shots="all" if require_contexts else args.shots,
        tasks=args.tasks,
        datasets=args.datasets,
        runs=args.runs,
        require_contexts=require_contexts,
        scan_root=args.scan_root,
        scannet_root=args.scannet_root,
        scannetpp_root=args.scannetpp_root,
        arkitscenes_root=args.arkitscenes_root,
        data_split=args.data_split,
    )
    if require_contexts:
        context_groups = {}
        for pair in pairs:
            key = (
                pair.model,
                pair.run_name,
                pair.dataset,
                pair.scene_id,
                pair.task_type,
                pair.gt_frame_id,
                pair.anchor_frame_id,
            )
            current = context_groups.get(key)
            if current is None or (pair.shot, len(pair.context_source_paths)) > (
                current.shot,
                len(current.context_source_paths),
            ):
                context_groups[key] = pair

        for pair in pairs:
            key = (
                pair.model,
                pair.run_name,
                pair.dataset,
                pair.scene_id,
                pair.task_type,
                pair.gt_frame_id,
                pair.anchor_frame_id,
            )
            source = context_groups[key]
            pair.raw_record["vggt_original_num_contexts"] = len(pair.context_source_paths)
            pair.raw_record["vggt_context_source_shot"] = source.shot
            pair.context_source_paths = list(source.context_source_paths)

        requested_shots = set(parse_int_list(args.shots))
        if requested_shots:
            pairs = [pair for pair in pairs if pair.shot in requested_shots]
        _apply_vggt_common_reference(pairs, args)
    if args.max_samples and args.max_samples > 0:
        pairs = pairs[:args.max_samples]
    return pairs


def _run_metric(args, module, metric_name: str, metric_keys, metric_dir: Path) -> None:
    if args.rank < 0 and args.num_gpus > 1:
        launch_metric_workers(args, metric_name, metric_keys)
        return

    if args.rank < 0:
        args.rank = 0
        args.world_size = 1
        if args.overwrite:
            safe_rmtree(metric_dir)
        elif args.skip_completed:
            prepare_resume_cache(metric_dir, metric_name, metric_keys)
        (metric_dir / "shards").mkdir(parents=True, exist_ok=True)
        save_json(metric_dir / "run_config.json", vars(args))

    require_contexts = metric_name == "vggt"
    pairs = _discover_for_metric(args, require_contexts=require_contexts)
    indexed = [p for i, p in enumerate(pairs) if i % args.world_size == args.rank]

    if args.save_pairs and args.rank == 0:
        from .common import pair_to_record
        save_json(metric_dir / "discovered_pairs.json", [pair_to_record(p) for p in pairs])

    reused, pending = partition_pairs(args, metric_name, metric_keys, indexed)
    print(
        f"[metric={metric_name}][rank={args.rank}] "
        f"pairs={len(indexed)} / total={len(pairs)}, "
        f"reused={len(reused)}, pending={len(pending)}"
    )

    if args.skip_completed:
        shard_path = metric_dir / "shards" / f"rank_{args.rank}.jsonl"
        if shard_path.exists():
            shard_path.unlink()
        for row in reused:
            append_jsonl(shard_path, row)

    # Avoid loading a large metric model when every current sample is a strict
    # content/configuration cache hit.
    if pending:
        module.run_worker(args, pending)

    if args.world_size == 1:
        rows = merge_shards(metric_dir, 1, metric_name, metric_keys)
        print(f"[metric={metric_name}] saved {len(rows)} rows to {metric_dir}")


def main() -> None:
    args = build_parser().parse_args()
    module, metric_name, metric_keys = _get_metric_info(args.metric)
    if args.overwrite and args.skip_completed:
        raise SystemExit("--overwrite and --skip_completed cannot be used together")
    if (
        (getattr(args, "vggt_common_reference_generation_results", "")
         or getattr(args, "vggt_canonical_json", ""))
        and metric_name != vggt_geometry.METRIC_NAME
    ):
        raise SystemExit(
            "--vggt_common_reference_generation_results is only valid with --metric vggt"
        )
    metric_dir = Path(args.metrics_root) / metric_name

    if args.rank < 0 and not args.overwrite and not args.skip_completed:
        has_existing_rows = (metric_dir / "results.json").is_file() or any(
            path.is_file() and path.stat().st_size > 0
            for path in (metric_dir / "shards").glob("rank_*.jsonl")
        )
        if has_existing_rows:
            raise SystemExit(
                f"Metric output already exists: {metric_dir}. Use exactly one of "
                "--overwrite or --skip_completed to avoid duplicate rows."
            )

    if args.rank < 0:
        try:
            with metric_run_lock(metric_dir):
                _run_metric(args, module, metric_name, metric_keys, metric_dir)
        except MetricRunLockedError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        _run_metric(args, module, metric_name, metric_keys, metric_dir)


if __name__ == "__main__":
    main()
