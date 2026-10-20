#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from PIL import Image

from .data_paths import canonical_dataset_name, image_path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class MetricRunLockedError(RuntimeError):
    pass


@dataclass
class EvalPair:
    model: str
    run_name: str
    run_dir: str
    dataset: str
    shot: int
    task_type: str
    sample_id: str
    scene_id: str
    gt_frame_id: str
    anchor_frame_id: str
    pred_path: str
    gt_path: str
    context_source_paths: List[str]
    context_vis_path: Optional[str]
    raw_record: Dict[str, Any]
    resume_schema_version: Optional[str] = None
    eval_key: Optional[str] = None
    input_sha256: Optional[str] = None
    metric_config_sha256: Optional[str] = None
    evaluation_sha256: Optional[str] = None


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: str | Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    invalid = 0
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
    if invalid:
        print(f"[Warning] ignored {invalid} invalid/incomplete JSONL rows: {path}")
    return rows


def save_csv(path: str | Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def parse_csv_list(value: str, default: Optional[Sequence[str]] = None) -> List[str]:
    if value is None or str(value).strip() == "":
        return list(default or [])
    value = str(value).strip()
    if value.lower() == "all":
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_int_list(value: str, default: Optional[Sequence[int]] = None) -> List[int]:
    if value is None or str(value).strip() == "":
        return list(default or [])
    value = str(value).strip()
    if value.lower() == "all":
        return []
    return [int(x.strip().replace("shot-", "")) for x in value.split(",") if x.strip()]


def parse_gpus(gpus: str, num_gpus: int) -> List[str]:
    if gpus:
        ids = [x.strip() for x in gpus.split(",") if x.strip()]
    else:
        ids = [str(i) for i in range(num_gpus)]
    if len(ids) < num_gpus:
        raise ValueError(f"Need {num_gpus} GPUs, but got {ids}")
    return ids[:num_gpus]


def preprocess_image(img: Image.Image, image_size: int = 512) -> Image.Image:
    img = img.convert("RGB")
    if image_size and image_size > 0:
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        img = img.resize((image_size, image_size), getattr(Image, "Resampling", Image).LANCZOS)
    return img


def load_image(path: str, image_size: int = 512) -> Image.Image:
    return preprocess_image(Image.open(path), image_size)


def is_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def safe_rmtree(path: str | Path) -> None:
    p = Path(path)
    if p.exists():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()


@contextmanager
def metric_run_lock(metric_dir: str | Path):
    """Prevent two parent jobs from mutating one metric directory at once."""
    metric_path = Path(metric_dir).resolve()
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = metric_path.parent / f".{metric_path.name}.run.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            holder = handle.read().strip() or "unknown process"
            raise MetricRunLockedError(
                f"Metric output is already being written: {metric_path} "
                f"(lock holder: {holder})"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _candidate_paths(path_str: str, output_root: Path, run_dir: Path) -> List[Path]:
    if not path_str:
        return []
    p = Path(path_str)
    if p.is_absolute():
        return [p]
    return [
        Path.cwd() / p,
        output_root.parent / p,
        output_root / p,
        run_dir / p,
    ]


def resolve_path(path_str: Optional[str], output_root: Path, run_dir: Path) -> Optional[str]:
    if not path_str:
        return None
    for p in _candidate_paths(path_str, output_root, run_dir):
        if p.exists():
            return str(p)
    p = Path(path_str)
    return str(p if p.is_absolute() else output_root.parent / p)


def _get_nested(obj: Dict[str, Any], keys: Sequence[str], default=None):
    cur: Any = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _find_constructed_image(
    run_dir: Path,
    shot: int,
    subdir: str,
    dataset: str,
    task_type: str,
    sample_id: str,
    exts: Sequence[str],
) -> Optional[str]:
    bases = [
        run_dir / f"shot-{shot}" / subdir / dataset / task_type,
        run_dir / subdir / dataset / task_type,
        run_dir / f"shot-{shot}" / subdir / task_type,
        run_dir / subdir / task_type,
        run_dir / f"shot-{shot}" / subdir,
        run_dir / subdir,
    ]
    for base in bases:
        for ext in exts:
            p = base / f"{sample_id}{ext}"
            if p.exists():
                return str(p)
    return None


def _shot_from_record(record: Dict[str, Any], fallback: Optional[int] = None) -> int:
    shot = record.get("shot", fallback)
    try:
        return int(shot)
    except Exception:
        return int(fallback or -1)


def _context_paths_from_dataset_root(
    record: Dict[str, Any],
    dataset: str,
    *,
    scannet_root: str,
    scannetpp_root: str,
    arkitscenes_root: str,
    data_split: str,
) -> List[str]:
    """Rebuild individual VGGT context paths with dataset-specific layouts."""
    scene_id = str(record.get("scene_id", ""))
    frame_ids = record.get("contexts_frame_id") or record.get("input_frame_ids") or []
    if not scene_id or not isinstance(frame_ids, list):
        return []

    out: List[str] = []
    for fid in frame_ids:
        path = image_path(
            dataset,
            scene_id,
            fid,
            scannet_root=scannet_root,
            scannetpp_root=scannetpp_root,
            arkitscenes_root=arkitscenes_root,
            data_split=data_split,
        )
        if not path or not Path(path).exists():
            return []
        out.append(path)
    return out


def _load_generation_records(run_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    shot_jsons = sorted(run_dir.glob("shot-*/generation_results.json"))
    if shot_jsons:
        for p in shot_jsons:
            shot_name = p.parent.name.replace("shot-", "")
            try:
                shot = int(shot_name)
            except Exception:
                shot = None
            cur = read_json(p)
            if isinstance(cur, list):
                for r in cur:
                    r = dict(r)
                    if shot is not None:
                        r.setdefault("shot", shot)
                    records.append(r)
        return records

    p = run_dir / "generation_results.json"
    if p.exists():
        cur = read_json(p)
        if isinstance(cur, list):
            return [dict(r) for r in cur]
    return []


def discover_eval_pairs(
    output_root: str,
    models: str = "all",
    shots: str = "all",
    tasks: str = "all",
    datasets: str = "all",
    runs: str = "all",
    require_contexts: bool = False,
    scan_root: str = "",
    scannet_root: str = "",
    scannetpp_root: str = "",
    arkitscenes_root: str = "",
    data_split: str = "val",
) -> List[EvalPair]:
    output_root_p = Path(output_root).resolve()
    model_filter = set(parse_csv_list(models))
    shot_filter = set(parse_int_list(shots))
    task_filter = set(parse_csv_list(tasks))
    dataset_filter = {
        canonical_dataset_name(x)
        for x in parse_csv_list(datasets)
    }
    run_filter = set(parse_csv_list(runs))
    scannet_root = scannet_root or scan_root

    if not output_root_p.exists():
        raise FileNotFoundError(f"benchmark output_root not found: {output_root_p}")

    pairs: List[EvalPair] = []
    model_dirs = [p for p in sorted(output_root_p.iterdir()) if p.is_dir()]
    for model_dir in model_dirs:
        model = model_dir.name
        if model_filter and model not in model_filter:
            continue
        run_dirs = [p for p in sorted(model_dir.iterdir()) if p.is_dir()]
        # Also support output_root/<model> itself being a run dir.
        if (model_dir / "generation_results.json").exists() or list(model_dir.glob("shot-*/generation_results.json")):
            run_dirs = [model_dir] + run_dirs
        for run_dir in run_dirs:
            run_name = run_dir.name
            if run_filter and run_name not in run_filter:
                continue
            records = _load_generation_records(run_dir)
            for r in records:
                if r.get("status") not in (None, "success"):
                    continue
                sample_id = str(r.get("sample_id") or r.get("id") or "")
                try:
                    dataset = canonical_dataset_name(r.get("dataset"))
                except ValueError:
                    continue
                task_type = str(r.get("task_type") or "default")
                shot = _shot_from_record(r)
                if not sample_id or shot < 0:
                    continue
                if shot_filter and shot not in shot_filter:
                    continue
                if task_filter and task_type not in task_filter:
                    continue
                if dataset_filter and dataset not in dataset_filter:
                    continue

                pred_raw = r.get("pred_image") or _get_nested(r, ["prediction", "image_path"])
                gt_raw = r.get("gt_image") or _get_nested(r, ["ground_truth", "image_path"])
                pred_path = resolve_path(pred_raw, output_root_p, run_dir) if pred_raw else None
                gt_path = resolve_path(gt_raw, output_root_p, run_dir) if gt_raw else None

                if not pred_path or not Path(pred_path).exists():
                    pred_path = _find_constructed_image(run_dir, shot, "pred", dataset, task_type, sample_id, [".png", ".jpg", ".jpeg", ".webp"])
                if not gt_path or not Path(gt_path).exists():
                    gt_path = _find_constructed_image(run_dir, shot, "gt", dataset, task_type, sample_id, [".jpg", ".png", ".jpeg", ".webp"])

                if not pred_path or not gt_path or not Path(pred_path).exists() or not Path(gt_path).exists():
                    continue

                expected_contexts = len(r.get("contexts_frame_id") or r.get("input_frame_ids") or [])

                src_contexts = (
                    _get_nested(r, ["contexts", "source_paths"], [])
                    or _get_nested(r, ["input", "image_paths"], [])
                    or []
                )
                context_source_paths = []
                for p in src_contexts:
                    rp = resolve_path(str(p), output_root_p, run_dir)
                    if rp and Path(rp).exists() and is_image_file(rp):
                        context_source_paths.append(rp)

                # If generation_results points to a stitched context visualization
                # or if raw paths are not reachable on the metric node, rebuild
                # individual paths from the corresponding dataset root.
                needs_rebuild = (
                    not context_source_paths
                    or (expected_contexts > 1 and len(context_source_paths) != expected_contexts)
                )
                if needs_rebuild:
                    rebuilt_contexts = _context_paths_from_dataset_root(
                        r,
                        dataset,
                        scannet_root=scannet_root,
                        scannetpp_root=scannetpp_root,
                        arkitscenes_root=arkitscenes_root,
                        data_split=data_split,
                    )
                    if rebuilt_contexts:
                        context_source_paths = rebuilt_contexts

                context_vis_path = None
                cands: List[Any] = []
                if isinstance(r.get("context_image"), str):
                    cands.append(r.get("context_image"))
                if isinstance(_get_nested(r, ["contexts", "image_path"]), str):
                    cands.append(_get_nested(r, ["contexts", "image_path"]))
                # In older versions context_images may be a list of individual images.
                if isinstance(r.get("context_images"), str):
                    cands.append(r.get("context_images"))
                for c in cands:
                    rp = resolve_path(str(c), output_root_p, run_dir)
                    if rp and Path(rp).exists():
                        context_vis_path = rp
                        break
                if context_vis_path is None:
                    context_vis_path = _find_constructed_image(run_dir, shot, "contexts", dataset, task_type, sample_id, [".jpg", ".png", ".jpeg", ".webp"])

                # Older benchmark_runner versions saved individual context images
                # under context_images / contexts.image_paths. Newer versions save a
                # single stitched visualization, which must NOT be used by VGGT.
                if not context_source_paths:
                    old_contexts = r.get("context_images") or _get_nested(r, ["contexts", "image_paths"], []) or []
                    if isinstance(old_contexts, list) and (expected_contexts <= 0 or len(old_contexts) == expected_contexts):
                        for p in old_contexts:
                            rp = resolve_path(str(p), output_root_p, run_dir) if p else None
                            if rp and Path(rp).exists() and is_image_file(rp):
                                context_source_paths.append(rp)

                if require_contexts and not context_source_paths:
                    continue

                pairs.append(EvalPair(
                    model=model,
                    run_name=run_name,
                    run_dir=str(run_dir),
                    dataset=dataset,
                    shot=shot,
                    task_type=task_type,
                    sample_id=sample_id,
                    scene_id=str(r.get("scene_id", "")),
                    gt_frame_id=str(r.get("gt_frame_id", "")),
                    anchor_frame_id=str(r.get("anchor_frame_id", "")),
                    pred_path=str(pred_path),
                    gt_path=str(gt_path),
                    context_source_paths=context_source_paths,
                    context_vis_path=context_vis_path,
                    raw_record=r,
                ))
    pairs.sort(key=lambda x: (x.model, x.run_name, x.dataset, x.shot, x.task_type, x.raw_record.get("global_index", 10**12), x.sample_id))
    return pairs


def pair_to_record(pair: EvalPair) -> Dict[str, Any]:
    record = {
        "model": pair.model,
        "run_name": pair.run_name,
        "dataset": pair.dataset,
        "shot": pair.shot,
        "task_type": pair.task_type,
        "sample_id": pair.sample_id,
        "scene_id": pair.scene_id,
        "gt_frame_id": pair.gt_frame_id,
        "anchor_frame_id": pair.anchor_frame_id,
        "pred_path": pair.pred_path,
        "gt_path": pair.gt_path,
        "context_source_paths": pair.context_source_paths,
        "context_vis_path": pair.context_vis_path,
    }
    for key in (
        "resume_schema_version",
        "eval_key",
        "input_sha256",
        "metric_config_sha256",
        "evaluation_sha256",
    ):
        value = getattr(pair, key, None)
        if value is not None:
            record[key] = value
    return record


def group_summary(
    rows: List[Dict[str, Any]],
    group_keys: Sequence[str],
    metric_keys: Sequence[str],
) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = tuple(r.get(k) for k in group_keys)
        groups[key].append(r)

    def group_sort_key(item):
        values = []
        for field, value in zip(group_keys, item[0]):
            if field == "shot":
                try:
                    values.append((0, int(str(value).replace("shot-", ""))))
                    continue
                except (TypeError, ValueError):
                    pass
            values.append((1, str(value)))
        return tuple(values)

    out: List[Dict[str, Any]] = []
    for key, rs in sorted(groups.items(), key=group_sort_key):
        item = {k: v for k, v in zip(group_keys, key)}
        item["num_records"] = len(rs)
        item["num_success"] = sum(bool(r.get("metric_success", True)) for r in rs)
        for mk in metric_keys:
            vals = []
            for r in rs:
                v = r.get(mk)
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    vals.append(float(v))
            item[f"mean_{mk}"] = sum(vals) / len(vals) if vals else None
            if len(vals) > 1:
                mean = item[f"mean_{mk}"]
                item[f"std_{mk}"] = (sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5
            else:
                item[f"std_{mk}"] = None
        out.append(item)
    return out


def write_metric_outputs(
    out_dir: str | Path,
    metric_name: str,
    rows: List[Dict[str, Any]],
    metric_keys: Sequence[str],
) -> None:
    out_dir = Path(out_dir)
    save_json(out_dir / "results.json", rows)
    save_csv(out_dir / "results.csv", rows)

    summaries = {
        "by_model_dataset_shot_task": group_summary(rows, ["model", "dataset", "shot", "task_type"], metric_keys),
        "by_model_dataset_shot": group_summary(rows, ["model", "dataset", "shot"], metric_keys),
        "by_model_dataset": group_summary(rows, ["model", "dataset"], metric_keys),
        "by_model_dataset_shot_scene": group_summary(rows, ["model", "dataset", "shot", "scene_id"], metric_keys),
        "by_model_shot_task": group_summary(rows, ["model", "shot", "task_type"], metric_keys),
        "by_model_shot": group_summary(rows, ["model", "shot"], metric_keys),
        "by_model": group_summary(rows, ["model"], metric_keys),
        "by_model_shot_scene": group_summary(rows, ["model", "shot", "scene_id"], metric_keys),
    }
    save_json(out_dir / "summary.json", {"metric": metric_name, "summaries": summaries})
    for name, table in summaries.items():
        save_csv(out_dir / f"summary_{name}.csv", table)


def merge_shards(metric_dir: str | Path, world_size: int, metric_name: str, metric_keys: Sequence[str]) -> List[Dict[str, Any]]:
    metric_dir = Path(metric_dir)
    rows: List[Dict[str, Any]] = []
    for rank in range(world_size):
        rows.extend(read_jsonl(metric_dir / "shards" / f"rank_{rank}.jsonl"))
    rows.sort(key=lambda x: (x.get("model", ""), x.get("run_name", ""), x.get("dataset", ""), int(x.get("shot", -1)), x.get("task_type", ""), x.get("sample_id", "")))
    write_metric_outputs(metric_dir, metric_name, rows, metric_keys)
    return rows


def launch_metric_workers(args, metric_name: str, metric_keys: Sequence[str]) -> None:
    metric_dir = Path(args.metrics_root) / metric_name
    if args.overwrite:
        safe_rmtree(metric_dir)
    elif getattr(args, "skip_completed", False):
        # Imported lazily to avoid a common -> resume -> common import cycle.
        from .resume import prepare_resume_cache
        prepare_resume_cache(metric_dir, metric_name, metric_keys)
    (metric_dir / "shards").mkdir(parents=True, exist_ok=True)
    save_json(metric_dir / "run_config.json", vars(args))

    gpu_ids = parse_gpus(args.gpus, args.num_gpus)
    base_cli = [x for x in sys.argv[1:]]
    procs = []
    for rank, gpu in enumerate(gpu_ids):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        cmd = [
            sys.executable, "-m", "benchmarker_metrics.run",
            *base_cli,
            "--rank", str(rank),
            "--world_size", str(args.num_gpus),
            "--num_gpus", "1",
        ]
        log_path = metric_dir / "shards" / f"rank_{rank}.log"
        log_f = open(log_path, "w", encoding="utf-8")
        print(f"[launch] metric={metric_name}, rank={rank}, gpu={gpu}, log={log_path}")
        procs.append((rank, subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT), log_f))

    failed = []
    for rank, p, log_f in procs:
        ret = p.wait()
        log_f.close()
        if ret != 0:
            failed.append((rank, ret))
    merge_shards(metric_dir, args.num_gpus, metric_name, metric_keys)
    if failed:
        raise SystemExit(f"Some metric workers failed: {failed}. Check {metric_dir / 'shards'}/*.log")
