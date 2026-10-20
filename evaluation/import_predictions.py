#!/usr/bin/env python3
"""Import sample-ID-named 512px RGB PNG predictions into the metric workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from types import SimpleNamespace

from PIL import Image

from sample_subset import FROZEN_JSONL, read_frozen_source

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "benchmark_runner"))
from benchmark.data_paths import image_path  # noqa: E402
from benchmark.io_utils import copy_gt_image  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_image(path: Path, *, prediction: bool = False) -> None:
    with Image.open(path) as image:
        image.load()
        if prediction and (image.format != "PNG" or image.mode != "RGB" or image.size != (512, 512)):
            raise ValueError(f"Prediction must be a 512x512 RGB PNG without conversion: {path}")


def import_predictions(args):
    canonical = args.canonical_jsonl.expanduser().resolve()
    canonical_bytes, frozen, _ = read_frozen_source(canonical)
    input_path = args.input_jsonl.expanduser().resolve()
    input_bytes = input_path.read_bytes()
    rows = [json.loads(line) for line in input_bytes.splitlines()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("Input JSONL must contain frozen record objects")
    by_id = {row["sample_id"]: row for row in frozen}
    ids = [row.get("sample_id") for row in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("Input JSONL must contain unique frozen sample IDs")
    for row in rows:
        if row != by_id.get(row.get("sample_id")):
            raise ValueError(f"Input sample differs from the frozen benchmark: {row.get('sample_id')}")
    for value in (args.model, args.run_name):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
            raise ValueError("Model and run names must be plain directory IDs")
    if args.model == "all":
        raise ValueError("'all' is reserved for model selection; use a distinct --model ID")
    predictions = args.predictions_dir.expanduser().resolve()
    if not predictions.is_dir():
        raise ValueError(f"Prediction directory does not exist: {predictions}")
    expected = {f"{sample_id}.png" for sample_id in ids}
    actual = {path.name for path in predictions.iterdir()
              if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}}
    if actual != expected:
        raise ValueError(f"Prediction IDs do not match input: missing={sorted(expected - actual)[:5]}, "
                         f"extra={sorted(actual - expected)[:5]}")
    roots = SimpleNamespace(scannet_root=args.scannet_root, scannetpp_root=args.scannetpp_root,
                            arkitscenes_root=args.arkitscenes_root, data_split=args.data_split)
    for dataset in {row["dataset"] for row in rows}:
        value = getattr(roots, dataset + "_root")
        if not value or not Path(value).expanduser().is_dir():
            raise ValueError(f"Set {dataset.upper()}_ROOT or --{dataset}-root to an existing directory")
        setattr(roots, dataset + "_root", str(Path(value).expanduser().resolve()))
    generation_root = args.generation_root.expanduser().resolve()
    run_dir = generation_root / args.model / args.run_name
    if run_dir.exists():
        raise ValueError(f"Import destination exists; choose a new --run-name: {run_dir}")
    protected = [predictions, input_path, canonical, HERE / "data"]
    protected += [Path(getattr(roots, name + "_root")) for name in {row["dataset"] for row in rows}]
    if any(run_dir == path or path in run_dir.parents for path in protected):
        raise ValueError("Import destination must be separate from source images and metadata")

    planned = []
    validated = set()
    for row in rows:
        pred = predictions / f"{row['sample_id']}.png"
        checked_image(pred, prediction=True)
        contexts = [Path(image_path(roots, row["dataset"], row["scene_id"], frame))
                    for frame in row["contexts_frame_id"]]
        gt = Path(image_path(roots, row["dataset"], row["scene_id"], row["gt_frame_id"]))
        for source in contexts + [gt]:
            if source not in validated:
                checked_image(source)
                validated.add(source)
        planned.append((row, pred, digest(pred), contexts, gt))
    metadata = {
        "schema_version": "mindview-external-predictions-v1", "model": args.model,
        "run_name": args.run_name, "input_jsonl": str(input_path),
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "canonical_sha256": hashlib.sha256(canonical_bytes).hexdigest(),
        "selected_sample_ids": ids, "predictions_dir": str(predictions),
        "prediction_sha256": {row["sample_id"]: sha for row, _, sha, _, _ in planned},
    }
    if args.dry_run:
        print(json.dumps({**metadata, "run_dir": str(run_dir), "dry_run": True}, indent=2))
        return run_dir
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.run_name}.import-", dir=run_dir.parent))
    try:
        records = []
        for index, (row, pred, pred_sha, contexts, gt) in enumerate(planned):
            relative = Path(f"shot-{row['shot']}")
            pred_relative = relative / "pred" / row["dataset"] / row["task_type"] / pred.name
            gt_relative = relative / "gt" / row["dataset"] / row["task_type"] / f"{row['sample_id']}.jpg"
            (staging / pred_relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pred, staging / pred_relative)
            if digest(staging / pred_relative) != pred_sha:
                raise ValueError(f"Prediction changed during import: {pred}")
            copy_gt_image(str(gt), str(staging / gt_relative), 512)
            pred_path, gt_path = str(run_dir / pred_relative), str(run_dir / gt_relative)
            sources = list(map(str, contexts))
            records.append({**row, "model": args.model, "global_index": index,
                            "status": "success", "pred_image": pred_path, "gt_image": gt_path,
                            "prediction": {"image_path": pred_path},
                            "ground_truth": {"image_path": gt_path, "source_path": str(gt)},
                            "contexts": {"source_paths": sources}, "input": {"image_paths": sources},
                            "extra": {"external_prediction": True}})
        for name, value in (("generation_results.json", records), ("import_manifest.json", metadata)):
            (staging / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        if run_dir.exists():
            raise ValueError(f"Import destination was created during validation: {run_dir}")
        staging.rename(run_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps({"run_dir": str(run_dir), "model": args.model, "samples": len(rows)}))
    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=Path(os.environ.get("INPUT_JSON", FROZEN_JSONL)))
    parser.add_argument("--canonical-jsonl", type=Path, default=FROZEN_JSONL)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Method ID used by metrics/report, including custom names")
    parser.add_argument("--run-name", default="external")
    parser.add_argument("--generation-root", type=Path,
                        default=Path(os.environ.get("GENERATION_ROOT", HERE.parent / "outputs/generations")))
    for dataset in ("scannet", "scannetpp", "arkitscenes"):
        parser.add_argument("--" + dataset + "-root", default=os.environ.get(dataset.upper() + "_ROOT", ""))
    parser.add_argument("--data-split", default=os.environ.get("DATA_SPLIT", "val"))
    parser.add_argument("--dry-run", action="store_true", help="Validate all images and print the plan without writing")
    import_predictions(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"[Error] {exc}") from exc
