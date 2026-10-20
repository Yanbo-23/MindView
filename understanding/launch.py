"""Run the three fixed BAGEL-SFT understanding protocols sequentially."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "understanding"
BENCHMARKS = ("sat-v2", "mmsi", "mindcube-tiny")
ITEMS = {"sat-v2": 300, "mmsi": 1000, "mindcube-tiny": 1050}
SOURCES = {"sat-v2": 150, "mmsi": 1000, "mindcube-tiny": 1050}
MAX_TOKENS = {"sat-v2": 1024, "mmsi": 2048, "mindcube-tiny": 4096}


def env(name: str, default=None):
    return os.environ.get(name) or default


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("benchmark", nargs="?", choices=("all", *BENCHMARKS), default="all")
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print commands without reading data/weights, downloading, or writing")
    mode.add_argument("--prepare-only", action="store_true", help="Validate data on CPU; do not load model weights or create results")
    result.add_argument("--allow-download", action="store_true", help="Allow the pinned SAT/MMSI downloads when no local parquet is supplied")
    result.add_argument("--model-path", default=env("BAGEL_MODEL_PATH", str(ROOT / "models/BAGEL-7B-MoT")))
    result.add_argument("--checkpoint-path", default=env("BAGEL_SFT_FINAL_CHECKPOINT"), help="Full BAGEL-SFT EMA; environment: BAGEL_SFT_FINAL_CHECKPOINT")
    result.add_argument("--output-root", default=env("UNDERSTANDING_OUTPUT_ROOT", str(Path(env("OUTPUTS_ROOT", str(ROOT / "outputs"))) / "understanding")))
    result.add_argument("--sat-parquet", default=env("SAT_V2_PARQUET", ""))
    result.add_argument("--mmsi-parquet", default=env("MMSI_PARQUET", ""))
    result.add_argument("--mindcube-data-root", default=env("MINDCUBE_DATA_ROOT", str(ROOT / "resources/understanding/MindCube/data")))
    result.add_argument("--limit", type=positive, help="Explicit partial run: cap source questions before SAT's two-order expansion")
    result.add_argument("--nproc-per-node", type=positive, default=env("UNDERSTANDING_NPROC_PER_NODE", "1"))
    result.add_argument("--gpus", default=env("CUDA_VISIBLE_DEVICES", env("GPU_LIST")), help="Optional CUDA_VISIBLE_DEVICES override; default one process")
    return result


def results_root(args: argparse.Namespace) -> Path:
    path = Path(args.output_root).expanduser().resolve()
    return path / "partial" / f"limit_{args.limit}" if args.limit is not None else path


def build_plan(args: argparse.Namespace) -> list[dict]:
    selected = BENCHMARKS if args.benchmark == "all" else (args.benchmark,)
    plan = []
    for name in selected:
        command = [sys.executable]
        if not args.prepare_only and args.nproc_per_node > 1:
            command += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={args.nproc_per_node}"]
        command += [str(HERE / "eval_spatial_benchmarks.py"), "--benchmark", name,
                    "--model-path", args.model_path, "--output-dir", str(results_root(args) / name)]
        # A dry run can show the required checkpoint variable without requiring
        # any locally installed weight file or reading its header.
        if not args.prepare_only:
            command += ["--checkpoint-path", args.checkpoint_path or "<BAGEL_SFT_FINAL_CHECKPOINT>"]
        if args.prepare_only:
            command += ["--prepare-only"]
        if args.limit is not None:
            command += ["--limit", str(args.limit)]
        if name == "sat-v2" and args.sat_parquet:
            command += ["--sat-parquet", args.sat_parquet]
        elif name == "mmsi" and args.mmsi_parquet:
            command += ["--mmsi-parquet", args.mmsi_parquet]
        elif name == "mindcube-tiny":
            command += ["--mindcube-data-root", args.mindcube_data_root]
        if name != "mindcube-tiny":
            command += ["--allow-download"] if args.allow_download else ["--offline"]
        selected_sources = min(args.limit, SOURCES[name]) if args.limit else SOURCES[name]
        plan.append({"benchmark": name, "source_questions": selected_sources,
                     "evaluation_items": selected_sources * (2 if name == "sat-v2" else 1),
                     "full_evaluation_items": ITEMS[name], "max_new_tokens": MAX_TOKENS[name],
                     "partial": args.limit is not None, "command": command})
    return plan


def preflight(args: argparse.Namespace) -> None:
    names = BENCHMARKS if args.benchmark == "all" else (args.benchmark,)
    if args.gpus is not None:
        devices = args.gpus.split(",")
        if (len(devices) < args.nproc_per_node or len(devices) != len(set(devices))
                or any(not item.strip() for item in devices)):
            raise ValueError("--gpus must provide distinct devices for every process")
    for name, location, variable in (("sat-v2", args.sat_parquet, "SAT_V2_PARQUET"), ("mmsi", args.mmsi_parquet, "MMSI_PARQUET")):
        if name not in names:
            continue
        if location:
            if not Path(location).expanduser().is_file():
                raise ValueError(f"Missing {variable} file: {location}")
        elif not args.allow_download:
            raise ValueError(f"Set {variable} to the pinned local parquet, or explicitly pass --allow-download")
    if "mindcube-tiny" in names and not Path(args.mindcube_data_root).expanduser().is_dir():
        raise ValueError(f"Missing MINDCUBE_DATA_ROOT: {args.mindcube_data_root}")
    if not args.prepare_only:
        if not args.checkpoint_path:
            raise ValueError("Set BAGEL_SFT_FINAL_CHECKPOINT or --checkpoint-path")
        model = Path(args.model_path).expanduser().resolve()
        checkpoint = Path(args.checkpoint_path).expanduser().resolve()
        if not model.is_dir() or not checkpoint.is_file():
            raise ValueError("BAGEL_MODEL_PATH and BAGEL_SFT_FINAL_CHECKPOINT must exist")
        output = results_root(args)
        if output == Path(output.anchor) or output == ROOT:
            raise ValueError("Use a dedicated understanding output directory")
        for protected in (HERE, ROOT / "sft", model, checkpoint.parent):
            if output == protected or output in protected.parents or protected in output.parents:
                raise ValueError(f"Understanding output overlaps protected input: {protected}")


def execution_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if args.gpus is not None:
        environment["CUDA_VISIBLE_DEVICES"] = args.gpus
    if args.prepare_only or args.nproc_per_node == 1:
        # A previous torchrun context must not turn a serial subprocess into
        # a distributed rank waiting for workers this launcher did not start.
        environment.update(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0", LOCAL_WORLD_SIZE="1")
    return environment


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "model": "bagel-sft-final", "partial": args.limit is not None,
                          "output_root": str(results_root(args)), "do_sample": False, "thinking": False,
                          "parameter_dtype": "bfloat16", "autocast_dtype": "bfloat16", "plan": plan}, indent=2))
        for item in plan:
            print("Command:", shlex.join(item["command"]))
        return 0
    try:
        preflight(args)
        environment = execution_environment(args)
        for item in plan:
            result = subprocess.run(item["command"], env=environment, cwd=ROOT, check=False)
            if not args.prepare_only:
                if str(ROOT) not in sys.path:
                    sys.path.insert(0, str(ROOT))
                from understanding.summarize import summarize
                summary = summarize(results_root(args), limit=args.limit)
                print(f"Understanding summary: {summary['status']} — {results_root(args) / 'summary.md'}", flush=True)
                if not result.returncode and summary["benchmarks"][item["benchmark"]]["status"] not in {"complete", "partial"}:
                    return 2
            if result.returncode:
                return result.returncode
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Understanding evaluation failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
