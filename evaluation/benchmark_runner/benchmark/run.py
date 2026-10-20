#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm
from PIL import Image

from .data_paths import image_path
from .io_utils import append_jsonl, copy_context_images, copy_gt_image, load_json_like, load_samples, preprocess_image, read_jsonl, safe_rmtree_file, save_json
from .models import MODEL_REGISTRY, build_model
from .tasks import get_task


def parse_gpus(gpus: str, num_gpus: int) -> List[str]:
    if gpus:
        ids = [x.strip() for x in gpus.split(",") if x.strip()]
    else:
        ids = [str(i) for i in range(num_gpus)]
    if len(ids) < num_gpus:
        raise ValueError(f"Need {num_gpus} GPUs, but got {ids}")
    return ids[:num_gpus]


def filter_samples(samples: List[Dict[str, Any]], task_filter: str, max_samples: int) -> List[Dict[str, Any]]:
    if task_filter and task_filter.lower() != "all":
        keep = {x.strip() for x in task_filter.split(",") if x.strip()}
        samples = [s for s in samples if s.get("task_type") in keep]
    if max_samples and max_samples > 0:
        samples = samples[:max_samples]
    return samples


def result_model_name(args) -> str:
    return getattr(args, "method_name", "") or args.model


def make_run_dir(args) -> Path:
    run_name = args.run_name or f"{args.model}_{Path(args.input_json).stem}"
    return Path(args.output_root) / result_model_name(args) / run_name


def generation_signature(args) -> Dict[str, Any]:
    """Sign input/configuration; inspect local weight metadata without reading GBs."""
    operational = {"input_json", "output_root", "run_name", "num_gpus", "gpus", "num_workers",
                   "rank", "world_size", "overwrite", "skip_completed", "device",
                   "offload_folder", "max_mem_per_gpu", "cpu_offload"}
    config = {key: value for key, value in vars(args).items() if key not in operational}
    resources = {}
    for key in ("model_path", "repo_path", "bagel_checkpoint_path", "emu_vq_path",
                "emu_tokenizer_path", "dreamomni2_flux_path"):
        value = getattr(args, key, "")
        if not value:
            continue
        path = Path(value).expanduser()
        if path.is_file():
            info = path.stat()
            resources[key] = {"path": str(path.resolve()), "size": info.st_size, "mtime_ns": info.st_mtime_ns}
        elif path.is_dir():
            # Immediate config/weight files and sharded weights in subdirectories.
            files = sorted(p for p in path.rglob("*") if p.is_file()
                           and p.suffix in {".safetensors", ".bin", ".json", ".model", ".py"}
                           and not any(part in {".git", "__pycache__"} for part in p.relative_to(path).parts))
            resources[key] = {"path": str(path.resolve()), "files": [
                [str(p.relative_to(path)), p.stat().st_size, p.stat().st_mtime_ns] for p in files
            ]}
    package = Path(__file__).resolve().parent
    implementation = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.glob("*.py"))}
    payload = {"schema": "generation-config-v1", "config": config, "resources": resources,
               "input_sha256": hashlib.sha256(Path(args.input_json).read_bytes()).hexdigest(),
               "implementation": implementation}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"sha256": digest, "payload": payload}


def prepare_run(args, run_dir: Path) -> None:
    signature = generation_signature(args)
    if args.overwrite:
        safe_rmtree_file(run_dir)
    elif run_dir.exists() and any(run_dir.iterdir()):
        config_path = run_dir / "run_config.json"
        previous = load_json_like(str(config_path)) if config_path.is_file() else {}
        if not isinstance(previous, dict) or previous.get("generation_signature", {}).get("sha256") != signature["sha256"]:
            raise ValueError(
                f"Generation configuration or input changed for {run_dir}. "
                "Use a new output directory/run name, or --overwrite to regenerate. "
                "Unsigned legacy outputs cannot be resumed safely."
            )
    (run_dir / "shards").mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "run_config.json", {**vars(args), "generation_signature": signature})


def make_base_record(sample: Dict[str, Any], args) -> Dict[str, Any]:
    task_type = sample.get("task_type", "default")
    return {
        "sample_id": sample.get("sample_id"),
        "dataset": sample.get("dataset"),
        "task_type": task_type,
        "model": result_model_name(args),
        "scene_id": sample.get("scene_id"),
        "gt_frame_id": str(sample.get("gt_frame_id")),
        "anchor_frame_id": str(sample.get("anchor_frame_id", "")),
        "shot": sample.get("shot"),
        "contexts_frame_id": [str(x) for x in sample.get("contexts_frame_id", [])],
        "instruction": sample.get("instruction", ""),
        "global_index": sample.get("_global_index"),
        "rank": args.rank,
    }


def expected_paths(sample: Dict[str, Any], args, run_dir: Path) -> Dict[str, Any]:
    task_type = sample.get("task_type", "default")
    sample_id = sample.get("sample_id")
    dataset = sample.get("dataset")
    scene_id = sample.get("scene_id")
    shot_dir = f"shot-{sample.get('shot', 0)}"

    context_source_paths = [
        image_path(args, dataset, scene_id, fid)
        for fid in sample.get("contexts_frame_id", [])
    ]
    gt_src_path = image_path(args, dataset, scene_id, sample.get("gt_frame_id"))

    return {
        "pred_path": str(run_dir / shot_dir / "pred" / dataset / task_type / f"{sample_id}.png"),
        "gt_save_path": str(run_dir / shot_dir / "gt" / dataset / task_type / f"{sample_id}.jpg"),
        "context_vis_path": str(run_dir / shot_dir / "contexts" / dataset / task_type / f"{sample_id}.jpg"),
        "context_source_paths": context_source_paths,
        "gt_src_path": gt_src_path,
    }


def valid_image_file(path: Optional[str]) -> bool:
    if not path:
        return False
    p = Path(path)
    if not p.exists() or not p.is_file() or p.stat().st_size <= 0:
        return False
    try:
        with Image.open(p) as img:
            img.verify()
        return True
    except Exception:
        return False


def sample_is_complete(sample: Dict[str, Any], args, run_dir: Path) -> bool:
    paths = expected_paths(sample, args, run_dir)
    required = [paths["pred_path"], paths["context_vis_path"]]
    if args.save_gt:
        required.append(paths["gt_save_path"])
    return all(valid_image_file(x) for x in required)


def build_skipped_record(sample: Dict[str, Any], args, run_dir: Path) -> Dict[str, Any]:
    paths = expected_paths(sample, args, run_dir)
    base_record = make_base_record(sample, args)
    context_images = [paths["context_vis_path"]]
    gt_image = paths["gt_save_path"] if args.save_gt else None
    pred_image = paths["pred_path"]
    return {
        **base_record,
        "input": {
            "image_paths": paths["context_source_paths"],
            "context_image_paths": context_images,
        },
        "contexts": {
            "image_paths": context_images,
            "source_paths": paths["context_source_paths"],
        },
        "prediction": {"image_path": pred_image},
        "ground_truth": {"image_path": gt_image, "source_path": paths["gt_src_path"]},
        "pred_image": pred_image,
        "gt_image": gt_image,
        "context_images": context_images,
        "status": "success",
        "time_sec": 0.0,
        "extra": {"skipped_completed": True},
    }


def launch_workers(args) -> None:
    run_dir = make_run_dir(args)
    prepare_run(args, run_dir)

    if args.num_gpus > 0:
        if args.num_workers > args.num_gpus:
            raise ValueError(
                f"num_workers={args.num_workers} exceeds num_gpus={args.num_gpus}; "
                "use --num_gpus 0 for CPU/API workers"
            )
        worker_devices: List[Optional[str]] = parse_gpus(args.gpus, args.num_gpus)[:args.num_workers]
    else:
        worker_devices = [None] * args.num_workers

    pending_plan_path = None
    if args.skip_completed and not args.overwrite:
        # Take one completion snapshot so every worker shards the same pending list.
        samples = filter_samples(load_samples(args.input_json), args.task, args.max_samples)
        pending_indices = [
            i for i, sample in enumerate(samples)
            if not sample_is_complete(sample, args, run_dir)
        ]
        pending_plan_path = run_dir / "shards" / "pending_indices.json"
        save_json(pending_plan_path, pending_indices)

    base_cli = [x for x in sys.argv[1:]]
    procs = []
    for rank, gpu in enumerate(worker_devices):
        env = os.environ.copy()
        if pending_plan_path is not None:
            env["BENCHMARK_PENDING_PLAN"] = str(pending_plan_path.resolve())
        if gpu is None:
            device_label = "cpu"
        else:
            env["CUDA_VISIBLE_DEVICES"] = gpu
            device_label = f"gpu={gpu}"
        cmd = [
            sys.executable, "-m", "benchmark.run",
            *base_cli,
            "--rank", str(rank),
            "--world_size", str(args.num_workers),
            "--num_gpus", "0" if gpu is None else "1",
        ]
        log_path = run_dir / "shards" / f"rank_{rank}.log"
        log_f = open(log_path, "w", encoding="utf-8")
        print(f"[launch] rank={rank}, {device_label}, log={log_path}")
        procs.append((rank, subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT), log_f))

    failed = []
    for rank, p, log_f in procs:
        ret = p.wait()
        log_f.close()
        if ret != 0:
            failed.append((rank, ret))

    summary = merge_results(run_dir, args.num_workers)
    if failed:
        raise SystemExit(f"Some workers failed: {failed}. Check {run_dir / 'shards'}/*.log")
    if summary["num_failed"]:
        raise SystemExit(f"Generation failed for {summary['num_failed']} samples; see {run_dir / 'generation_results.json'}")


def save_prediction(img, path: str, save_image_size: int) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = img.convert("RGB")
    if save_image_size and save_image_size > 0:
        img = preprocess_image(img, save_image_size)
    img.save(path)
    return str(path)


def run_worker(args) -> None:
    run_dir = make_run_dir(args)
    shard_path = run_dir / "shards" / f"rank_{args.rank}.jsonl"

    # Always rebuild the current shard manifest. This never deletes generated images;
    # it only prevents duplicated JSONL records when resuming an interrupted run.
    if shard_path.exists():
        shard_path.unlink()

    samples = filter_samples(load_samples(args.input_json), args.task, args.max_samples)
    indexed = [(i, s) for i, s in enumerate(samples) if i % args.world_size == args.rank]

    pending = []
    skipped = 0
    if args.skip_completed and not args.overwrite:
        pending_plan_path = os.environ.get("BENCHMARK_PENDING_PLAN")
        if pending_plan_path:
            pending_indices = load_json_like(pending_plan_path)
            pending_index_set = set(pending_indices)
            for i, sample in indexed:
                if i not in pending_index_set:
                    append_jsonl(shard_path, build_skipped_record(sample, args, run_dir))
                    skipped += 1
            pending = [
                (sample_index, samples[sample_index])
                for pending_index, sample_index in enumerate(pending_indices)
                if pending_index % args.world_size == args.rank
            ]
        else:
            for item in indexed:
                _, sample = item
                if sample_is_complete(sample, args, run_dir):
                    append_jsonl(shard_path, build_skipped_record(sample, args, run_dir))
                    skipped += 1
                else:
                    pending.append(item)
    else:
        pending = indexed

    print(
        f"[rank {args.rank}] total={len(indexed)}, "
        f"skipped_completed={skipped}, pending={len(pending)}"
    )
    if not pending:
        print(f"[rank {args.rank}] no pending samples; model loading is skipped.")
        return

    model = build_model(args.model, args)

    for _, sample in tqdm(pending, desc=f"rank {args.rank}"):
        t0 = time.time()
        base_record = make_base_record(sample, args)
        try:
            task_type = sample.get("task_type", "default")
            task = get_task(task_type, args)
            req = task.build_request(sample, run_dir)

            gt_image = None
            if args.save_gt:
                gt_image = copy_gt_image(req.gt_src_path, req.gt_save_path, args.save_image_size)
            context_images = copy_context_images(
                src_paths=req.image_paths,
                frame_ids=[str(x) for x in sample.get("contexts_frame_id", [])],
                dst_dir=req.context_save_dir,
                image_size=args.save_image_size,
                sample_id=sample["sample_id"],
            )

            if valid_image_file(req.pred_path) and not args.overwrite:
                pred_image = req.pred_path
                extra = {"skipped_existing_pred": True}
            else:
                img, extra = model.infer(req)
                pred_image = save_prediction(img, req.pred_path, args.save_image_size)

            record = {
                **base_record,
                "input": {
                    "image_paths": req.image_paths,
                    "context_image_paths": context_images,
                },
                "contexts": {
                    "image_paths": context_images,
                    "source_paths": req.image_paths,
                },
                "prediction": {"image_path": pred_image},
                "ground_truth": {"image_path": gt_image, "source_path": req.gt_src_path},
                "pred_image": pred_image,
                "gt_image": gt_image,
                "context_images": context_images,
                "status": "success",
                "time_sec": round(time.time() - t0, 4),
                "extra": extra,
            }
        except Exception as e:
            record = {
                **base_record,
                "input": {"image_paths": []},
                "contexts": {"image_paths": [], "source_paths": []},
                "prediction": {"image_path": None},
                "ground_truth": {"image_path": None},
                "pred_image": None,
                "gt_image": None,
                "context_images": [],
                "status": "failed",
                "error": repr(e),
                "time_sec": round(time.time() - t0, 4),
            }
            print(f"[rank {args.rank}][error] {sample.get('sample_id')}: {repr(e)}")
        append_jsonl(shard_path, record)


def merge_results(run_dir: Path, world_size: int) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    for rank in range(world_size):
        records.extend(read_jsonl(run_dir / "shards" / f"rank_{rank}.jsonl"))
    records.sort(key=lambda x: (x.get("global_index") if x.get("global_index") is not None else 10**12, x.get("sample_id", "")))

    save_json(run_dir / "generation_results.json", records)

    by_task = defaultdict(list)
    by_dataset = defaultdict(list)
    by_dataset_task = defaultdict(lambda: defaultdict(list))
    by_shot = defaultdict(list)
    for r in records:
        task = r.get("task_type", "default")
        dataset = r.get("dataset", "unknown")
        shot = str(r.get("shot", "unknown"))
        by_task[task].append(r)
        by_dataset[dataset].append(r)
        by_dataset_task[dataset][task].append(r)
        by_shot[shot].append(r)
    task_dir = run_dir / "generation_results_by_task"
    for task, rs in by_task.items():
        save_json(task_dir / f"{task}.json", rs)

    dataset_dir = run_dir / "generation_results_by_dataset"
    dataset_task_dir = run_dir / "generation_results_by_dataset_task"
    for dataset, rs in by_dataset.items():
        save_json(dataset_dir / f"{dataset}.json", rs)
        for task, task_records in by_dataset_task[dataset].items():
            save_json(dataset_task_dir / dataset / f"{task}.json", task_records)

    summary = {
        "num_samples": len(records),
        "num_success": sum(r.get("status") == "success" for r in records),
        "num_failed": sum(r.get("status") == "failed" for r in records),
        "by_dataset": {k: len(v) for k, v in sorted(by_dataset.items())},
        "by_task": {k: len(v) for k, v in sorted(by_task.items())},
        "by_shot": {k: len(v) for k, v in sorted(by_shot.items(), key=lambda x: int(x[0]))},
        "by_dataset_task": {
            dataset: {task: len(rs) for task, rs in sorted(tasks.items())}
            for dataset, tasks in sorted(by_dataset_task.items())
        },
    }
    save_json(run_dir / "summary.json", summary)
    print(f"[merge] saved: {run_dir / 'generation_results.json'}")
    print(f"[merge] summary: {summary}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--input_json", required=True)
    p.add_argument("--scannet_root", default="", help="ScanNet root: <root>/<scene>/color/<frame>.jpg")
    p.add_argument("--scannetpp_root", default="", help="ScanNet++ processed-data root")
    p.add_argument("--arkitscenes_root", default="", help="ARKitScenes processed-data root")
    p.add_argument("--data_split", default="val", help="Split directory used by ScanNet++ and ARKitScenes")
    p.add_argument("--scan_root", default="", help=argparse.SUPPRESS)
    p.add_argument("--output_root", default=str(Path(__file__).resolve().parents[3] / "outputs/generations"))
    p.add_argument("--run_name", default=None)
    p.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    p.add_argument(
        "--method_name",
        default="",
        help="Optional result label used for the output directory and records; model loading still uses --model.",
    )
    p.add_argument("--task", default="all", help="all or comma-separated task_type names")
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--gpus", default="", help="e.g. 0,1,2,3; empty means 0..num_gpus-1")
    p.add_argument("--num_workers", type=int, default=None, help="Parallel sample workers; defaults to num_gpus, or 1 when num_gpus=0")
    p.add_argument("--rank", type=int, default=-1)
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip_completed", dest="skip_completed", action="store_true", default=False, help="Skip samples whose pred/gt/context images already exist and are readable.")
    p.add_argument("--no_skip_completed", dest="skip_completed", action="store_false")
    p.add_argument("--save_gt", dest="save_gt", action="store_true", default=True)
    p.add_argument("--no_save_gt", dest="save_gt", action="store_false")
    p.add_argument(
        "--allow_variable_contexts",
        action="store_true",
        help=(
            "Allow logical shot-10 samples to contain 6--10 context images. "
            "Disabled by default."
        ),
    )

    # Common image/model args
    p.add_argument("--input_image_size", type=int, default=512)
    p.add_argument("--save_image_size", type=int, default=512, help=">0 resizes saved predictions; <=0 keeps raw model size")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_path", default="")
    p.add_argument("--repo_path", default="", help="Optional repo path for local model imports, e.g. BAGEL repo")

    # FLUX args
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--cpu_offload", action="store_true", default=False)
    p.add_argument("--generator_device", default="cuda")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--num_inference_steps", type=int, default=4)

    # Qwen-Image-Edit-2511 args. Kept separate from the FLUX2 arguments above
    # so registering Qwen never changes another model's defaults.
    p.add_argument("--qwen_num_inference_steps", type=int, default=40)

    # Isolated local-model adapters.
    p.add_argument("--emu_vq_path", default="")
    p.add_argument("--emu_tokenizer_path", default="")
    p.add_argument("--dreamomni2_flux_path", default="")

    # API args
    p.add_argument("--base_url", default="")
    p.add_argument("--api_key_env", default="OPENAI_API_KEY")
    p.add_argument("--model_name", default="gemini-3-pro-image-preview")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--output_size", default="1024x1024")
    p.add_argument("--quality", default="auto", choices=["auto", "low", "medium", "high", "standard"])
    p.add_argument("--output_format", default="png", choices=["png", "jpeg", "webp"])
    p.add_argument("--input_fidelity", choices=["low", "high"], default=None)

    # BAGEL args
    p.add_argument("--bagel_checkpoint_path", default="", help="Optional BAGEL weight file; empty uses <model_path>/ema.safetensors")
    p.add_argument("--max_mem_per_gpu", default="80GiB")
    p.add_argument("--offload_folder", default="/tmp/offload")
    p.add_argument("--cfg_text_scale", type=float, default=4.0)
    p.add_argument("--cfg_img_scale", type=float, default=2.0)
    p.add_argument("--cfg_interval_start", type=float, default=0.0)
    p.add_argument("--cfg_interval_end", type=float, default=1.0)
    p.add_argument("--timestep_shift", type=float, default=3.0)
    p.add_argument("--num_timesteps", type=int, default=50)
    p.add_argument("--cfg_renorm_min", type=float, default=0.0)
    p.add_argument("--cfg_renorm_type", default="text_channel")
    p.add_argument("--bagel_thinking", type=int, choices=[0, 1], default=1)
    p.add_argument("--sensenova_u15_thinking", type=int, choices=[0, 1], default=0)
    p.add_argument("--sensenova_u1_thinking", type=int, choices=[0, 1], default=0)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.num_gpus < 0:
        parser.error("--num_gpus must be >= 0")
    if args.num_workers is None:
        args.num_workers = args.num_gpus if args.num_gpus > 0 else 1
    if args.num_workers <= 0:
        parser.error("--num_workers must be > 0")
    if args.num_gpus == 0:
        if args.device == "cuda":
            args.device = "cpu"

    if args.rank < 0 and args.num_workers > 1:
        launch_workers(args)
        return
    if args.rank < 0:
        args.rank = 0
        args.world_size = 1
        run_dir = make_run_dir(args)
        prepare_run(args, run_dir)
    run_worker(args)
    if args.world_size == 1:
        summary = merge_results(make_run_dir(args), 1)
        if summary["num_failed"]:
            raise SystemExit(f"Generation failed for {summary['num_failed']} samples; see generation_results.json")


if __name__ == "__main__":
    main()
