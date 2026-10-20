"""Validate and launch the MindView BAGEL SFT recipe without local setup hooks."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import tempfile

SFT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SFT_ROOT.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(SFT_ROOT))
from data.mindview_schema import MindViewDataError, MindViewSchema  # noqa: E402

RELEASE_ROWS = 20000
RELEASE_SHA256 = "24d973dc5b2f967d105a6f2a55f46d8f2b0c2d5b262dbd785a60e10ee84627d5"
MODEL_ASSETS = (
    "llm_config.json", "vit_config.json", "ae.safetensors", "ema.safetensors",
    "tokenizer_config.json", "vocab.json", "merges.txt",
)


def env(name, default=None):
    return os.environ.get(name) or default


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def parser():
    result = argparse.ArgumentParser(
        description="MindView shot-6 BAGEL SFT; defaults reproduce the 20K/10K-step recipe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    outputs = Path(env("OUTPUTS_ROOT", str(REPO_ROOT / "outputs")))
    result.add_argument("--training-jsonl", default=env("TRAINING_JSONL", env("MINDVIEW_SNAPSHOT")),
                        help="Training snapshot; environment: TRAINING_JSONL")
    result.add_argument("--model-path", default=env("BAGEL_MODEL_PATH", str(REPO_ROOT / "models/BAGEL-7B-MoT")),
                        help="Official BAGEL model assets; environment: BAGEL_MODEL_PATH")
    result.add_argument("--output-root", default=env("SFT_OUTPUT_ROOT", str(outputs / "sft")),
                        help="Persistent run directory; environment: SFT_OUTPUT_ROOT")
    result.add_argument("--dataset-config", default=str(SFT_ROOT / "data/configs/mindview_sft.yaml"))
    result.add_argument("--benchmark-jsonl", default=env("BENCHMARK_JSONL", str(REPO_ROOT / "evaluation/data/merged_dataset_final.jsonl")),
                        help="Benchmark records used to reject overlapping training scenes")
    for dataset in ("scannet", "scannetpp", "arkitscenes"):
        result.add_argument(f"--{dataset}-root", default=env(f"{dataset.upper()}_ROOT"),
                            help=f"Original RGB root; environment: {dataset.upper()}_ROOT")
    result.add_argument("--expected-rows", type=positive, default=env("SFT_EXPECTED_ROWS", str(RELEASE_ROWS)),
                        help="Expected unique shot-6 records; override explicitly for custom data")
    result.add_argument("--snapshot-sha256", default=env("SFT_SNAPSHOT_SHA256", RELEASE_SHA256),
                        help="Expected snapshot SHA256; use 'auto' to bind a custom snapshot")
    result.add_argument("--gpus", default=env("GPU_LIST"), help="Comma-separated CUDA_VISIBLE_DEVICES override")
    result.add_argument("--nproc-per-node", type=positive, default=env("SFT_NPROC_PER_NODE", "8"))
    result.add_argument("--nnodes", type=positive, default=env("SFT_NNODES", "1"))
    result.add_argument("--node-rank", type=nonnegative, default=env("SFT_NODE_RANK", "0"))
    result.add_argument("--master-addr", default=env("MASTER_ADDR", "127.0.0.1"))
    result.add_argument("--master-port", type=positive, default=env("MASTER_PORT", "29500"))
    result.add_argument("--total-steps", type=positive, default=env("SFT_TOTAL_STEPS", "10000"))
    result.add_argument("--warmup-steps", type=nonnegative, default=env("SFT_WARMUP_STEPS", "200"))
    result.add_argument("--save-every", type=positive, default=env("SFT_SAVE_EVERY", "2000"))
    result.add_argument("--gradient-accumulation-steps", type=positive,
                        default=env("SFT_GRADIENT_ACCUMULATION_STEPS", "1"))
    result.add_argument("--num-workers", type=positive, default=env("SFT_NUM_WORKERS", "1"))
    result.add_argument("--lr", type=float, default=env("SFT_LR", "1e-5"))
    result.add_argument("--min-lr", type=float, default=env("SFT_MIN_LR", "1e-6"))
    result.add_argument("--sharding-strategy", choices=("FULL_SHARD", "HYBRID_SHARD"), default="FULL_SHARD")
    result.add_argument("--num-shard", type=positive, default=1)
    result.add_argument("--num-replicate", type=positive, default=1)
    result.add_argument("--min-free-gib", type=nonnegative, default=env("SFT_MIN_FREE_GIB", "1300"))
    result.add_argument("--check-images", action="store_true", help="Check every referenced RGB file exists")
    result.add_argument("--dry-run", action="store_true", help="Validate and print effective config; write nothing and use no GPU")
    return result


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def path(value):
    return Path(os.path.expandvars(os.path.expanduser(os.fspath(value)))).resolve()


def overlaps(left, right):
    return left == right or left in right.parents or right in left.parents


def prepare(args):
    if not args.training_jsonl:
        raise ValueError("Set TRAINING_JSONL or --training-jsonl")
    if args.node_rank >= args.nnodes:
        raise ValueError("node-rank must be less than nnodes")
    if args.master_port > 65535:
        raise ValueError("master-port must be at most 65535")
    if not 0 < args.min_lr <= args.lr:
        raise ValueError("Learning rates must satisfy 0 < min-lr <= lr")
    world_size = args.nnodes * args.nproc_per_node
    if args.sharding_strategy == "HYBRID_SHARD" and args.num_shard * args.num_replicate != world_size:
        raise ValueError("HYBRID_SHARD requires num-shard * num-replicate == world size")
    gpus = args.gpus.split(",") if args.gpus is not None else None
    if gpus is not None and (len(gpus) != args.nproc_per_node or len(set(gpus)) != len(gpus)
                             or any(not gpu.strip() for gpu in gpus)):
        raise ValueError("--gpus must list one distinct device per local process")
    snapshot, model, output, source_config = map(path, (
        args.training_jsonl, args.model_path, args.output_root, args.dataset_config,
    ))
    benchmark = path(args.benchmark_jsonl)
    if not benchmark.is_file():
        raise ValueError(f"Missing benchmark JSONL for training-scene isolation: {benchmark}")
    benchmark_scenes = set()
    with benchmark.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank benchmark JSONL line: {benchmark}:{number}")
            row = json.loads(line)
            if not isinstance(row, dict) or not all(isinstance(row.get(key), str) and row[key] for key in ("dataset", "scene_id")):
                raise ValueError(f"Invalid benchmark scene identity: {benchmark}:{number}")
            benchmark_scenes.add((row["dataset"], row["scene_id"]))
    if not benchmark_scenes:
        raise ValueError("Benchmark JSONL must contain scenes for isolation validation")
    for asset in MODEL_ASSETS:
        item = model / asset
        if not item.is_file() or not item.stat().st_size:
            raise ValueError(f"Missing or empty BAGEL model asset: {item}")
    if not snapshot.is_file():
        raise ValueError(f"Missing training JSONL: {snapshot}")
    roots = {}
    for dataset in ("scannet", "scannetpp", "arkitscenes"):
        value = getattr(args, f"{dataset}_root")
        if value:
            root = path(value)
            if not root.is_dir():
                raise ValueError(f"Missing {dataset} image root: {root}")
            roots[dataset] = root
    schema = MindViewSchema()
    schema.image_roots = roots
    uids, lineages, scene_ids, datasets = set(), set(), set(), Counter()
    digest = hashlib.sha256()
    with snapshot.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            if not raw.strip():
                raise ValueError(f"Blank JSONL line: {snapshot}:{line_number}")
            row = json.loads(raw)
            schema._validate_row(row, snapshot, line_number)
            if row["shot"] != 6:
                raise ValueError(f"SFT requires shot=6: {snapshot}:{line_number}")
            lineage = (row["dataset"], row["source_family"], row["source_sample_id"], row["shot"])
            if row["record_uid"] in uids or lineage in lineages:
                raise ValueError(f"Duplicate UID or source lineage: {snapshot}:{line_number}")
            uids.add(row["record_uid"])
            lineages.add(lineage)
            scene_ids.add((row["dataset"], row["scene_id"]))
            datasets[row["dataset"]] += 1
            if args.check_images:
                root = roots[row["dataset"]]
                for relative in row["contexts"]["6"] + [row["target_image"]]:
                    image = (root / relative).resolve()
                    if root not in image.parents or not image.is_file():
                        raise ValueError(f"Missing RGB file or path outside dataset root: {image}")
    if len(uids) != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} rows, found {len(uids)}")
    if len(uids) < world_size * args.num_workers:
        raise ValueError("Snapshot must provide at least one record per rank and data worker")
    shared_scenes = scene_ids & benchmark_scenes
    if shared_scenes:
        raise ValueError(f"Training scenes overlap the benchmark: {sorted(shared_scenes)[:10]}")
    snapshot_digest = digest.hexdigest()
    if args.snapshot_sha256 != "auto" and snapshot_digest != args.snapshot_sha256:
        raise ValueError("Snapshot SHA256 mismatch; explicitly use --snapshot-sha256 auto for custom data")
    if output == Path(output.anchor) or output == REPO_ROOT:
        raise ValueError("Output directory must be a dedicated run directory")
    for protected in (SFT_ROOT, model, snapshot, benchmark, source_config, *roots.values()):
        if overlaps(output, protected):
            raise ValueError(f"Output directory overlaps protected input: {protected}")
    import yaml
    dataset_config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    if not isinstance(dataset_config, dict) or set(dataset_config) != {"mindview_sft"}:
        raise ValueError("Dataset config must contain only the mindview_sft group")
    config = dataset_config["mindview_sft"]
    if config.get("dataset_names") != ["mindview_shot6_20k"]:
        raise ValueError("Dataset config must use the MindView snapshot registry")
    config.update(image_roots={key: str(value) for key, value in roots.items()},
                  expected_manifest_sha256=snapshot_digest, num_used_data=[len(uids)])
    if (config.get("strict_schema") is not True or config.get("enable_condition_dropout") is not False
            or config.get("expected_schema_version") != schema.expected_schema_version
            or config.get("expected_preprocess_id") != schema.expected_preprocess_id):
        raise ValueError("Dataset config changed the strict schema, preprocessing, or no-dropout contract")
    expected_transforms = {
        "image_transform_args": dict(image_stride=16, max_image_size=512, min_image_size=512, max_pixels=262144),
        "vit_image_transform_args": dict(image_stride=14, max_image_size=518, min_image_size=518, max_pixels=268324),
    }
    if any(config.get(name) != value for name, value in expected_transforms.items()):
        raise ValueError("Dataset config changed the release VAE/ViT image transforms")
    runtime_config = output / "mindview_sft.json"
    command = [sys.executable, "-m", "torch.distributed.run"]
    if args.nnodes == 1:
        command += ["--standalone", f"--nproc_per_node={args.nproc_per_node}"]
    else:
        command += [f"--nnodes={args.nnodes}", f"--node_rank={args.node_rank}",
                    f"--nproc_per_node={args.nproc_per_node}", f"--master_addr={args.master_addr}",
                    f"--master_port={args.master_port}"]
    command += [str(SFT_ROOT / "train/pretrain_unified_navit.py")]
    training = dict(
        dataset_config_file=str(runtime_config), model_path=str(model), layer_module="Qwen2MoTDecoderLayer",
        max_latent_size=64, resume_from=str(model), finetune_from_hf=True, auto_resume=True,
        resume_model_only=True, finetune_from_ema=True, visual_gen=True, visual_und=True,
        freeze_vae=True, freeze_vit=True, freeze_llm=False, freeze_und=False,
        text_cond_dropout_prob=0.0, vae_cond_dropout_prob=0.0, vit_cond_dropout_prob=0.0,
        use_flex=True, total_steps=args.total_steps, warmup_steps=args.warmup_steps, lr_scheduler="cosine",
        log_every=10, lr=args.lr, min_lr=args.min_lr, max_grad_norm=0.5, ce_weight=1.0, mse_weight=1.0,
        gradient_accumulation_steps=args.gradient_accumulation_steps, global_seed=42, data_seed=42,
        num_workers=args.num_workers, expected_num_tokens=8192, prefer_buffer_before=8192,
        ema=0.9995, timestep_shift=4.0, max_num_tokens=16384, max_num_tokens_per_sample=16384,
        sharding_strategy=args.sharding_strategy, num_replicate=args.num_replicate, num_shard=args.num_shard,
        cpu_offload=False, wandb_offline=True, wandb_project="mindview", wandb_name="mindview_sft",
        save_every=args.save_every, save_final_checkpoint=True,
        results_dir=str(output / "logs"), checkpoint_dir=str(output / "checkpoints"),
    )
    for name, value in training.items():
        command += [f"--{name}", str(value)]
    specification = dict(schema="mindview_sft_run_v1", snapshot=str(snapshot),
                         snapshot_sha256=snapshot_digest, dataset_config=dataset_config,
                         benchmark=str(benchmark), benchmark_sha256=sha256(benchmark),
                         training=training, world_size=world_size)
    for name, content in (("run_spec.json", specification), ("mindview_sft.json", dataset_config)):
        existing = output / name
        if existing.exists() and json.loads(existing.read_text()) != content:
            raise ValueError(f"Existing run configuration differs: {existing}; use a separate output directory")
    environment = os.environ.copy()
    environment.update(TRAINING_JSONL=str(snapshot), SFT_SNAPSHOT_ROWS=str(len(uids)),
                       PYTHONPATH=str(SFT_ROOT) + (os.pathsep + env("PYTHONPATH") if env("PYTHONPATH") else ""))
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.gpus is not None:
        environment["CUDA_VISIBLE_DEVICES"] = args.gpus
    return dict(command=command, environment=environment, output=output, config=dataset_config,
                specification=specification, rows=len(uids), datasets=dict(datasets), scenes=len(scene_ids),
                benchmark_scenes=len(benchmark_scenes))


def write_json(destination, value):
    with tempfile.NamedTemporaryFile("w", dir=destination.parent, delete=False, encoding="utf-8") as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, destination)


def main():
    arguments = parser().parse_args()
    try:
        prepared = prepare(arguments)
        print(json.dumps({"rows": prepared["rows"], "datasets": prepared["datasets"],
                          "scenes": prepared["scenes"], "world_size": prepared["specification"]["world_size"],
                          "benchmark_scenes": prepared["benchmark_scenes"], "scene_overlap": 0,
                          "snapshot_sha256": prepared["specification"]["snapshot_sha256"],
                          "dataset_config": prepared["config"], "output_root": str(prepared["output"])}, indent=2))
        print("Launch command:", shlex.join(prepared["command"]), flush=True)
        if arguments.dry_run:
            print("Dry run passed; no model loaded, no GPU used, no files written.")
            return
        if arguments.gpus is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpus
        import torch
        if torch.cuda.device_count() < arguments.nproc_per_node:
            raise ValueError("Insufficient visible CUDA devices for nproc-per-node")
        if any(torch.cuda.get_device_capability(index)[0] < 8 for index in range(arguments.nproc_per_node)):
            raise ValueError("BF16 training requires Ampere-or-newer CUDA devices")
        ancestor = prepared["output"]
        while not ancestor.exists():
            ancestor = ancestor.parent
        if shutil.disk_usage(ancestor).free < arguments.min_free_gib * 1024 ** 3:
            raise ValueError(f"Training requires {arguments.min_free_gib} GiB free checkpoint space")
        output = prepared["output"]
        output.mkdir(parents=True, exist_ok=True)
        # Every node uses the same frozen config; changes must use a new run directory.
        import fcntl
        with (output / ".config.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for name, content in (("run_spec.json", prepared["specification"]), ("mindview_sft.json", prepared["config"])):
                target = output / name
                if target.exists():
                    if json.loads(target.read_text()) != content:
                        raise ValueError(f"Existing run configuration differs: {target}")
                else:
                    write_json(target, content)
        (output / "logs").mkdir(exist_ok=True)
        (output / "checkpoints").mkdir(exist_ok=True)
        os.chdir(SFT_ROOT)
        os.execvpe(prepared["command"][0], prepared["command"], prepared["environment"])
    except (OSError, ValueError, MindViewDataError) as error:
        raise SystemExit(f"SFT preflight failed: {error}") from error


if __name__ == "__main__":
    main()
