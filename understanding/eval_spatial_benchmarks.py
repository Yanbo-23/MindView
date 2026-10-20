#!/usr/bin/env python3
"""Evaluate BAGEL checkpoints on registered spatial-understanding benchmarks.

SAT-v2 examples:

  # One A800
  CUDA_VISIBLE_DEVICES=0 python understanding/eval_spatial_benchmarks.py \
    --benchmark sat-v2 \
    --checkpoint-path /path/to/ema.safetensors \
    --output-dir /path/to/results

  # Eight A800s; every process owns one complete model and one data shard
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --standalone --nproc_per_node=8 \
    understanding/eval_spatial_benchmarks.py \
    --benchmark sat-v2 \
    --checkpoint-path /path/to/ema.safetensors \
    --output-dir /path/to/results
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from understanding.spatial_benchmarks import BENCHMARKS, get_benchmark_spec
from understanding.spatial_benchmarks.bagel_model import (
    BagelUnderstandingRunner,
    checkpoint_identity,
)
from understanding.spatial_benchmarks.base import EvaluationSample, EvaluationTask
from understanding.spatial_benchmarks.results import (
    JsonlShardWriter,
    OutputRunLock,
    append_launch,
    completed_eval_ids,
    ensure_manifest,
    finalize_results,
    spec_fingerprint,
    utc_now,
)
from understanding.spatial_benchmarks.runtime import DistributedRuntime

DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "BAGEL-7B-MoT"
PREDICTION_SCHEMA = "bagel_spatial_prediction_v2"
EVALUATOR_VERSION = "bagel_spatial_eval_v2"


def build_parser(argv: list[str] | None = None) -> argparse.ArgumentParser:
    preparser = argparse.ArgumentParser(add_help=False)
    preparser.add_argument(
        "--benchmark", choices=sorted(BENCHMARKS), default="sat-v2"
    )
    preliminary, _ = preparser.parse_known_args(argv)
    benchmark_spec = get_benchmark_spec(preliminary.benchmark)

    parser = argparse.ArgumentParser(
        description="Evaluate base or SFT BAGEL on spatial understanding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmark", choices=sorted(BENCHMARKS), default=preliminary.benchmark
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="BAGEL directory containing tokenizer and model configs.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="",
        help="Base/SFT weights. Empty selects <model-path>/ema.safetensors.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Persistent result directory; required unless --prepare-only is used.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Smoke-test this many source questions before any adapter-specific "
            "benchmark expansion."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help=(
            "Maximum generated answer tokens. The selected adapter's frozen "
            f"default is {benchmark_spec.default_max_new_tokens}."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--log-every", type=int, default=1, help="Progress interval per rank."
    )
    parser.add_argument(
        "--fsync-every",
        type=int,
        default=10,
        help="Fsync each result shard after this many appended records.",
    )
    parser.add_argument(
        "--oom-retries",
        type=int,
        default=1,
        help="Number of per-sample CUDA OOM retries after clearing the cache.",
    )
    parser.add_argument(
        "--allow-incomplete-checkpoint",
        action="store_true",
        help="Allow a numeric training checkpoint directory without _SUCCESS.",
    )
    parser.add_argument("--allow-download", action="store_true", help="Explicitly allow downloading pinned benchmark data")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Resolve and validate benchmark data without loading BAGEL or using CUDA.",
    )
    benchmark_spec.add_arguments(parser)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser(argv)
    args = parser.parse_args(argv)
    benchmark_spec = get_benchmark_spec(args.benchmark)
    if hasattr(args, "offline"):
        args.offline = args.offline or not args.allow_download
    if args.max_new_tokens is None:
        args.max_new_tokens = benchmark_spec.default_max_new_tokens
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.log_every <= 0:
        parser.error("--log-every must be positive")
    if args.fsync_every <= 0:
        parser.error("--fsync-every must be positive")
    if args.oom_retries < 0:
        parser.error("--oom-retries must be non-negative")
    if not args.prepare_only and not args.output_dir:
        parser.error("--output-dir is required for model evaluation")
    return args


def _safe_output_dir(
    raw_path: str, model_path: str, checkpoint_path: str | Path
) -> Path:
    output_dir = Path(raw_path).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    forbidden = {
        Path(output_dir.anchor), REPO_ROOT.resolve(),
        Path(model_path).expanduser().resolve(), checkpoint.parent,
    }
    if checkpoint.parent.name.isdigit():
        forbidden.add(checkpoint.parent.parent)
    if output_dir in forbidden:
        raise RuntimeError(f"Refusing unsafe broad --output-dir: {output_dir}")
    return output_dir


def _validate_task_index(tasks: list[EvaluationTask]) -> None:
    expected_indexes = list(range(len(tasks)))
    observed_indexes = [task.global_index for task in tasks]
    if observed_indexes != expected_indexes:
        raise RuntimeError("Benchmark tasks do not have dense canonical global indexes")
    eval_ids = [task.eval_id for task in tasks]
    if len(eval_ids) != len(set(eval_ids)):
        raise RuntimeError("Benchmark adapter emitted duplicate eval_ids")


def _base_record(
    *,
    task: EvaluationTask,
    runtime: DistributedRuntime,
    checkpoint_path: str,
    run_fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema_version": PREDICTION_SCHEMA,
        "timestamp_utc": utc_now(),
        "eval_id": task.eval_id,
        "global_index": task.global_index,
        "source_index": task.source_index,
        "source_id": task.source_id,
        "category": task.category,
        "variant": task.variant,
        "rank": runtime.rank,
        "world_size": runtime.world_size,
        "local_rank": runtime.local_rank,
        "checkpoint_path": checkpoint_path,
        "run_fingerprint": run_fingerprint,
    }


def _inference_record(
    *,
    task: EvaluationTask,
    adapter: Any,
    runner: BagelUnderstandingRunner,
    runtime: DistributedRuntime,
    checkpoint_path: str,
    run_fingerprint: str,
    max_new_tokens: int,
    oom_retries: int,
) -> dict[str, Any]:
    record = _base_record(
        task=task,
        runtime=runtime,
        checkpoint_path=checkpoint_path,
        run_fingerprint=run_fingerprint,
    )
    sample: EvaluationSample | None = None
    try:
        sample = adapter.materialize(task)
        record.update(
            {
                "question": sample.metadata.get("question"),
                "question_type": sample.metadata.get("question_type"),
                "difficulty": sample.metadata.get("difficulty"),
                "official_index": sample.metadata.get("official_index"),
                "official_group_index": sample.metadata.get("official_group_index"),
                "prompt": sample.prompt,
                "options": list(sample.options),
                "target_letter": sample.target_letter,
                "target_text": sample.target_text,
                "num_images": len(sample.images),
            }
        )
        response = None
        elapsed = None
        for attempt in range(oom_retries + 1):
            try:
                response, elapsed = runner.generate(
                    images=sample.images,
                    prompt=sample.prompt,
                    max_new_tokens=max_new_tokens,
                )
                break
            except runner.torch.OutOfMemoryError:
                if attempt >= oom_retries:
                    raise
                gc.collect()
                runner.torch.cuda.empty_cache()
                runtime.log(
                    f"CUDA OOM for {task.eval_id}; retry {attempt + 1}/{oom_retries}"
                )

        if response is None or elapsed is None:
            raise RuntimeError("BAGEL returned no response")
        record.update(
            {
                "status": "ok",
                "raw_response": response,
                "elapsed_seconds": elapsed,
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "temperature": 0.0,
                **adapter.score_response(sample, response),
            }
        )
    except Exception as exc:
        message = str(exc).lower()
        is_oom = isinstance(exc, runner.torch.OutOfMemoryError) or (
            "out of memory" in message and "cuda" in message
        )
        if is_oom:
            gc.collect()
            runner.torch.cuda.empty_cache()
        elif any(
            marker in message
            for marker in (
                "cuda error",
                "device-side assert",
                "illegal memory access",
                "unspecified launch failure",
                "misaligned address",
                "cublas_status",
                "cudnn_status",
            )
        ):
            raise
        record.update(
            {
                "status": "inference_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if sample is not None:
            for image in sample.images:
                close = getattr(image, "close", None)
                if close is not None:
                    close()
            sample.images.clear()
    return record


def _print_summary(adapter: Any, metrics: dict[str, Any]) -> None:
    print()
    for line in adapter.summary_lines(metrics):
        print(line)


def run(args: argparse.Namespace) -> int:
    runtime = DistributedRuntime(prepare_only=args.prepare_only)
    run_lock: OutputRunLock | None = None
    try:
        benchmark_spec = get_benchmark_spec(args.benchmark)
        adapter = None
        preparation_error = None
        try:
            adapter = benchmark_spec.adapter_class.from_args(args, runtime)
        except Exception:  # noqa: BLE001 - synchronize the traceback across ranks
            preparation_error = traceback.format_exc()
        runtime.raise_if_any_rank_failed(
            preparation_error, label=f"prepare benchmark {args.benchmark}"
        )
        assert adapter is not None
        tasks = list(adapter.tasks)
        _validate_task_index(tasks)
        manifest_fragment = adapter.manifest_fragment
        source_count = manifest_fragment.get(
            "selected_source_rows", manifest_fragment.get("raw_rows", "unknown")
        )
        runtime.log(
            f"Validated benchmark={adapter.name}; source_items={source_count}; "
            f"evaluation_items={len(tasks)}",
            primary_only=True,
        )

        if args.prepare_only:
            if runtime.is_primary:
                print(
                    json.dumps(adapter.manifest_fragment, ensure_ascii=False, indent=2)
                )
            return 0

        model_path = Path(args.model_path).expanduser().resolve()
        checkpoint_path = (
            Path(args.checkpoint_path).expanduser().resolve()
            if args.checkpoint_path
            else model_path / "ema.safetensors"
        )
        model_identity = runtime.primary_call(
            lambda: checkpoint_identity(
                model_path,
                checkpoint_path,
                allow_incomplete_checkpoint=args.allow_incomplete_checkpoint,
            ),
            label="validate BAGEL checkpoint",
        )
        checkpoint_path = Path(model_identity["checkpoint_path"])
        output_dir = _safe_output_dir(args.output_dir, str(model_path), checkpoint_path)
        if runtime.is_primary:
            run_lock = OutputRunLock(output_dir)
        runtime.primary_call(
            lambda: run_lock.acquire() if run_lock is not None else None,
            label="lock result directory",
        )

        frozen_spec = {
            "evaluator_version": EVALUATOR_VERSION,
            "benchmark": dict(adapter.manifest_fragment),
            "model": model_identity,
            "scope": {"partial": args.limit is not None, "source_limit": args.limit},
            "decoding": {
                "do_sample": False,
                "temperature": 0.0,
                "thinking": False,
                "max_new_tokens": args.max_new_tokens,
                "seed": args.seed,
                "image_transform_source": "understanding/image_transform.yaml:vlm_sft",
            },
        }
        frozen_spec["model_execution"] = {
            "parameter_dtype": "torch.bfloat16",
            "autocast_dtype": "torch.bfloat16",
            "checkpoint_loading": "stream_safetensors_tensor_by_tensor",
            "replication": "one_complete_model_per_process",
        }
        run_fingerprint = spec_fingerprint(frozen_spec)

        def setup_output() -> dict[str, Any]:
            ensure_manifest(output_dir, frozen_spec)
            processed = completed_eval_ids(
                output_dir,
                adapter,
                run_fingerprint=run_fingerprint,
                record_schema=PREDICTION_SCHEMA,
                checkpoint_path=str(checkpoint_path),
            )
            append_launch(
                output_dir,
                {
                    "started_at_utc": utc_now(),
                    "world_size": runtime.world_size,
                    "processed_before_launch": len(processed),
                    "run_fingerprint": run_fingerprint,
                },
            )
            return {
                "processed": processed,
                "already_complete": (output_dir / "_SUCCESS").is_file(),
            }

        setup = runtime.primary_call(setup_output, label="prepare result directory")
        processed = set(setup["processed"])
        if setup["already_complete"]:
            metrics = runtime.primary_call(
                lambda: finalize_results(
                    output_dir,
                    adapter,
                    run_fingerprint=run_fingerprint,
                    record_schema=PREDICTION_SCHEMA,
                    checkpoint_path=str(checkpoint_path),
                    partial=args.limit is not None,
                    source_limit=args.limit,
                ),
                label="verify completed result shards",
            )
            if runtime.is_primary:
                print(f"Run is already complete: {output_dir}")
                _print_summary(adapter, metrics)
            return 0

        pending = [
            task
            for task in tasks
            if task.eval_id not in processed
            and task.global_index % runtime.world_size == runtime.rank
        ]
        runtime.log(
            f"Assigned {len(pending)} pending items "
            f"({len(processed)}/{len(tasks)} already complete globally)"
        )

        runner = None
        load_error = None
        if pending:
            try:
                runtime.log(f"Loading BAGEL on {runtime.device}")
                runner = BagelUnderstandingRunner(
                    model_path=model_path,
                    checkpoint_path=checkpoint_path,
                    device=runtime.device,
                    seed=args.seed,
                )
                runtime.log(
                    f"Loaded {runner.num_parameters / 1e9:.3f}B parameters; "
                    f"ignored {len(runner.unused_checkpoint_keys)} generation-only keys"
                )
            except Exception:  # noqa: BLE001 - synchronize model-load failures
                load_error = traceback.format_exc()
        runtime.raise_if_any_rank_failed(load_error, label="load BAGEL")

        with JsonlShardWriter(
            output_dir, runtime.rank, fsync_every=args.fsync_every
        ) as writer:
            for local_index, task in enumerate(pending, start=1):
                assert runner is not None
                record = _inference_record(
                    task=task,
                    adapter=adapter,
                    runner=runner,
                    runtime=runtime,
                    checkpoint_path=str(checkpoint_path),
                    run_fingerprint=run_fingerprint,
                    max_new_tokens=args.max_new_tokens,
                    oom_retries=args.oom_retries,
                )
                writer.write(record)
                if (
                    local_index % args.log_every == 0
                    or local_index == len(pending)
                    or record["status"] != "ok"
                ):
                    prediction = record.get("prediction", "ERROR")
                    runtime.log(
                        f"[{local_index}/{len(pending)}] {task.eval_id} "
                        f"status={record['status']} pred={prediction}"
                    )

        runtime.barrier()
        metrics = runtime.primary_call(
            lambda: finalize_results(
                output_dir,
                adapter,
                run_fingerprint=run_fingerprint,
                record_schema=PREDICTION_SCHEMA,
                checkpoint_path=str(checkpoint_path),
                partial=args.limit is not None,
                source_limit=args.limit,
            ),
            label="merge and score result shards",
        )
        if runtime.is_primary:
            _print_summary(adapter, metrics)
            print(f"Results: {output_dir}")
        return 0 if metrics["complete"] else 2
    finally:
        if run_lock is not None:
            run_lock.release()
        runtime.close()


def main() -> None:
    args = parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
