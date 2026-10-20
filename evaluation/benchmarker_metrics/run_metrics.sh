#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"
BENCHMARK_OUTPUT_ROOT="${BENCHMARK_OUTPUT_ROOT:-${GENERATION_ROOT}}"
MODELS="${MODELS:-all}"
SHOTS="${SHOTS:-3,6,10}"
TASKS="${TASKS:-all}"
DATASETS="${DATASETS:-all}"
RUNS="${RUNS:-all}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
OVERWRITE="${OVERWRITE:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
NUM_GPUS="${NUM_GPUS:-1}"
GPU_ARGS=(--num_gpus "${NUM_GPUS}")
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${NUM_GPUS}" -gt 0 ]]; then
  IFS=',' read -ra visible <<< "${CUDA_VISIBLE_DEVICES}"
  NUM_GPUS="${#visible[@]}"
  GPU_ARGS=(--num_gpus "${NUM_GPUS}" --gpus "${CUDA_VISIBLE_DEVICES}")
fi
CLIP_DINO_PYTHON="${CLIP_DINO_PYTHON:-${PYTHON_BIN}}"
QWEN_VL_PYTHON="${QWEN_VL_PYTHON:-${PYTHON_BIN}}"
VGGT_PYTHON="${VGGT_PYTHON:-${CLIP_DINO_PYTHON:-${PYTHON_BIN}}}"
VGGT_REPO_PATH="${VGGT_REPO_PATH:-${REPO_ROOT}/vggt}"
CLIP_MODEL_NAME="${CLIP_MODEL_NAME:-openai/clip-vit-large-patch14}"
DINO_MODEL_NAME="${DINO_MODEL_NAME:-facebook/dinov2-large}"
QWEN_MODEL_NAME="${QWEN_MODEL_NAME:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
VGGT_MODEL_NAME="${VGGT_MODEL_NAME:-${VGGT_MODEL_PATH:-facebook/VGGT-1B}}"
RUN_CLIP_DINO="${RUN_CLIP_DINO:-1}"
RUN_QWEN_VL="${RUN_QWEN_VL:-1}"
RUN_VGGT="${RUN_VGGT:-1}"
USE_FLASH_ATTN="${USE_FLASH_ATTN:-0}"
for flag in OVERWRITE SKIP_COMPLETED RUN_CLIP_DINO RUN_QWEN_VL RUN_VGGT USE_FLASH_ATTN; do
  [[ "${!flag}" == "0" || "${!flag}" == "1" ]] || { echo "[Error] ${flag} must be 0 or 1" >&2; exit 1; }
done
[[ "${OVERWRITE}${SKIP_COMPLETED}" == "10" || "${OVERWRITE}${SKIP_COMPLETED}" == "01" ]] || {
  echo "[Error] Enable exactly one of OVERWRITE or SKIP_COMPLETED." >&2; exit 1;
}
SELECTED_METRICS=()
[[ "${RUN_CLIP_DINO}" == "0" ]] || SELECTED_METRICS+=(clip_dino)
[[ "${RUN_QWEN_VL}" == "0" ]] || SELECTED_METRICS+=(qwen_vl)
[[ "${RUN_VGGT}" == "0" ]] || SELECTED_METRICS+=(vggt)
(( ${#SELECTED_METRICS[@]} > 0 )) || { echo "[Error] No metric selected." >&2; exit 1; }
SELECTED_METRICS_CSV="$(IFS=,; echo "${SELECTED_METRICS[*]}")"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
mkdir -p "${METRICS_ROOT}"
exec {METRICS_LOCK_FD}>"${METRICS_ROOT}/.run_metrics.lock"
flock -n "${METRICS_LOCK_FD}" || { echo "[Error] Another metrics job is already writing: ${METRICS_ROOT}" >&2; exit 1; }

# ========================= 5. Input snapshot =================================
# Generation workers append to shards while they are running. Build an isolated
# metadata snapshot so metrics always see a stable generation_results.json and
# never modify or depend on a partially-written live shard.
LIVE_BENCHMARK_OUTPUT_ROOT="${BENCHMARK_OUTPUT_ROOT}"
SNAPSHOT_PARENT="${SNAPSHOT_PARENT:-${METRICS_ROOT}/input_snapshots}"
mkdir -p "${SNAPSHOT_PARENT}"
SNAPSHOT_ROOT="$(mktemp -d "${SNAPSHOT_PARENT}/snapshot_$(date +%Y%m%d_%H%M%S)_XXXXXX")"

evaluation_python "${PYTHON_BIN}" - "${LIVE_BENCHMARK_OUTPUT_ROOT}" "${SNAPSHOT_ROOT}" "${MODELS}" <<'PY'
import json
import os
import sys
from pathlib import Path

source_root = Path(sys.argv[1]).resolve()
snapshot_root = Path(sys.argv[2]).resolve()
selected_models = None if sys.argv[3] == "all" else set(sys.argv[3].split(","))

if not source_root.is_dir():
    raise SystemExit(f"[Snapshot][Error] benchmark output root not found: {source_root}")


def read_shards(shard_paths):
    records = []
    invalid_lines = 0
    for shard_path in shard_paths:
        with shard_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    # A worker may be appending the final line during this read.
                    invalid_lines += 1
    return records, invalid_lines


def deduplicate_successes(records):
    unique = {}
    for record in records:
        if record.get("status") not in (None, "success"):
            continue
        sample_id = str(record.get("sample_id") or record.get("id") or "")
        if not sample_id:
            continue
        key = (
            sample_id,
            str(record.get("shot", "")),
            str(record.get("dataset", "")),
            str(record.get("task_type", "")),
        )
        unique[key] = record

    def sort_key(record):
        try:
            index = int(record.get("global_index"))
        except (TypeError, ValueError):
            index = 10**12
        return index, str(record.get("sample_id") or record.get("id") or "")

    return sorted(unique.values(), key=sort_key)


total_records = 0
for model_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
    if selected_models is not None and model_dir.name not in selected_models:
        continue
    for run_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
        shard_paths = sorted((run_dir / "shards").glob("rank_*.jsonl"))
        records, invalid_lines = read_shards(shard_paths)
        source_kind = "shards"

        # Finished or legacy runs may only have the merged JSON file.
        if not records:
            merged_path = run_dir / "generation_results.json"
            if merged_path.is_file():
                try:
                    loaded = json.loads(merged_path.read_text(encoding="utf-8"))
                    records = loaded if isinstance(loaded, list) else []
                    source_kind = "generation_results.json"
                except json.JSONDecodeError:
                    invalid_lines += 1

        records = deduplicate_successes(records)
        if not records:
            continue

        snapshot_run_dir = snapshot_root / model_dir.name / run_dir.name
        snapshot_run_dir.mkdir(parents=True, exist_ok=True)

        # Symlinks keep constructed/relative image paths usable without copying
        # any generated images into the snapshot.
        for shot_dir in sorted(run_dir.glob("shot-*")):
            if shot_dir.is_dir():
                (snapshot_run_dir / shot_dir.name).symlink_to(
                    shot_dir.resolve(), target_is_directory=True
                )

        output_path = snapshot_run_dir / "generation_results.json"
        temporary_path = output_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary_path, output_path)

        total_records += len(records)
        print(
            f"[Snapshot] {model_dir.name}/{run_dir.name}: "
            f"{len(records)} successful records from {source_kind}; "
            f"ignored incomplete lines={invalid_lines}"
        )

if total_records == 0:
    raise SystemExit(f"[Snapshot][Error] no completed samples found under: {source_root}")

print(f"[Snapshot] total successful records: {total_records}")
print(f"[Snapshot] saved to: {snapshot_root}")
PY

BENCHMARK_OUTPUT_ROOT="${SNAPSHOT_ROOT}"

COMMON_ARGS=(
  --benchmark_output_root "${BENCHMARK_OUTPUT_ROOT}"
  --scannet_root "${SCANNET_ROOT}"
  --scannetpp_root "${SCANNETPP_ROOT}"
  --arkitscenes_root "${ARKITSCENES_ROOT}"
  --data_split "${DATA_SPLIT}"
  --metrics_root "${METRICS_ROOT}"
  --models "${MODELS}"
  --shots "${SHOTS}"
  --tasks "${TASKS}"
  --datasets "${DATASETS}"
  --runs "${RUNS}"
  --max_samples "${MAX_SAMPLES}"
  --image_size "${IMAGE_SIZE}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
  COMMON_ARGS+=(--overwrite)
fi
if [[ "${SKIP_COMPLETED}" == "1" ]]; then
  COMMON_ARGS+=(--skip_completed)
fi

run_metric_with_python() {
  local python_bin="$1"
  local metric_name="$2"
  shift 2

  echo
  echo "======================================================================"
  echo "[Metric] ${metric_name}"
  echo "[Python] ${python_bin}"
  echo "[GPU]    NUM_GPUS=${NUM_GPUS}"
  echo "[GPU]    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "[GPU]    GPU_ARGS=${GPU_ARGS[*]}"
  echo "======================================================================"

  evaluation_python "${python_bin}" -m benchmarker_metrics.run \
    --metric "${metric_name}" \
    "${COMMON_ARGS[@]}" \
    "${GPU_ARGS[@]}" \
    "$@"
}

if [[ "${RUN_CLIP_DINO}" == "1" ]]; then
  run_metric_with_python "${CLIP_DINO_PYTHON}" "clip_dino" \
    --clip_model_name "${CLIP_MODEL_NAME}" \
    --dino_model_name "${DINO_MODEL_NAME}"
fi

if [[ "${RUN_QWEN_VL}" == "1" ]]; then
  QWEN_ARGS=(--qwen_model_name "${QWEN_MODEL_NAME}")
  if [[ "${USE_FLASH_ATTN}" == "1" ]]; then
    QWEN_ARGS+=(--use_flash_attn)
  fi
  run_metric_with_python "${QWEN_VL_PYTHON}" "qwen_vl" "${QWEN_ARGS[@]}"
fi

if [[ "${RUN_VGGT}" == "1" ]]; then
  export PYTHONPATH="${VGGT_REPO_PATH}:${SCRIPT_DIR}:${PYTHONPATH:-}"
  VGGT_ARGS=()
  [[ -z "${VGGT_CANONICAL_JSON:-}" ]] || VGGT_ARGS+=(--vggt_canonical_json "${VGGT_CANONICAL_JSON}")
  [[ -z "${VGGT_COMMON_REFERENCE_GENERATION_RESULTS:-}" ]] || VGGT_ARGS+=(--vggt_common_reference_generation_results "${VGGT_COMMON_REFERENCE_GENERATION_RESULTS}")
  run_metric_with_python "${VGGT_PYTHON}" "vggt" "${VGGT_ARGS[@]}" \
    --vggt_repo_path "${VGGT_REPO_PATH}" \
    --vggt_model_name "${VGGT_MODEL_NAME}"
fi

# Combined raw summaries are diagnostic; build_metric_report.py defines final aggregation.
evaluation_python "${PYTHON_BIN}" -m benchmarker_metrics.summarize \
  --metrics_root "${METRICS_ROOT}" \
  --metrics "${SELECTED_METRICS_CSV}"

echo
echo "All metrics finished. Results are under: ${METRICS_ROOT}"
