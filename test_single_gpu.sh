#!/usr/bin/env bash
# Sequential BAGEL-SFT acceptance: 10 random C6 examples, six metrics, three QA benchmarks.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
usage() {
  cat <<'HELP'
Usage: bash test_single_gpu.sh [OPTIONS]
  --config FILE                Explicit shell configuration (or CONFIG_FILE).
  --gpu ID                     One CUDA device; default: first visible device, otherwise 0.
  --seed INTEGER               Random subset seed; default: 42. Generation keeps seed 42.
  --output-root DIR            Default: outputs/single_gpu_test under this repository.
  --allow-download             Allow pinned SAT-v2/MMSI downloads when not present locally.
  --understanding-limit N       Optional diagnostic source-question limit; default: FULL benchmarks.
  --dry-run                    Show the exact sequence; no GPU, downloads, or output writes.
  -h, --help                   Show this help.

Normal order: sample 10 C6 cases -> data preflight -> BAGEL-SFT generation ->
CLIP/DINO + Qwen judge + VGGT -> subset report -> SAT-v2 -> MMSI -> MindCube-Tiny.
Ten-case MindView scores are smoke-test results, not the full benchmark score.
Requires prepared environments, BAGEL assets/checkpoint, RGB/depth and QA datasets.
See README.md, evaluation/README.md, and understanding/README.md.
HELP
}

dry_run=0; allow_download=0; cli_gpu=""; cli_output=""; subset_seed=42; qa_limit=""
while (( $# )); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) dry_run=1; shift ;;
    --allow-download) allow_download=1; shift ;;
    --config|--gpu|--output-root|--seed|--understanding-limit)
      (( $# >= 2 )) && [[ -n "$2" ]] || { echo "[error] $1 requires a value" >&2; exit 2; }
      case "$1" in
        --config) CONFIG_FILE="$2" ;;
        --gpu) cli_gpu="$2" ;;
        --output-root) cli_output="$2" ;;
        --seed) subset_seed="$2" ;;
        --understanding-limit) qa_limit="$2" ;;
      esac
      shift 2 ;;
    *) echo "[error] Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$subset_seed" =~ ^[0-9]+$ ]] || { echo "[error] --seed must be a non-negative integer" >&2; exit 2; }
[[ -z "$qa_limit" || "$qa_limit" =~ ^[1-9][0-9]*$ ]] || { echo "[error] --understanding-limit must be positive" >&2; exit 2; }
if [[ -n "${CONFIG_FILE:-}" ]]; then
  [[ -f "$CONFIG_FILE" ]] || { echo "[error] Missing CONFIG_FILE: $CONFIG_FILE" >&2; exit 2; }
  set -a; source "$CONFIG_FILE"; set +a
fi
[[ "${WORLD_SIZE:-1}" == 1 && "${RANK:-0}" == 0 ]] || {
  echo "[error] Run this script once, outside torchrun/distributed launchers" >&2; exit 2;
}
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export BAGEL_PYTHON="${BAGEL_PYTHON:-$PYTHON_BIN}"
export EVAL_PYTHON="${EVAL_PYTHON:-$BAGEL_PYTHON}"
export CLIP_DINO_PYTHON="${CLIP_DINO_PYTHON:-$PYTHON_BIN}"
export QWEN_VL_PYTHON="${QWEN_VL_PYTHON:-$PYTHON_BIN}"
export VGGT_PYTHON="${VGGT_PYTHON:-$PYTHON_BIN}"
export BAGEL_MODEL_PATH="${BAGEL_MODEL_PATH:-$ROOT/models/BAGEL-7B-MoT}"
for variable in BAGEL_MODEL_PATH BAGEL_SFT_FINAL_CHECKPOINT SCANNET_ROOT SCANNETPP_ROOT ARKITSCENES_ROOT \
                SAT_V2_PARQUET MMSI_PARQUET MINDCUBE_DATA_ROOT; do
  [[ -n "${!variable:-}" ]] || continue
  printf -v "$variable" '%s' "$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "${!variable}")"
  export "$variable"
done
gpu="${cli_gpu:-${TEST_GPU:-${CUDA_VISIBLE_DEVICES:-0}}}"; gpu="${gpu%%,*}"
[[ -n "$gpu" && "$gpu" != -1 && "$gpu" != *[[:space:]]* ]] || {
  echo "[error] Choose one CUDA device with --gpu" >&2; exit 2;
}
[[ -z "$cli_gpu" || "$cli_gpu" != *,* ]] || { echo "[error] --gpu accepts exactly one device" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="$gpu" NUM_GPUS=1 GPU_LIST="$gpu" UNDERSTANDING_NPROC_PER_NODE=1
export SKIP_CONFIG=1 PYTHONDONTWRITEBYTECODE=1
export OVERWRITE=0 SKIP_COMPLETED=1 REPORT_OVERWRITE=1
export SEED=42 BAGEL_THINKING=0 SAVE_GT=1 SAVE_IMAGE_SIZE=512 INPUT_IMAGE_SIZE=512
export IMAGE_SIZE=512 DATA_SPLIT=val
export MAX_SAMPLES=-1 TASK=all TASKS=all DATASETS=all SHOTS=6 RUNS=all
export RUN_CLIP_DINO=1 RUN_QWEN_VL=1 RUN_VGGT=1 ALLOW_VARIABLE_CONTEXTS=0
export REPORT_PROTOCOL=selected
export REPORT_CONFIG="$ROOT/evaluation/benchmarker_metrics/metric_report_config.json"
unset VGGT_COMMON_REFERENCE_GENERATION_RESULTS
canonical="$ROOT/evaluation/data/merged_dataset_final.jsonl"
test_root="$("$PYTHON_BIN" - "${cli_output:-${TEST_OUTPUT_ROOT:-$ROOT/outputs/single_gpu_test}}" "$ROOT" <<'PY'
from pathlib import Path
import os, sys
out, root = (Path(p).expanduser().resolve() for p in sys.argv[1:])
protected = [root / name for name in ('evaluation', 'sft', 'understanding', 'vggt', 'assets', 'models', 'resources', 'ckpt')]
protected += [Path(os.environ[key]).expanduser().resolve() for key in
              ('BAGEL_MODEL_PATH', 'SCANNET_ROOT', 'SCANNETPP_ROOT', 'ARKITSCENES_ROOT') if os.environ.get(key)]
if out == root or out in root.parents or any(out == p or out in p.parents or p in out.parents for p in protected):
    raise SystemExit('[error] Test output must be a dedicated directory outside protected inputs/source')
print(out)
PY
)"
export INPUT_JSON="$test_root/subset.jsonl" VGGT_CANONICAL_JSON="$canonical"
export GENERATION_ROOT="$test_root/generations" BENCHMARK_OUTPUT_ROOT="$test_root/generations"
export METRICS_ROOT="$test_root/metrics" REPORT_ROOT="$test_root/reports"
export UNDERSTANDING_OUTPUT_ROOT="$test_root/understanding"
export RUN_TAG="single_gpu_c6_seed${subset_seed}"
sample=("$PYTHON_BIN" "$ROOT/evaluation/sample_subset.py" --source-jsonl "$canonical"
        --count 10 --seed "$subset_seed" --output-jsonl "$INPUT_JSON")
qa=(bash "$ROOT/understanding/run.sh" all --nproc-per-node 1 --gpus "$gpu"
    --model-path "$BAGEL_MODEL_PATH" --checkpoint-path "${BAGEL_SFT_FINAL_CHECKPOINT:-}")
[[ "$allow_download" == 0 ]] || qa+=(--allow-download)
[[ -z "$qa_limit" ]] || qa+=(--limit "$qa_limit")
print_command() { printf '  '; printf '%q ' "$@"; printf '\n'; }
if [[ "$dry_run" == 1 ]]; then
  printf '[plan] One CUDA device: %s; output: %s\n' "$gpu" "$test_root"
  "${sample[@]}" --dry-run
  print_command "${qa[@]}" --prepare-only
  for phase in generate metrics report; do print_command bash "$ROOT/evaluation/run.sh" "$phase" bagel-sft-final; done
  "${qa[@]}" --dry-run
  echo '[plan] No inference, downloads, or output files were created.'
  exit 0
fi

# Fail before expensive generation if any selected interpreter lacks a usable GPU.
declare -A checked=()
for interpreter in "$BAGEL_PYTHON" "$CLIP_DINO_PYTHON" "$QWEN_VL_PYTHON" "$VGGT_PYTHON" "$EVAL_PYTHON"; do
  [[ -z "${checked[$interpreter]:-}" ]] || continue
  checked["$interpreter"]=1
  "$interpreter" - <<'PY'
import sys
try:
    import torch
except ImportError:
    raise SystemExit('[error] Install the required PyTorch environment for ' + sys.executable)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit('[error] This test requires exactly one usable CUDA GPU per configured interpreter')
print('[preflight] CUDA:', torch.cuda.get_device_name(0))
PY
done
"$BAGEL_PYTHON" -c 'from flash_attn import flash_attn_varlen_func' >/dev/null 2>&1 || {
  echo '[error] mindview_bagel is missing FlashAttention; install a matching flash_attn wheel in that environment' >&2
  exit 2
}
[[ -s "${BAGEL_SFT_FINAL_CHECKPOINT:-}" ]] || { echo '[error] Set BAGEL_SFT_FINAL_CHECKPOINT to the completed EMA' >&2; exit 2; }
for asset in llm_config.json vit_config.json tokenizer_config.json tokenizer.json vocab.json merges.txt ae.safetensors; do
  [[ -s "$BAGEL_MODEL_PATH/$asset" ]] || { echo "[error] Missing BAGEL asset: $BAGEL_MODEL_PATH/$asset" >&2; exit 2; }
done
for variable in SCANNET_ROOT SCANNETPP_ROOT ARKITSCENES_ROOT; do
  [[ -d "${!variable:-}" ]] || { echo "[error] Set $variable to the extracted dataset root" >&2; exit 2; }
done
mkdir -p "$test_root/logs"
exec {test_lock}>"$test_root/.test.lock"
flock -n "$test_lock" || { echo '[error] Another test is using this output directory' >&2; exit 2; }
trap 'status=$?; if (( status != 0 )); then echo "[failed] Test stopped; inspect logs in $test_root/logs" >&2; fi' EXIT
stage() {
  local name="$1"; shift
  printf '\n[stage] %s\n' "$name"
  "$@" 2>&1 | tee "$test_root/logs/$name.log"
}
stage 00_subset "${sample[@]}"
stage 01_understanding_data "${qa[@]}" --prepare-only
stage 02_generate bash "$ROOT/evaluation/run.sh" generate bagel-sft-final
stage 03_metrics bash "$ROOT/evaluation/run.sh" metrics bagel-sft-final
stage 04_report bash "$ROOT/evaluation/run.sh" report bagel-sft-final
stage 05_understanding "${qa[@]}"
"$PYTHON_BIN" - "$test_root" "$gpu" "$qa_limit" <<'PY'
import json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
qa_root = root / 'understanding'
if sys.argv[3]:
    qa_root = qa_root / 'partial' / ('limit_' + sys.argv[3])
required = [root / 'reports/report_summary.json', qa_root / 'summary.json']
for path in required:
    if not path.is_file():
        raise SystemExit(f'[error] Missing completed result: {path}')
subset = json.loads((root / 'subset.manifest.json').read_text())
report, understanding = (json.loads(path.read_text()) for path in required)
if (report.get('models') != ['bagel-sft-final'] or report.get('dataset', {}).get('num_samples') != 10
        or report['dataset'].get('sha256') != subset['subset_sha256']):
    raise SystemExit('[error] MindView report must cover this exact ten-case BAGEL-SFT subset')
scores = {'dino_i', 'semantic', 'spatial', 'quality', 'vggt_rscore', 'vggt_tscore'}
if not scores.issubset(report.get('score_fields', [])):
    raise SystemExit('[error] MindView report is missing primary metric components')
expected_status = 'partial' if sys.argv[3] else 'complete'
benchmarks = understanding.get('benchmarks', {})
if (understanding.get('selected_runs_complete') is not True
        or understanding.get('status') != expected_status
        or set(benchmarks) != {'sat-v2', 'mmsi', 'mindcube-tiny'}):
    raise SystemExit('[error] All three requested understanding protocols must finish successfully')
checkpoint = str(Path(os.environ['BAGEL_SFT_FINAL_CHECKPOINT']).resolve())
if any(row.get('status') != expected_status or row.get('checkpoint', {}).get('checkpoint_path') != checkpoint
       for row in benchmarks.values()):
    raise SystemExit('[error] Understanding results have incomplete scope or a different checkpoint')
record = {'status': 'completed', 'method': 'bagel-sft-final', 'cuda_visible_device': sys.argv[2],
          'mindview_scope': '10 random C6 smoke cases; not a full benchmark score',
          'checkpoint_path': checkpoint, 'subset': subset,
          'understanding_scope': 'full' if not sys.argv[3] else 'partial source-question limit=' + sys.argv[3],
          'reports': [str(path) for path in required]}
(root / 'test_summary.json').write_text(json.dumps(record, indent=2) + '\n')
print('[complete]', root / 'test_summary.json')
PY
