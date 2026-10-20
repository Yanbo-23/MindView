#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"
FINAL_MODELS=(bagel bagel-sft-final flux2 qwen-image-edit-2511 emu3.5 dreamomni2 sensenova-u1 sensenova-u1.5 gpt-image-2 nano-banana-pro gpt-image-1.5 nano-banana)
if (( $# > 0 )); then SELECTED_MODELS=("$@"); else IFS=',' read -ra SELECTED_MODELS <<< "${MODELS:-all}"; fi
[[ "${#SELECTED_MODELS[@]}" != "1" || "${SELECTED_MODELS[0]}" != "all" ]] || SELECTED_MODELS=("${FINAL_MODELS[@]}")
METRICS_ROOT_BASE="${METRICS_ROOT}"
BENCHMARK_OUTPUT_ROOT="${BENCHMARK_OUTPUT_ROOT:-${GENERATION_ROOT}}"
declare -A seen=()
for method in "${SELECTED_MODELS[@]}"; do
  [[ -n "${method}" && "${method}" != */* && "${method}" != .* ]] || { echo "[Error] Invalid model directory ID: ${method}" >&2; exit 1; }
  [[ -z "${seen[${method}]:-}" ]] || { echo "[Error] Duplicate model: ${method}" >&2; exit 1; }
  seen["${method}"]=1
  [[ -d "${BENCHMARK_OUTPUT_ROOT}/${method}" ]] || { echo "[Error] Missing generation directory: ${BENCHMARK_OUTPUT_ROOT}/${method}" >&2; exit 1; }
done
for method in "${SELECTED_MODELS[@]}"; do
  MODELS="${method}" METRICS_ROOT="${METRICS_ROOT_BASE}/${method}" BENCHMARK_OUTPUT_ROOT="${BENCHMARK_OUTPUT_ROOT}" \
    bash "${SCRIPT_DIR}/run_metrics.sh"
done
