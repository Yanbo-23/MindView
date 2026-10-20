#!/usr/bin/env bash
# Shared paths and optional explicit configuration; never reads shell startup files.
EVALUATION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EVALUATION_ROOT}/.." && pwd)"
if [[ "${SKIP_CONFIG:-0}" != "1" && -n "${CONFIG_FILE:-}" && "${EVALUATION_CONFIG_LOADED:-}" != "${CONFIG_FILE}" ]]; then
  [[ -f "${CONFIG_FILE}" ]] || { echo "[Error] CONFIG_FILE does not exist: ${CONFIG_FILE}" >&2; return 1; }
  set -a
  source "${CONFIG_FILE}"
  set +a
  export EVALUATION_CONFIG_LOADED="${CONFIG_FILE}"
fi
export PYTHON_BIN="${PYTHON_BIN:-python}"
export OUTPUTS_ROOT="${OUTPUTS_ROOT:-${REPO_ROOT}/outputs}"
export GENERATION_ROOT="${GENERATION_ROOT:-${OUTPUTS_ROOT}/generations}"
export METRICS_ROOT="${METRICS_ROOT:-${OUTPUTS_ROOT}/metrics}"
export REPORT_ROOT="${REPORT_ROOT:-${OUTPUTS_ROOT}/reports}"
export INPUT_JSON="${INPUT_JSON:-${EVALUATION_ROOT}/data/merged_dataset_final.jsonl}"
export SCANNET_ROOT="${SCANNET_ROOT:-}"
export SCANNETPP_ROOT="${SCANNETPP_ROOT:-}"
export ARKITSCENES_ROOT="${ARKITSCENES_ROOT:-}"
export DATA_SPLIT="${DATA_SPLIT:-val}"
for path_variable in INPUT_JSON OUTPUTS_ROOT GENERATION_ROOT METRICS_ROOT REPORT_ROOT \
  SCANNET_ROOT SCANNETPP_ROOT ARKITSCENES_ROOT VGGT_CANONICAL_JSON \
  BENCHMARK_OUTPUT_ROOT BAGEL_REPO_PATH VGGT_REPO_PATH EMU3P5_REPO_PATH \
  DREAMOMNI2_REPO_PATH SENSENOVA_REPO_PATH; do
  if [[ -n "${!path_variable:-}" && "${!path_variable}" != /* ]]; then
    printf -v "${path_variable}" '%s/%s' "${PWD}" "${!path_variable}"
    export "${path_variable}"
  fi
done

evaluation_python() {
  local interpreter="$1"
  shift
  command -v "${interpreter}" >/dev/null 2>&1 || {
    echo "[Error] Python interpreter not found: ${interpreter}" >&2; return 1;
  }
  "${interpreter}" "$@"
}
