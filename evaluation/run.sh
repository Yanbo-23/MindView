#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"
phase="${1:-help}"
(( $# == 0 )) || shift
if [[ "${phase}" == "help" || "${phase}" == "--help" || "${phase}" == "-h" ]]; then
  cat <<'HELP'
Usage: bash evaluation/run.sh all|generate|metrics|report|check [MODEL ...]
  generate  Generate selected methods (default: all configured methods).
  metrics   Evaluate selected methods with CLIP/DINO, Qwen judge and VGGT.
  report    Validate complete selected metrics and write the task-macro report.
  all       Run generate, metrics and report in sequence.
  check     Check the frozen data, scripts and selected report when metrics exist.
Configuration: CONFIG_FILE=/absolute/path/config.env, or exported environment variables.
REPORT_PROTOCOL=selected enables a report over INPUT_JSON.
HELP
  exit 0
fi
case "${phase}" in all|generate|metrics|report|check) ;; *) echo "[Error] Unknown phase: ${phase}" >&2; exit 2 ;; esac
models="${MODELS:-all}"
if (( $# > 0 )); then models="$(IFS=,; echo "$*")"; fi
if [[ "${phase}" == "check" ]]; then
  evaluation_python "${PYTHON_BIN}" "${HERE}/check.py"
  for script in "${HERE}"/*.sh "${HERE}/benchmark_runner"/*.sh "${HERE}/benchmarker_metrics"/*.sh; do bash -n "${script}"; done
  if [[ ! -d "${METRICS_ROOT}" ]]; then
    echo "[check] no metric outputs yet; run metrics before report validation"
    exit 0
  fi
fi
if [[ "${phase}" == "all" || "${phase}" == "generate" ]]; then bash "${HERE}/run_generate.sh" "$@"; fi
if [[ "${phase}" == "all" || "${phase}" == "metrics" ]]; then bash "${HERE}/run_metrics.sh" "$@"; fi
if [[ "${phase}" == "all" || "${phase}" == "report" || "${phase}" == "check" ]]; then
  args=(--dataset-jsonl "${INPUT_JSON}" --metrics-root "${METRICS_ROOT}" --output-dir "${REPORT_ROOT}" --models "${models}" --protocol "${REPORT_PROTOCOL:-paper}")
  [[ -z "${REPORT_CONFIG:-}" ]] || args+=(--config "${REPORT_CONFIG}")
  if [[ "${phase}" == "check" ]]; then args+=(--check-only); elif [[ "${REPORT_OVERWRITE:-0}" == "1" ]]; then args+=(--overwrite); fi
  evaluation_python "${PYTHON_BIN}" "${HERE}/benchmarker_metrics/build_metric_report.py" "${args[@]}"
fi
