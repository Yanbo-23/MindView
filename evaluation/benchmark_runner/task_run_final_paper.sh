#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"
FINAL_MODELS=(bagel bagel-sft-final flux2 qwen-image-edit-2511 emu3.5 dreamomni2 sensenova-u1 sensenova-u1.5 gpt-image-2 nano-banana-pro gpt-image-1.5 nano-banana)
if (( $# > 0 )); then SELECTED_MODELS=("$@"); else IFS=',' read -ra SELECTED_MODELS <<< "${MODELS:-all}"; fi
[[ "${#SELECTED_MODELS[@]}" != "1" || "${SELECTED_MODELS[0]}" != "all" ]] || SELECTED_MODELS=("${FINAL_MODELS[@]}")
declare -A seen=()
for method in "${SELECTED_MODELS[@]}"; do
  case "${method}" in bagel|bagel-sft-final|flux2|qwen-image-edit-2511|emu3.5|dreamomni2|sensenova-u1|sensenova-u1.5|gpt-image-2|nano-banana-pro|gpt-image-1.5|nano-banana|dummy) ;;
    *) echo "[Error] Unknown model: ${method}" >&2; exit 1 ;;
  esac
  [[ -z "${seen[${method}]:-}" ]] || { echo "[Error] Duplicate model: ${method}" >&2; exit 1; }
  seen["${method}"]=1
done
for method in "${SELECTED_MODELS[@]}"; do
  backend="${method}"
  checkpoint="${BAGEL_CHECKPOINT_PATH:-}"
  case "${method}" in
    bagel-sft-final)
      backend=bagel
      checkpoint="${BAGEL_SFT_FINAL_CHECKPOINT:-}"
      [[ -s "${checkpoint}" ]] || { echo "[Error] Set BAGEL_SFT_FINAL_CHECKPOINT to the trained EMA checkpoint." >&2; exit 1; }
      ;;
    nano-banana-pro) backend=nanobanna-pro ;;
  esac
  RUN_MODELS="${backend}" METHOD_NAME="${method}" OUTPUT_ROOT="${GENERATION_ROOT}" BAGEL_CHECKPOINT_PATH="${checkpoint}" \
    bash "${SCRIPT_DIR}/run_benchmark.sh"
done
