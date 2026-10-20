#!/usr/bin/env bash
set -euo pipefail

# Local model workers shard samples across visible GPUs; API workers use CPU.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"

NUM_GPUS="${NUM_GPUS:-1}"
API_NUM_WORKERS="${API_NUM_WORKERS:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${GENERATION_ROOT}}"
TASK="${TASK:-all}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
SAVE_IMAGE_SIZE="${SAVE_IMAGE_SIZE:-512}"
INPUT_IMAGE_SIZE="${INPUT_IMAGE_SIZE:-512}"
SEED="${SEED:-42}"
OVERWRITE="${OVERWRITE:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
SAVE_GT="${SAVE_GT:-1}"
ALLOW_VARIABLE_CONTEXTS="${ALLOW_VARIABLE_CONTEXTS:-0}"
RUN_TAG="${RUN_TAG:-$(basename "${INPUT_JSON}" .jsonl)}"
METHOD_NAME="${METHOD_NAME:-}"
RUN_MODELS="${RUN_MODELS:-all}"
BAGEL_THINKING="${BAGEL_THINKING:-0}"
SENSENOVA_U15_THINKING="${SENSENOVA_U15_THINKING:-0}"
SENSENOVA_U1_THINKING="${SENSENOVA_U1_THINKING:-0}"

BAGEL_PYTHON="${BAGEL_PYTHON:-${PYTHON_BIN}}"
FLUX_PYTHON="${FLUX_PYTHON:-${PYTHON_BIN}}"
QWEN_PYTHON="${QWEN_IMAGE_PYTHON:-${QWEN_PYTHON:-${PYTHON_BIN}}}"
EMU3P5_PYTHON="${EMU3P5_PYTHON:-${PYTHON_BIN}}"
DREAMOMNI2_PYTHON="${DREAMOMNI2_PYTHON:-${PYTHON_BIN}}"
SENSENOVA_PYTHON="${SENSENOVA_PYTHON:-${PYTHON_BIN}}"
API_PYTHON="${API_PYTHON:-${PYTHON_BIN}}"

BAGEL_REPO_PATH="${BAGEL_REPO_PATH:-${REPO_ROOT}/sft}"
BAGEL_MODEL_PATH="${BAGEL_MODEL_PATH:-${REPO_ROOT}/models/BAGEL-7B-MoT}"
BAGEL_MAX_MEM_PER_GPU="${BAGEL_MAX_MEM_PER_GPU:-80GiB}"
BAGEL_OFFLOAD_FOLDER="${BAGEL_OFFLOAD_FOLDER:-${OUTPUTS_ROOT}/offload/bagel}"
BAGEL_CHECKPOINT_PATH="${BAGEL_CHECKPOINT_PATH:-}"
FLUX_MODEL_PATH="${FLUX_MODEL_PATH:-black-forest-labs/FLUX.2-klein-9B}"
FLUX_DTYPE="${FLUX_DTYPE:-bfloat16}"
FLUX_CPU_OFFLOAD="${FLUX_CPU_OFFLOAD:-1}"
FLUX_HEIGHT="${FLUX_HEIGHT:-512}"
FLUX_WIDTH="${FLUX_WIDTH:-512}"
FLUX_GUIDANCE_SCALE="${FLUX_GUIDANCE_SCALE:-1.0}"
FLUX_NUM_INFERENCE_STEPS="${FLUX_NUM_INFERENCE_STEPS:-4}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-Qwen/Qwen-Image-Edit-2511}"
QWEN_NUM_INFERENCE_STEPS="${QWEN_NUM_INFERENCE_STEPS:-40}"
EMU3P5_REPO_PATH="${EMU3P5_REPO_PATH:-}"
EMU3P5_MODEL_PATH="${EMU3P5_MODEL_PATH:-}"
EMU3P5_VQ_PATH="${EMU3P5_VQ_PATH:-}"
EMU3P5_TOKENIZER_PATH="${EMU3P5_TOKENIZER_PATH:-}"
DREAMOMNI2_REPO_PATH="${DREAMOMNI2_REPO_PATH:-}"
DREAMOMNI2_MODEL_PATH="${DREAMOMNI2_MODEL_PATH:-}"
DREAMOMNI2_FLUX_PATH="${DREAMOMNI2_FLUX_PATH:-}"
SENSENOVA_REPO_PATH="${SENSENOVA_REPO_PATH:-}"
SENSENOVA_U15_MODEL_PATH="${SENSENOVA_U15_MODEL_PATH:-${REPO_ROOT}/models/SenseNova-U1.5-8B-MoT}"
SENSENOVA_U1_MODEL_PATH="${SENSENOVA_U1_MODEL_PATH:-${REPO_ROOT}/models/SenseNova-U1-8B-MoT}"
SENSENOVA_DTYPE="${SENSENOVA_DTYPE:-bfloat16}"
API_BASE_URL="${API_BASE_URL:-}"
API_KEY_ENV="${API_KEY_ENV:-OPENAI_API_KEY}"
NANOBANNA_MODEL_NAME="${NANOBANNA_MODEL_NAME:-gemini-3-pro-image-preview}"
NANO_BANANA_MODEL_NAME="${NANO_BANANA_MODEL_NAME:-gemini-2.5-flash-image}"
GPT_IMAGE_MODEL_NAME="${GPT_IMAGE_MODEL_NAME:-gpt-image-2}"
GPT_IMAGE_15_MODEL_NAME="${GPT_IMAGE_15_MODEL_NAME:-gpt-image-1.5}"
GPT_IMAGE_15_INPUT_FIDELITY="${GPT_IMAGE_15_INPUT_FIDELITY:-high}"
GPT_IMAGE_OUTPUT_SIZE="${GPT_IMAGE_OUTPUT_SIZE:-1024x1024}"
GPT_IMAGE_QUALITY="${GPT_IMAGE_QUALITY:-auto}"
GPT_IMAGE_OUTPUT_FORMAT="${GPT_IMAGE_OUTPUT_FORMAT:-png}"

[[ -s "${INPUT_JSON}" ]] || { echo "[Error] INPUT_JSON is missing or empty: ${INPUT_JSON}" >&2; exit 1; }
for flag in OVERWRITE SKIP_COMPLETED SAVE_GT ALLOW_VARIABLE_CONTEXTS FLUX_CPU_OFFLOAD; do
  [[ "${!flag}" == "0" || "${!flag}" == "1" ]] || { echo "[Error] ${flag} must be 0 or 1" >&2; exit 1; }
done
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# ========================= 5. GPU args =======================================
# Important:
# In Slurm, CUDA_VISIBLE_DEVICES may be something like 6,7.
# benchmark.run launches one worker per GPU and sets CUDA_VISIBLE_DEVICES
# inside each worker. Therefore we must pass the Slurm-assigned ids to --gpus.
GPU_ARGS=(
  --num_gpus "${NUM_GPUS}"
)

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU_ARGS+=(--gpus "${CUDA_VISIBLE_DEVICES}")

  IFS=',' read -ra VISIBLE_GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
  VISIBLE_GPU_COUNT="${#VISIBLE_GPU_ARRAY[@]}"

  if [[ "${VISIBLE_GPU_COUNT}" -ne "${NUM_GPUS}" ]]; then
    echo "[Warning] NUM_GPUS=${NUM_GPUS}, but CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} has ${VISIBLE_GPU_COUNT} entries."
    echo "[Warning] Reset NUM_GPUS to ${VISIBLE_GPU_COUNT} to match Slurm allocation."
    NUM_GPUS="${VISIBLE_GPU_COUNT}"
    GPU_ARGS=(
      --num_gpus "${NUM_GPUS}"
      --gpus "${CUDA_VISIBLE_DEVICES}"
    )
  fi

  GPU_INFO="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
else
  GPU_INFO="CUDA_VISIBLE_DEVICES is unset; benchmark.run will use cuda:0..cuda:$((NUM_GPUS - 1))"
fi

# ========================= 6. Common args ====================================
COMMON_ARGS=(
  --input_json "${INPUT_JSON}"
  --scannet_root "${SCANNET_ROOT}"
  --scannetpp_root "${SCANNETPP_ROOT}"
  --arkitscenes_root "${ARKITSCENES_ROOT}"
  --data_split "${DATA_SPLIT}"
  --output_root "${OUTPUT_ROOT}"
  --task "${TASK}"
  --max_samples "${MAX_SAMPLES}"
  --save_image_size "${SAVE_IMAGE_SIZE}"
  --input_image_size "${INPUT_IMAGE_SIZE}"
  --seed "${SEED}"
)

if [[ -n "${METHOD_NAME}" ]]; then
  COMMON_ARGS+=(--method_name "${METHOD_NAME}")
fi

if [[ "${ALLOW_VARIABLE_CONTEXTS}" == "1" ]]; then
  COMMON_ARGS+=(--allow_variable_contexts)
elif [[ "${ALLOW_VARIABLE_CONTEXTS}" != "0" ]]; then
  echo "[Error] ALLOW_VARIABLE_CONTEXTS must be 0 or 1, got: ${ALLOW_VARIABLE_CONTEXTS}"
  exit 1
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  COMMON_ARGS+=(--overwrite)
fi

if [[ "${SKIP_COMPLETED}" == "1" ]]; then
  COMMON_ARGS+=(--skip_completed)
else
  COMMON_ARGS+=(--no_skip_completed)
fi

if [[ "${SAVE_GT}" == "1" ]]; then
  COMMON_ARGS+=(--save_gt)
else
  COMMON_ARGS+=(--no_save_gt)
fi

run_with_env() {
  local python_bin="$1" display_name="$2" execution_mode="$3"
  shift 3
  local -a parallel_args
  if [[ "${execution_mode}" == "api" ]]; then
    parallel_args=(--num_gpus 0 --num_workers "${API_NUM_WORKERS}" --device cpu)
  else
    parallel_args=("${GPU_ARGS[@]}")
  fi
  echo "[Run] model=${display_name}, python=${python_bin}, workers=${parallel_args[*]}"
  evaluation_python "${python_bin}" -m benchmark.run "${COMMON_ARGS[@]}" "${parallel_args[@]}" "$@"
}

run_with_python() {
  local python_bin="$1" display_name="$2"
  shift 2
  run_with_env "${python_bin}" "${display_name}" gpu "$@"
}

check_api_key() {
  if [[ -z "${!API_KEY_ENV:-}" ]]; then
    echo "[Error] Environment variable ${API_KEY_ENV} is not set."
    echo "        Please run: export ${API_KEY_ENV}=your_api_key"
    exit 1
  fi
}

should_run_model() {
  local model="$1"

  if [[ "${RUN_MODELS}" == "all" ]]; then
    return 0
  fi

  local item
  IFS=',' read -ra RUN_MODEL_ARRAY <<< "${RUN_MODELS}"
  for item in "${RUN_MODEL_ARRAY[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ "${item}" == "${model}" ]]; then
      return 0
    fi
  done

  return 1
}

if [[ "${RUN_MODELS}" != "all" ]]; then
  IFS=',' read -ra requested_models <<< "${RUN_MODELS}"
  for model in "${requested_models[@]}"; do
    case "${model}" in bagel|flux2|qwen-image-edit-2511|emu3.5|dreamomni2|sensenova-u1|sensenova-u1.5|nanobanna-pro|nano-banana|gpt-image-2|gpt-image-1.5|dummy) ;;
      *) echo "[Error] Unknown backend model: ${model}" >&2; exit 1 ;;
    esac
  done
fi
if should_run_model "emu3.5" && [[ -z "${EMU3P5_REPO_PATH}" ]]; then
  echo "[Error] Emu3.5 requires EMU3P5_REPO_PATH with its official inference modules." >&2; exit 1
fi
if should_run_model "dreamomni2" && [[ ! -f "${DREAMOMNI2_REPO_PATH}/dreamomni2/pipeline_dreamomni2.py" || ! -f "${DREAMOMNI2_REPO_PATH}/utils/vprocess.py" ]]; then
  echo "[Error] DreamOmni2 requires the official repository in DREAMOMNI2_REPO_PATH (dreamomni2/pipeline_dreamomni2.py and utils/vprocess.py). See evaluation/benchmark_runner/README.md." >&2; exit 1
fi
if should_run_model "nanobanna-pro" || should_run_model "nano-banana"; then
  [[ -n "${API_BASE_URL}" ]] || { echo "[Error] Gemini adapters require API_BASE_URL for an OpenAI-compatible image gateway." >&2; exit 1; }
fi
if should_run_model "dummy"; then
  run_with_env "${PYTHON_BIN}" dummy api --model dummy --run_name "dummy_${RUN_TAG}"
fi

# ========================= 7. Run models sequentially =========================
echo "[Config] RUN_MODELS=${RUN_MODELS}"
echo "[Config] SEED=${SEED}"
echo "[Config] OVERWRITE=${OVERWRITE}, SKIP_COMPLETED=${SKIP_COMPLETED}"
echo "[Config] ALLOW_VARIABLE_CONTEXTS=${ALLOW_VARIABLE_CONTEXTS}"

# 7.1 BAGEL
BAGEL_ARGS=(
  --model bagel
  --run_name "bagel_${RUN_TAG}"
  --model_path "${BAGEL_MODEL_PATH}"
  --max_mem_per_gpu "${BAGEL_MAX_MEM_PER_GPU}"
  --offload_folder "${BAGEL_OFFLOAD_FOLDER}"
  --cfg_text_scale 4.0
  --cfg_img_scale 2.0
  --cfg_interval_start 0.0
  --cfg_interval_end 1.0
  --timestep_shift 3.0
  --num_timesteps 50
  --cfg_renorm_min 0.0
  --cfg_renorm_type global
  --bagel_thinking "${BAGEL_THINKING}"
)

if [[ -n "${BAGEL_REPO_PATH}" ]]; then
  BAGEL_ARGS+=(--repo_path "${BAGEL_REPO_PATH}")
fi

if [[ -n "${BAGEL_CHECKPOINT_PATH}" ]]; then
  BAGEL_ARGS+=(--bagel_checkpoint_path "${BAGEL_CHECKPOINT_PATH}")
fi

if should_run_model "bagel"; then
  run_with_env "${BAGEL_PYTHON}" "bagel" gpu "${BAGEL_ARGS[@]}"
else
  echo "[Skip] model=bagel"
fi

# 7.2 FLUX2
FLUX_ARGS=(
  --model flux2
  --run_name "flux2_${RUN_TAG}"
  --model_path "${FLUX_MODEL_PATH}"
  --dtype "${FLUX_DTYPE}"
  --height "${FLUX_HEIGHT}"
  --width "${FLUX_WIDTH}"
  --guidance_scale "${FLUX_GUIDANCE_SCALE}"
  --num_inference_steps "${FLUX_NUM_INFERENCE_STEPS}"
  --generator_device cuda
)

if [[ "${FLUX_CPU_OFFLOAD}" == "1" ]]; then
  FLUX_ARGS+=(--cpu_offload)
fi

if should_run_model "flux2"; then
  run_with_env "${FLUX_PYTHON}" "flux2" gpu "${FLUX_ARGS[@]}"
else
  echo "[Skip] model=flux2"
fi

# 7.2a Qwen-Image-Edit-2511: use its configured Python runtime.
QWEN_ARGS=(
  --model qwen-image-edit-2511
  --run_name "qwen-image-edit-2511_${RUN_TAG}"
  --model_path "${QWEN_MODEL_PATH}"
  --qwen_num_inference_steps "${QWEN_NUM_INFERENCE_STEPS}"
  --generator_device cuda
)

if should_run_model "qwen-image-edit-2511"; then
  run_with_python "${QWEN_PYTHON}" "qwen-image-edit-2511" "${QWEN_ARGS[@]}"
else
  echo "[Skip] model=qwen-image-edit-2511"
fi

# 7.2b SenseNova-U1.5: use its configured Python runtime
SENSENOVA_U15_ARGS=(
  --model sensenova-u1.5
  --run_name "sensenova-u1.5_${RUN_TAG}"
  --repo_path "${SENSENOVA_REPO_PATH}"
  --model_path "${SENSENOVA_U15_MODEL_PATH}"
  --dtype "${SENSENOVA_DTYPE}"
  --sensenova_u15_thinking "${SENSENOVA_U15_THINKING}"
)

if should_run_model "sensenova-u1.5"; then
  run_with_python "${SENSENOVA_PYTHON}" "sensenova-u1.5" "${SENSENOVA_U15_ARGS[@]}"
else
  echo "[Skip] model=sensenova-u1.5"
fi

# 7.2c SenseNova-U1: use the same configured Python runtime
SENSENOVA_U1_ARGS=(
  --model sensenova-u1
  --run_name "sensenova-u1_${RUN_TAG}"
  --repo_path "${SENSENOVA_REPO_PATH}"
  --model_path "${SENSENOVA_U1_MODEL_PATH}"
  --dtype "${SENSENOVA_DTYPE}"
  --sensenova_u1_thinking "${SENSENOVA_U1_THINKING}"
)

if should_run_model "sensenova-u1"; then
  run_with_python "${SENSENOVA_PYTHON}" "sensenova-u1" "${SENSENOVA_U1_ARGS[@]}"
else
  echo "[Skip] model=sensenova-u1"
fi

# 7.3 nanobanna-pro: uses API_PYTHON
if should_run_model "nanobanna-pro"; then
  check_api_key

  run_with_env "${API_PYTHON}" "nanobanna-pro" api \
    --model nanobanna-pro \
    --run_name "nanobanna-pro_${RUN_TAG}" \
    --model_name "${NANOBANNA_MODEL_NAME}" \
    --base_url "${API_BASE_URL}" \
    --api_key_env "${API_KEY_ENV}" \
    --temperature 1.0
else
  echo "[Skip] model=nanobanna-pro"
fi

# 7.3a Nano Banana: same API path as Nano Banana Pro, with the official stable model ID
if should_run_model "nano-banana"; then
  check_api_key

  run_with_env "${API_PYTHON}" "nano-banana" api \
    --model nano-banana \
    --run_name "nano-banana_${RUN_TAG}" \
    --model_name "${NANO_BANANA_MODEL_NAME}" \
    --base_url "${API_BASE_URL}" \
    --api_key_env "${API_KEY_ENV}" \
    --temperature 1.0
else
  echo "[Skip] model=nano-banana"
fi

# 7.4 gpt-image-2: uses API_PYTHON
if should_run_model "gpt-image-2"; then
  check_api_key

  run_with_env "${API_PYTHON}" "gpt-image-2" api \
    --model gpt-image-2 \
    --run_name "gpt-image-2_${RUN_TAG}" \
    --model_name "${GPT_IMAGE_MODEL_NAME}" \
    --base_url "${API_BASE_URL}" \
    --api_key_env "${API_KEY_ENV}" \
    --output_size "${GPT_IMAGE_OUTPUT_SIZE}" \
    --quality "${GPT_IMAGE_QUALITY}" \
    --output_format "${GPT_IMAGE_OUTPUT_FORMAT}"
else
  echo "[Skip] model=gpt-image-2"
fi

# 7.4a GPT Image 1.5: same Images Edit request as GPT Image 2, with configurable input fidelity
if should_run_model "gpt-image-1.5"; then
  check_api_key

  run_with_env "${API_PYTHON}" "gpt-image-1.5" api \
    --model gpt-image-1.5 \
    --run_name "gpt-image-1.5_${RUN_TAG}" \
    --model_name "${GPT_IMAGE_15_MODEL_NAME}" \
    --base_url "${API_BASE_URL}" \
    --api_key_env "${API_KEY_ENV}" \
    --output_size "${GPT_IMAGE_OUTPUT_SIZE}" \
    --quality "${GPT_IMAGE_QUALITY}" \
    --output_format "${GPT_IMAGE_OUTPUT_FORMAT}" \
    --input_fidelity "${GPT_IMAGE_15_INPUT_FIDELITY}"
else
  echo "[Skip] model=gpt-image-1.5"
fi

# 7.5 Emu3.5: use its configured Python runtime.
EMU3P5_ARGS=(
  --model emu3.5
  --run_name "emu3.5_${RUN_TAG}"
  --repo_path "${EMU3P5_REPO_PATH}"
  --model_path "${EMU3P5_MODEL_PATH}"
  --emu_vq_path "${EMU3P5_VQ_PATH}"
  --emu_tokenizer_path "${EMU3P5_TOKENIZER_PATH}"
)

if should_run_model "emu3.5"; then
  run_with_python "${EMU3P5_PYTHON}" "emu3.5" "${EMU3P5_ARGS[@]}"
else
  echo "[Skip] model=emu3.5"
fi

# 7.6 DreamOmni2: direct multi-image input in its isolated Python runtime.
DREAMOMNI2_ARGS=(
  --model dreamomni2
  --run_name "dreamomni2_${RUN_TAG}"
  --repo_path "${DREAMOMNI2_REPO_PATH}"
  --model_path "${DREAMOMNI2_MODEL_PATH}"
  --dreamomni2_flux_path "${DREAMOMNI2_FLUX_PATH}"
)

if should_run_model "dreamomni2"; then
  run_with_python "${DREAMOMNI2_PYTHON}" "dreamomni2" "${DREAMOMNI2_ARGS[@]}"
else
  echo "[Skip] model=dreamomni2"
fi

echo
echo "All selected models finished. Results are under: ${OUTPUT_ROOT}"
