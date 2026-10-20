# Generation adapters

Use `bash evaluation/run.sh generate MODEL ...` from the repository root. Supported methods are `bagel`, `bagel-sft-final`, `flux2`, `qwen-image-edit-2511`, `emu3.5`, `dreamomni2`, `sensenova-u1`, `sensenova-u1.5`, `gpt-image-2`, `nano-banana-pro`, `gpt-image-1.5`, and `nano-banana`. No model list means all twelve. `bagel-sft-final` uses the BAGEL backend with `BAGEL_SFT_FINAL_CHECKPOINT`. The `dummy` adapter copies an input image for CPU pipeline smoke checks.

`task_run_final_paper.sh` selects public method names and calls `run_benchmark.sh`; the latter launches `python -m benchmark.run`. Direct Python use requires this directory on `PYTHONPATH`:

```bash
PYTHONPATH=evaluation/benchmark_runner python -m benchmark.run \
  --model dummy --num_gpus 0 --device cpu \
  --input_json evaluation/data/merged_dataset_final.jsonl \
  --scannet_root /datasets/scannet/scans \
  --scannetpp_root /datasets/scannetpp \
  --arkitscenes_root /datasets/arkitscenes --max_samples 3
```

The shell accepts `INPUT_JSON`, the three dataset roots, `DATA_SPLIT`, `TASK`, `MAX_SAMPLES`, `INPUT_IMAGE_SIZE`, `SAVE_IMAGE_SIZE`, `SEED`, `NUM_GPUS`, `API_NUM_WORKERS`, `RUN_TAG`, and the interpreter/model variables documented in [evaluation](../README.md). Defaults preserve the released inference settings; each saved prediction and GT is 512 × 512. Local models use one worker per selected GPU. API models use CPU workers. The public `nano-banana-pro` method maps to the historical internal backend ID `nanobanna-pro`.

Inputs contain `sample_id`, `dataset`, `task_type`, `scene_id`, `gt_frame_id`, `anchor_frame_id`, `shot`, `contexts_frame_id`, and `instruction`. Context counts equal shot (3, 6 or 10). Only `ALLOW_VARIABLE_CONTEXTS=1` permits logical C10 with 6–10 retained contexts. Qwen Image Edit and Emu3.5 use the original anchor/contact-sheet protocol; DreamOmni2 uses all individual contexts in order. Instructions and model algorithms are retained.

Generation writes predictions, GT copies, context montages, response artifacts, per-worker shards, `generation_results.json`, `summary.json`, and signed `run_config.json` under `GENERATION_ROOT/<method>/<backend>_<RUN_TAG>/`. Resume checks the input JSONL hash, generation settings, implementation hashes and local model path/stat metadata before checking saved images. A changed seed, prompt/input, checkpoint or model metadata requires a new run directory or `OVERWRITE=1`. Parallel worker counts and GPU assignment can change during resume. Legacy unsigned outputs require regeneration. The signature does not hash the bytes of every dataset frame or multi-GB weight.

BAGEL imports the bundled `sft/` runtime by default. Emu3.5 and SenseNova depend on separately installed upstream runtimes.

## Obtain generation resources

Download benchmark RGB/depth and configure the source roots using the
[shared resource guide](../../README.md#download-resources). Generation models
and API credentials are separate from the MindView datasets. Official sources
checked on 2026-10-16 are:

| Method | Official resources | Runtime configuration |
|---|---|---|
| BAGEL / BAGEL-SFT | [BAGEL base](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT/tree/5019f57d168e5816e8f3f701b17cc816bb7cf24b); [published SFT EMA and instructions](../../sft/README.md) | `BAGEL_MODEL_PATH`; SFT also needs `BAGEL_SFT_FINAL_CHECKPOINT` |
| FLUX.2 | [FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/tree/92196c8e11f7b6cf2b7493e037d8c5345c559216) | `FLUX_MODEL_PATH` |
| Qwen Image Edit | [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511/tree/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9) | `QWEN_MODEL_PATH` |
| Emu3.5 | [Official code and installation](https://github.com/baaivision/Emu3.5); [Emu3.5 weights](https://huggingface.co/BAAI/Emu3.5); [vision tokenizer](https://huggingface.co/BAAI/Emu3.5-VisionTokenizer) | `EMU3P5_REPO_PATH`, `EMU3P5_MODEL_PATH`, `EMU3P5_VQ_PATH`, optional `EMU3P5_TOKENIZER_PATH` |
| SenseNova-U1 | [Official code and installation](https://github.com/OpenSenseNova/SenseNova-U1); [U1 weights](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT/tree/bfa9b436503cb8aed4f2bc60e3236710cc77468d) | `SENSENOVA_REPO_PATH`, `SENSENOVA_U1_MODEL_PATH` |
| SenseNova-U1.5 | Same official runtime; [U1.5 weights](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT/tree/9feeeab8a2792514d109cd34589342a2cc1d4ab2) | `SENSENOVA_REPO_PATH`, `SENSENOVA_U15_MODEL_PATH` |
| DreamOmni2 | Fixed code, model and FLUX revisions are listed below. | `DREAMOMNI2_REPO_PATH`, `DREAMOMNI2_MODEL_PATH`, `DREAMOMNI2_FLUX_PATH` |

Use `hf download <model-id> --revision <full-commit> --local-dir <absolute-directory>`
to preserve a chosen model snapshot, then point its path variable at that
directory. Model-card access controls also apply: FLUX.2 is gated and may require
accepting its conditions and logging in with an approved personal account.
Install each runtime's own dependencies in its configured Python environment.
Availability of a model repository does not verify generation with that runtime.

The Emu3.5 adapter imports `configs.test_x2i_common`, `inference`,
`src.utils.model_utils`, and `src.utils.vis_proto` from its external checkout;
the configured runtime must provide all four. Keep the exact model variant,
tokenizers, configuration, and runtime revision together. Compatibility with
the latest upstream checkout has not been validated. The four API methods use
the endpoint and credentials described below; no downloadable checkpoint is
distributed with MindView for those methods.

## DreamOmni2

Set `DREAMOMNI2_REPO_PATH` to a checkout of the [official repository](https://github.com/JIA-Lab-research/DreamOmni2/blob/b0b98e03bfc95c2bf0f27a5f9ca33d9b16a525b5/inference_gen.py) containing `dreamomni2/pipeline_dreamomni2.py` and `utils/vprocess.py`, and install its dependencies in `DREAMOMNI2_PYTHON`:

```bash
git clone https://github.com/JIA-Lab-research/DreamOmni2 /models/DreamOmni2-code
git -C /models/DreamOmni2-code checkout b0b98e03bfc95c2bf0f27a5f9ca33d9b16a525b5
```

Provide `DREAMOMNI2_MODEL_PATH` with `vlm-model/` and `gen_lora/`, plus `DREAMOMNI2_FLUX_PATH` with the FLUX base snapshot. The historical model revisions are `xiabs/DreamOmni2@38eb68e8f546bec36680180fd922e61b7112a7e1` and `black-forest-labs/FLUX.1-Kontext-dev@24e9dedc4ef646698dc8eb4e18ae2cec3c9fea0d`. If paths are unset, the wrapper resolves only already cached copies of those exact revisions.

The included `benchmark/dreamomni_runtime.py` replaces the unavailable experiment-specific `compare_benchmark_shots.py`. It uses the official pipeline/VLM preprocessing and the recovered benchmark constants: generation suffix ` It is generation task.`, deterministic VLM generation capped at 4096 tokens, LoRA weight 1, 30 diffusion steps, guidance 3.5, and 1024 × 1024 output. The prompt extraction matches the saved raw/refined benchmark artifact. The adapter keeps individual context order and paired-shot seed derivation; the wrapper explicitly uses a generator on the configured device. These settings and calls are reviewable, but pixel-level equality with the missing historical wrapper has not been tested on GPU. The upstream [Apache-2.0 license](../third_party/dreamomni2/LICENSE) and [wrapper notice](../third_party/dreamomni2/NOTICE.md) are retained.

For API methods, store the key in the variable named by `API_KEY_ENV` (default `OPENAI_API_KEY`). An empty `API_BASE_URL` uses the OpenAI client's standard endpoint for GPT Image. Gemini image adapters require an explicitly configured compatible gateway. Model IDs, output size, quality and format remain configurable; no private gateway is included.

To add a backend, register a `BaseModel` implementation in `benchmark/models.py`. Its `infer` receives an `InferenceRequest` and returns a PIL image plus metadata. Task validation and frame-path resolution live in `benchmark/tasks.py` and `benchmark/data_paths.py`.
