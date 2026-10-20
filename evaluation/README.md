# Evaluation

Run BAGEL-SFT generation, evaluate predictions, and build validated six-component reports. The frozen index is `data/merged_dataset_final.jsonl`: 1,500 aligned scene/query/target cases, stored as 4,500 C3/C6/C10 records. RGB/depth assets and model weights are separate downloads.

## Run existing models

From the repository root, prepare [resources and configuration](../README.md#download-resources), then run:

```bash
CONFIG_FILE=/absolute/path/config.env bash evaluation/run.sh all bagel bagel-sft-final
# Or execute stages separately:
bash evaluation/run.sh generate bagel
bash evaluation/run.sh metrics bagel
bash evaluation/run.sh report bagel
bash evaluation/run.sh check
```

Pass the same `CONFIG_FILE` to every stage, or export configuration variables directly. It loads only when explicitly set; shell startup files and Conda environments are never activated. Use `bagel-sft-final` for the bundled setup. `check` validates the bundled metadata/hash and script syntax without loading models. It also checks reports if metrics exist, but does not inspect downloaded RGB/depth files.

| Methods | Interpreter | Required runtime/model settings |
| --- | --- | --- |
| `bagel`, `bagel-sft-final` | `BAGEL_PYTHON` | Bundled `sft/`, `BAGEL_MODEL_PATH`; SFT also needs `BAGEL_SFT_FINAL_CHECKPOINT` |

Interpreter overrides default to `PYTHON_BIN`. The bootstrap installs CLIP/DINO,
Qwen-VL, and VGGT in the shared `mindview_metrics` environment, exposed through
`CLIP_DINO_PYTHON`, `QWEN_VL_PYTHON`, and `VGGT_PYTHON`. The frozen judge is
`Qwen/Qwen3-VL-30B-A3B-Instruct`. See
[generation](benchmark_runner/README.md) and [metrics](benchmarker_metrics/README.md)
for backend options.

Defaults are `outputs/generations/<method>/<run>/`, `outputs/metrics/<method>/`, and `outputs/reports/`. Override `GENERATION_ROOT`, `METRICS_ROOT`, and `REPORT_ROOT` to isolate runs. `NUM_GPUS` controls local workers; `API_NUM_WORKERS` controls API parallelism. Generation/metrics default to safe resume (`OVERWRITE=0`, `SKIP_COMPLETED=1`). Generation rejects changed input/configuration signatures; metrics compare image/depth bytes and runtime fingerprints. Local weight signatures use path/stat metadata, so retain pinned snapshots. `REPORT_OVERWRITE=1` atomically replaces an existing report.

## Inputs and protocols

The [MindView-Bench snapshot](https://huggingface.co/datasets/wangyanbo/MindView-Bench/tree/f89c6d8342cb0ed5ff098115967e23813ec250ce) supplies all benchmark RGB and millimetre-depth assets. Use one fixed commit for metadata and TARs; follow the root [download and extraction commands](../README.md#download-resources). Set roots to the extracted dataset directories:

```text
SCANNET_ROOT:     <root>/<scene>/color/<frame>.jpg
SCANNETPP_ROOT:   <root>/color/<split>/<scene>/<frame>.jpg
ARKITSCENES_ROOT: <root>/color/<split>/<scene>/<scene>_<frame>.png
```

`DATA_SPLIT=val` is the default. Depth paths replace `color` with `depth` and end in `.png`; values are millimetres. VGGT requires individual source RGB/depth frames, including the complete ten-view scoring reference and target.

Every input row keeps these nine frozen fields: `sample_id`, `dataset`, `task_type`, `scene_id`, `gt_frame_id`, `anchor_frame_id`, `shot`, `contexts_frame_id`, and `instruction`. Preserve IDs, prompt text, target, and context order. Standard `shot` values are 3, 6, and 10 with exactly that many contexts. Model inputs never include the target image. Instructions are passed unchanged except for the documented Qwen/Emu anchor/contact-sheet adaptation.

| Evaluation | `INPUT_JSON` | `REPORT_PROTOCOL` |
| --- | --- | --- |
| Full C3/C6/C10 | Bundled frozen index, or downloaded `context_scaling/metadata.jsonl` | `paper` (default) |
| Official C6, 1,500 cases | Downloaded `data/test.jsonl` | `selected` |
| Smoke-test subset | Output of `sample_subset.py` | `selected` |

For C6-only or subset runs also set `VGGT_CANONICAL_JSON` to the **full 4,500-record index**. This restores the same ten scoring views for every C3/C6/C10 prediction while preserving saved GT/prediction encodings. Keep these settings in `CONFIG_FILE`, or export them with `CONFIG_FILE` unset. For example, after configuring dataset roots and model interpreters:

```bash
export INPUT_JSON="$PWD/resources/MindView-Bench/data/test.jsonl"
export VGGT_CANONICAL_JSON="$PWD/evaluation/data/merged_dataset_final.jsonl"
export REPORT_PROTOCOL=selected
export GENERATION_ROOT="$PWD/outputs/official_c6/generations"
export METRICS_ROOT="$PWD/outputs/official_c6/metrics"
export REPORT_ROOT="$PWD/outputs/official_c6/reports"
bash evaluation/run.sh all bagel
```

For the full protocol restore the frozen `INPUT_JSON`, `REPORT_PROTOCOL=paper`, and a separate output location.

### Reproducible ten-example smoke test

The sampler uses only the 1,500 frozen C6 rows. Seeded SHA256 ranking chooses cases, and output keeps the original JSONL lines in source order:

```bash
python evaluation/sample_subset.py \
  --source-jsonl evaluation/data/merged_dataset_final.jsonl \
  --count 10 --seed 42 --output-jsonl outputs/sample10/input.jsonl --dry-run
# Remove --dry-run to write the subset and input.manifest.json.
```

The manifest records source/subset SHA256, seed, selected IDs, and task counts. An identical request reuses identical outputs; changed or incomplete outputs are rejected. Frozen inputs cannot be overwritten. Set `INPUT_JSON` to this subset, `REPORT_PROTOCOL=selected`, `VGGT_CANONICAL_JSON` to the full index, and use isolated generation/metrics/report roots. Set `MAX_SAMPLES=-1` to avoid additional truncation, then call the same `evaluation/run.sh generate`, `metrics`, and `report` stages.

Ten random cases need not cover all seven tasks. Their report averages only the observed task means and is a pipeline smoke test, **not a paper benchmark score**.

## Evaluate images generated outside this repository

Place one **512 × 512, RGB-mode PNG** per selected sample in a flat folder, named exactly `<sample_id>.png`. The folder must contain exactly the selected image IDs. Keep the matching frozen input rows in `INPUT_JSON`; use the C6/subset settings above and configure the dataset roots. Import without resizing or recolouring predictions:

```bash
python evaluation/import_predictions.py \
  --input-jsonl "$INPUT_JSON" --predictions-dir /absolute/path/predictions \
  --model my-model --run-name external --generation-root "$GENERATION_ROOT" \
  --scannet-root "$SCANNET_ROOT" --scannetpp-root "$SCANNETPP_ROOT" \
  --arkitscenes-root "$ARKITSCENES_ROOT"
bash evaluation/run.sh metrics my-model
bash evaluation/run.sh report my-model
```

`--dry-run` validates without writing. The importer checks every row against the frozen index, rejects duplicate/unknown/mismatched IDs and unreadable or incorrectly sized predictions, and obtains GT and ordered contexts from the canonical metadata plus dataset roots. Predictions are copied byte-for-byte; GT is saved with the runner's standard 512px centre-square preprocessing. Existing run directories are rejected. It writes no scores; all metrics are computed by the normal metric modules. Custom method names are supported by metrics and report discovery.

The discovery contract is `GENERATION_ROOT/<model>/<run>/generation_results.json`, a JSON **list** of per-sample objects. Imported records contain the nine original fields plus:

| Field | Value |
| --- | --- |
| `model`, `status` | Directory method ID and `"success"` |
| `prediction.image_path` | Absolute path to the saved 512px RGB prediction PNG |
| `ground_truth.image_path` | Absolute path to the runner-preprocessed target JPEG |
| `ground_truth.source_path` | Original dataset target RGB path, used to locate metric depth |
| `contexts.source_paths` | Ordered individual source RGB paths; never a stitched montage |
| `input.image_paths` | Same individual context paths |
| `pred_image`, `gt_image` | Compatibility aliases for prediction/GT saved paths |

Normal generation also records optional context montages and response artifacts. The loader supports these nested image paths and compatibility aliases; a bare image folder or an `images` JSON field alone is insufficient. Use the importer to produce a complete record instead of inventing GT/context paths or manually joining IDs.

## Scores and validation

The six components are DINO-I × 100, Semantic, Spatial, Quality, VGGT-RScore, and VGGT-TScore. CLIP-I is auxiliary. The frozen VLM judge receives GT/prediction for Semantic/Spatial and prediction alone for Quality. VGGT scores are `100 * clip(1 - rotation_error_deg / 60, 0, 1)` and `100 * clip(1 - translation_error_m / 2, 0, 1)`.

Reports require all selected IDs in all three metric modules, validate identity/configuration/ranges, average samples within tasks, and equally average task means. Avg6 equally averages the six components; the full protocol also gives C3/C6/C10 equal weight. Reports contain full-precision values and provenance; per-module summaries are diagnostic. Closed API images are not guaranteed to reproduce pixel for pixel; downloaded-layout GPU validation remains separate from CPU checks.

```bash
PYTHONPATH=evaluation/benchmark_runner:evaluation/benchmarker_metrics \
  python -m pytest evaluation/tests evaluation/benchmarker_metrics/tests
```

CPU tests cover entrypoints, deterministic sampling, external-image import, report coverage/aggregation, and strict resume. Tests importing metric modules skip when PyTorch is unavailable; they do not download weights or run GPU inference.
