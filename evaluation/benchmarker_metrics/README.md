# Metrics and reports

From the repository root:

```bash
bash evaluation/run.sh metrics bagel flux2
bash evaluation/run.sh report bagel flux2
REPORT_OVERWRITE=1 bash evaluation/run.sh report bagel flux2
```

The metrics launcher reads completed generations from `GENERATION_ROOT` and writes `METRICS_ROOT/<method>/`. Each selected method is evaluated independently. It takes an isolated snapshot of generation metadata, retains paths to images, and holds output locks while workers and summaries run. A partially written shard line is ignored during snapshotting. Input RGB/depth datasets and metric weights must be available separately.

The three modules preserve their scoring algorithms:

- `clip_dino`: GT–prediction CLIP-I and DINO-I feature cosine similarity.
- `qwen_vl`: independent semantic/spatial GT–prediction judgements and prediction-only quality judgement. Judge v2 emits 0–10 scores and multiplies by 10. The quality view preserves the prediction's aspect ratio.
- `vggt`: jointly estimates poses from individual contexts, GT and prediction. Within a normal C3/C6/C10 group all settings use the matching C10 reference contexts. Sensor depth calibrates translation to metres. RScore is `100 * clip(1 - rotation_error_deg / 60, 0, 1)`; TScore is `100 * clip(1 - translation_error_m / 2, 0, 1)`.

The bootstrap uses the shared `mindview_metrics` interpreter for
`CLIP_DINO_PYTHON`, `QWEN_VL_PYTHON`, and `VGGT_PYTHON`. The default judge is
`Qwen/Qwen3-VL-30B-A3B-Instruct`. The released scoring protocol uses this judge.
`VGGT_REPO_PATH` defaults to the bundled `vggt/`; `VGGT_MODEL_NAME`/`VGGT_MODEL_PATH`
accepts a model ID or local snapshot. `RUN_CLIP_DINO`, `RUN_QWEN_VL`, and
`RUN_VGGT` can disable modules for intermediate work; final reports still require
all six components. `SHOTS`, `TASKS`, `DATASETS`, `RUNS`, and `MAX_SAMPLES` filter
metric discovery.

Install the bundled `vggt/` package and a compatible OpenCV build into
`mindview_metrics`.

The bootstrap pins Transformers 4.57.6 with the matching tokenizer and Hugging Face Hub versions for the Qwen3-VL judge.

## Obtain metric checkpoints

These official repositories and immutable revisions were reachable anonymously on
2026-10-16. Install the corresponding `evaluation/requirements-*.txt` in each
metric environment. Record the chosen checkpoint revision with your run; the
availability check does not establish GPU equivalence with historical results.

| Metric | Official model and checked revision | Configuration |
|---|---|---|
| CLIP-I (auxiliary) | [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14/tree/32bd64288804d66eefd0ccbe215aa642df71cc41), `32bd64288804d66eefd0ccbe215aa642df71cc41` | `CLIP_MODEL_NAME` |
| DINO-I | [facebook/dinov2-large](https://huggingface.co/facebook/dinov2-large/tree/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c), `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c` | `DINO_MODEL_NAME` |
| Semantic / Spatial / Quality | [Qwen/Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/tree/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c), `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c` | `QWEN_MODEL_NAME` |
| RScore / TScore | [facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B/tree/860abec7937da0a4c03c41d3c269c366e82abdf9), `860abec7937da0a4c03c41d3c269c366e82abdf9` | `VGGT_MODEL_PATH` or `VGGT_MODEL_NAME` |

From the repository root, download local snapshots to preserve these revisions:

```bash
hf download openai/clip-vit-large-patch14 \
  --revision 32bd64288804d66eefd0ccbe215aa642df71cc41 \
  --local-dir "$PWD/models/clip-vit-large-patch14"
hf download facebook/dinov2-large \
  --revision 47b73eefe95e8d44ec3623f8890bd894b6ea2d6c \
  --local-dir "$PWD/models/dinov2-large"
hf download Qwen/Qwen3-VL-30B-A3B-Instruct \
  --revision 9c4b90e1e4ba969fd3b5378b57d966d725f1b86c \
  --local-dir "$PWD/models/Qwen3-VL-30B-A3B-Instruct"
hf download facebook/VGGT-1B config.json model.safetensors \
  --revision 860abec7937da0a4c03c41d3c269c366e82abdf9 \
  --local-dir "$PWD/models/VGGT-1B"
```

Set the four configuration variables to these absolute snapshot directories.
RGB/depth download, extraction, and fixed C10 reference configuration are in
[evaluation](../README.md) and the [shared resource guide](../../README.md#download-resources).

`OVERWRITE=0`, `SKIP_COMPLETED=1` are the defaults. Metric cache reuse requires successful, finite results matching the content hashes of actual prediction/GT/context/depth inputs, code/configuration signatures and runtime fingerprints. Use `OVERWRITE=1 SKIP_COMPLETED=0` to recompute.

The authoritative report builder is `build_metric_report.py` with `metric_report_config.json`:

```bash
python evaluation/benchmarker_metrics/build_metric_report.py --models bagel,flux2 --check-only
python evaluation/benchmarker_metrics/build_metric_report.py --models bagel,flux2 --overwrite
```

It reads per-sample `results.json` from each model/module, checks complete frozen sample coverage and identities, then averages within tasks and equally across tasks. DINO-I is multiplied by 100; CLIP-I is excluded from Avg6. RScore/TScore are averaged from their stored per-sample scores, rather than reconstructed from mean raw errors. Full-precision CSV/JSON and provenance manifests are written atomically to `outputs/reports`; rounded `.paper.csv` files are presentation copies. Raw `benchmarker_metrics.summarize` tables remain diagnostic summaries and are not the final task-macro report.

Tests cover report aggregation/scaling/coverage, metric selection, common-reference inputs and strict resume. Run lightweight tests without model weights:

```bash
PYTHONPATH=evaluation/benchmark_runner:evaluation/benchmarker_metrics python -m pytest evaluation/tests evaluation/benchmarker_metrics/tests
```

Tests that import metric modules require PyTorch and skip explicitly when it is unavailable. They do not load model weights or execute GPU inference.
