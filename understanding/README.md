# BAGEL-SFT spatial understanding

This workflow evaluates the configured BAGEL-SFT EMA on exactly three
understanding benchmarks. It shares the model runtime in `../sft/` and keeps
the original VLM image transform in `image_transform.yaml`: maximum/minimum
image size 980/378, stride 14, and 2,007,040 maximum pixels. This is separate
from generation/SFT image preprocessing.

| Benchmark | Fixed protocol | Full inference calls | Maximum new tokens |
|---|---|---:|---:|
| `sat-v2` | 150 source questions, original then reversed option order | 300 | 1,024 |
| `mmsi` | MMSI-Bench Vanilla | 1,000 | 2,048 |
| `mindcube-tiny` | MindCube TinyBench raw-QA | 1,050 | 4,096 |

All three use greedy decoding, no thinking, seed 0, BF16 model parameters,
and BF16 autocast. The default is one complete model on one visible CUDA GPU.
The workflow does not run a vanilla/SFT comparison or substitute historical
paper scores. Scores come from the selected checkpoint's saved answers.

## Environment and model

Use the BAGEL Python 3.10 environment prepared as described in
[SFT](../sft/README.md), including its documented FlashAttention installation:

```bash
python -m pip install -r understanding/requirements.txt
export EVAL_PYTHON=/absolute/path/to/bagel-environment/bin/python
export BAGEL_MODEL_PATH=/absolute/path/to/BAGEL-7B-MoT
export BAGEL_SFT_FINAL_CHECKPOINT=/absolute/path/to/ema.safetensors
```

`BAGEL_MODEL_PATH` supplies the official configs and tokenizer;
`BAGEL_SFT_FINAL_CHECKPOINT` supplies the full MindView EMA, including the BF16
release format. The understanding loader streams only required model tensors
and casts floating parameters to BF16; generation-only checkpoint keys are
reported and ignored. It requires every understanding parameter and handles
non-persistent rotary buffers. It does not require the generation VAE.
See [Download resources](../README.md#download-resources) for BAGEL assets and
the SFT release. A checkpoint inside a numeric training-step directory must
have its `_SUCCESS` marker.

The interpreter fallback is `EVAL_PYTHON`, `BAGEL_PYTHON`, `PYTHON_BIN`, then
`python3`. An explicit `CONFIG_FILE` is loaded unless `SKIP_CONFIG=1`; command
line options override the resulting environment. No fixed Conda installation
or shell startup file is used.

## Benchmark data

Existing local data is used by default. SAT/MMSI are downloaded only when
`--allow-download` is explicitly supplied and their local parquet variable is
unset. This always selects the fixed dataset revision and checks its SHA256.
MindCube requires its official archive to be downloaded and extracted first.

To download the pinned files yourself, run from the repository root with the
Hugging Face CLI installed:

```bash
hf download array/SAT-v2 data/test-00000-of-00001.parquet --repo-type dataset \
  --revision 77d83da806d0a31f837be7b36d0dfbda230d8864 \
  --local-dir "$PWD/resources/understanding/SAT-v2"
export SAT_V2_PARQUET="$PWD/resources/understanding/SAT-v2/data/test-00000-of-00001.parquet"

hf download RunsenXu/MMSI-Bench MMSI_Bench.parquet --repo-type dataset \
  --revision ec7c92bfaf7728fcca1d61e3e224e190af309436 \
  --local-dir "$PWD/resources/understanding/MMSI-Bench"
export MMSI_PARQUET="$PWD/resources/understanding/MMSI-Bench/MMSI_Bench.parquet"

hf download MLL-Lab/MindCube data.zip --repo-type dataset \
  --revision 9c941b46a6bd65b6914669ef7a579948fc9c8467 \
  --local-dir "$PWD/resources/understanding/MindCube"
python -m zipfile -e "$PWD/resources/understanding/MindCube/data.zip" \
  "$PWD/resources/understanding/MindCube"
export MINDCUBE_DATA_ROOT="$PWD/resources/understanding/MindCube/data"
```

MindCube must contain `raw/MindCube_tinybench.jsonl` and `other_all_image/`.
The CPU preflight checks all 428 unique image files as well as the 1,050-row
JSONL, prompt digest, distributions, formats, and protocol identity.

The fixed source files are:

| File | SHA256 |
|---|---|
| SAT-v2 test parquet | `4880cf7b331d4398cd537d5faf96d26fa880ce9b6fd3c7dcdb35e253fa16091a` |
| MMSI_Bench.parquet | `72a9c94699ca88f3eba79f330ab28965ee537bccd7cbf563265982a6ff8b74b4` |
| MindCube_tinybench.jsonl | `0289eb82d81ff9aa0201ae75f86da7fd1924cf23dc202856d0579a0effd22ac8` |

Scoring preserves the pinned official implementations:
[SAT lmms-eval](https://github.com/arijitray1993/lmms-eval/tree/8856d847809d5586059fae171e8039c5af0301e1/lmms_eval/tasks/sat_real),
[MMSI-Bench](https://github.com/InternRobotics/MMSI-Bench/tree/13e58a2b8b30d880d7e8a1e4a6aa1c0feda94cac),
and [MindCube](https://github.com/mll-lab-nu/MindCube/tree/b8b7062adf6d3e49d588a7d014a0a787553d09ec).
SAT's primary score uses its official target-aware compatible fallback; strict
accuracy, strict parse rate, and both-order accuracy are separate diagnostics.
MMSI uses the official case-sensitive extractor without an LLM fallback.
MindCube uses the official raw-QA prompt, response cleaning, case-sensitive
parser priority, and setting filter. MMSI Circular is outside this public suite.

## Run and resume

```bash
# Print all three commands with no model/data reads, downloads, or result writes.
bash understanding/run.sh all --dry-run

# Validate the existing benchmark data on CPU without loading model weights.
bash understanding/run.sh all --prepare-only

# SAT/MMSI may be fetched at their fixed revisions if local paths are unset.
bash understanding/run.sh all --prepare-only --allow-download

# Run all 2,350 inference calls sequentially on the selected GPU.
CUDA_VISIBLE_DEVICES=0 bash understanding/run.sh all

# Run just one of the three protocols.
bash understanding/run.sh sat-v2
bash understanding/run.sh mmsi
bash understanding/run.sh mindcube-tiny
```

`--prepare-only` creates no score or result directory. It may populate the
standard Hugging Face cache only when downloading was explicitly allowed.
Use `--help` for local path and output overrides. `--nproc-per-node N` optionally
launches one complete model per GPU with disjoint modulo sharding; no examples
are repeated or padded. The default remains one process, independently of
generation's `NUM_GPUS` or older benchmark-specific limit variables.

Full results default to `outputs/understanding/`, overridden by
`UNDERSTANDING_OUTPUT_ROOT` or `--output-root`:

```text
understanding/
├── sat-v2/
├── mmsi/
├── mindcube-tiny/
├── summary.json
├── summary.csv
└── summary.md
```

Each benchmark stores `run_manifest.json`, append-only `shards/rank-*.jsonl`,
canonical `predictions.jsonl`, `metrics.json`, and `_SUCCESS` only after all
selected items succeed. A process lock prevents concurrent writers. The
manifest binds the data hash/protocol, decoding, image transform, tokenizer
assets, and checkpoint identity. Checkpoint identity uses its size, mtime, and
a first/middle/last sampled hash; this is resume protection, not a full-file
integrity checksum or a hardcoded private checkpoint restriction.

Rerun the same command to validate and reuse completed answers. Changed model,
data, or protocol settings require a separate output directory. Summaries
recompute official scores from saved raw answers, check complete denominators
and SAT option pairs, and reject mixed checkpoint/tokenizer/transform identities.
The suite is marked complete only when all three full runs pass these checks.
To regenerate summaries without model inference:

```bash
python understanding/summarize.py --results-root outputs/understanding
```

For an explicit short diagnostic, add `--limit N`. It limits source questions
before SAT's two-order expansion and writes into
`UNDERSTANDING_OUTPUT_ROOT/partial/limit_N/`, including that directory's own
summary files. Both metrics and summaries identify the run as partial, even
when every selected item succeeded. Such scores are not full-benchmark results.

## CPU verification

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s understanding/tests -v
```

Parser, protocol, sharding, resume, launcher, and raw-answer summary checks run
on CPU. Tiny tensor-loader tests are skipped when PyTorch, accelerate, or
safetensors is absent. These checks do not establish successful full-model GPU
loading, generation, or published benchmark scores. The current release changes
have not been GPU-tested.

Bundled BAGEL-derived code retains the [Apache-2.0 license](LICENSE). Benchmark
datasets retain their upstream terms.
