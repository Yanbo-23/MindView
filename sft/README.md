# MindView BAGEL SFT

This directory fine-tunes BAGEL on six ordered context RGB images, the query,
and one target RGB image per record. Use `run.sh` for the frozen MindView
recipe; the underlying BAGEL trainer's generic defaults describe a different
experiment.

| Setting | Release recipe |
| --- | --- |
| Data | One 20,000-row, shot-6 snapshot in a single MindView dataset group; no other BAGEL datasets or shot resampling |
| Images | Center-square crop; VAE 512×512, stride 16; ViT 518×518, stride 14; original image bytes and context order |
| Updates | One node, eight GPUs; 10,000 optimizer updates; gradient accumulation 1; one data worker per rank |
| Packing | One packed sequence per rank; target/prefer-buffer 8,192 tokens; maximum sample/pack 16,384; buffer size 50 |
| Optimizer | AdamW, `lr=1e-5`, betas `(0.9, 0.95)`, epsilon `1e-15`, weight decay 0, gradient clipping 0.5 |
| Schedule | 200 linear warmup updates, then cosine to `min_lr=1e-6` over the fixed 10,000 updates |
| EMA / diffusion | EMA decay 0.9995 after each optimizer update; timestep shift 4.0 |
| Trainable parameters | VAE and ViT frozen; LLM and understanding expert trainable; both visual branches enabled |
| Loss / conditioning | Target VAE prediction MSE; contexts and prompt have no supervised loss; all text/VAE/ViT condition-dropout probabilities 0 |
| Randomness | `global_seed=42`, actual rank seed `42 * world_size + rank` (336–343 by default); `data_seed=42`, dataset `shuffle_seed=0` |
| Execution | CUDA BF16 autocast, FSDP FULL_SHARD, no CPU offload; save every 2,000 updates and after the final update |

Eight ranks produce eight packed sequences per update. For this fixed six-view
layout, each row's image tokens already exceed the 8,192-token packing target,
so each accepted pack contains one row: 80,000 consumed rows, or four nominal
passes over 20,000 rows. Oversize rows are skipped by the underlying trainer;
packing and effective batch size must not be generalized to other data layouts.

These settings match the original release launcher and dataset configuration;
the trainer, optimizer/scheduler implementation, image transforms, and sample
construction remain the same. The exported EMA has no embedded training-run
metadata, and the original full optimizer/log bundle is not distributed.
Configuration reproduction does not establish bit-for-bit reproduction of the
published checkpoint's historical training run.

Use Python 3.10 and install the existing BAGEL dependency versions:

```bash
python -m pip install -r sft/requirements.txt
python -m pip install flash_attn==2.5.8 --no-build-isolation
```

The public [MindView-Train dataset](https://huggingface.co/datasets/wangyanbo/MindView-Train)
contains the frozen **20,000 training records from 3,406 scenes**, together with
**104,716 unique RGB images in 58 TAR archives**. The training JSONL SHA256 is
`24d973dc5b2f967d105a6f2a55f46d8f2b0c2d5b262dbd785a60e10ee84627d5`,
matching the launcher's default frozen snapshot. Its scenes are disjoint from
MindView-Bench when identified by dataset and scene ID.

From the repository root, download the annotations and image archives at the
same immutable commit, then verify and extract them:

```bash
python -m pip install -U huggingface_hub
hf download wangyanbo/MindView-Train \
  --repo-type dataset \
  --revision c838213f3358013345b5352fdef71a76dfb0d222 \
  --local-dir resources/MindView-Train

# Optional: verify every package file and TAR member without extracting.
python resources/MindView-Train/scripts/prepare_data.py \
  --repo resources/MindView-Train --verify-only
python resources/MindView-Train/scripts/prepare_data.py \
  --repo resources/MindView-Train --output resources/data
```

These commands use an immutable public commit. The complete snapshot is
approximately 15.3 GB. Extraction
restores `resources/data/scannet/`, `scannetpp/`, and `arkitscenes/`. Train and
Bench can be extracted into this same `resources/data` directory: their scenes
are disjoint, matching existing files are reused, and conflicting files are
rejected. See the repository's [resource downloads](../README.md#download-resources)
for the Bench package. The bundled verifier checks file and TAR-member hashes;
run it on the downloaded snapshot before training.

Download the complete official [BAGEL base model](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT)
at its pinned revision, including configurations, tokenizer, VAE, and weights:

```bash
hf download ByteDance-Seed/BAGEL-7B-MoT \
  --revision 5019f57d168e5816e8f3f701b17cc816bb7cf24b \
  --local-dir models/BAGEL-7B-MoT
```

The training JSONL follows `mindview_bagel_single_shot_v1`,
with `shot=6`, one `contexts["6"]` list of six relative RGB paths, and a separate
`target_image`. Paths are relative to the configured dataset roots. The
loader validates unique record IDs and source lineages, requires the anchor
first for anchor tasks, and rejects target images in the contexts. Images are
center-square cropped, then resized to 512 pixels for the VAE and 518 pixels
for ViT. Missing or corrupt images stop training.

From the repository root, configure the inputs and output directory:

```bash
unset CONFIG_FILE  # Alternatively put these assignments into your explicit config file.
export BAGEL_PYTHON=python
export BAGEL_MODEL_PATH="$PWD/models/BAGEL-7B-MoT"
export TRAINING_JSONL="$PWD/resources/MindView-Train/data/train.jsonl"
export SCANNET_ROOT="$PWD/resources/data/scannet"
export SCANNETPP_ROOT="$PWD/resources/data/scannetpp"
export ARKITSCENES_ROOT="$PWD/resources/data/arkitscenes"
export SFT_OUTPUT_ROOT=/path/to/persistent/sft-run

SFT_RECIPE=(
  --dataset-config "$PWD/sft/data/configs/mindview_sft.yaml"
  --expected-rows 20000
  --snapshot-sha256 24d973dc5b2f967d105a6f2a55f46d8f2b0c2d5b262dbd785a60e10ee84627d5
  --nnodes 1 --node-rank 0 --nproc-per-node 8
  --total-steps 10000 --warmup-steps 200
  --lr 1e-5 --min-lr 1e-6 --save-every 2000
  --gradient-accumulation-steps 1 --num-workers 1
  --sharding-strategy FULL_SHARD --num-shard 1 --num-replicate 1
)

# Optional: validate the full recipe without importing PyTorch or using GPUs.
bash sft/run.sh "${SFT_RECIPE[@]}" --check-images --dry-run

# Start the eight-GPU training run.
bash sft/run.sh "${SFT_RECIPE[@]}"
```

`--dry-run` validates model-asset presence, the complete record schema, dataset
roots, row count, snapshot SHA256, and the effective training configuration.
It also rejects any dataset/scene identity shared with the frozen benchmark
in `evaluation/data/merged_dataset_final.jsonl`, including custom training
snapshots. `--benchmark-jsonl` selects a separate benchmark when needed.
It prints that configuration and the launch command without importing PyTorch,
loading weights, using GPUs, or writing files. Add `--check-images` to check
all referenced RGB files exist. Only roots for datasets present in the JSONL
are required. `CONFIG_FILE=/path/to/config.env` optionally loads a shared
configuration; no shell startup files or Conda activation are used.

Keep the supplied dataset YAML, frozen JSONL, base-model revision, image
transforms, worker/world size, packing limits, and complete schedule unchanged
for this recipe. The explicit arguments above prevent inherited `SFT_*`
schedule/batch overrides from changing the run. A custom snapshot, changed
schedule, different sharding, or different world size is a separate experiment.

`bash sft/run.sh --help` lists overrides. For a separate short run on a valid
small shot-6 snapshot, explicitly select its count and bind its own checksum:

```bash
bash sft/run.sh --training-jsonl /path/to/small-shot6.jsonl \
  --expected-rows 8 --snapshot-sha256 auto \
  --gpus 0 --nproc-per-node 1 --total-steps 2 --warmup-steps 0 \
  --save-every 1 --min-free-gib 0 --output-root /path/to/smoke-run --dry-run
```

Remove `--dry-run` to train. Distributed settings are configurable through
`--nnodes`, `--node-rank`, `--master-addr`, and `--master-port`; launch once on
each node with the same shared inputs and output directory. The default uses
FSDP FULL_SHARD. `--sharding-strategy HYBRID_SHARD` also requires
`--num-shard * --num-replicate` to equal the world size. Existing run settings
are frozen in `run_spec.json` and `mindview_sft.json`; changed settings require
a different output directory.

Training writes `logs/` and `checkpoints/<step>/`. Auto-resume uses only a
checkpoint with `_SUCCESS`, both model files, scheduler/data state, and all
expected optimizer shards. The default final checkpoint is
`checkpoints/0009999/ema.safetensors`; set `BAGEL_SFT_FINAL_CHECKPOINT` to this
file for evaluation, while retaining the original BAGEL model
directory for configs, tokenizer, and VAE. Five complete default checkpoints
need roughly 1.09 TiB, so the launcher defaults to a 1,300 GiB free-space check.

The public [MindView-BAGEL-SFT model repository](https://huggingface.co/wangyanbo/MindView-BAGEL-SFT)
provides the released BF16 EMA for inference:

```bash
hf download wangyanbo/MindView-BAGEL-SFT ema.safetensors \
  --revision b38fa540324242d500d2dd9b0543840d0df96ced \
  --local-dir models/MindView-BAGEL-SFT
export BAGEL_SFT_FINAL_CHECKPOINT="$PWD/models/MindView-BAGEL-SFT/ema.safetensors"
```

The checkpoint is **29,214,685,336 bytes**, with SHA256
`504eb0958540b4873623e87eae8a971e9af545117ad0b4eb240e8400935143ac`.
It contains all 1,223 EMA tensors cast from the original FP32 EMA using native
PyTorch BF16 round-to-nearest-even. Every element was checked against both the
native cast and an independent rounding implementation; BF16 rounding is lossy.
This is an inference weight export, without optimizer, scheduler, dataloader,
non-EMA model, or complete training-resume state. Resume training only from a
complete training checkpoint, not this export. GPU training and inference
reproduction with the published package remain unverified.

CPU checks:

```bash
python -m pip install -r sft/requirements-dev.txt
cd sft
python -m pytest
```

The BAGEL code retains its Apache-2.0 license in `LICENSE`.
