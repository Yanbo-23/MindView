# VGGT inference runtime

This directory contains the VGGT model, heads, layers, and image/pose utilities
used by MindView's geometric metrics. The metric implementation is
in `evaluation/benchmarker_metrics/`; VGGT itself is a shared dependency.

Install it into the environment selected by `VGGT_PYTHON`:

```bash
python -m pip install -e vggt
```

Install matching CUDA-compatible PyTorch and torchvision builds for that
environment separately; image preprocessing imports torchvision.
Model weights are external inputs; configure the model asset variables
documented in the evaluation workflow. Upstream demos, example
images, and VGGT training are outside this release.

The original [VGGT license](LICENSE.txt) is retained.
