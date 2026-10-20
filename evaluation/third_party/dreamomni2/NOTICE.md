# DreamOmni2 wrapper attribution

`evaluation/benchmark_runner/benchmark/dreamomni_runtime.py` adapts the generation entrypoint from [DreamOmni2](https://github.com/JIA-Lab-research/DreamOmni2/blob/b0b98e03bfc95c2bf0f27a5f9ca33d9b16a525b5/inference_gen.py) (upstream commit `b0b98e03bfc95c2bf0f27a5f9ca33d9b16a525b5`), licensed under Apache-2.0. The original license is retained in this directory. The official pipeline and its image-processing helpers are external dependencies, not copied here.

Benchmark constants and the two model snapshot revisions were recovered from the original per-sample DreamOmni2 generation metadata. The historical experiment wrapper was unavailable; the public wrapper preserves documented settings but GPU equivalence has not been established.
