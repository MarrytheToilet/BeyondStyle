# `src/`

Library code, imported by every entry-point script under `scripts/`.

```
src/
├── config.py        .env loader, output-path resolvers (`get_*_dir`)
├── data/            corpus loaders, prefix/continuation splitter,
│                    paired-sample dataclasses, prompt builders
├── extraction/      per-layer last-token hidden-state extractor
├── steering/        v_src estimators (raw / nullspace / fisher),
│                    forward hooks, SAE, deep / vector interpretability
├── generation/      base + API + local-HF generators, steered hook generator
├── evaluation/      `SteeredHFLM` wrapper for lm-evaluation-harness
└── visualization/   palette + interpretability plotting helpers
```

`src/config.py::get_<section>_dir()` is the single source of truth for
output paths; honour `BEYONDSTYLE_OUTPUT_DIR` to redirect everything to a
sibling root (used by `scripts/run_pooled_analysis.sh`).
