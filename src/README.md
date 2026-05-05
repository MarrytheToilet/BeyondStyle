# `src/`

Library code, imported by every entry-point script under `scripts/`.

```
src/
├── config.py        .env loader, output-path resolvers (`get_*_dir`)
├── data/            corpus loaders, prefix/continuation splitter,
│                    paired-sample dataclasses, prompt builders
├── extraction/      per-layer last-token hidden-state extractor
├── steering/        v_src estimators (raw / nullspace / fisher),
│                    forward hooks, deep / vector interpretability
├── generation/      base + API + local-HF generators, steered hook generator
├── evaluation/      `SteeredHFLM` wrapper for lm-evaluation-harness
└── visualization/   palette + interpretability plotting helpers
```

`src/config.py::get_<section>_dir()` is the single source of truth for
output paths.  Set `OUTPUT_DIR` in `.env` to redirect everything to a
sibling root — `scripts/run_pooled_analysis.sh` uses
`OUTPUT_DIR=outputs_pool` to keep the mean-pooled re-run isolated from
the canonical `outputs/`.
