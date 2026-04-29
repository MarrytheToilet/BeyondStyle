# `outputs/`

Experimental artefacts. The directory tree itself is tracked, but everything
inside (`*.json`, `*.pt`, `*.png`, …) is git-ignored.

```
outputs/
└── {en, zh}/
    ├── 01_data_corpus/                        §3.1 Pair construction
    │   ├── content_confound_results.json      P1–P5 numbers
    │   └── figures/data_corpus_summary.png    cosine / confound / regression panels
    │
    ├── 02_separability/<model>/<dataset>/     §3.3 Direction estimation
    │   ├── layer_metrics.json + _summary.md   per-(layer, method) AUC / Sep / Hull / Fisher
    │   ├── heldout_auc.json   + _summary.md   5-fold held-out AUC
    │   └── figures/{layer_auc,layer_geometry,heldout_auc,...}.png
    │
    ├── 03_steering_vectors/<model>/<dataset>/ §3.2 + §3.4 Steering
    │   ├── states/         per-layer hidden-state caches (.pt)
    │   ├── models/         trained v_src per (layer, method) (.pt)
    │   └── generation/     α-sweep generations (generation_<method>.json)
    │
    ├── 04_lmeval/<model>/                     §4.3 LM-eval-harness
    │   ├── baseline.json + <method>_L<L>_a<α>.json     raw task outputs
    │   ├── summary_*.md                                paper-ready tables
    │   └── figures/{eval3d_raw,eval3d_norm,heatmap}/   per-method 3D + heatmap
    │
    ├── 05_perception/winston/                 §4.4 AI-detector perception
    │   ├── winston_<method>_L<L>.json
    │   └── summary_<method>_L<L>.md, summary_combined_L<L>.md
    │
    └── 06_interpretability/<model>/<dataset>/ §3.5 + §4.5 Interp
        ├── interpretability_results.json + vector_interp_results.json
        └── figures/figXX_*.{md,json,png}      one figure per paper readout
```

Conventions:

- Every JSON result ships with a `*_summary.md` (or `figXX_*.md`) that is
  drop-in for the paper / appendix.
- Filenames embed the salient axes (`L<layer>`, `a<alpha>`, `<method>`)
  so they sort sensibly under `ls`.
- A parallel `outputs_pool/` (same layout) holds the mean-pooled
  hidden-state version, produced by `scripts/run_pooled_analysis.sh`.
