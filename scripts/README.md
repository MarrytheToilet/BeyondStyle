# `scripts/` — entry points

One script per pipeline step. Every script reads `.env` (see
`src/config.py`) and accepts CLI overrides; pass `--help` for the full
flag list.

## Pipeline order

| # | Script | Purpose |
|---|---|---|
| 0 | `build_prompts.py` | Sample neutral held-out prompts from the source corpus. |
| 0 | `build_dataset.py` | Build paired (human, AI-chat, AI-free) continuations. |
| 1 | `steering_pipeline.py` | Extract per-layer hidden states, train $v_{\text{src}}$ (raw / nullspace / fisher), and run $\alpha$-sweep generation. |
| 2 | `probe_sketch_confound.py` | Five content-confound probes (P1–P5) on the paired corpus. |
| 2 | `eval_heldout_separability.py` | 5-fold held-out AUC for each (layer, method). |
| 3 | `eval_lm_benchmark.py` | lm-eval-harness grid under steering. |
| 3 | `eval_winston.py` | Winston AI-detection scoring of generated text. |
| 3 | `eval_human_scoring.py` | Build blind LLM-judge / human-rating sheets. |
| 4 | `run_interpretability.py` | Token projection + vocabulary lens + contrastive n-grams + construction templates + layer trajectory. |

## Plotting

| Script | Reads | Writes |
|---|---|---|
| `plot_data_corpus_figs.py` | `01_data_corpus/content_confound_results.json` | `01_data_corpus/figures/data_corpus_summary.png` |
| `plot_separability_figs.py` | `02_separability/.../{layer_metrics,heldout_auc}.json` | `02_separability/.../figures/{layer_auc,layer_geometry,heldout_auc}.png` |
| `plot_heldout_separability.py` | `02_separability/.../heldout_auc.json` | extra held-out figure |
| `plot_eval_3d.py` | `04_lmeval/<model>/*.json` | `04_lmeval/<model>/figures/{eval3d_raw,eval3d_norm,heatmap}/*.png` |

## Convenience runners

| Shell | What it does |
|---|---|
| `run_steering_full.sh` | EN + ZH steering generation followed by lm-eval grid. |
| `run_pooled_analysis.sh {en,zh}` | Re-run separability + interpretability with mean-pooled hidden states (writes to `outputs_pool/`). |

All long-running jobs are best launched inside `tmux` / `nohup`.
