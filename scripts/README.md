# `scripts/` — entry points

One script per pipeline step.  All scripts read `.env` via `src/config.py`
and accept CLI overrides; pass `--help` for the full flag list.

## Pipeline order

| # | Script | Purpose |
|---|---|---|
| 0 | `build_prompts.py` | Sample neutral held-out prompts from the source corpus. |
| 0 | `build_dataset.py` | Build paired (human, AI-chat, AI-free) continuations.  Supports `--corpus {cnn_dailymail, openwebtext, ...}` and a mixed multi-generator pool. |
| 1 | `steering_pipeline.py` | Extract per-layer hidden states, train $v_{\text{src}}$ (raw / nullspace / fisher), and run the α-sweep generation grid. |
| 2 | `probe_sketch_confound.py` | Five content-confound probes (P1–P5) on the paired corpus. |
| 2 | `eval_heldout_separability.py` | 5-fold random hold-out AUC per (layer, method). |
| 2 | `eval_generator_holdout.py` | Hold-out-by-generator AUC (train on N−1 generators, test on the held-out one). |
| 2 | `eval_cross_domain_transfer.py` | Apply a direction trained on dataset A to hidden states of dataset B; reports AUC, accuracy, projection gap, Cohen's d. |
| 3 | `eval_cross_domain_steering.py` | Apply a direction trained on dataset A as an α-sweep steering hook on prefixes from dataset B. |
| 3 | `eval_lm_benchmark.py` | lm-eval-harness grid under steering. |
| 3 | `eval_winston.py` | Winston AI-detection scoring of generated text. |
| 3 | `eval_pangram.py` | Pangram AI-detection scoring (alternative detector). |
| 3 | `eval_human_scoring.py` | Build blind LLM-judge / human-rating sheets. |
| 4 | `run_interpretability.py` | Token projection + vocabulary lens + contrastive n-grams + construction templates + layer trajectory. |
| 4 | `refresh_token_projection.py` | Re-extract token-level projection for one (layer, method) and merge into an existing `interpretability_results.json` (refreshes fig01 with a larger top-k). |

## Plotting

| Script | Reads | Writes |
|---|---|---|
| `plot_data_corpus_figs.py` | `01_data_corpus/content_confound_results.json` | `01_data_corpus/figures/data_corpus_summary.png` |
| `plot_separability_figs.py` | `02_separability/.../{insample_metrics,random_holdout}.json` | `02_separability/.../figures/{insample_auc,layer_geometry,random_holdout_auc}.png` |
| `plot_heldout_separability.py` | `02_separability/.../random_holdout.json` | extra held-out figure |
| `plot_eval_3d.py` | `04_lmeval/<model>/*.json` | `04_lmeval/<model>/figures/{eval3d_raw,eval3d_norm,heatmap}/*.png` |

## Convenience runners

| Shell | What it does |
|---|---|
| `run_steering_full.sh` | EN + ZH steering generation followed by lm-eval grid (assumes states/ + models/ already on disk). |
| `run_pooled_analysis.sh {en,zh}` | Re-run separability + interpretability with mean-pooled hidden states (writes to `outputs_pool/` via `OUTPUT_DIR`). |

## Output isolation

All scripts write under
`{OUTPUT_DIR}/{lang}/{section}/{model_id}/{dataset_tag}/…` so that
multiple datasets (`mixed_500`, `openwebtext_mixed_300`, …) and detector
runs (`winston/mixed_500`, `winston/cross_owt_from_mixed_500`, …) never
overwrite each other.  `eval_winston.py` and `eval_pangram.py` accept
`--out-tag <subdir>` to make this explicit.

All long-running jobs are best launched inside `tmux` / `nohup` with
unbuffered output (`python -u`).
