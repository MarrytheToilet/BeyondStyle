# What Makes Text Feel AI-Generated?
### Steering and Interpreting AI-Like Discourse Directions in Hidden States

This repository contains the full code and experimental pipeline for the paper.
We learn a per-layer linear direction
$v_{\text{src}}^{(\ell)} \in \mathbb{R}^{d}$
in a frozen LLM that separates **human-written** from **AI-generated** continuations
of the same shared prefix, then study what the direction encodes and use it as
an additive activation-steering knob to make a frozen LLM sound more or less
"AI-like" at inference time.

The pipeline covers:

1. **Paired (human, AI) continuation construction**, with controlled topic/entity
   sketches to suppress content confounds  (`scripts/build_dataset.py`).
2. **Hidden-state extraction** and $v_{\text{src}}$ training in three flavours
   — raw mean-diff, within-class null-space projection, Fisher LDA
   (`scripts/steering_pipeline.py`).
3. **Content-confound / sketch-leakage controls**
   (`scripts/probe_sketch_confound.py`) and **layer / hold-out separability**
   evaluation (`scripts/eval_heldout_separability.py`).
4. **Cross-domain & cross-generator generalisation** of the learned direction
   on OpenWebText with a heterogeneous generator pool
   (`scripts/eval_cross_domain_transfer.py`,
   `scripts/eval_cross_domain_steering.py`,
   `scripts/eval_generator_holdout.py`).
5. **α-sweep generation, lm-eval-harness, and AI-detector scoring**
   (`scripts/eval_lm_benchmark.py`, `scripts/eval_winston.py`,
   `scripts/eval_pangram.py`).
6. **Token / vocabulary / construction interpretability** of $v_{\text{src}}$
   (`scripts/run_interpretability.py`).

## Repository layout

```
src/                    Python package (config, data, extraction, steering, eval, viz)
scripts/                Entry-point CLIs (one per pipeline step; pass --help)
data/
  raw/{en,zh}/          source corpora cache         (git-ignored)
  prompts/{en,zh}/      held-out test prompts
  paired/{en,zh}/       paired (human, AI) datasets
outputs/{en,zh}/        all experimental artefacts, organised by paper section
  01_data_corpus/         §3.1 pair construction & content-confound checks
  02_separability/        §3.3 / §4.2 separability + cross-domain transfer
  03_steering_vectors/    §3.2 / §3.4 v_src + steered generations
  04_lmeval/              §4.3 lm-eval-harness sweeps
  05_perception/          §4.4 detector scoring (Winston / Pangram)
  06_interpretability/    §3.5 / §4.5 vector + deep interpretability
```

`outputs/` and `data/raw/` are git-ignored; only the directory layout and
top-level READMEs are tracked.  All script outputs are isolated by
`{model_id}/{dataset_tag}` so multiple datasets (e.g. `mixed_500` vs
`openwebtext_mixed_300`) never collide.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env            # fill in model paths / API keys
python -c "from src.config import print_config; print_config()"

# 1) Paired dataset (English, CNN/DailyMail; 500 pairs from 3 generators)
python scripts/build_dataset.py --language en -n 500 \
    --output data/paired/en/paired_mixed_500.json

# 2) Hidden states + v_src + α-sweep generation
python scripts/steering_pipeline.py \
    --dataset data/paired/en/paired_mixed_500.json --multi \
    --denoise raw nullspace fisher --alphas -2 -1 0 1 2

# 3) Content-confound controls + held-out separability
python scripts/probe_sketch_confound.py --language en
python scripts/eval_heldout_separability.py --language en

# 4) AI-detector scoring of the α-sweep generations
python scripts/eval_winston.py --layer 15 --alpha -2 0 2 -n 100

# 5) lm-eval grid + 3D / heatmap figures
python scripts/eval_lm_benchmark.py --mode grid --language en
python scripts/plot_eval_3d.py --eval-dir outputs/en/04_lmeval/<model>

# 6) Interpretability (token / construction / trajectory)
python scripts/run_interpretability.py --language en
```

For Chinese, set `--language zh` and point `LOCAL_MODEL_PATH` at a Chinese
student model (the default in `.env.example` is `Qwen3-4B`).

### Cross-domain experiment (OpenWebText)

To check whether the direction is **not** a CNN/DailyMail news artefact:

```bash
# 1) Build paired OpenWebText corpus (300 pairs, 3 generators)
python scripts/build_dataset.py --language en -n 300 \
    --corpus openwebtext --no-mixed \
    --output data/paired/en/paired_openwebtext_mixed_300.json

# 2) Extract hidden states for the same target model (no new generation)
python scripts/steering_pipeline.py \
    --dataset data/paired/en/paired_openwebtext_mixed_300.json --multi \
    --denoise raw nullspace --skip_generation

# 3) Cross-domain transfer AUC (CNN-trained direction → OWT, and reverse)
python scripts/eval_cross_domain_transfer.py
python scripts/eval_cross_domain_transfer.py --reverse

# 4) Generator-held-out AUC (train on 2 generators, test on the third)
python scripts/eval_generator_holdout.py \
    --dataset data/paired/en/paired_openwebtext_mixed_300.json

# 5) Apply CNN-trained direction to OWT prefixes for steered generation
python scripts/eval_cross_domain_steering.py --layers 15 --alphas -2 0 2

# 6) Score the cross-domain steered text (writes to a separate sub-folder
#    so it never overwrites the in-domain Winston results)
python scripts/eval_winston.py \
    --input outputs/en/03_steering_vectors/<model>/openwebtext_mixed_300/generation/from_mixed_500/generation_mean_diff_nullspace.json \
    --method nullspace --layer 15 --alpha -2 0 2 -n 100
```

## Outputs by paper section

| Folder | Paper section | Main artefacts |
|---|---|---|
| `01_data_corpus/` | §3.1 Pair construction | `content_confound_results.json`, `figures/data_corpus_summary.png` |
| `02_separability/<model>/<dataset>/` | §3.3 / §4.2 Direction estimation | `insample_metrics.json`, `random_holdout.json`, `generator_holdout.json` |
| `02_separability/<model>/cross_domain/` | §4.2 Cross-domain transfer | `transfer_summary.{json,md}` |
| `03_steering_vectors/<model>/<dataset>/` | §3.2 / §3.4 Steering | `states/`, `models/`, `generation/generation_*.json` |
| `03_steering_vectors/<model>/<tgt>/generation/from_<src>/` | §4.2 Cross-domain steering | `generation_mean_diff_nullspace.json` |
| `04_lmeval/<model>/` | §4.3 lm-eval-harness | task JSONs, `summary_*.md`, `figures/{eval3d_*,heatmap}/*.png` |
| `05_perception/{winston,pangram}/<dataset>/` | §4.4 Detector scoring | `winston_*_L*.json`, `summary_*_L*.md` |
| `06_interpretability/<model>/<dataset>/` | §3.5 / §4.5 Interpretability | `interpretability_results.json`, `vector_interp_results.json`, `figures/figXX_*.{md,json,png}` |

Every JSON result ships with a paper-ready `*_summary.md` (or `figXX_*.md`).

## Configuration

All paths / API keys are read from `.env` via `src/config.py`.  See
`.env.example` for the canonical list.  Output paths are derived as

```
{OUTPUT_DIR}/{lang}/{section}/{model_id}/{dataset_tag}/...
```

where `dataset_tag = basename(paired_file).removeprefix("paired_")`.
Always pass `--dataset` explicitly to keep results from different
datasets (`mixed_500`, `openwebtext_mixed_300`, …) cleanly separated.

## Citation

If you use this code or the produced datasets, please cite the paper
*"What Makes Text Feel AI-Generated? Steering and Interpreting AI-Like
Discourse Directions in Hidden States."*
