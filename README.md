# BeyondStyle

**Source-Subspace Steering of LLM Hidden Representations.** We learn a per-layer
direction $v_{\text{src}}^{(\ell)}$ that separates human-written from
model-generated continuations and use it as an additive activation-steering lever
to push a frozen LLM toward either source at inference time.

The repository covers the full pipeline:

1. paired (human, AI) continuation construction (`scripts/build_dataset.py`),
2. hidden-state extraction and $v_{\text{src}}$ training in three flavours —
   raw mean-diff, within-class null-space, Fisher LDA (`scripts/steering_pipeline.py`),
3. content-confound / sketch-leakage controls (`scripts/probe_sketch_confound.py`),
4. layer / hold-out separability evaluation
   (`scripts/eval_heldout_separability.py`),
5. $\alpha$-sweep generation, lm-eval-harness, Winston AI-detection
   (`scripts/eval_lm_benchmark.py`, `scripts/eval_winston.py`),
6. token / vocabulary / construction interpretability
   (`scripts/run_interpretability.py`).

## Layout

```
src/                Python package (config, data, extraction, steering, eval, viz)
scripts/            Entry-point CLIs (one per pipeline step, --help on each)
data/               Paired datasets and held-out prompts
outputs/{en,zh}/    All experimental artefacts, organised by paper section
                    01_data_corpus  02_separability  03_steering_vectors
                    04_lmeval       05_perception    06_interpretability
```

`outputs/` and `data/raw/` are git-ignored; only the directory layout and
shipped READMEs are tracked.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env            # then fill in model paths / API keys

# 1) paired dataset (English, CNN/DailyMail)
python scripts/build_dataset.py --language en -n 500

# 2) hidden states + v_src + α-sweep generation
python scripts/steering_pipeline.py --language en --multi \
    --denoise raw nullspace fisher --alphas -2 -1 0 1 2

# 3) content-confound controls + held-out separability
python scripts/probe_sketch_confound.py --language en
python scripts/eval_heldout_separability.py --language en

# 4) AI-detection scoring
python scripts/eval_winston.py --language en

# 5) lm-eval grid + 3D / heatmap figures
python scripts/eval_lm_benchmark.py --mode grid --language en
python scripts/plot_eval_3d.py --eval-dir outputs/en/04_lmeval/<model>

# 6) interpretability
python scripts/run_interpretability.py --language en
```

For Chinese, set `--language zh` and point `BEYONDSTYLE_MODEL_PATH` at a
Chinese student (the default in `.env.example` is `Qwen3-4B`).

## Outputs by paper section

| Folder | Paper section | Main artefact |
|---|---|---|
| `01_data_corpus/` | §3.1 Pair construction | `content_confound_results.json` + `data_corpus_summary.png` |
| `02_separability/` | §3.3 Direction estimation | `layer_metrics.json`, `heldout_auc.json` (+ `*_summary.md`) |
| `03_steering_vectors/` | §3.2 / §3.4 Steering | `states/`, `models/`, `generation/` |
| `04_lmeval/` | §4.3 LM-eval-harness | task JSONs, `summary_*.md`, `figures/{eval3d_*,heatmap}/*.png` |
| `05_perception/` | §4.4 Detector | `winston_*_L*.json`, `summary_*_L*.md` |
| `06_interpretability/` | §3.5 / §4.5 Interp | `interpretability_results.json`, `vector_interp_results.json`, `figures/figXX_*.{md,json,png}` |

Every `.json` result ships with a paper-ready `*_summary.md` (or `figXX_*.md`).
