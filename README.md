# BeyondStyle

**Source Subspace Steering in LLM Hidden Representations**

This project investigates internal representation subspaces that separate
human-written from AI-generated text continuations. Given the same prefix,
human and AI continuations occupy distinct regions in hidden-state space.
We learn a *source direction* **v_src** and use it for activation steering,
pushing generation toward more human-like or more AI-like styles.

## Pipeline

```
CNN/DailyMail ─► prefix/continuation split ─► AI continuation (API model)
      │                                              │
      │                              ┌───────────────┤
      │                     guided (light       free (no
      │                     topic+entity        constraints)
      │                     constraint)
      └──── paired dataset (human + AI guided + AI free) ◄──┘
                             │
                    local model (extraction)
                             │
                  hidden-state extraction (10 layers)
                             │
                   learn steering vector v_src
                      (additive mean_diff)
                             │
                  intervention experiments
                   (alpha-sweep x test prompts)
```

### Two-model design

| Role | Model | Config key |
|------|-------|------------|
| Dataset construction (AI continuations) | configurable (default: Qwen3.5-397B-A17B) | `BEYONDSTYLE_MODEL_NAME` |
| Hidden-state extraction & steering | configurable (default: Llama-3.1-8B-Instruct) | `BEYONDSTYLE_MODEL_PATH` |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API key and model paths
cp .env.example .env   # edit SILICONFLOW_API_KEY, BEYONDSTYLE_MODEL_NAME, BEYONDSTYLE_MODEL_PATH

# 3. Build paired dataset from CNN/DailyMail (300 samples, dual AI continuations)
python scripts/build_dataset.py --corpus cnn_dailymail -n 300 -w 8

# 4. Run full multi-layer steering pipeline (with denoising)
#    Dataset and model are auto-detected from .env (or pass --dataset / --model)
python scripts/steering_pipeline.py --multi --denoise raw nullspace fisher

# 5. Generate analysis figures (auto-detects dataset, includes ablation plots)
python scripts/run_analysis.py

# 6. (Optional) Override dataset or prompts explicitly
python scripts/steering_pipeline.py --multi \
  --dataset outputs/dataset/paired_dataset_Qwen3.5-397B-A17B.json \
  --test_prompts_file data/prompts/long.json

# 7. (Optional) Single-layer quick run
python scripts/steering_pipeline.py --layer 10
```

## Project Structure

```
BeyondStyle/
├── src/                          # Core library
│   ├── config.py                 #   Centralized config (model path, output dirs, dataset tag)
│   ├── data/                     #   Corpus loading, splitting, AI generation
│   │   ├── corpus.py             #     Corpus loader (CNN/DailyMail primary; AG News, Wikipedia)
│   │   ├── splitter.py           #     Prefix-continuation splitter
│   │   ├── continuation_gen.py   #     AI continuation generator
│   │   ├── prompt_builder.py     #     Light/full content-aligned prompts
│   │   └── paired_dataset.py     #     PairedSample dataclass
│   ├── extraction/               #   Hidden-state extraction
│   │   ├── hidden_states.py      #     HiddenStatesExtractor
│   │   └── paired_extractor.py   #     PairedExtractor + PairedHiddenStates
│   ├── steering/                 #   Steering methods
│   │   ├── base.py               #     BaseSteering abstract class
│   │   ├── mean_diff.py          #     MeanDiffSteering (v_src = mean difference)
│   │   ├── denoise.py            #     Denoising: NullSpace, Fisher LDA
│   │   ├── hooks.py              #     Activation hooks for injection
│   │   └── metrics.py            #     Geometry metrics (Fisher, hull distance)
│   ├── generation/               #   Text generation
│   │   ├── backends.py           #     API / local model backends
│   │   ├── steered.py            #     SteeredGenerator (α-sweep)
│   │   ├── generator.py          #     Base generator
│   │   └── prompts.py            #     Prompt templates
│   ├── evaluation/               #   Evaluation tools
│   │   ├── steered_model.py      #     SteeredHFLM wrapper for lm-eval
│   │   └── __init__.py           #     Exports: SteeredHFLM, apply_steering_hooks, load_hflm
│   └── visualization/            #   Plotting
│       ├── paired_plots.py       #     Projection, PCA, similarity, shift
│       ├── steering_plots.py     #     Metrics, ROC, separation landscape
│       ├── ablation_plots.py     #     Denoising comparison, projection violins
│       └── palette.py            #     Style setup and color palette
│
├── scripts/                      # Entry-point scripts
│   ├── build_dataset.py          #   Step 1: build paired dataset
│   ├── steering_pipeline.py      #   Step 2: extract → train → generate
│   ├── run_analysis.py           #   Step 3: generate all analysis figures
│   ├── eval_lm_benchmark.py      #   Eval: lm-eval benchmarks (MMLU, WikiText, etc.)
│   ├── eval_gptzero.py           #   Eval: GPTZero AI detection scoring
│   └── eval_human_scoring.py     #   Eval: human AI-feel blind scoring
│
├── notebooks/
│   └── steering_analysis.ipynb   #   Interactive analysis notebook
│
├── data/
│   ├── raw/                      #   CNN/DailyMail articles (1000 samples)
│   │   └── cnn_dailymail_1000.jsonl
│   ├── prompts/                  #   Test prompts for steering experiments
│   │   ├── default.json          #     Default set (20 medium-length prefixes)
│   │   ├── short.json            #     15 short prefixes (~15 words)
│   │   ├── medium.json           #     20 medium prefixes (~30 words)
│   │   ├── long.json             #     15 long prefixes (~50 words)
│   │   └── all.json              #     Combined 50 prompts
│   └── test_prompts.json         #   Legacy alias → same as prompts/default.json
│
├── outputs/                      # Generated outputs (gitignored)
│   ├── dataset/                  #   paired_dataset_{api_model}.json
│   ├── steering/{model_id}/      #   Organized by extraction model
│   │   └── {dataset_tag}/        #     Sub-dir per dataset (derived from filename)
│   │       ├── states/           #       Cached hidden-state tensors per layer
│   │       ├── models/           #       Trained steering models per layer
│   │       ├── generation/       #       Generation outputs (α-sweep per method)
│   │       └── layer_metrics.json#       Per-layer metrics summary
│   ├── figures/{model_id}/       #   Analysis figures
│   │   └── {dataset_tag}/        #     fig01–fig10
│   └── eval/                     #   Evaluation results
│       ├── {model_id}/           #     Per-model eval results
│       │   ├── baseline.json     #       α=0.0 scores
│       │   ├── fisher_L14_a+1.0.json  # Per-run results
│       │   ├── raw/              #       Full lm-eval raw output
│       │   └── summary_latest.json    # Comparison table
│       ├── gptzero_results.json  #     GPTZero detection probabilities
│       └── human_scoring/        #     Human scoring sheets and analysis
│
├── .env                          # API keys and config (not committed)
├── .env.example                  # Template
└── requirements.txt
```

## Configuration

All settings are controlled via `.env` (copy from `.env.example`). No hardcoded model names or dataset paths.

| Variable | Purpose | Default |
|----------|---------|---------|
| `BEYONDSTYLE_BACKEND` | API backend for dataset construction | `siliconflow` |
| `BEYONDSTYLE_MODEL_NAME` | API model for generating AI continuations | `Qwen/Qwen3.5-397B-A17B` |
| `BEYONDSTYLE_MODEL_PATH` | Local model for extraction & steering | `../../models/Llama-3.1-8B-Instruct` |
| `BEYONDSTYLE_DATASET` | Paired dataset path | auto-detect latest in `outputs/dataset/` |
| `BEYONDSTYLE_TEST_PROMPTS` | Test prompts JSON file | `data/prompts/default.json` |
| `BEYONDSTYLE_OUTPUT_DIR` | Base output directory | `outputs` |
| `SILICONFLOW_API_KEY` | SiliconFlow API key | *(required)* |

Scripts accept `--dataset`, `--model`, `--test_prompts_file` CLI flags to override `.env` values.

## Corpus Options

| Corpus | Source | Text style | Token targets |
|--------|--------|------------|---------------|
| **CNN/DailyMail** (default) | `cnn_dailymail` (HuggingFace) | Full-length news articles, rich detail | prefix=100, cont=200 |
| AG News | `fancyzhx/ag_news` (HuggingFace) | Short news snippets (alternative) | prefix=50, cont=80 |
| Wikipedia | Pre-downloaded JSONL | Encyclopedic, formal (alternative) | prefix=128, cont=256 |

Switch corpus with `--corpus`:
```bash
python scripts/build_dataset.py --corpus cnn_dailymail -n 300   # default, recommended
python scripts/build_dataset.py --corpus ag_news -n 300         # alternative: shorter articles
python scripts/build_dataset.py --corpus wikipedia -n 50        # alternative: formal style
```

### Dual AI continuation modes

By default, each sample gets **two** AI continuations:
- **Guided** (`ai_continuation_chat`): light topic+entity constraint extracted from the human text
- **Free** (`ai_continuation_free`): unconstrained "just continue" prompt

```bash
python scripts/build_dataset.py --free-only -n 300     # only free continuations
python scripts/build_dataset.py --no-free -n 300       # only guided continuations
python scripts/build_dataset.py --content-align -n 300  # full two-step content alignment
```

## Key Results (249-sample, CNN/DailyMail + Qwen3.5-397B + Llama-3.1-8B-Instruct)

### Steering method

We use **additive mean-difference steering** (`mean_diff`): compute v_src as the
mean difference between AI and human hidden states, then add α·v_src to
activations at inference time. Positive α steers toward AI style; negative α
steers toward human style.

Each sample has dual AI continuations: **guided** (light topic+entity constraint)
used for training the steering vector, and **free** (unconstrained) for comparison.
CNN/DailyMail articles provide ~145-word continuations — much longer than AG News
(~60 words), allowing richer style signal extraction.

### Per-layer metrics

| Layer | AUC | Accuracy | Fisher Ratio | Cohen's d | Effect |
|-------|-----|----------|--------------|-----------|--------|
| 4 | 1.000 | 99.2% | 947.0 | 4.85 | huge |
| 8 | 1.000 | 98.6% | 941.1 | 5.11 | huge |
| 10 | 1.000 | 99.0% | 1087.8 | 6.06 | huge |
| 12 | 1.000 | 98.6% | 1475.3 | 5.84 | huge |
| **14** | **1.000** | **98.8%** | **1336.8** | **6.59** | **huge** |
| **16** | **1.000** | **99.4%** | **1359.4** | **7.05** | **huge** |
| 20 | 1.000 | 99.2% | 1341.1 | 6.83 | huge |
| 24 | 1.000 | 99.2% | 1084.3 | 6.39 | huge |
| 28 | 0.999 | 98.8% | 834.9 | 6.15 | huge |
| 32 | 1.000 | 98.6% | 707.8 | 5.42 | huge |

### Highlights

- **Near-perfect discrimination**: AUC ≥ 0.999 across ALL 10 layers tested
- **Best layer: L16** — highest accuracy (99.4%), Fisher ratio 1359, Cohen's d = 7.05 (huge)
- **All layers show huge effects** (Cohen's d 4.85–7.05, all p < 1e-200)
- **Massive improvement over AG News** (v3): Cohen's d jumped from 1.27–2.00 to 4.85–7.05
- **CNN/DailyMail advantage**: longer texts (~145 words vs ~60) give 3× stronger separation
- **Cross-domain steering works**: test prompts from CNN/DM articles respond to learned vectors

### Denoising ablation

We test whether the Human-AI direction captured by raw mean-diff is "pure" or entangled
with content. Three methods extract directions that are increasingly different
from the naive mean-difference while maintaining perfect discrimination:

| Method | Approach | AUC (all layers) | Best Accuracy | cos(v, v_raw) | Interpretation |
|--------|----------|-------------------|---------------|---------------|----------------|
| **Raw MeanDiff** | v = mean(AI − Human) | 1.000 | 99.4% (L16) | 1.000 | Baseline — captures content + source |
| **NullSpace** | Project out top-k *within-class* PCs | 1.000 | 100% (L8+) | 0.630 | Content-cleaned; retains full signal |
| **Fisher LDA** | v = S_w⁻¹ @ d_mean (optimal discriminant) | 1.000 | 100% (all) | 0.294 | **Most pure** — near-orthogonal, perfect AUC |

**Key findings**:
- **All three methods achieve AUC=1.0 at every layer**. Denoising maintains performance while
  discovering genuinely different directions (cos = 0.63 and 0.29 respectively).
- **Fisher LDA** (cos=0.29) finds a near-orthogonal direction that still perfectly separates
  human from AI — the strongest evidence for a genuine source-identity subspace.
- PairedPCA and Combined methods were tested but removed — they achieved AUC ≤ 0.84 due to
  over-aggressive denoising that discarded part of the source signal.

Run denoising with:
```bash
python scripts/steering_pipeline.py --multi --denoise raw nullspace fisher --skip_generation
```

### Analysis figures

Generated by `python scripts/run_analysis.py`:

| Figure | Description |
|--------|-------------|
| `fig01_projection_distributions.png` | Projection of hidden states onto v_src per layer |
| `fig02_pca_embedding.png` | 2D PCA of human vs AI hidden states |
| `fig03_direction_similarity.png` | Cosine similarity of v_src across layers |
| `fig04_embedding_shift.png` | Per-sample embedding shift from human to AI |
| `fig05_layer_metrics.png` | AUC, accuracy, separation across layers |
| `fig06_roc_curves.png` | ROC curves per layer |
| `fig08_separation_landscape.png` | Separation landscape (violin) across layers |
| `fig09_denoise_comparison.png` | AUC, accuracy, direction similarity per denoising method |
| `fig10_projection_violins.png` | H vs A projection distributions per method |

## Steering α convention

| α value | Direction | Effect |
|---------|-----------|--------|
| α < 0 | → Human | More natural, casual, detail-rich |
| α = 0 | Baseline | Unsteered generation |
| α > 0 | → AI | More formal, structured, generalized |

## Evaluation

Three evaluation tools measure the impact of steering on model behavior:

### 1. lm-eval Benchmarks

Measures performance degradation on standard NLP benchmarks at different steering strengths.
The model is loaded **once**; hooks are swapped between runs — no OOM on 24GB GPUs.
Results are saved per-run with structured naming and full raw lm-eval output preserved.

**Supported tasks** (10 benchmarks):

| Group | Tasks |
|-------|-------|
| Reasoning | MMLU, ARC-Easy, ARC-Challenge, OpenBookQA |
| Commonsense | HellaSwag, WinoGrande, PIQA, BoolQ |
| Reading | RACE |
| Perplexity | WikiText-2 |

```bash
# Quick sanity check (limit 10 examples)
python scripts/eval_lm_benchmark.py --alpha 0.0 1.0 --tasks wikitext --limit 10

# Full single-layer sweep (all 10 tasks, 7 alpha values)
python scripts/eval_lm_benchmark.py --alpha 0.0 -0.5 -1.0 -2.0 0.5 1.0 2.0

# Multi-layer grid sweep (layers × alphas × denoise methods)
python scripts/eval_lm_benchmark.py --mode grid \
    --layers 10 14 16 20 --alpha 0.0 -1.0 1.0 --denoise fisher raw

# Resume interrupted grid (skips already-completed runs)
python scripts/eval_lm_benchmark.py --mode grid --resume

# Test specific task group
python scripts/eval_lm_benchmark.py --task-group reasoning --alpha 0.0 1.0
```

**Output structure** (`outputs/eval/{model_id}/`):
```
├── baseline.json                    # α=0.0 scores (shared across layers)
├── fisher_L14_a+1.0.json          # per-run structured results
├── fisher_L14_a-1.0.json
├── raw_L16_a+2.0.json
├── raw/                             # full lm-eval raw output (preserved)
│   ├── baseline_raw.json
│   ├── fisher_L14_a+1.0_raw.json
│   └── ...
├── summary_20260412_201659.json    # timestamped comparison
└── summary_latest.json              # latest comparison
```

### 2. GPTZero AI Detection

Generates steered text at different α values and submits to GPTZero for AI detection probability. Requires `GPTZERO_API_KEY` in `.env`.

```bash
# Generate samples only (no API call)
python scripts/eval_gptzero.py --alpha 0.0 -1.0 1.0 --generate-only

# Full pipeline (generate + detect)
python scripts/eval_gptzero.py --alpha 0.0 -1.0 -2.0 1.0 2.0
```

### 3. Human AI-feel Scoring

Blind evaluation workflow: generates steered text, shuffles/blinds samples, provides interactive scoring interface (1–5 scale), then computes correlation between α and human ratings.

```bash
# Step 1: Generate blind scoring sheet
python scripts/eval_human_scoring.py generate --alpha 0.0 -1.0 1.0

# Step 2: Score samples interactively (1=human-like, 5=AI-like)
python scripts/eval_human_scoring.py score

# Step 3: Analyze results (Pearson r, Kendall τ, comparison table)
python scripts/eval_human_scoring.py analyze
```

## Requirements

- Python 3.10+
- PyTorch with CUDA
- GPU with ≥16 GB VRAM (RTX 3090 tested)
- SiliconFlow API key for dataset construction
