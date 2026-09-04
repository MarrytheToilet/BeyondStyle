# BeyondStyle

BeyondStyle traces corpus-conditioned machine--human realization signatures from surface measurements into language-model representations and back into generated text. The implementation keeps one lexical, syntactic, or discourse construct fixed across four tests:

```text
matched continuations
    -> surface recurrence across generators, domains, and interfaces
    -> leave-one-generator-out hidden-state readout
    -> Jacobian-pulled fixed-token transport against refitted nulls
    -> paired free-generation realization with off-target checks
```

The current English experiments use 500 proposition-conditioned human--machine pairs from four domains and four generators, with a matched prefix-only transfer corpus. Readout experiments cover Llama 3.1 8B, Qwen3 4B, and Gemma 4 E2B-IT. Fixed-token transport is evaluated in Llama and Qwen; held-out generation is evaluated in all three model families.

## What is implemented

- extraction of lexical, syntactic, discourse, and nuisance measurements;
- generator- and domain-held-out surface-direction audits;
- continuous hidden-state readouts evaluated on both source sides;
- source-consistent Jacobian pullbacks into earlier intervention layers;
- nuisance-nullspace projection and complete refitted-pipeline nulls;
- decode-only activation steering with untouched prompt prefill;
- prompt-clustered bootstrap inference over paired generations and seeds;
- feature-level, semantic-similarity, dose, baseline, and prompt-heterogeneity audits.

The current result artifacts record stable lexical and syntactic recurrence and held-out readout, lexical transport beyond the fitted null in Llama and Qwen, and positive lexical target and selectivity estimates in held-out generation for Llama, Qwen, and Gemma. Machineward intervention lowers function-word rate and raises MATTR-50 lexical diversity in all three families. Discourse remains an off-target measurement, and syntactic transport remains within the matched null range.

## Repository layout

```text
src/                    reusable data, extraction, analysis, generation,
                        steering, evaluation, and runtime modules
scripts/                command-line pipeline and audit entry points
data/corpora/           pinned corpus registries
data/prefixes/          frozen natural-language prefix banks
data/paired/            proposition-conditioned and prefix-only pairs
data/prompts/           document-disjoint generation prompts
outputs/                ignored experiment artifacts; only README is tracked
requirements/           runtime, analysis, evaluation, and development layers
lm_eval_tasks/          local lm-eval task definitions
tests/                  unit and integration tests
```

Model weights and generated outputs do not belong in the repository. Keep models under an external `MODEL_ROOT`; `.env`, raw downloads, logs, caches, checkpoints, and output artifacts are ignored.

## Installation

Use Python 3.10 or 3.11. Install the PyTorch build that matches the server GPU and CUDA driver before installing the project requirements.

```bash
git clone https://github.com/MarrytheToilet/BeyondStyle.git
cd BeyondStyle

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install the appropriate PyTorch wheel first.
python -m pip install -r requirements.txt
cp .env.example .env
python -m pip check
```

Set the generation API and local model paths in `.env`. The repository accepts an OpenAI-compatible endpoint through `API_BASE_URL`, `API_KEY`, and `API_MODEL` or `API_MODELS`. Never commit `.env` or model access tokens.

For GPU-specific installation and migration notes, see [`SERVER_RUNBOOK.md`](SERVER_RUNBOOK.md).

## Pipeline

Inspect each entry point with `--help` before launching a full run. The main stages are:

```bash
# Data construction and validation
python scripts/download_hf_raw_datasets.py --help
python scripts/build_prefix_bank.py --help
python scripts/validate_prefix_bank.py --language en
python scripts/synthesize_paired_data.py --help
python scripts/synthesize_prefix_only_control.py --help

# Surface measurements and hidden readout
python scripts/extract_linguistic_features.py --language en
bash scripts/run_hidden_formal.sh en
python scripts/eval_generator_holdout.py --help
python scripts/build_concept_patterns.py --help

# Pullback construction and fixed-token tests
python scripts/build_causal_pullback_directions.py --help
python scripts/freeze_causal_pullback_axes.py --help
python scripts/audit_layerwise_concept_transport.py --help
python scripts/audit_pullback_matched_nulls.py --help

# Held-out paired generation and analysis
python scripts/run_steering_generation.py --help
python scripts/analyze_seed_robustness.py --help
python scripts/audit_generation_semantics.py --help
```

Formal output paths follow:

```text
outputs/{language}/{stage}/{model}/{dataset_tag}/...
```

Each run records model identity, layer conventions, prompt and data hashes, seeds, intervention configuration, and output manifests. Completed output trees are intentionally ignored; copy only the required frozen inputs and manifests when moving experiments between machines.

## Reproducibility invariants

- Surface targets are fitted without using free-generation outcomes.
- The evaluated generator is excluded from nuisance fitting, surface scoring, and readout fitting in its fold.
- Human- and machine-side readout patterns are combined only when their orientations agree.
- The intervention pole is inherited from the machine-oriented downstream readout.
- Prompt prefill is unchanged; steering begins on generated-token forwards.
- Fixed-token learned and null directions reuse identical token trajectories.
- Null replicates rebuild the complete fitted pipeline after breaking only the linguistic correspondence.
- Generation inference is clustered by prompt after averaging paired seed differences within prompt.

## Tests

```bash
python -m pytest -q
```

Run the CUDA smoke checks before a long model job:

```bash
python scripts/check_environment.py --help
python scripts/smoke_layer_alignment.py --help
python scripts/smoke_steering_runtime.py --help
```

## Responsible use

The measured directions are corpus-conditioned contrasts, not authorship detectors. They must not be used to accuse individual writers, adjudicate misconduct, or penalize formal, non-native, neurodivergent, or technical writing. Activation steering can also be misused to obscure machine provenance; deployments should preserve provenance and disclose generated content where required.
