# BeyondStyle

BeyondStyle tests whether a corpus-conditioned machine--human realization
signature can be followed from surface measurements into language-model
representations and back into generated text. The implementation keeps the
same lexical, syntactic, or discourse construct fixed across four stages:

```text
matched continuations
    -> recurrence across generators, domains, and interfaces
    -> leave-one-generator-out hidden-state readout
    -> fixed-token transport against refitted pipeline nulls
    -> held-out paired generation with off-target checks
```

The frozen English data contain 500 proposition-conditioned human--machine
pairs from four domains and four generators, plus a matched prefix-only
transfer corpus. Readout experiments cover Llama 3.1 8B, Qwen3 4B, and Gemma
4 E2B-IT. Fixed-token transport is evaluated in Llama and Qwen; held-out
generation is evaluated in all three model families.

The current artifacts show stable lexical and syntactic recurrence and
held-out readout. Lexical transport exceeds the fitted null in Llama and
Qwen, and the held-out lexical intervention has positive target and
selectivity estimates in Llama, Qwen, and Gemma. Discourse is retained as an
off-target measurement, while syntactic transport remains inside the matched
null range.

## Repository layout

```text
src/                    reusable data, feature, readout, steering, and
                        evaluation modules
scripts/                server-independent command-line entry points
data/corpora/           pinned corpus registries
data/prefixes/          frozen natural-language prefix banks
data/paired/            frozen proposition-conditioned and prefix-only pairs
data/prompts/           document-disjoint development and test prompts
requirements/           runtime, analysis, evaluation, and test dependencies
lm_eval_tasks/          local lm-eval task definitions
tests/                  unit and integration tests
outputs/                generated artifacts; ignored except for its README
```

Manuscript sources, model weights, generated outputs, server provisioning
helpers, logs, and caches are intentionally excluded from the code repository.
Figure-building scripts remain tracked so every reported visualization can be
regenerated from the numerical artifacts.

## Installation

Use Python 3.10 or 3.11. Install the PyTorch wheel appropriate for the local
CUDA driver first; the shared requirements deliberately do not replace it.

```bash
git clone https://github.com/MarrytheToilet/BeyondStyle.git
cd BeyondStyle

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install a CUDA-compatible torch build here.
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m pip check
python -m pytest -q
```

The reproduction below starts from the tracked frozen corpora, so it does not
need a text-generation API. Copy `.env.example` to `.env` only when rebuilding
the paired corpus. Never commit `.env`, access tokens, or model weights.

Download the three observer models separately and place them under one
directory with these exact leaf names:

```text
$MODEL_ROOT/Llama-3.1-8B-Instruct
$MODEL_ROOT/Qwen3-4B
$MODEL_ROOT/gemma-4-E2B-it
```

Then set an absolute path and verify the runtime:

```bash
export MODEL_ROOT=/absolute/path/to/models
python scripts/check_environment.py
```

## Reproduce the English evidence chain

Run every command from the repository root. Outputs are written under
`outputs/en/`; every stage records input hashes, seeds, model identity, and
layer conventions in a manifest.

### 1. Surface signatures and interface transfer

```bash
python scripts/extract_linguistic_features.py \
  --language en \
  --dataset data/paired/en/proposition_aligned/pairs.jsonl \
  --out-dir outputs/en/05_linguistic/features \
  --bootstrap 2000 --permutations 5000 --seed 20260725

python scripts/extract_linguistic_features.py \
  --language en \
  --dataset data/paired/en/prefix_only_control/pairs.jsonl \
  --out-dir outputs/en/05_linguistic/prefix_only_control/features \
  --bootstrap 2000 --permutations 5000 --seed 20260725

python scripts/evaluate_prefix_only_transfer.py \
  --surface-only \
  --model-specs \
    Llama-3.1-8B-Instruct:7:32 \
    Qwen3-4B:8:36 \
    gemma-4-E2B-it:8:35
```

The feature-inventory sensitivity checks consume those two frozen feature
tables:

```bash
python scripts/analyze_feature_jackknife.py
python scripts/analyze_feature_subset_sensitivity.py
python scripts/analyze_feature_pca_sensitivity.py
python scripts/analyze_feature_redundancy.py
```

### 2. Hidden states and generator-held-out readouts

`run_hidden_formal.sh` is the only multi-model extraction wrapper retained in
the public repository. It is parameterized by `MODEL_ROOT` and does not assume
a scheduler, GPU number, cache location, or server filesystem.

```bash
MODEL_ROOT="$MODEL_ROOT" bash scripts/run_hidden_formal.sh en

for model in Llama-3.1-8B-Instruct Qwen3-4B gemma-4-E2B-it; do
  MODEL_ROOT="$MODEL_ROOT" \
    bash scripts/run_prefix_only_hidden_control.sh "$model"
done

python scripts/evaluate_prefix_only_transfer.py \
  --model-specs \
    Llama-3.1-8B-Instruct:7:32 \
    Qwen3-4B:8:36 \
    gemma-4-E2B-it:8:35
```

Build the source-consistent readouts at every depth used by the transport and
configuration analyses:

```bash
python scripts/build_concept_patterns.py \
  --states-root outputs/en/03_steering_vectors \
  --features outputs/en/05_linguistic/features/paired_feature_differences.csv \
  --out-dir outputs/en/05_linguistic/concept_patterns_layerwise \
  --representation mean_pool --holdout generator --null-k 10 \
  --groups lexical syntactic discourse \
  --model-specs \
    Llama-3.1-8B-Instruct:3 Llama-3.1-8B-Instruct:7 \
    Llama-3.1-8B-Instruct:11 Llama-3.1-8B-Instruct:15 \
    Llama-3.1-8B-Instruct:20 Llama-3.1-8B-Instruct:24 \
    Llama-3.1-8B-Instruct:28 Llama-3.1-8B-Instruct:32 \
    Qwen3-4B:3 Qwen3-4B:8 Qwen3-4B:12 Qwen3-4B:17 \
    Qwen3-4B:22 Qwen3-4B:27 Qwen3-4B:31 Qwen3-4B:36 \
    gemma-4-E2B-it:3 gemma-4-E2B-it:8 gemma-4-E2B-it:12 \
    gemma-4-E2B-it:17 gemma-4-E2B-it:21 gemma-4-E2B-it:26 \
    gemma-4-E2B-it:30 gemma-4-E2B-it:35
```

### 3. Jacobian pullbacks and fixed-token transport

The confirmatory transport configurations are Llama L7-to-L32 and Qwen
L8-to-L36. The following shell function only abbreviates repeated,
server-independent CLI calls:

```bash
build_transport_axis () {
  model="$1"; input_layer="$2"; anchor_layer="$3"
  layer_tag="$(printf '%02d' "$input_layer")"
  anchor_tag="$(printf '%02d' "$anchor_layer")"
  states="outputs/en/03_steering_vectors/$model/proposition_aligned/mean_pool"
  patterns="outputs/en/05_linguistic/concept_patterns_layerwise/$model/mean_pool"
  pullback="outputs/en/05_linguistic/causal_pullback/$model/mean_pool"

  python scripts/build_causal_pullback_directions.py \
    --model "$MODEL_ROOT/$model" \
    --pairs data/paired/en/proposition_aligned/pairs.jsonl \
    --states "$states/states/paired_states_layer$layer_tag.pt" \
    --input-directions-dir "$patterns/layer$layer_tag/directions" \
    --downstream-directions-dir "$patterns/layer$anchor_tag/directions" \
    --input-layer "$input_layer" --downstream-layer "$anchor_layer" \
    --max-pairs 32 --batch-size 1 --null-k 10 \
    --torch-dtype bf16 --attention-backend sdpa --seed 8236 \
    --out-dir "$pullback"

  python scripts/freeze_causal_pullback_axes.py \
    --source-checkpoint "$states/models/mean_diff_nullspace_layer$layer_tag.pt" \
    --lexical-checkpoint "$pullback/layer$layer_tag/directions/lexical.pt" \
    --syntactic-checkpoint "$pullback/layer$layer_tag/directions/syntactic.pt" \
    --sealed-prompt-start 34 --random-controls 8 --seed 8236 \
    --out-dir "$pullback/frozen_axes"
}

build_transport_axis Llama-3.1-8B-Instruct 7 32
build_transport_axis Qwen3-4B 8 36

for model in Llama-3.1-8B-Instruct Qwen3-4B; do
  MODEL_ROOT="$MODEL_ROOT" \
    bash scripts/run_canonical_behavior_model.sh "$model"
  MODEL_ROOT="$MODEL_ROOT" \
    MAX_PROMPTS=100 NULL_CONTROLS=100 \
    bash scripts/run_confirmatory_hidden_audits.sh "$model"
done
```

The fixed-token audit uses the same 100 token trajectories for the learned
directions and every null refit. `audit_pullback_matched_nulls.py` verifies the
trajectory hashes before comparing depth-normalized transport areas.

### 4. Held-out generation

Configuration selection uses prompts 0--19; prompts 20--33 remain unused; the
test uses prompts 34--199. The frozen generation settings are:

| Model | Intervention layer | Symmetric endpoint |
|---|---:|---:|
| Llama 3.1 8B | 7 | 2.0 |
| Qwen3 4B | 22 | 3.0 |
| Gemma 4 E2B-IT | 3 | 0.25 |

Build and freeze the three selected pullbacks under a separate root, then run
the untouched test. The commands are identical across architectures apart
from layer and endpoint values:

```bash
build_generation_axis () {
  model="$1"; input_layer="$2"; anchor_layer="$3"
  layer_tag="$(printf '%02d' "$input_layer")"
  anchor_tag="$(printf '%02d' "$anchor_layer")"
  states="outputs/en/03_steering_vectors/$model/proposition_aligned/mean_pool"
  patterns="outputs/en/05_linguistic/concept_patterns_layerwise/$model/mean_pool"
  root="outputs/en/05_linguistic/causal_pullback/$model/mean_pool/selected_generation"

  python scripts/build_causal_pullback_directions.py \
    --model "$MODEL_ROOT/$model" \
    --pairs data/paired/en/proposition_aligned/pairs.jsonl \
    --states "$states/states/paired_states_layer$layer_tag.pt" \
    --input-directions-dir "$patterns/layer$layer_tag/directions" \
    --downstream-directions-dir "$patterns/layer$anchor_tag/directions" \
    --input-layer "$input_layer" --downstream-layer "$anchor_layer" \
    --max-pairs 32 --batch-size 1 --null-k 10 \
    --torch-dtype bf16 --attention-backend sdpa --seed 8236 \
    --out-dir "$root"

  python scripts/freeze_causal_pullback_axes.py \
    --source-checkpoint "$states/models/mean_diff_nullspace_layer$layer_tag.pt" \
    --lexical-checkpoint "$root/layer$layer_tag/directions/lexical.pt" \
    --syntactic-checkpoint "$root/layer$layer_tag/directions/syntactic.pt" \
    --sealed-prompt-start 34 --random-controls 8 --seed 8236 \
    --out-dir "$root/frozen_axes"
}

build_generation_axis Llama-3.1-8B-Instruct 7 32
build_generation_axis Qwen3-4B 22 36
build_generation_axis gemma-4-E2B-it 3 35

run_selected_generation () {
  model="$1"; layer="$2"; strength="$3"
  root="outputs/en/05_linguistic/causal_pullback/$model/mean_pool/selected_generation"
  generated="outputs/en/03_steering_vectors/$model/proposition_aligned/mean_pool/selected_generation"
  analysis="$root/heldout_analysis"
  chat_args=()
  if [ "$model" = "gemma-4-E2B-it" ]; then
    chat_args=(--chat-template --chat-continuation-instruction)
  fi

  python scripts/run_steering_generation.py \
    --model "$MODEL_ROOT/$model" --language en \
    --representations mean_pool --dataset-tag proposition_aligned \
    --prompts data/prompts/en/formal.json \
    --components lexical --component-family causal_pullback \
    --component-root "$root/frozen_axes" --layer "$layer" \
    --strengths "-$strength" "$strength" \
    --prompt-start 34 --max-prompts 166 \
    --num-seeds 3 --base-seed 8236 --max-new-tokens 128 \
    --torch-dtype bf16 --attention-backend sdpa \
    "${chat_args[@]}" \
    --output-subdir selected_generation

  python scripts/analyze_seed_robustness.py \
    --input-dir "$generated" --out-dir "$analysis" \
    --model-id "$model" --direction-family canonical_pullback \
    --observational-features \
      outputs/en/05_linguistic/features/paired_feature_differences.csv \
    --components lexical --endpoint-strength "$strength" \
    --bootstrap 10000 --permutations 20000

  python scripts/analyze_generation_feature_inventory.py \
    --differences "$analysis/paired_component_differences.csv" \
    --observational-features \
      outputs/en/05_linguistic/features/paired_feature_differences.csv \
    --out-dir "$analysis/lexical_features" \
    --component lexical --group lexical \
    --bootstrap 10000 --permutations 20000
}

run_selected_generation Llama-3.1-8B-Instruct 7 2
run_selected_generation Qwen3-4B 22 3
run_selected_generation gemma-4-E2B-it 3 0.25
```

For the paired-output semantic audit, pass the generated directory above to
`audit_generation_semantics.py` and use an independent local encoder or Hugging
Face model identifier:

```bash
python scripts/audit_generation_semantics.py \
  --input-dir outputs/en/03_steering_vectors/Llama-3.1-8B-Instruct/proposition_aligned/mean_pool/selected_generation \
  --out-dir outputs/en/05_linguistic/causal_pullback/Llama-3.1-8B-Instruct/mean_pool/selected_generation/heldout_analysis/semantics \
  --model-path sentence-transformers/all-MiniLM-L6-v2 \
  --components lexical --endpoint-strength 2 \
  --bootstrap 10000 --permutations 20000
```

Repeat the semantic command with the matching Qwen or Gemma paths and
endpoint. Additional baseline, calibration, and prompt-level analyses are
indexed in [`scripts/README.md`](scripts/README.md).

### 5. Figures

After the numerical stages finish, regenerate the paper figures directly from
their CSV/JSON artifacts:

```bash
python scripts/plot_fig_surface_signature.py
python scripts/plot_fig_hidden_readout.py
python scripts/plot_fig_transport_null.py
python scripts/plot_fig_generation.py
python scripts/plot_projected_lexical_field.py
```

The scripts write vector PDFs under `ACL27/figures/` and raster previews under
`ACL27/build/previews/`. Those generated directories remain ignored; the
figure-building source code is tracked.

## Data regeneration

The paired corpora and their manifests are tracked, so API regeneration is not
part of result reproduction. To rebuild them, configure an OpenAI-compatible
endpoint in `.env`, then inspect and run:

```bash
python scripts/download_hf_raw_datasets.py --help
python scripts/build_prefix_bank.py --help
python scripts/validate_prefix_bank.py --language en
python scripts/synthesize_paired_data.py --help
python scripts/synthesize_prefix_only_control.py --help
```

API generations need not be byte-identical. The committed pair files and
manifests are the frozen experimental inputs.

## Reproducibility invariants

- Surface targets are fitted without using free-generation outcomes.
- The evaluated generator is excluded from every nuisance, score, and readout
  fit in its fold.
- Human- and machine-side patterns are combined only when their orientations
  agree.
- Prompt prefill is unchanged; steering begins on generated-token forwards.
- Learned and null fixed-token runs reuse identical token trajectories.
- Null replicates rebuild the fitted pipeline after breaking only the
  linguistic correspondence.
- Generation inference resamples prompts after averaging paired seed
  differences within each prompt.

## Responsible use

The measured directions are corpus-conditioned contrasts, not authorship
detectors. They must not be used to accuse individual writers, adjudicate
misconduct, or penalize formal, non-native, neurodivergent, or technical
writing. Activation steering can also obscure machine provenance; deployments
should preserve provenance and disclose generated content where required.
