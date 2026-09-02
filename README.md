# BeyondStyle

BeyondStyle studies a narrow, automatically measurable object:

> **content-conditioned machine–human realization signatures**

The project asks which lexical and syntactic differences remain when a
machine realizes an explicit proposition plan extracted from a matched human
continuation, how those differences map to hidden states, and whether the same
hidden route produces the expected change in free generation.

The ACL paper is in [`ACL27/paper.tex`](ACL27/paper.tex). The current research
specification and live completion checklist are in
[`ACL_RESEARCH_PLAN.md`](ACL_RESEARCH_PLAN.md).

## Evidence chain

```text
500 proposition-conditioned human–machine pairs
    → frozen lexical / syntactic / discourse measurements
    → generator-held-out hidden readouts
    → canonical Jacobian pullbacks
    → pipeline-matched fixed-token intervention trace
    → separate free-generation response matrix
```

The stages are not treated as equivalent:

- a source contrast is not a universal authorship definition;
- a linear readout is not a causal result;
- a fixed-token readout response is not automatically a generation control;
- detector scores and human impressions do not define the construct.

> **Non-detection warning.** The released measurements are conditional
> corpus-level research contrasts, not an authorship detector. Do not use
> them to accuse, rank, deny access to, or otherwise make decisions about
> individual writers. Formal, non-native, neurodivergent, or technical human
> writing may share measured properties with generated text.

## Confirmatory scope

- Language: English.
- Target models: Llama 3.1 8B Instruct and Qwen3 4B.
- Representation: continuation mean.
- Hidden estimator: within-class nullspace readout and Jacobian pullback.
- Intervention: decode-only, prompt prefill untouched, strengths \(-1,+1\).
- Surface targets: lexical and syntactic.
- Off-target measurement: discourse.
- Hidden null: 100 within-stratum target permutations that repeat the
  readout→pullback→fixed-token pipeline.
- Behavior null: eight orthonormal residual-stream axes.

Chinese, last-token states, Gemma, detector APIs, and LM-eval outputs are not
required for the current ACL claim.

## Canonical layout

```text
data/
  raw/{en,zh}/hf/                   pinned Hugging Face subsets
  prefixes/{en,zh}/                 natural-language prefix banks
  paired/en/proposition_aligned/    500 formal English pairs
  prompts/en/formal.json            document-disjoint steering prompts
src/
  analysis/                         reusable estimators and inference
  extraction/                       hidden-state extraction
  generation/                       generation backends
  steering/                         nullspace directions and hooks
scripts/                            auditable pipeline entry points
outputs/en/05_linguistic/           canonical linguistic and causal results
ACL27/                              ACL manuscript and three main figures
paper/reviews/                      original review record
```

Models remain outside the repository under `MODEL_ROOT` (default:
`../../models`).

## Main commands

```bash
# Validate the canonical English data.
python scripts/validate_prefix_bank.py --language en

# Extract continuation-mean hidden states and automatic language features.
bash scripts/run_hidden_formal.sh en
python scripts/extract_linguistic_features.py --language en

# Fit cross-source readouts and canonical pullbacks.
python scripts/build_concept_patterns.py --help
python scripts/build_causal_pullback_directions.py --help
python scripts/freeze_causal_pullback_axes.py --help

# Fixed-token traces and strict matched nulls.
python scripts/audit_layerwise_concept_transport.py --help
bash scripts/run_matched_null_audits.sh

# Learned axes plus the eight-axis free-generation null.
bash scripts/run_canonical_behavior_ensemble.sh

# Rebuild all paper figures.
python scripts/plot_acl_story_figures.py --out-dir ACL27/figures

# Run the repository test suite.
python -m pytest -q
```

## Reproducibility invariants

- Never use free-generation outcomes to orient a causal axis.
- The positive pole is the canonical pullback of the machine-oriented
  downstream readout.
- Never modify prompt-prefill representations during steering.
- Never use a fitted endpoint as the only hidden causal evidence.
- Preserve real hidden-side nuisance structure in matched nulls; permute only
  target correspondence and its paired feature-side nuisance rows.
- Report the full lexical–syntactic–discourse response matrix.
- Do not subtract one privileged random direction from a learned behavior
  effect.
- Record input hashes, model identity, layer convention, direction hashes,
  seeds, prompt indices, and hook audits in manifests.

## Current result boundary

The surface signature is distributed rather than a single marker. Lexical and
syntactic scores are readable across held-out generators in both target
models.

Canonical lexical and syntactic pullbacks give fixed-token responses against
machine-oriented layerwise readouts. Under the corrected nuisance-preserving
pipeline null, lexical capacity is larger at all six non-anchor layers in
both models.  For syntax, prompt-bootstrap intervals are positive at 5/6
Llama and 4/6 Qwen non-anchor layers, while 3/6 and 1/6, respectively,
exceed every one of the eight finite matched-null fits.

Free generation does not yield portable component control. Lexical target
effects are unresolved in both models. The canonical syntactic effect is
unresolved in Llama and reverses toward the human-oriented surface signature
in Qwen. Eight random axes, the full cross-component response matrix, and
content/quality guardrails are complete.

See [`SERVER_RUNBOOK.md`](SERVER_RUNBOOK.md) for environment migration.
