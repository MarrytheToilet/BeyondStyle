# Command-line scripts

The public scripts are server-independent entry points. Run them from the
repository root, inspect Python interfaces with `--help`, and set model paths
through `MODEL_ROOT` or explicit `--model` arguments. The exact end-to-end
commands and frozen model configurations are in the root
[`README.md`](../README.md).

Machine provisioning, file transfer, tmux scheduling, multi-GPU launch
bundles, and interruption recovery are intentionally ignored. They do not
define an estimand and are not needed to reproduce one. Figure builders remain
tracked because they are part of result reproduction.

## Core evidence chain

| Stage | Entry point | Purpose |
|---|---|---|
| Frozen data | `download_hf_raw_datasets.py` | Save pinned native rows when rebuilding the corpus. |
| Frozen data | `build_prefix_bank.py`, `validate_prefix_bank.py` | Build and verify model-independent prefix windows. |
| Frozen data | `audit_prefix_tokenizers.py` | Measure frozen windows under observer tokenizers without redefining them. |
| Frozen data | `synthesize_paired_data.py` | Construct proposition-conditioned human--machine pairs. |
| Interface control | `synthesize_prefix_only_control.py` | Construct the matched prefix-only transfer corpus. |
| Surface signature | `extract_linguistic_features.py` | Extract the lexical, syntactic, discourse, and nuisance inventory. |
| Hidden states | `steering_pipeline.py` | Extract layer states and fit source/nullspace checkpoints. |
| Source-separation diagnostic | `eval_generator_holdout.py` | Evaluate source discrimination with one generator held out. |
| Hidden readout | `build_concept_patterns.py` | Fit source-consistent continuous-score readouts and evaluate generator-held-out correlations. |
| Interface transfer | `evaluate_prefix_only_transfer.py` | Apply plan-fitted surface axes and readouts to prefix-only examples. |
| Pullback | `build_causal_pullback_directions.py` | Pull an anchor readout into an earlier intervention layer. |
| Pullback | `freeze_causal_pullback_axes.py` | Freeze pole, scale, held-out boundary, and random controls. |
| Transport | `audit_layerwise_concept_transport.py` | Trace fixed-token responses through independently fitted readouts. |
| Transport null | `audit_pullback_matched_nulls.py` | Refit the full pipeline after stratified correspondence permutation. |
| Generation | `run_steering_generation.py` | Run resumable, generated-token-only paired interventions. |
| Generation | `analyze_seed_robustness.py` | Estimate prompt-level effects after averaging paired seeds. |
| Feature audit | `analyze_generation_feature_inventory.py` | Test every feature in a steered surface group with BH correction. |
| Semantic audit | `audit_generation_semantics.py` | Measure paired-output similarity using an independent encoder. |

## Retained wrappers

These wrappers contain stable experiment definitions rather than
machine-specific scheduling:

| Wrapper | Scope |
|---|---|
| `run_hidden_formal.sh` | Extract and evaluate all three English observer models. |
| `run_prefix_only_hidden_control.sh` | Extract prefix-only states for one named observer model. |
| `run_canonical_behavior_model.sh` | Produce one-model frozen trajectories and behavior controls. |
| `run_confirmatory_hidden_audits.sh` | Run fixed-token transport and 100 matched null refits for one model. |
| `run_seed_robustness_audit.sh` | Run a symmetric multi-seed endpoint for one frozen axis. |
| `run_direction_baseline_audit.sh` | Evaluate encoding and source-contrast baselines under the same protocol. |
| `run_decoding_regime_audit.sh` | Re-evaluate a frozen direction under declared decoding regimes. |

Each wrapper accepts `MODEL_ROOT`; none hard-codes a server path, GPU count,
Hugging Face proxy, or job scheduler.

## Robustness and interpretation

| Entry point | Purpose |
|---|---|
| `analyze_feature_jackknife.py` | Refit each surface group after deleting one feature. |
| `analyze_feature_subset_sensitivity.py` | Enumerate all two- and three-feature deletions of the lexical group. |
| `analyze_feature_pca_sensitivity.py` | Test correlated-coordinate truncation. |
| `analyze_feature_redundancy.py` | Report paired-difference correlation and effective rank. |
| `audit_semantic_alignment.py` | Compare content alignment across the two corpus interfaces. |
| `audit_pullback_design_sensitivity.py` | Reuse frozen gradients across fit-size and nullspace-rank settings. |
| `analyze_development_selection_stability.py` | Replay the frozen development rule under prompt resampling. |
| `select_qwen_layer_strength.py` | Freeze the Qwen cell from a completed development grid. |
| `select_gemma_layer_strength.py` | Freeze the Gemma cell from a completed development grid. |
| `analyze_prompt_heterogeneity.py` | Audit domain, length, sign reversal, and representative prompts. |
| `compare_generation_directions.py` | Compare canonical and baseline directions on matched estimands. |
| `evaluate_blind_lexical_realization.py` | Run an optional blind external evaluation of paired outputs. |
| `analyze_token_projections.py` | Analyze contextual token projections with generator cross-fitting. |
| `analyze_signed_transport.py` | Summarize sign-oriented fixed-token transport. |

The jackknife and subset scripts take no command-line options and use the
feature paths written by the root README. Development-selection replay also
takes no options, but requires the separately completed per-model development
grids under `injection_layer_sweep/` (and the two Llama development analyses).
Running only the frozen held-out settings does not produce those grids. The
selection scripts must never be pointed at test-set effects to retune a result.

## Supporting analyses

The following scripts are useful for secondary diagnostics but are not
prerequisites for the four-stage result:

- `eval_heldout_separability.py`, `analyze_hidden_geometry.py`,
  `analyze_projection_mapping.py`, and `analyze_nullspace_concept_readout.py`;
- `decompose_hidden_language.py`, `analyze_component_steering.py`, and
  `audit_hidden_propagation.py`;
- `eval_lm_benchmark.py`, `eval_pangram.py`, and
  `summarize_decoding_regimes.py`;
- `merge_generation_shards.py` and `refresh_separability_figures.py`.

## Figure reproduction

| Entry point | Output |
|---|---|
| `paper_figstyle.py` | Shared dimensions, palette, typography, and export helpers. |
| `plot_fig_surface_signature.py` | Surface recurrence and stability figure. |
| `plot_fig_hidden_readout.py` | Generator-held-out readout figure. |
| `plot_fig_transport_null.py` | Fixed-token transport and fitted-null figure. |
| `plot_fig_generation.py` | Held-out target, selectivity, and seed figure. |
| `plot_projected_lexical_field.py` | Contextual lexical-field visualization. |
| `plot_positive_generation_result.py` | `appendix_generation_robustness.pdf`; uses Llama prompt-level estimates only. |
| `plot_acl_evidence_figures.py --figures surface` | `appendix_surface_transfer.pdf`; needs interface-transfer outputs. |
| `plot_acl_evidence_figures.py --figures hidden` | `appendix_hidden_robustness.pdf`; also needs the design-sensitivity audit. |

The builders read CSV/JSON artifacts under `outputs/` and write vector PDFs
and raster previews. They do not run model inference. The lexical-field
builder also computes expression summaries from existing token projections;
it is not just an image-format conversion. Its upstream GPU step is
`analyze_token_projections.py`, documented in the root README.

Retired story-layout variants and unused plotting functions have been removed.
The current generation appendix is produced only by
`plot_positive_generation_result.py`, so the transfer/sensitivity builder
cannot replace it with an older layout. The manually prepared overview asset
is separate from these numerical plotters.

## Runtime checks

```bash
python scripts/check_environment.py
python scripts/smoke_layer_alignment.py --help
python scripts/smoke_steering_runtime.py --help
python scripts/check_lm_eval_data.py --help
```

Generated files follow the common layout
`outputs/{language}/{stage}/{model}/{dataset_tag}/...`. Output trees are
ignored; manifests within them record the exact inputs and runtime settings.
