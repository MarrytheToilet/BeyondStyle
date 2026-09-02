# `scripts/`

Each file is one auditable entry point. All scripts read `.env` through
`src/config.py`; use `--help` for complete arguments.

## Canonical pipeline

| Stage | Script | Role |
|---|---|---|
| Raw data | `download_hf_raw_datasets.py` | Save fixed-size native rows from pinned Hugging Face shards. |
| Prefixes | `build_prefix_bank.py` | Build model-independent natural-language prefix/continuation windows. |
| Validation | `validate_prefix_bank.py` | Reconstruct and verify every frozen prefix window. |
| Token audit | `audit_prefix_tokenizers.py` | Measure, but never redefine, windows under target tokenizers. |
| Pair synthesis | `synthesize_paired_data.py` | Build the single proposition-aligned EN/ZH paired corpus. |
| Construct control | `synthesize_prefix_only_control.py` | Generate the matched English prefix-only control without exposing a proposition plan. |
| Construct transfer | `run_prefix_only_hidden_control.sh`, `evaluate_prefix_only_transfer.py` | Extract the two primary models and test plan-fitted surface/hidden axes on prefix-only continuations; surface axes use generator-leave-one-out cross-fitting, and `--surface-only` runs without hidden states. |
| Semantic alignment audit | `audit_semantic_alignment.py` | Compare proposition-conditioned and prefix-only pairs with a frozen independent encoder and within-source derangements; the encoder never enters fitting or selection. |
| Confirmatory hidden audit | `run_confirmatory_hidden_audits.sh` | Run 100-prompt direction ablations and 100 complete pipeline-matched nulls for one primary model. |
| Formal prompts | `build_formal_prompts.py` | Select document-disjoint prompts from unused source documents. |
| Hidden states | `steering_pipeline.py` | Extract last-token/mean states and fit nullspace plus raw control. |
| Random holdout | `eval_heldout_separability.py` | Estimate fold-held-out source separability. |
| Generator holdout | `eval_generator_holdout.py` | Test unseen-generator transfer and select robust layers on development data. |
| Geometry | `analyze_hidden_geometry.py` | Analyze cross-layer, cross-generator and cross-model paired geometry. |
| Linguistic features | `extract_linguistic_features.py` | Extract the frozen lexical/syntactic/discourse inventory and paired statistics. |
| Projection cartography | `analyze_projection_mapping.py` | Nullspace source-coordinate trajectories plus train-fitted generator/domain-held-out surface-to-axis mapping. |
| Nullspace concept readout | `analyze_nullspace_concept_readout.py` | Human→AI and AI→Human linguistic readouts from the broader nullspace subspace, with nuisance controls and permutation baselines. |
| Concept patterns | `build_concept_patterns.py` | Build cross-source lexical/syntactic directions with generator-holdout validation and fixed activation-norm calibration. |
| Token projection | `analyze_token_projections.py` | Leave-one-generator-out nullspace token coordinates, position trajectories, exact same-token contextual contrasts, and source-balanced empirical token fields. |
| Hidden-language reconstruction audit | `decompose_hidden_language.py` | Estimator audit for full source-axis reconstruction; not the primary hidden–surface mapping analysis. |
| Component causal analysis | `analyze_component_steering.py` | Compare ±1 component effects against the observational linguistic signature. |
| Hidden propagation audit | `audit_hidden_propagation.py` | Hold tokens fixed and trace the symmetric −1/0/+1 component response across decoder layers. |
| Layerwise concept transport | `audit_layerwise_concept_transport.py` | Test whether an injected concept rotates into independently fitted downstream concept readouts and record per-prompt fixed-trajectory token hashes. |
| Causal pullback directions | `build_causal_pullback_directions.py` | Pull downstream concept readouts back through the frozen model Jacobian and constrain the intervention to the within-class nullspace. |
| Freeze causal pullback axes | `freeze_causal_pullback_axes.py` | Freeze the canonical Jacobian-pullback pole without behavior-based sign selection and build an eight-axis random ensemble. |
| Pullback matched-null audit | `audit_pullback_matched_nulls.py` | Permute surface targets within corpus×generator strata, refit readouts, apply the same Jacobian pullback, and compare fixed-token traces only after exact learned/null trajectory-hash identity is verified. |
| Generation | `run_steering_generation.py` | Resumable, calibrated, generated-token-only steering. |
| Utility | `eval_lm_benchmark.py` | Compact lm-eval non-inferiority guardrail. |
| Detector | `eval_pangram.py` | One auxiliary black-box response; never a primary endpoint. |

## Runtime and figures

| Script | Role |
|---|---|
| `check_environment.py` | Validate Python, packages, CUDA, attention and model configs. |
| `check_lm_eval_data.py` | Prefetch or offline-check the fixed lm-eval tasks. |
| `smoke_layer_alignment.py` | Check extraction/intervention layer identity. |
| `smoke_steering_runtime.py` | Assert untouched prefill and cached decode-only intervention. |
| `refresh_separability_figures.py` | Apply the canonical pink/blue figure style to held-out results. |
| `plot_acl_story_figures.py` | Rebuild the three ACL figures from frozen observations, including injection/propagation/anchor distinctions. |
| `build_acl_result_tables.py` | Rebuild the exact ACL appendix result tables from frozen CSVs and record every source hash. |
| `run_hidden_formal.sh` | Run formal extraction, separability and geometry for one language. |
| `run_steering_full.sh` | Reproduce existing nullspace source-direction generations. |
| `run_canonical_behavior_ensemble.sh` | Generate canonical lexical/syntactic interventions plus eight frozen random axes and analyze the complete surface-response matrix. |
| `run_seed_robustness_audit.sh` | Generate three paired decode seeds for the two learned axes of one model. |
| `run_direction_baseline_audit.sh` | Run the same three-seed protocol for encoding-pattern and nullspace source-difference baselines. |
| `run_direction_baselines_dual_gpu.sh` | Run the two target models concurrently on two GPUs, then analyze and regenerate the ACL table. |
| `analyze_seed_robustness.py` | Preserve prompt×seed observations, report each seed, and bootstrap prompt means after averaging paired seeds within prompt. |
| `audit_pullback_design_sensitivity.py` | Reuse frozen per-pair gradients to audit pullback-axis stability across nullspace ranks and fit-pair counts. |

The formal output layout is:

```text
outputs/{language}/{stage}/{target_model}/proposition_aligned/
    {last_token,mean_pool}/...
```

Current source-direction generation files may contain development conditions
beyond \(-1,+1\). New confirmatory analyses must select only the symmetric
\(-1,+1\) pair. Wider grids are diagnostics, not paper hypotheses.

The component analysis includes prompt-overlap, entity/number, length,
repetition, and fragment checks. An optional local sentence-transformer adds
paired-output semantic-similarity guardrails without an API or human judge.
