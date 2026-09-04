# `outputs/`

Experiment outputs are organized by language, stage, model, and dataset tag. The directory itself is retained for documentation, while generated artifacts are ignored by Git.

```text
outputs/{language}/
  02_separability/{model}/{dataset_tag}/
  03_steering_vectors/{model}/{dataset_tag}/
  04_lmeval/{model}/{dataset_tag}/
  05_linguistic/
    features/
    prefix_only_control/
    concept_patterns/
    concept_patterns_layerwise/
    causal_pullback/{model}/
    token_projection/{model}/
    projection_mapping/
    hidden_language/
    external_evaluation/
    compute_audit/
```

Each analysis directory should retain the inputs or their hashes, model and layer conventions, random seeds, configuration fingerprint, summary tables, and a manifest linking derived files to their sources. Large hidden-state tensors, model checkpoints, raw generations, logs, plots, and intermediate shards stay outside version control.

Do not compare or merge output directories unless their manifests agree on the paired-corpus hash, prompt split, feature dictionary, model revision, layer convention, pooling rule, and intervention configuration.
