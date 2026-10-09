# `src/`

Reusable library code for the formal pipeline:

```text
config.py          environment and canonical output paths
layer_indexing.py  one-based decoder-layer convention
data/              natural windows, paired I/O and proposition validation
extraction/        exact last-token and continuation-mean hidden states
steering/          nullspace source direction, raw control, calibration, hooks
generation/        local/API generation and steered decoding
evaluation/        lm-eval steered-model wrapper
runtime/           dtype, attention, device and inference helpers
analysis/          paired hidden-geometry analysis
visualization/     shared pink/blue plotting style
```

All user-facing layers are one-based decoder-block outputs. Layer \(L\) is
extracted and intervened on at module `layers[L - 1]`.

Formal text generation uses `decode_only`: prompt activations and prompt KV
cache remain identical to baseline, and intervention begins on cached
generated-token forwards. The runtime multiplies the requested strength by
the scale stored in the direction checkpoint. Source-direction checkpoints
and canonical pullback checkpoints can use different calibrations; preserve
their metadata rather than assuming a common unit across direction families.

The package does not implement a human-participant study. The optional blind
generation evaluator uses model judgments, which must not be reported as
human annotation or perceptual validation.
