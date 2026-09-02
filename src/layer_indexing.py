"""Shared layer-numbering conventions for extraction and intervention.

Many Hugging Face causal language models return ``hidden_states`` as

    [embedding_output, block_1_output, ..., block_N_output]

but some architectures apply a final norm before storing the last item. Decoder
modules are stored with zero-based indices
``layers[0] ... layers[N - 1]``.  BeyondStyle exposes decoder layers to users
with one-based numbers and captures their forward outputs directly, so public
layer ``L`` always maps to ``layers[L - 1]`` for both extraction and
intervention.
"""

LAYER_CONVENTION = "decoder_block_output_1_based"


def _validate_layer_number(layer: int, num_hidden_layers: int) -> None:
    if isinstance(layer, bool) or not isinstance(layer, int):
        raise TypeError(f"layer must be an integer, got {type(layer).__name__}")
    if layer < 1 or layer > num_hidden_layers:
        raise ValueError(
            f"layer must be in [1, {num_hidden_layers}] under "
            f"{LAYER_CONVENTION}, got {layer}"
        )


def hidden_state_index(layer: int, num_hidden_layers: int) -> int:
    """Return the usual HF tuple slot for a one-based decoder layer.

    This mapping is useful for diagnostics. It is not used for aligned
    extraction because the last tuple item can be post-final-norm.
    """
    _validate_layer_number(layer, num_hidden_layers)
    return layer


def decoder_block_index(layer: int, num_hidden_layers: int) -> int:
    """Map a public one-based decoder layer to a zero-based module index."""
    _validate_layer_number(layer, num_hidden_layers)
    return layer - 1
