"""Regression tests for extraction/intervention layer alignment."""

import unittest

import torch.nn as nn

from src.layer_indexing import (
    LAYER_CONVENTION,
    decoder_block_index,
    hidden_state_index,
)
from src.steering.hooks import get_layer_module


class _Backbone(nn.Module):
    def __init__(self, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(2, 2) for _ in range(n_layers))


class _CausalLM(nn.Module):
    def __init__(self, n_layers: int):
        super().__init__()
        self.model = _Backbone(n_layers)


class _NestedBackbone(nn.Module):
    def __init__(self, n_layers: int):
        super().__init__()
        self.language_model = _Backbone(n_layers)


class _NestedCausalLM(nn.Module):
    def __init__(self, n_layers: int):
        super().__init__()
        self.model = _NestedBackbone(n_layers)


class LayerIndexingTest(unittest.TestCase):
    def test_public_layer_maps_to_matching_state_and_block(self):
        self.assertEqual(LAYER_CONVENTION, "decoder_block_output_1_based")
        self.assertEqual(hidden_state_index(1, 4), 1)
        self.assertEqual(decoder_block_index(1, 4), 0)
        self.assertEqual(hidden_state_index(4, 4), 4)
        self.assertEqual(decoder_block_index(4, 4), 3)

    def test_zero_and_out_of_range_layers_are_rejected(self):
        for invalid in (0, -1, 5):
            with self.subTest(layer=invalid):
                with self.assertRaises(ValueError):
                    hidden_state_index(invalid, 4)
                with self.assertRaises(ValueError):
                    decoder_block_index(invalid, 4)

    def test_non_integer_layer_is_rejected(self):
        for invalid in (True, 1.0, "1"):
            with self.subTest(layer=invalid):
                with self.assertRaises(TypeError):
                    hidden_state_index(invalid, 4)

    def test_llama_style_module_mapping_is_one_based(self):
        model = _CausalLM(4)
        self.assertIs(get_layer_module(model, 1), model.model.layers[0])
        self.assertIs(get_layer_module(model, 4), model.model.layers[3])

    def test_nested_language_model_layout_is_supported(self):
        model = _NestedCausalLM(3)
        self.assertIs(
            get_layer_module(model, 2),
            model.model.language_model.layers[1],
        )

    def test_module_mapping_rejects_invalid_public_layer(self):
        model = _CausalLM(2)
        with self.assertRaises(ValueError):
            get_layer_module(model, 0)
        with self.assertRaises(ValueError):
            get_layer_module(model, 3)


if __name__ == "__main__":
    unittest.main()
