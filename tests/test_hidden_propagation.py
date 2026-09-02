import numpy as np
import torch
from torch import nn

from scripts.audit_hidden_propagation import (
    capture_symmetric_conditions,
    capture_symmetric_direction_batch,
    filter_valid_anchor_records,
)
from src.analysis.hidden_propagation import (
    normalized_position_slices,
    orthogonal_random_directions,
    summarize_counterfactual_states,
)
from src.steering import MeanDiffSteering


def test_filter_valid_anchor_records_reports_empty_trajectories():
    records = [
        {
            "prompt_id": "valid",
            "outputs": {"0.0": {"text": "A continuation."}},
        },
        {
            "prompt_id": "whitespace",
            "outputs": {"0.0": {"text": "\n  "}},
        },
        {
            "prompt_id": "missing_strength",
            "outputs": {"1.0": {"text": "Different condition."}},
        },
    ]

    valid, excluded = filter_valid_anchor_records(records, 0.0)

    assert [row["prompt_id"] for row in valid] == ["valid"]
    assert excluded == ["whitespace", "missing_strength"]


def test_normalized_position_slices_cover_every_token():
    slices = normalized_position_slices(7, bins=4)

    assert slices[0] == ("all", slice(0, 7))
    covered = []
    for _, token_slice in slices[1:]:
        covered.extend(range(token_slice.start, token_slice.stop))
    assert covered == list(range(7))


def test_counterfactual_summary_recovers_injected_direction():
    baseline = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 4.0],
            [3.0, 0.0, 5.0],
            [4.0, -1.0, 6.0],
        ],
    )
    lexical = np.array([1.0, 0.0, 0.0])
    syntactic = np.array([0.0, 1.0, 0.0])
    scale = 2.5
    condition_states = {
        -1.0: {8: baseline - scale * lexical},
        0.0: {8: baseline},
        1.0: {8: baseline + scale * lexical},
    }

    rows = summarize_counterfactual_states(
        condition_states,
        {
            "lexical": lexical,
            "syntactic": syntactic,
        },
        target="lexical",
        intervention_scale=scale,
        position_bins=2,
    )
    all_rows = {
        row["readout"]: row
        for row in rows
        if row["position_scope"] == "all"
    }

    assert np.isclose(all_rows["lexical"]["projection_effect"], scale)
    assert np.isclose(all_rows["lexical"]["projection_retention"], 1.0)
    assert np.isclose(all_rows["lexical"]["target_alignment"], 1.0)
    assert np.isclose(all_rows["lexical"]["curvature_norm"], 0.0)
    assert np.isclose(all_rows["syntactic"]["projection_effect"], 0.0)


def test_counterfactual_summary_rejects_mismatched_shapes():
    states = {
        -1.0: {8: np.zeros((2, 3))},
        0.0: {8: np.zeros((2, 3))},
        1.0: {8: np.zeros((3, 3))},
    }

    try:
        summarize_counterfactual_states(
            states,
            {"lexical": np.array([1.0, 0.0, 0.0])},
            target="lexical",
            intervention_scale=1.0,
        )
    except ValueError as error:
        assert "shapes do not match" in str(error)
    else:
        raise AssertionError("mismatched token shapes should fail")


def test_random_controls_are_deterministic_and_orthogonal():
    forbidden = {
        "source": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        "concept": np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
    }

    first = orthogonal_random_directions(
        forbidden,
        count=2,
        seed=17,
    )
    second = orthogonal_random_directions(
        forbidden,
        count=2,
        seed=17,
    )

    assert np.allclose(first, second)
    assert np.allclose(first @ first.T, np.eye(2), atol=1e-10)
    assert np.allclose(first[:, :2], 0.0, atol=1e-10)


def test_symmetric_capture_batches_negative_and_positive_conditions():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Identity(), nn.Identity()])

    class TinyExtractor:
        def __init__(self):
            self.model = TinyModel()
            self.device = torch.device("cpu")

        def _prepare_continuation_batch(
            self,
            prefixes,
            continuations,
            _max_length,
        ):
            assert prefixes == ["Prefix", "Prefix"]
            assert continuations == ["Continuation", "Continuation"]
            return (
                {"input_ids": torch.zeros((2, 4), dtype=torch.long)},
                [
                    {"continuation_start": 1, "continuation_end": 3},
                    {"continuation_start": 1, "continuation_end": 3},
                ],
            )

        def forward_model(self, input_ids, **_kwargs):
            hidden = torch.zeros((*input_ids.shape, 2))
            for layer in self.model.model.layers:
                hidden = layer(hidden)
            return hidden

    steering = MeanDiffSteering(hidden_size=2, layer=1)
    with torch.no_grad():
        steering.v_src.copy_(torch.tensor([1.0, 0.0]))
        steering.projection_std.fill_(2.0)
        steering.is_fitted.fill_(True)

    negative, positive = capture_symmetric_conditions(
        TinyExtractor(),
        "Prefix",
        "Continuation",
        [1, 2],
        steering=steering,
    )

    for layer in (1, 2):
        assert np.allclose(negative[layer][:, 0], -2.0)
        assert np.allclose(positive[layer][:, 0], 2.0)
        assert np.allclose(negative[layer][:, 1], 0.0)


def test_symmetric_direction_batch_keeps_direction_rows_separate():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Identity()])

    class TinyExtractor:
        def __init__(self):
            self.model = TinyModel()
            self.device = torch.device("cpu")

        def _prepare_continuation_batch(
            self,
            prefixes,
            continuations,
            _max_length,
        ):
            count = len(prefixes)
            return (
                {"input_ids": torch.zeros((count, 3), dtype=torch.long)},
                [
                    {"continuation_start": 1, "continuation_end": 3}
                    for _ in range(count)
                ],
            )

        def forward_model(self, input_ids, **_kwargs):
            hidden = torch.zeros((*input_ids.shape, 2))
            return self.model.model.layers[0](hidden)

    steerings = []
    for name, vector in (
        ("x", torch.tensor([1.0, 0.0])),
        ("y", torch.tensor([0.0, 1.0])),
    ):
        steering = MeanDiffSteering(hidden_size=2, layer=1)
        with torch.no_grad():
            steering.v_src.copy_(vector)
            steering.projection_std.fill_(3.0)
            steering.is_fitted.fill_(True)
        steerings.append((name, steering))

    result = capture_symmetric_direction_batch(
        TinyExtractor(),
        "Prefix",
        "Continuation",
        [1],
        steerings=steerings,
    )

    x_negative, x_positive = result["x"]
    y_negative, y_positive = result["y"]
    assert np.allclose(x_negative[1], [[-3.0, 0.0], [-3.0, 0.0]])
    assert np.allclose(x_positive[1], [[3.0, 0.0], [3.0, 0.0]])
    assert np.allclose(y_negative[1], [[0.0, -3.0], [0.0, -3.0]])
    assert np.allclose(y_positive[1], [[0.0, 3.0], [0.0, 3.0]])
