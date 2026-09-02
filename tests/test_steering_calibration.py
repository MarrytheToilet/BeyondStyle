import torch

from src.steering.base import SteeringConfig
from src.steering.mean_diff import MeanDiffSteering


def test_projection_std_strength_has_cross_model_units():
    human = torch.tensor([[0.0], [2.0]])
    ai = torch.tensor([[4.0], [6.0]])
    steering = MeanDiffSteering(hidden_size=1, layer=1)
    metrics = steering.fit(human, ai)

    hidden = torch.zeros(1, 1, 1)
    delta = steering.compute_delta(
        hidden,
        SteeringConfig(
            layer=1,
            strength=2.0,
            direction="ai",
            strength_unit="projection_std",
        ),
    )

    assert torch.isclose(
        steering.projection_std,
        torch.tensor(2.0).sqrt(),
    )
    assert metrics["projection_std"] == steering.projection_std.item()
    assert torch.allclose(
        delta,
        torch.full_like(hidden, 2.0 * (2.0**0.5)),
    )


def test_absolute_strength_remains_available_as_diagnostic_units():
    steering = MeanDiffSteering(hidden_size=2, layer=1)
    steering.fit(
        torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        torch.tensor([[2.0, 0.0], [3.0, 0.0]]),
    )
    hidden = torch.zeros(1, 1, 2)

    delta = steering.compute_delta(
        hidden,
        SteeringConfig(
            layer=1,
            strength=0.5,
            direction="human",
            strength_unit="absolute",
        ),
    )

    assert torch.allclose(delta, torch.tensor([[[-0.5, 0.0]]]))


def test_checkpoint_without_embedded_calibration_loads_neutrally(tmp_path):
    steering = MeanDiffSteering(hidden_size=2, layer=1)
    steering.fit(
        torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        torch.tensor([[2.0, 0.0], [3.0, 0.0]]),
    )
    old_state = steering.state_dict()
    old_state.pop("projection_std")
    checkpoint = tmp_path / "old.pt"
    torch.save({
        "state_dict": old_state,
        "class_name": "MeanDiffSteering",
        "hidden_size": 2,
        "layer": 1,
        "metadata": {},
    }, checkpoint)

    loaded = MeanDiffSteering.load(checkpoint)

    assert loaded.projection_std.item() == 1.0
    assert torch.equal(loaded.v_src, steering.v_src)
