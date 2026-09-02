import torch

from src.steering.base import SteeringConfig
from src.steering.hooks import ActivationHook


class OnesSteering:
    def compute_delta(self, hidden_states, config):
        return torch.ones_like(hidden_states) * (
            config.get_sign() * config.strength
        )


def test_decode_only_leaves_prefill_object_exactly_unchanged():
    hook = ActivationHook(
        OnesSteering(),
        SteeringConfig(layer=1, strength=2.0, direction="ai"),
        "decode_only",
    )
    hidden = torch.zeros(2, 5, 3)
    output = (hidden, "cache")

    returned = hook(None, (), output)

    assert returned is output
    assert torch.equal(returned[0], hidden)
    assert hook.stats == {
        "forward_calls": 1,
        "prefill_calls_skipped": 1,
        "steered_calls": 0,
        "steered_token_positions": 0,
    }


def test_decode_only_modifies_only_newest_position_after_prefill():
    hook = ActivationHook(
        OnesSteering(),
        SteeringConfig(layer=1, strength=2.0, direction="ai"),
        "decode_only",
    )
    hook(None, (), torch.zeros(2, 4, 3))
    decoded = hook(None, (), torch.zeros(2, 3, 3))

    assert torch.equal(decoded[:, :-1], torch.zeros(2, 2, 3))
    assert torch.equal(decoded[:, -1], torch.full((2, 3), 2.0))
    assert hook.stats == {
        "forward_calls": 2,
        "prefill_calls_skipped": 1,
        "steered_calls": 1,
        "steered_token_positions": 1,
    }


def test_all_mode_still_supports_non_generation_forward_passes():
    hook = ActivationHook(
        OnesSteering(),
        SteeringConfig(layer=1, strength=1.0, direction="human"),
        "all",
    )
    hidden = torch.zeros(1, 4, 2)

    returned = hook(None, (), hidden)

    assert torch.equal(returned, torch.full_like(hidden, -1.0))
    assert hook.stats["prefill_calls_skipped"] == 0
    assert hook.stats["steered_token_positions"] == 4


def test_hook_reset_starts_a_new_prefill_phase():
    hook = ActivationHook(
        OnesSteering(),
        SteeringConfig(layer=1),
        "decode_only",
    )
    hook(None, (), torch.zeros(1, 2, 2))
    hook(None, (), torch.zeros(1, 1, 2))
    hook.reset()

    hidden = torch.zeros(1, 3, 2)
    assert hook(None, (), hidden) is hidden
    assert hook.stats["prefill_calls_skipped"] == 1
    assert hook.stats["steered_calls"] == 0


def test_unknown_position_mode_is_rejected():
    try:
        ActivationHook(
            OnesSteering(),
            SteeringConfig(layer=1),
            "prompt_and_decode",
        )
    except ValueError as error:
        assert "Unknown steering position mode" in str(error)
    else:
        raise AssertionError("invalid mode should fail")
