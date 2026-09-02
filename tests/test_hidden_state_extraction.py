import torch

from src.extraction.hidden_states import (
    HiddenStatesExtractor,
    continuation_separator,
)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens):
        ids = [ord(character) + 1 for character in text]
        return ([1] + ids) if add_special_tokens else ids


def fake_extractor():
    extractor = HiddenStatesExtractor.__new__(HiddenStatesExtractor)
    extractor.tokenizer = CharacterTokenizer()
    extractor.device = "cpu"
    extractor.num_layers = 1
    extractor.capture_calls = 0

    def capture(inputs, layers):
        extractor.capture_calls += 1
        batch_size = inputs["input_ids"].shape[0]
        sequence_length = inputs["input_ids"].shape[1]
        states = torch.arange(
            batch_size * sequence_length * 2,
            dtype=torch.float32,
        ).reshape(batch_size, sequence_length, 2)
        return {layer: states for layer in layers}

    extractor._capture_layer_outputs = capture
    return extractor


def test_natural_language_separator_is_model_independent():
    assert continuation_separator("A complete sentence.", "Next one.") == " "
    assert continuation_separator("前文结束。", "下一句开始。") == ""
    assert continuation_separator("Already spaced. ", "Next one.") == ""


def test_last_state_is_final_continuation_token_only():
    extractor = fake_extractor()
    result = extractor.extract_continuation(
        "Prefix.",
        "Continuation.",
        pooling="last",
        layers=[1],
    )
    sequence_length = result.token_count
    expected = torch.tensor(
        [(sequence_length - 1) * 2, (sequence_length - 1) * 2 + 1],
        dtype=torch.float32,
    )

    assert result.prefix_token_count == len(
        extractor.tokenizer.encode("Prefix.", add_special_tokens=True)
    )
    assert result.continuation_token_count == len(" Continuation.")
    assert torch.equal(result.hidden_states[1], expected)


def test_mean_state_excludes_every_prefix_token():
    extractor = fake_extractor()
    result = extractor.extract_continuation(
        "Prefix.",
        "Continuation.",
        pooling="full_mean",
        layers=[1],
    )
    all_states = torch.arange(
        result.token_count * 2,
        dtype=torch.float32,
    ).reshape(result.token_count, 2)
    expected = all_states[result.prefix_token_count :].mean(dim=0)

    assert torch.equal(result.hidden_states[1], expected)


def test_two_poolings_share_one_model_forward():
    extractor = fake_extractor()
    results = extractor.extract_continuation_poolings(
        "Prefix.",
        "Continuation.",
        poolings=["last", "full_mean"],
        layers=[1],
    )

    assert extractor.capture_calls == 1
    assert set(results) == {"last", "full_mean"}
    assert not torch.equal(
        results["last"].hidden_states[1],
        results["full_mean"].hidden_states[1],
    )


def test_batched_extraction_preserves_each_continuation_boundary():
    extractor = fake_extractor()
    results = extractor.extract_continuation_poolings_batch(
        prefixes=["A.", "Longer prefix."],
        continuations=["B.", "Second continuation."],
        poolings=["last", "full_mean"],
        layers=[1],
    )

    assert extractor.capture_calls == 1
    assert len(results) == 2
    assert results[0]["last"].continuation_token_count == len(" B.")
    assert results[1]["last"].continuation_token_count == len(
        " Second continuation."
    )
    assert not torch.equal(
        results[0]["last"].hidden_states[1],
        results[1]["last"].hidden_states[1],
    )


def test_token_extraction_returns_only_continuation_states_and_ids():
    extractor = fake_extractor()
    result = extractor.extract_continuation_tokens(
        "Prefix.",
        "Continuation.",
        layers=[1],
    )

    expected_ids = torch.tensor(
        extractor.tokenizer.encode(
            " Continuation.",
            add_special_tokens=False,
        )
    )
    assert extractor.capture_calls == 1
    assert torch.equal(result.token_ids, expected_ids)
    assert result.hidden_states[1].shape == (len(expected_ids), 2)
    assert result.continuation_token_count == len(expected_ids)
    assert result.separator == " "


def test_batched_token_extraction_keeps_unpadded_lengths():
    extractor = fake_extractor()
    results = extractor.extract_continuation_tokens_batch(
        prefixes=["A.", "Long prefix."],
        continuations=["B.", "Longer continuation."],
        layers=[1],
    )

    assert extractor.capture_calls == 1
    assert results[0].hidden_states[1].shape[0] == len(" B.")
    assert results[1].hidden_states[1].shape[0] == len(
        " Longer continuation."
    )
    assert results[0].token_count < results[1].token_count


def test_only_two_continuation_poolings_are_supported():
    extractor = fake_extractor()
    try:
        extractor.extract_continuation(
            "Prefix.",
            "Continuation.",
            pooling="middle",
            layers=[1],
        )
    except ValueError as error:
        assert "choose 'last' and/or 'full_mean'" in str(error)
    else:
        raise AssertionError("unsupported pooling should fail")
