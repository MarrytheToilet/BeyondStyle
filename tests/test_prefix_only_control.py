from scripts.synthesize_prefix_only_control import (
    PROMPT_FAMILY,
    make_control_pair,
    prefix_only_quality,
    prompt_messages,
    run_record,
)


def source_record():
    return {
        "schema": "proposition_aligned_pair",
        "pair_id": "planned-1",
        "sample_id": "sample-1",
        "language": "en",
        "domain": "journalistic",
        "source": "news",
        "source_document": {"row_id": "1"},
        "prefix": "The committee met again on Tuesday.",
        "human_continuation": (
            "The hidden reference wording must never appear in a prompt."
        ),
        "ai_continuation": "An existing planned continuation.",
        "lengths": {
            "length_unit": "word",
            "prefix_units": 7,
            "human_continuation_units": 40,
            "ai_continuation_units": 5,
            "ai_to_human_ratio": 0.125,
        },
        "content_control": {
            "method": "proposition_plan",
            "plan": {
                "propositions": ["Secret plan statement."],
                "required_entities": ["Secret entity"],
                "discourse_endpoint": "Secret endpoint",
            },
        },
        "generation": {
            "generator_model": "model-a",
            "temperature": 0.7,
        },
        "provenance": {"prefix_window_sha256": "abc"},
    }


def test_prefix_only_prompt_exposes_no_reference_or_plan():
    record = source_record()
    messages = prompt_messages(record)
    payload = "\n".join(message["content"] for message in messages)

    assert record["prefix"] in payload
    assert record["human_continuation"] not in payload
    assert "Secret plan statement" not in payload
    assert "Secret entity" not in payload
    assert "Secret endpoint" not in payload


def test_prefix_only_pair_preserves_matched_assignment():
    record = source_record()
    quality = {
        "units": 42,
        "length_ratio": 1.05,
        "accepted": True,
    }

    pair = make_control_pair(
        record,
        "A fresh continuation.",
        quality,
        generation_attempt=2,
        temperature=0.6,
    )

    assert pair["sample_id"] == record["sample_id"]
    assert pair["prefix"] == record["prefix"]
    assert pair["human_continuation"] == record["human_continuation"]
    assert (
        pair["generation"]["generator_model"]
        == record["generation"]["generator_model"]
    )
    assert pair["generation"]["prompt_family"] == PROMPT_FAMILY
    assert pair["generation"]["temperature"] == 0.6
    assert pair["content_control"]["generator_received_proposition_plan"] is False


def test_prefix_only_generation_is_resumable(tmp_path):
    class FakeApi:
        def __init__(self):
            self.calls = []

        def chat(self, model, messages, **kwargs):
            self.calls.append((model, messages, kwargs))
            return (
                "Officials discussed several practical options during the "
                "meeting and agreed to review the available evidence before "
                "making a final decision. Further details will be released "
                "after the next scheduled session."
            )

    api = FakeApi()
    work_path = tmp_path / "sample-1.json"
    first = run_record(
        api,
        source_record(),
        retries=2,
        temperature=0.7,
        fingerprint="fingerprint",
        work_path=work_path,
    )
    second = run_record(
        api,
        source_record(),
        retries=2,
        temperature=0.7,
        fingerprint="fingerprint",
        work_path=work_path,
    )

    assert first["status"] == "accepted"
    assert second["status"] == "accepted"
    assert len(api.calls) == 1


def test_narrative_cannot_is_not_misclassified_as_api_refusal():
    record = source_record()
    narrative = (
        "I cannot bear the thought of her leaving without another word, "
        "yet I know the choice belongs to her alone. The room remained "
        "quiet while everyone waited for the final letter to arrive."
    )

    quality = prefix_only_quality(narrative, record)

    assert "refusal_or_capability_disclaimer" not in quality["issues"]
    assert quality["accepted"] is True


def test_explicit_generation_refusal_is_still_rejected():
    record = source_record()
    refusal = (
        "As an AI, I cannot continue this document or generate the requested "
        "prose. Please provide a different task with more context."
    )

    quality = prefix_only_quality(refusal, record)

    assert "refusal_or_capability_disclaimer" in quality["issues"]
    assert quality["accepted"] is False
