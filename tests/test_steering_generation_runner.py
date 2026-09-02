import json

from scripts.run_steering_generation import (
    load_prompts,
    order_prompts_by_length,
    resolve_component_assets,
    select_robust_layer,
)
from scripts.analyze_component_steering import (
    contrast_pairs_for_components,
)
from src.generation.steered import (
    SteeredGenerationConfig,
    SteeredGenerator,
)


def test_prompt_loader_assigns_stable_unique_ids(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps(["First natural prefix.", {"id": "given", "text": "Second."}]),
        encoding="utf-8",
    )

    first = load_prompts(path, "en")
    second = load_prompts(path, "en")

    assert first == second
    assert first[0]["prompt_id"].startswith("en_0000_")
    assert first[1]["prompt_id"] == "given"


def test_robust_layer_maximizes_worst_generator_then_mean(tmp_path):
    path = tmp_path / "generator_holdout.json"
    path.write_text(
        json.dumps({
            "results": {
                "3": {
                    "g1": {"nullspace": {"auc": 0.80}},
                    "g2": {"nullspace": {"auc": 0.90}},
                },
                "8": {
                    "g1": {"nullspace": {"auc": 0.82}},
                    "g2": {"nullspace": {"auc": 0.83}},
                },
                "12": {
                    "g1": {"nullspace": {"auc": 0.82}},
                    "g2": {"nullspace": {"auc": 0.86}},
                },
            }
        }),
        encoding="utf-8",
    )

    selected, table = select_robust_layer(path, "nullspace")

    assert selected == 12
    assert table[0]["worst_generator_auc"] == 0.82
    assert table[0]["mean_generator_auc"] == 0.84


def test_prompt_length_order_is_descending_and_stable():
    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            del kwargs
            return {
                "input_ids": [
                    list(range(len(text.split())))
                    for text in texts
                ]
            }

    prompts = [
        {"prompt_index": 0, "text": "one two"},
        {"prompt_index": 1, "text": "one two three four"},
        {"prompt_index": 2, "text": "one two three four"},
        {"prompt_index": 3, "text": "one"},
    ]

    ordered = order_prompts_by_length(
        prompts,
        FakeTokenizer(),
        max_prompt_tokens=32,
    )

    assert [row["prompt_index"] for row in ordered] == [1, 2, 0, 3]


def test_component_asset_resolution(tmp_path):
    root = tmp_path / "model" / "last_token"
    layer = root / "layer11"
    (layer / "directions").mkdir(parents=True)
    (layer / "decomposition.json").write_text(
        json.dumps({"explained_ratio": 0.2}),
        encoding="utf-8",
    )
    (layer / "directions" / "explained.pt").write_bytes(b"checkpoint")

    selected, checkpoint, selection, table = resolve_component_assets(
        root,
        "explained",
    )

    assert selected == 11
    assert checkpoint.name == "explained.pt"
    assert selection.name == "decomposition.json"
    assert table == [{"explained_ratio": 0.2}]


def test_symmetric_comparison_does_not_require_zero_baseline():
    generator = object.__new__(SteeredGenerator)
    generator.last_generation_stats = {}

    def fake_generate(prompts, steering, config, seed):
        del steering, seed
        generator.last_generation_stats = {
            "direction": config.steering_direction,
            "strength": config.steering_strength,
        }
        return [config.steering_direction for _ in prompts]

    generator._generate_batch = fake_generate
    result = generator.generate_batch_comparison(
        ["prompt"],
        steering=object(),
        strengths=[-1.0, 1.0],
        config=SteeredGenerationConfig(seed=7, paired_sampling=True),
    )

    assert set(result) == {-1.0, 1.0}
    assert result[-1.0] == ["human"]
    assert result[1.0] == ["ai"]


def test_concept_contrasts_are_filtered_to_available_components():
    pairs = contrast_pairs_for_components(
        ["source", "lexical", "syntactic", "residual", "random"],
    )

    assert ("lexical", "source") in pairs
    assert ("syntactic", "residual") in pairs
    assert ("explained", "source") not in pairs
