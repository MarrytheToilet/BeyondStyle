from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_registry(language: str) -> dict:
    path = PROJECT_ROOT / "data" / "corpora" / f"{language}.yaml"
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_registries_have_disjoint_language_roots():
    en = load_registry("en")
    zh = load_registry("zh")

    assert en["language"] == "en"
    assert zh["language"] == "zh"
    assert en["download_root"] == "data/raw/en/hf"
    assert zh["download_root"] == "data/raw/zh/hf"
    assert en["download_root"] != zh["download_root"]


def test_default_sources_are_pinned_and_audited():
    for language in ("en", "zh"):
        registry = load_registry(language)
        defaults = [
            spec
            for spec in registry["sources"].values()
            if spec.get("default_download")
        ]
        assert defaults
        for spec in defaults:
            assert spec["enabled"] is True
            assert spec["tier"] == "core"
            assert len(spec["source_revision"]) == 40
            assert len(spec["snapshot_commit"]) == 40
            assert spec["snapshot_patterns"]
            assert spec["prefix_file_patterns"]
            assert 200 <= spec["sample_rows"] <= 5000
            assert spec["sample_seed"] == 20260723
            assert spec["sampling_strategy"] == "shard_balanced_contiguous_v1"
            assert spec["expected_files"] > 0
            assert spec["expected_bytes"] > 0
            assert spec["license"]
            assert spec["license_evidence"].startswith("https://")


def test_excluded_chinese_sources_cannot_download_by_default():
    registry = load_registry("zh")
    for name in ("thucnews_mirror", "classical_chinese"):
        spec = registry["sources"][name]
        assert spec["enabled"] is False
        assert spec["default_download"] is False
        assert spec["exclusion_reason"]

def test_no_government_report_source_is_registered():
    for language in ("en", "zh"):
        registry = load_registry(language)
        dataset_ids = [
            spec["dataset_id"].lower()
            for spec in registry["sources"].values()
        ]
        assert all("gov_report" not in dataset_id for dataset_id in dataset_ids)
