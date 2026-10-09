from pathlib import Path

import pytest

from scripts.audit_generation_semantics import local_encoder_path


def test_local_semantic_encoder_directory_is_resolved(tmp_path, monkeypatch):
    encoder = tmp_path / "encoder"
    encoder.mkdir()
    monkeypatch.chdir(tmp_path)
    assert local_encoder_path(Path("encoder")) == encoder


def test_hub_identifier_is_not_silently_treated_as_local(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="hf download"):
        local_encoder_path(Path("sentence-transformers/all-MiniLM-L6-v2"))


def test_encoder_file_is_not_a_model_directory(tmp_path):
    checkpoint = tmp_path / "model.bin"
    checkpoint.touch()
    with pytest.raises(ValueError, match="existing local directory"):
        local_encoder_path(checkpoint)
