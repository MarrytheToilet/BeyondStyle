import numpy as np
import torch

from scripts.freeze_causal_pullback_axes import (
    load_checkpoint,
    main,
    save_checkpoint,
)


def test_checkpoint_roundtrip_preserves_canonical_pole(tmp_path):
    path = tmp_path / "axis.pt"
    vector = np.array([-2.0, 1.0, 0.0], dtype=np.float32)

    save_checkpoint(
        path,
        vector,
        layer=8,
        scale=0.25,
        metadata={"orientation_uses_generation_outcomes": False},
    )
    loaded, layer, scale, metadata = load_checkpoint(path)

    assert layer == 8
    assert np.isclose(scale, 0.25)
    assert loaded[0] < 0
    assert np.allclose(loaded, vector / np.linalg.norm(vector))
    assert metadata["orientation_uses_generation_outcomes"] is False
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["metadata"]["orientation_uses_generation_outcomes"] is False


def test_freeze_uses_shared_pullback_scale_not_source_scale(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "source.pt"
    lexical = tmp_path / "lexical.pt"
    syntactic = tmp_path / "syntactic.pt"
    out_dir = tmp_path / "frozen"
    save_checkpoint(
        source,
        np.array([1, 0, 0, 0, 0], dtype=np.float32),
        layer=8,
        scale=1.0,
        metadata={"method": "source"},
    )
    save_checkpoint(
        lexical,
        np.array([0, 1, 0, 0, 0], dtype=np.float32),
        layer=8,
        scale=0.25,
        metadata={"method": "pullback", "downstream_layer": 12},
    )
    save_checkpoint(
        syntactic,
        np.array([0, 0, 1, 0, 0], dtype=np.float32),
        layer=8,
        scale=0.25,
        metadata={"method": "pullback", "downstream_layer": 12},
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "freeze_causal_pullback_axes.py",
            "--source-checkpoint",
            str(source),
            "--lexical-checkpoint",
            str(lexical),
            "--syntactic-checkpoint",
            str(syntactic),
            "--random-controls",
            "2",
            "--out-dir",
            str(out_dir),
        ],
    )

    main()

    for name in ("source", "lexical", "syntactic", "random", "random_01"):
        _, layer, scale, metadata = load_checkpoint(
            out_dir / "layer08" / "directions" / f"{name}.pt"
        )
        assert layer == 8
        assert np.isclose(scale, 0.25)
        assert np.isclose(metadata["source_checkpoint_original_scale"], 1.0)
