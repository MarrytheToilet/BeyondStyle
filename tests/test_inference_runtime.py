import torch

from src.runtime.inference import (
    model_source_fingerprint,
    resolve_attention_backend,
    resolve_torch_dtype,
    stable_seed,
)


def test_explicit_dtype_resolution():
    assert resolve_torch_dtype("bf16", "cpu") is torch.bfloat16
    assert resolve_torch_dtype("fp16", "cpu") is torch.float16
    assert resolve_torch_dtype("fp32", "cuda") is torch.float32


def test_explicit_attention_backend_is_preserved():
    assert resolve_attention_backend("sdpa") == "sdpa"
    assert resolve_attention_backend("eager") == "eager"


def test_auto_attention_uses_profiled_portable_sdpa():
    assert resolve_attention_backend("auto") == "sdpa"


def test_stable_seed_is_reproducible_and_field_sensitive():
    first = stable_seed(8236, "model", "layer12", 0)
    assert first == stable_seed(8236, "model", "layer12", 0)
    assert first != stable_seed(8236, "model", "layer12", 1)


def test_local_model_fingerprint_is_portable(tmp_path):
    first = tmp_path / "first" / "model"
    second = tmp_path / "second" / "model"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    for directory in (first, second):
        (directory / "config.json").write_text('{"model_type":"test"}')
        (directory / "tokenizer_config.json").write_text('{"legacy":false}')
    assert model_source_fingerprint(str(first)) == model_source_fingerprint(
        str(second)
    )
