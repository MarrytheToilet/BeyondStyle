"""Portable inference configuration for local and server-side experiments."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
from pathlib import Path
from typing import Any, Iterator

import torch


DTYPE_NAMES = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}
ATTENTION_BACKENDS = {"auto", "sdpa", "flash_attention_2", "eager"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_source_fingerprint(model_name: str) -> dict[str, Any]:
    """Return a path-portable identity for a local directory or Hub ID."""
    path = Path(model_name).expanduser()
    if path.is_dir():
        return {
            "model_id": path.name,
            "source_type": "local_directory",
            "config_sha256": (
                _sha256_file(path / "config.json")
                if (path / "config.json").is_file()
                else None
            ),
            "tokenizer_config_sha256": (
                _sha256_file(path / "tokenizer_config.json")
                if (path / "tokenizer_config.json").is_file()
                else None
            ),
        }
    return {
        "model_id": model_name.rstrip("/").split("/")[-1],
        "source_type": "hub_id",
        "hub_id": model_name,
        "config_sha256": None,
        "tokenizer_config_sha256": None,
    }


def configure_torch_runtime(*, allow_tf32: bool = True) -> None:
    """Enable safe throughput optimizations that do not quantize the model."""
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32


def resolve_torch_dtype(
    requested: str | torch.dtype,
    device: str = "cuda",
) -> torch.dtype:
    """Resolve an explicit or hardware-aware inference dtype."""
    if isinstance(requested, torch.dtype):
        return requested
    name = requested.lower()
    if name != "auto":
        if name not in DTYPE_NAMES:
            raise ValueError(
                f"Unknown dtype {requested!r}; choose auto, bf16, fp16, or fp32"
            )
        return DTYPE_NAMES[name]
    if device.startswith("cuda") and torch.cuda.is_available():
        return (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
    return torch.float32


def attention_backend_diagnostics() -> dict[str, Any]:
    """Report usable attention implementations without raising.

    Importing ``flash_attn`` is the only reliable check for binary wheels:
    package metadata alone cannot reveal a CUDA/PyTorch ABI mismatch.
    """
    diagnostics: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "sdpa_available": hasattr(
            torch.nn.functional,
            "scaled_dot_product_attention",
        ),
        "flash_attention_2_available": False,
        "flash_attention_2_error": None,
    }
    if not torch.cuda.is_available():
        diagnostics["flash_attention_2_error"] = "CUDA is unavailable"
        return diagnostics
    try:
        importlib.import_module("flash_attn")
    except (ImportError, OSError) as error:
        diagnostics["flash_attention_2_error"] = (
            f"{type(error).__name__}: {error}"
        )
        return diagnostics
    diagnostics["flash_attention_2_available"] = True
    return diagnostics


def resolve_attention_backend(requested: str = "auto") -> str:
    """Resolve the attention implementation.

    ``auto`` deliberately selects PyTorch SDPA. On the validated RTX 3090
    stack it outperforms external FlashAttention 2 for Qwen3/Qwen3.5, and it
    also supports Gemma 4 attention heads wider than FlashAttention 2's
    256-dimension limit. FlashAttention remains available as an explicit
    model-specific benchmark option.
    """
    if requested not in ATTENTION_BACKENDS:
        raise ValueError(
            f"Unknown attention backend {requested!r}; "
            f"choose one of {sorted(ATTENTION_BACKENDS)}"
        )
    if requested == "auto":
        return "sdpa"
    return requested


def _device_map(device: str):
    if device == "auto":
        return "auto"
    if device == "cuda":
        return {"": 0}
    if device.startswith("cuda:"):
        return {"": int(device.split(":", 1)[1])}
    if device == "cpu":
        return {"": "cpu"}
    return device


def model_load_kwargs(
    *,
    device: str,
    dtype: str | torch.dtype = "auto",
    attention_backend: str = "auto",
    trust_remote_code: bool = True,
) -> tuple[dict, dict]:
    """Build Transformers load kwargs plus a JSON-serializable resolution log."""
    resolved_dtype = resolve_torch_dtype(dtype, device)
    resolved_attention = resolve_attention_backend(attention_backend)
    kwargs = {
        "dtype": resolved_dtype,
        "device_map": _device_map(device),
        "attn_implementation": resolved_attention,
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote_code,
    }
    info = {
        "requested_device": device,
        "device_map": str(kwargs["device_map"]),
        "requested_dtype": str(dtype),
        "resolved_dtype": str(resolved_dtype).replace("torch.", ""),
        "requested_attention_backend": attention_backend,
        "resolved_attention_backend": resolved_attention,
        "attention_resolution_policy": (
            "profiled_portable_sdpa"
            if attention_backend == "auto"
            else "explicit"
        ),
        "allow_tf32": bool(
            torch.cuda.is_available()
            and torch.backends.cuda.matmul.allow_tf32
        ),
    }
    return kwargs, info


def input_device(model) -> torch.device:
    """Return the device that owns the token embedding for sharded models."""
    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, RuntimeError):
        return next(model.parameters()).device


@contextmanager
def seeded_rng(seed: int | None, device: torch.device) -> Iterator[None]:
    """Isolate and reproducibly seed CPU/CUDA sampling state."""
    if seed is None:
        yield
        return
    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(int(seed))
        if cuda_devices:
            torch.cuda.manual_seed_all(int(seed))
        yield


def stable_seed(base_seed: int, *parts: object) -> int:
    """Derive an order-independent 31-bit seed from stable experiment fields."""
    payload = "\x1f".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)
