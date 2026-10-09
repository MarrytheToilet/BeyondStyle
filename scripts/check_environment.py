#!/usr/bin/env python3
"""Validate a BeyondStyle runtime without loading model weights.

The report is safe to attach to experiment logs: environment variables are
reported only as configured/unconfigured and secret values are never printed.
"""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


REQUIRED_PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "datasets",
    "safetensors",
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
)
OPTIONAL_PACKAGES = (
    "lm-eval",
    "spacy",
    "sentence-transformers",
    "wordfreq",
    "matplotlib",
    "seaborn",
    "openai",
    "flash-attn",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check CUDA, Python dependencies, disk, and local models.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Local model directory or Hub ID; repeat for multiple models.",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow AutoConfig to contact the Hub for non-local model IDs.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail instead of warn when CUDA is unavailable.",
    )
    parser.add_argument(
        "--min-free-disk-gb",
        type=float,
        default=20.0,
        help="Required free space on the filesystem containing the repository.",
    )
    parser.add_argument(
        "--pip-check",
        action="store_true",
        help="Run `python -m pip check` and include its output.",
    )
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def package_versions() -> tuple[dict[str, str | None], list[str]]:
    versions: dict[str, str | None] = {}
    errors: list[str] = []
    for package in REQUIRED_PACKAGES + OPTIONAL_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
            if package in REQUIRED_PACKAGES:
                errors.append(f"missing required package: {package}")
    return versions, errors


def text_config(config: Any) -> Any:
    return getattr(config, "text_config", config)


def first_attr(obj: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def inspect_model(model_name: str, allow_network: bool) -> tuple[dict, str | None]:
    from transformers import AutoConfig

    expanded = Path(model_name).expanduser()
    is_local = expanded.is_dir()
    resolved_name = str(expanded.resolve()) if is_local else model_name
    record: dict[str, Any] = {
        "requested": model_name,
        "resolved": resolved_name,
        "local_directory": is_local,
    }
    if is_local:
        size_bytes = sum(
            path.stat().st_size
            for path in expanded.rglob("*")
            if path.is_file()
        )
        record["size_gb"] = round(size_bytes / 1024**3, 3)
        record["has_config_json"] = (expanded / "config.json").is_file()
        record["has_tokenizer_config"] = (
            expanded / "tokenizer_config.json"
        ).is_file()
    elif not allow_network:
        return record, (
            f"{model_name}: not a local directory; use --allow-network only "
            "when a Hub lookup is intended"
        )

    try:
        config = AutoConfig.from_pretrained(
            resolved_name,
            local_files_only=is_local or not allow_network,
            trust_remote_code=True,
        )
    except Exception as error:
        return record, f"{model_name}: {type(error).__name__}: {error}"

    language_config = text_config(config)
    record.update({
        "model_type": getattr(config, "model_type", None),
        "architecture": (getattr(config, "architectures", None) or [None])[0],
        "hidden_size": first_attr(
            language_config,
            ("hidden_size", "d_model", "n_embd"),
        ),
        "decoder_layers": first_attr(
            language_config,
            ("num_hidden_layers", "n_layer", "num_layers"),
        ),
        "configured_dtype": str(
            first_attr(config, ("dtype", "torch_dtype"))
        ),
    })
    return record, None


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    versions, errors = package_versions()
    warnings: list[str] = []

    try:
        import torch
        from src.runtime import (
            attention_backend_diagnostics,
            resolve_attention_backend,
        )
    except Exception as error:
        errors.append(f"cannot import torch/runtime: {type(error).__name__}: {error}")
        torch = None

    python_ok = sys.version_info >= (3, 10)
    if not python_ok:
        errors.append("Python 3.10 or newer is required")

    cuda: dict[str, Any] = {"available": False, "devices": []}
    attention: dict[str, Any] = {}
    if torch is not None:
        cuda["available"] = torch.cuda.is_available()
        cuda["torch_cuda_version"] = torch.version.cuda
        cuda["cudnn_version"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            cuda["bf16_supported"] = torch.cuda.is_bf16_supported()
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                cuda["devices"].append({
                    "index": index,
                    "name": properties.name,
                    "compute_capability": (
                        f"{properties.major}.{properties.minor}"
                    ),
                    "total_memory_gb": round(
                        properties.total_memory / 1024**3,
                        2,
                    ),
                })
        elif args.require_cuda:
            errors.append("CUDA is required but torch.cuda.is_available() is false")
        else:
            warnings.append("CUDA is unavailable; only CPU smoke tests can run")
        attention = attention_backend_diagnostics()
        attention["auto_backend"] = resolve_attention_backend("auto")
        if not attention["flash_attention_2_available"]:
            warnings.append(
                "FlashAttention 2 is unavailable as an explicit benchmark "
                "backend; the automatic runtime policy uses PyTorch SDPA"
            )

    disk = shutil.disk_usage(ROOT)
    free_disk_gb = disk.free / 1024**3
    if free_disk_gb < args.min_free_disk_gb:
        errors.append(
            f"free disk {free_disk_gb:.1f} GiB is below "
            f"{args.min_free_disk_gb:.1f} GiB"
        )

    model_records = []
    for model_name in args.model:
        record, error = inspect_model(model_name, args.allow_network)
        model_records.append(record)
        if error:
            errors.append(error)

    pip_check = None
    if args.pip_check:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True,
            text=True,
            check=False,
        )
        pip_check = {
            "returncode": completed.returncode,
            "output": (completed.stdout + completed.stderr).strip(),
        }
        if completed.returncode:
            errors.append("pip dependency check failed")

    report = {
        "ok": not errors,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": versions,
        "cuda": cuda,
        "attention": attention,
        "disk": {
            "path": str(ROOT),
            "free_gb": round(free_disk_gb, 2),
            "required_free_gb": args.min_free_disk_gb,
        },
        "environment_variables_configured": {
            name: bool(os.environ.get(name))
            for name in (
                "API_BASE_URL",
                "API_KEY",
                "API_MODELS",
                "HF_HOME",
                "HF_TOKEN",
                "OUTPUT_DIR",
            )
        },
        "models": model_records,
        "pip_check": pip_check,
        "warnings": warnings,
        "errors": errors,
    }

    print(f"BeyondStyle environment: {'OK' if report['ok'] else 'FAILED'}")
    print(
        f"Python {report['python']['version']} · "
        f"torch {versions.get('torch')} · "
        f"transformers {versions.get('transformers')}"
    )
    if cuda.get("devices"):
        for device in cuda["devices"]:
            print(
                f"CUDA {device['index']}: {device['name']} · "
                f"{device['total_memory_gb']:.2f} GiB · "
                f"compute {device['compute_capability']}"
            )
    flash_status = (
        "available"
        if attention.get("flash_attention_2_available")
        else "unavailable"
    )
    print(
        f"Attention auto: {attention.get('auto_backend', 'unresolved')} · "
        f"FlashAttention 2 explicit: {flash_status}"
    )
    print(f"Free disk: {report['disk']['free_gb']:.2f} GiB")
    for model in model_records:
        print(
            f"Model: {model['requested']} · "
            f"{model.get('model_type', 'unresolved')} · "
            f"L={model.get('decoder_layers', '?')} · "
            f"d={model.get('hidden_size', '?')}"
        )
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")

    if args.json_out:
        atomic_json(args.json_out, report)
        print(f"JSON report: {args.json_out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
