"""
Centralized Configuration for BeyondStyle.

Loads .env, derives model identifiers, and builds output paths.
All scripts should use this module instead of hardcoding paths.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ── .env loading ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def get_model_path() -> str:
    """Get local model path from env. Required for steering/extraction.
    
    Relative paths are resolved against PROJECT_ROOT so that scripts
    work regardless of the current working directory.
    """
    path = os.environ.get("BEYONDSTYLE_MODEL_PATH")
    if not path:
        raise ValueError(
            "BEYONDSTYLE_MODEL_PATH not set. "
            "Set it in .env or export it before running."
        )
    resolved = (PROJECT_ROOT / path).resolve()
    if resolved.is_dir():
        return str(resolved)
    return path


def get_model_id(model_path: Optional[str] = None) -> str:
    """
    Derive a short model identifier from a model path or name.

    Examples:
        ../../models/Llama-3.1-8B-Instruct  →  Llama-3.1-8B-Instruct
        /home/user/models/Qwen2.5-3B-Instruct  →  Qwen2.5-3B-Instruct
        Qwen/Qwen2.5-7B-Instruct  →  Qwen2.5-7B-Instruct

    The model_id is used as a subdirectory name in outputs.
    """
    if model_path is None:
        model_path = get_model_path()

    # Strip trailing slashes, get the last path component
    name = Path(model_path).name
    if not name or name == ".":
        name = Path(model_path).parent.name

    return name


def extract_dataset_tag(dataset_path: str) -> str:
    """Extract API model tag from dataset filename.

    Examples:
        paired_dataset_Qwen3.5-397B-A17B.json  →  Qwen3.5-397B-A17B
        paired_dataset_Qwen3.5-397B-A17B.json  →  Qwen3.5-397B-A17B
        paired_dataset.json                     →  ""
    """
    name = Path(dataset_path).stem
    prefix = "paired_dataset_"
    if name.startswith(prefix) and len(name) > len(prefix):
        return name[len(prefix):]
    return ""


def get_output_base() -> Path:
    """Get base output directory (default: outputs/)."""
    return Path(os.environ.get("BEYONDSTYLE_OUTPUT_DIR", "outputs"))


def get_steering_dir(model_id: Optional[str] = None,
                     dataset_tag: Optional[str] = None) -> Path:
    """
    Get model-specific steering output directory.

    Returns: outputs/steering/{model_id}/{dataset_tag}/ (if tag given)
             outputs/steering/{model_id}/                (if no tag)
    """
    if model_id is None:
        model_id = get_model_id()
    base = get_output_base() / "steering" / model_id
    if dataset_tag:
        base = base / dataset_tag
    return base


def get_figures_dir(model_id: Optional[str] = None,
                    dataset_tag: Optional[str] = None) -> Path:
    """
    Get model-specific figures output directory.

    Returns: outputs/figures/{model_id}/{dataset_tag}/ (if tag given)
             outputs/figures/{model_id}/                (if no tag)
    """
    if model_id is None:
        model_id = get_model_id()
    base = get_output_base() / "figures" / model_id
    if dataset_tag:
        base = base / dataset_tag
    return base


def get_dataset_dir() -> Path:
    """Get dataset output directory."""
    return get_output_base() / "dataset"


def get_dataset_path() -> str:
    """Get dataset path from env or auto-detect latest in outputs/dataset/.

    Resolution order:
        1. BEYONDSTYLE_DATASET env var (explicit path)
        2. Auto-detect latest paired_dataset_*.json in outputs/dataset/
        3. Fall back to outputs/dataset/paired_dataset.json
    """
    env_path = os.environ.get("BEYONDSTYLE_DATASET")
    if env_path:
        return env_path

    dataset_dir = get_dataset_dir()
    if dataset_dir.is_dir():
        candidates = sorted(
            dataset_dir.glob("paired_dataset_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return str(candidates[0])

    return str(dataset_dir / "paired_dataset.json")


def get_test_prompts_path() -> str:
    """Get test prompts path from env or default.

    Resolution order:
        1. BEYONDSTYLE_TEST_PROMPTS env var
        2. data/prompts/default.json
        3. data/test_prompts.json (legacy fallback)
    """
    env_path = os.environ.get("BEYONDSTYLE_TEST_PROMPTS")
    if env_path:
        return env_path

    default = PROJECT_ROOT / "data" / "prompts" / "default.json"
    if default.is_file():
        return str(default)

    legacy = PROJECT_ROOT / "data" / "test_prompts.json"
    if legacy.is_file():
        return str(legacy)

    return str(default)


def print_config():
    """Print current resolved configuration."""
    print("BeyondStyle Configuration")
    print("=" * 50)
    print(f"  Backend      : {os.environ.get('BEYONDSTYLE_BACKEND', 'local')}")
    print(f"  API model    : {os.environ.get('BEYONDSTYLE_MODEL_NAME', '(not set)')}")
    print(f"  Model path   : {os.environ.get('BEYONDSTYLE_MODEL_PATH', '(not set)')}")
    try:
        print(f"  Model ID     : {get_model_id()}")
    except ValueError:
        print(f"  Model ID     : (model path not set)")
    print(f"  Dataset      : {get_dataset_path()}")
    print(f"  Test prompts : {get_test_prompts_path()}")
    print(f"  Output base  : {get_output_base()}")
    print("=" * 50)
