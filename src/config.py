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
        paired_Qwen3.5-397B-A17B.json           →  Qwen3.5-397B-A17B
        paired_dataset_Qwen3.5-397B-A17B.json    →  Qwen3.5-397B-A17B  (legacy)
        paired_dataset.json                       →  ""
    """
    name = Path(dataset_path).stem
    # New layout: paired_{tag}
    if name.startswith("paired_") and not name.startswith("paired_dataset"):
        tag = name[len("paired_"):]
        return tag if tag else ""
    # Legacy layout: paired_dataset_{tag}
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


def get_dataset_dir(language: str = "en") -> Path:
    """Get paired dataset directory for a given language.

    New layout: data/paired/{lang}/
    Falls back to legacy outputs/dataset/ if the new dir doesn't exist yet.
    """
    new_dir = PROJECT_ROOT / "data" / "paired" / language
    if new_dir.is_dir():
        return new_dir
    # Legacy fallback
    legacy = get_output_base() / "dataset"
    if legacy.is_dir():
        return legacy
    return new_dir


def get_dataset_path(language: str = "en") -> str:
    """Get dataset path from env or auto-detect latest paired dataset.

    Resolution order:
        1. BEYONDSTYLE_DATASET env var (explicit path)
        2. Auto-detect latest paired_*.json in data/paired/{lang}/
        3. Legacy fallback: outputs/dataset/paired_dataset_*.json
        4. Fall back to data/paired/{lang}/paired_dataset.json
    """
    env_path = os.environ.get("BEYONDSTYLE_DATASET")
    if env_path:
        return env_path

    # New layout
    new_dir = PROJECT_ROOT / "data" / "paired" / language
    if new_dir.is_dir():
        candidates = sorted(
            new_dir.glob("paired_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return str(candidates[0])

    # Legacy layout
    legacy_dir = get_output_base() / "dataset"
    if legacy_dir.is_dir():
        candidates = sorted(
            legacy_dir.glob("paired_dataset_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return str(candidates[0])

    return str(new_dir / "paired_dataset.json")


def get_test_prompts_path(language: str = "en") -> str:
    """Get test prompts path from env or default.

    Resolution order:
        1. BEYONDSTYLE_TEST_PROMPTS env var
        2. data/prompts/{language}/default.json (new layout)
        3. data/prompts/default.json (legacy)
        4. data/test_prompts.json (legacy fallback)
    """
    env_path = os.environ.get("BEYONDSTYLE_TEST_PROMPTS")
    if env_path:
        return env_path

    # New layout: data/prompts/{lang}/default.json
    new_default = PROJECT_ROOT / "data" / "prompts" / language / "default.json"
    if new_default.is_file():
        return str(new_default)

    # Legacy layout: data/prompts/default.json
    legacy_default = PROJECT_ROOT / "data" / "prompts" / "default.json"
    if legacy_default.is_file():
        return str(legacy_default)

    legacy = PROJECT_ROOT / "data" / "test_prompts.json"
    if legacy.is_file():
        return str(legacy)

    return str(new_default)


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
    print(f"  Dataset (en) : {get_dataset_path('en')}")
    print(f"  Dataset (zh) : {get_dataset_path('zh')}")
    print(f"  Prompts (en) : {get_test_prompts_path('en')}")
    print(f"  Prompts (zh) : {get_test_prompts_path('zh')}")
    print(f"  Output base  : {get_output_base()}")
    print("=" * 50)


def get_raw_dir(language: str = "en") -> Path:
    """Get raw corpus cache directory: data/raw/{language}/."""
    return PROJECT_ROOT / "data" / "raw" / language


# ── Layer selection utilities ─────────────────────────────

def auto_select_layers(
    num_layers: int,
    n_select: int = 4,
    skip_first: int = 2,
) -> list:
    """Pick *n_select* evenly-spaced layers from a model.

    Skips the first *skip_first* layers (typically embedding-like)
    and always includes the final layer.

    Examples:
        auto_select_layers(32) -> [8, 16, 24, 32]
        auto_select_layers(36) -> [9, 18, 27, 36]
        auto_select_layers(32, n_select=6) -> [5, 10, 16, 21, 27, 32]
    """
    first = skip_first + 1  # 1-indexed; e.g. layer 3 for skip_first=2
    last = num_layers
    if n_select <= 1:
        return [last]
    step = (last - first) / (n_select - 1)
    layers = [round(first + i * step) for i in range(n_select)]
    # Guarantee last layer is included
    layers[-1] = last
    return sorted(set(layers))


def get_sweep_layers(model_or_path, n_select: int = 4) -> list:
    """Derive *n_select* evenly-spaced layers from a model path or config.

    Accepts:
        - a string / Path to a local HF checkpoint (reads config.json)
        - a PreTrainedModel instance (reads .config)
        - an int (interpreted as num_hidden_layers directly)
    """
    if isinstance(model_or_path, int):
        return auto_select_layers(model_or_path, n_select)

    # HuggingFace model object
    if hasattr(model_or_path, "config"):
        n = getattr(model_or_path.config, "num_hidden_layers", 32)
        return auto_select_layers(n, n_select)

    # Local path — peek at config.json
    import json as _json
    cfg_path = Path(model_or_path).expanduser().resolve() / "config.json"
    if cfg_path.is_file():
        with open(cfg_path) as f:
            cfg = _json.load(f)
        n = cfg.get("num_hidden_layers", 32)
        return auto_select_layers(n, n_select)

    # Fallback to 32 (Llama default)
    return auto_select_layers(32, n_select)
