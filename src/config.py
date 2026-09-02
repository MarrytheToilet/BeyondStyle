"""
Centralized configuration loader.

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


def get_api_key(required: bool = True) -> Optional[str]:
    """Return the canonical OpenAI-compatible generation API key."""
    value = os.environ.get("API_KEY")
    if required and not value:
        raise ValueError(
            "API_KEY not set. Add it to .env or export it before running."
        )
    return value


def get_api_base_url() -> str:
    """Return the shared OpenAI-compatible generation endpoint."""
    return os.environ.get("API_BASE_URL", "https://api.openai.com/v1")


def get_api_models() -> list[str]:
    """Return the configured model pool, or the single-model fallback."""
    models = [
        model.strip()
        for model in os.environ.get("API_MODELS", "").split(",")
        if model.strip()
    ]
    if models:
        return models
    single = os.environ.get("API_MODEL", "").strip()
    return [single] if single else []


def get_model_path() -> str:
    """Get local model path from env. Required for steering/extraction.
    
    Relative paths are resolved against PROJECT_ROOT so that scripts
    work regardless of the current working directory.
    """
    path = os.environ.get("LOCAL_MODEL_PATH")
    if not path:
        raise ValueError(
            "LOCAL_MODEL_PATH not set. "
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
    """Return the canonical dataset directory name."""
    name = Path(dataset_path).stem
    path = Path(dataset_path)
    if name == "pairs" and path.parent.name:
        return path.parent.name
    raise ValueError(
        f"Expected canonical .../<dataset_tag>/pairs.jsonl, got {dataset_path}"
    )


def resolve_language(
    language: Optional[str] = None,
    dataset_path: Optional[str] = None,
) -> str:
    """Resolve language from explicit flag or dataset path.

    Priority:
        1. Explicit ``language`` argument
        2. Infer from dataset path  (``data/paired/zh/...`` → ``"zh"``)
        3. Default ``"en"``
    """
    if language:
        return language
    if dataset_path:
        parts = Path(dataset_path).resolve().parts
        # Look for data/paired/{lang}/ in the path
        for i, p in enumerate(parts):
            if p == "paired" and i + 1 < len(parts):
                candidate = parts[i + 1]
                if len(candidate) == 2 and candidate.isalpha():
                    return candidate
    return "en"


def get_output_base() -> Path:
    """Get base output directory (default: outputs/)."""
    path = Path(os.environ.get("OUTPUT_DIR", "outputs")).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _section_dirname(section: str) -> str:
    """Map a logical section name to its on-disk directory name.

    The numbered prefixes mirror the experiment order:
    01 data validation, 02 source geometry, 03 steering, 04 guardrails,
    05 linguistic decomposition and 06 auxiliary detector response.
    """
    aliases = {
        "data_corpus": "01_data_corpus",
        "separability": "02_separability",
        "steering": "03_steering_vectors",
        "lmeval": "04_lmeval",
        "eval": "04_lmeval",
        "linguistic": "05_linguistic",
        "detector": "06_detector",
    }
    return aliases.get(section, section)


def _lang_output_dir(
    section: str,
    model_id: Optional[str] = None,
    dataset_tag: Optional[str] = None,
    language: str = "en",
) -> Path:
    """Build ``outputs/{lang}/{section_dir}/{model_id}[/{dataset_tag}]``."""
    if model_id is None:
        model_id = get_model_id()
    new = get_output_base() / language / _section_dirname(section) / model_id
    if dataset_tag:
        new = new / dataset_tag
    return new


def get_steering_dir(model_id: Optional[str] = None,
                     dataset_tag: Optional[str] = None,
                     language: str = "en") -> Path:
    """Get the canonical model-specific steering output directory."""
    return _lang_output_dir("steering", model_id, dataset_tag, language)


def get_eval_dir(model_id: Optional[str] = None,
                 language: str = "en") -> Path:
    """Get the canonical lm-eval output directory."""
    return _lang_output_dir("eval", model_id, language=language)


def get_separability_dir(model_id: Optional[str] = None,
                         dataset_tag: Optional[str] = None,
                         language: str = "en") -> Path:
    """Get separability (§3.3 + §4.2) output directory."""
    return _lang_output_dir("separability", model_id, dataset_tag, language)


def get_linguistic_dir(language: str = "en") -> Path:
    """Get the automatic linguistic-feature and decomposition directory."""
    return get_output_base() / language / _section_dirname("linguistic")


def get_detector_dir(language: str = "en", subdir: str = "pangram") -> Path:
    """Get the auxiliary detector-response directory."""
    return get_output_base() / language / _section_dirname("detector") / subdir


def get_dataset_dir(language: str = "en") -> Path:
    """Get the paired-data root for one language."""
    return PROJECT_ROOT / "data" / "paired" / language


def get_dataset_path(language: str = "en") -> str:
    """Get the canonical paired dataset path.

    Resolution order:
        1. PAIRED_DATASET env var (explicit path)
        2. data/paired/{lang}/proposition_aligned/pairs.jsonl
    """
    env_path = os.environ.get("PAIRED_DATASET")
    if env_path:
        path = Path(env_path).expanduser()
        return str(
            path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
        )

    canonical = (
        PROJECT_ROOT
        / "data"
        / "paired"
        / language
        / "proposition_aligned"
        / "pairs.jsonl"
    )
    return str(canonical)


def get_test_prompts_path(language: str = "en") -> str:
    """Get test prompts path from env or default.

    Resolution order:
        1. TEST_PROMPTS env var
        2. data/prompts/{language}/formal.json
    """
    env_path = os.environ.get("TEST_PROMPTS")
    if env_path:
        path = Path(env_path).expanduser()
        return str(
            path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
        )

    return str(PROJECT_ROOT / "data" / "prompts" / language / "formal.json")


def print_config():
    """Print current resolved configuration."""
    print("Project Configuration")
    print("=" * 50)
    print(f"  Backend      : {os.environ.get('GEN_BACKEND', 'local')}")
    print(f"  API model    : {os.environ.get('API_MODEL', '(not set)')}")
    print(f"  Model path   : {os.environ.get('LOCAL_MODEL_PATH', '(not set)')}")
    try:
        print(f"  Model ID     : {get_model_id()}")
    except ValueError:
        print(f"  Model ID     : (model path not set)")
    print(f"  Dataset (en) : {get_dataset_path('en')}")
    print(f"  Dataset (zh) : {get_dataset_path('zh')}")
    print(f"  Prompts (en) : {get_test_prompts_path('en')}")
    print(f"  Prompts (zh) : {get_test_prompts_path('zh')}")
    print(f"  Output base  : {get_output_base()}")
    print(f"  Output (en)  : {get_output_base()}/en/")
    print(f"  Output (zh)  : {get_output_base()}/zh/")
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


def get_sweep_layers(model_or_path, n_select: int = 8) -> list:
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
        config = model_or_path.config
        text_config = getattr(config, "text_config", config)
        n = getattr(text_config, "num_hidden_layers", 32)
        return auto_select_layers(n, n_select)

    # Local path — peek at config.json
    import json as _json
    cfg_path = Path(model_or_path).expanduser().resolve() / "config.json"
    if cfg_path.is_file():
        with open(cfg_path) as f:
            cfg = _json.load(f)
        n = cfg.get(
            "num_hidden_layers",
            (cfg.get("text_config") or {}).get("num_hidden_layers", 32),
        )
        return auto_select_layers(n, n_select)

    # Fallback to 32 (Llama default)
    return auto_select_layers(32, n_select)
