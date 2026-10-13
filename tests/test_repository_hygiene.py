"""Keep documented entry points and tracked source files usable in exports."""

import ast
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def source_files(suffix):
    """Inspect distributed code, not ignored server helpers in a working tree."""
    if (ROOT / ".git").exists() and shutil.which("git"):
        names = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
            text=True,
        ).split("\0")
        return sorted({ROOT / n for n in names if n.endswith(suffix) and (ROOT / n).is_file()})
    return sorted(p for folder in ("src", "scripts", "tests", "data")
                  for p in (ROOT / folder).rglob(f"*{suffix}"))


def test_python_sources_parse():
    for path in source_files(".py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path.relative_to(ROOT)))


def test_shell_entry_points_parse():
    if not shutil.which("bash"):
        pytest.skip("Bash is required for shell entry-point validation")
    for path in source_files(".sh"):
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        assert result.returncode == 0, f"{path.name}: {result.stderr}"


def test_readme_entry_points_exist():
    for path in (ROOT / "README.md", ROOT / "scripts/README.md", ROOT / "assets/README.md"):
        text = path.read_text(encoding="utf-8")
        references = re.findall(r"(?:scripts/)?([a-z][a-z0-9_]+\.(?:py|sh))\b", text)
        assert references
        for name in references:
            assert (ROOT / "scripts" / name).is_file(), f"{path.name}: missing {name}"


def test_readme_local_links_exist():
    docs = [ROOT / name for name in ("README.md", "scripts/README.md", "data/README.md", "assets/README.md")]
    for path in docs:
        for target in re.findall(r"\]\(([^\s)]+)\)", path.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            assert (path.parent / target.split("#", 1)[0]).exists(), f"Broken link in {path.name}: {target}"


def test_readme_bash_examples_parse():
    if not shutil.which("bash"):
        pytest.skip("Bash is required for README example validation")
    blocks = re.findall(r"```bash\n(.*?)```", (ROOT / "README.md").read_text(encoding="utf-8"), re.S)
    assert blocks
    for index, block in enumerate(blocks, 1):
        result = subprocess.run(["bash", "-n"], input=block, capture_output=True, text=True)
        assert result.returncode == 0, f"README Bash block {index}: {result.stderr}"


def test_tokenizer_audits_use_portable_model_ids():
    for language in ("en", "zh"):
        path = ROOT / "data/prefixes" / language / "balanced_natural/tokenizer_audit.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        for model, details in report["models"].items():
            assert details["tokenizer_id"] == model
            assert "tokenizer_path" not in details


def test_tracked_data_files_parse():
    for path in source_files(".json"):
        if path.is_relative_to(ROOT / "data"):
            json.loads(path.read_text(encoding="utf-8"))
    for path in source_files(".jsonl"):
        if path.is_relative_to(ROOT / "data"):
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if line.strip():
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as error:
                        pytest.fail(f"Invalid JSON in {path.relative_to(ROOT)}:{line_number}: {error.msg}")
