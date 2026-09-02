#!/usr/bin/env python3
"""Prefetch or offline-validate lm-eval task datasets before reserving a GPU."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
import types


ROOT = Path(__file__).resolve().parent.parent
CUSTOM_TASK_DIR = ROOT / "lm_eval_tasks"


DEFAULT_TASKS = [
    "piqa",
    "openbookqa",
    "sciq",
    "arc_challenge",
    "social_iqa",
    "copa",
    "winogrande",
    "wic",
    "rte",
    "boolq",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Forbid network access and fail if any task is not cached.",
    )
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


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
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    # transformers>=5 no longer exposes a vision class imported
    # unconditionally by lm-eval 0.4.12.
    sys.modules.setdefault(
        "lm_eval.models.hf_vlms",
        types.ModuleType("lm_eval.models.hf_vlms"),
    )
    from lm_eval.tasks import TaskManager

    manager = TaskManager(include_path=CUSTOM_TASK_DIR)
    records = []
    for task in args.tasks:
        started = time.perf_counter()
        error_text = None
        try:
            loaded = manager.load([task])
            loaded_count = len(loaded)
        except Exception as error:
            loaded_count = 0
            error_text = f"{type(error).__name__}: {error}"
        elapsed = time.perf_counter() - started
        records.append({
            "task": task,
            "ok": error_text is None,
            "loaded_objects": loaded_count,
            "elapsed_seconds": elapsed,
            "error": error_text,
        })
        status = "OK" if error_text is None else "FAILED"
        print(f"{task}: {status} ({elapsed:.1f}s)")
        if error_text:
            print(f"  {error_text}")

    report = {
        "created_at": datetime.now().isoformat(),
        "offline": args.offline,
        "hf_home": os.environ.get("HF_HOME"),
        "datasets_cache": os.environ.get("HF_DATASETS_CACHE"),
        "ok": all(record["ok"] for record in records),
        "tasks": records,
    }
    if args.json_out:
        atomic_json(args.json_out, report)
        print(f"Report: {args.json_out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
