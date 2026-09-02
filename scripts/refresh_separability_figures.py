#!/usr/bin/env python3
"""Redraw every canonical EN/ZH per-folder separability figure from JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_generator_holdout import plot_per_generator
from scripts.eval_heldout_separability import plot_per_layer


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def canonical_result_dirs(output_root: Path, languages: list[str]) -> list[Path]:
    dirs: set[Path] = set()
    for language in languages:
        base = output_root / language / "02_separability"
        for representation in ("last_token", "mean_pool"):
            dirs.update(
                path.parent
                for path in base.glob(
                    f"*/proposition_aligned/{representation}/random_holdout.json"
                )
            )
    return sorted(dirs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=("en", "zh"),
        default=("en", "zh"),
    )
    args = parser.parse_args()

    result_dirs = canonical_result_dirs(args.output_root, list(args.languages))
    if not result_dirs:
        raise FileNotFoundError(
            f"No canonical separability results under {args.output_root}"
        )

    random_count = 0
    generator_count = 0
    for result_dir in result_dirs:
        random_json = result_dir / "random_holdout.json"
        generator_json = result_dir / "generator_holdout.json"
        if not generator_json.exists():
            raise FileNotFoundError(
                f"Missing generator-held-out result beside {random_json}"
            )

        plot_per_layer(
            load_json(random_json),
            str(result_dir / "random_holdout_per_layer.png"),
        )
        random_count += 1
        plot_per_generator(
            load_json(generator_json),
            str(result_dir / "generator_holdout_per_generator.png"),
        )
        generator_count += 1
        print(f"refreshed  {result_dir}")

    print(
        f"done: {random_count} random-holdout + "
        f"{generator_count} generator-holdout figures "
        "(PNG and PDF)"
    )


if __name__ == "__main__":
    main()
