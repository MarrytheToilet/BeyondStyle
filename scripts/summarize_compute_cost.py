#!/usr/bin/env python3
"""Summarize recorded compute for the confirmatory ACL experiment chain."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
PULLBACK_ROOT = ROOT / "outputs/en/05_linguistic/causal_pullback"
GENERATION_ROOT = ROOT / "outputs/en/03_steering_vectors"
CORPUS_MANIFEST = ROOT / "data/paired/en/proposition_aligned/manifest.json"
QUALITY_REPORT = ROOT / "data/paired/en/proposition_aligned/quality_report.json"
OUT_DIR = ROOT / "outputs/en/05_linguistic/compute_audit"
MODELS = ("Llama-3.1-8B-Instruct", "Qwen3-4B")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def generation_seconds(model: str, subdir: str) -> tuple[float, int, list[Path]]:
    root = (
        GENERATION_ROOT
        / model
        / "proposition_aligned/mean_pool"
        / subdir
    )
    paths = sorted(root.glob("generation_component_*_manifest.json"))
    seconds = 0.0
    tokens = 0
    for path in paths:
        payload = load(path)
        seconds += float(payload["elapsed_generation_seconds"])
        tokens += int(payload["generated_tokens"])
    return seconds, tokens, paths


def main() -> None:
    rows: list[dict[str, object]] = []
    sources: list[Path] = [CORPUS_MANIFEST, QUALITY_REPORT]
    quality = load(QUALITY_REPORT)
    rows.append(
        {
            "stage": "corpus construction",
            "model": "four hosted APIs",
            "device": "provider-managed; local GPU not used",
            "wall_seconds": "",
            "gpu_process_hours": "",
            "generated_tokens": "",
            "work_units": (
                f"{quality['n_pairs']} accepted pairs; "
                f"{quality['rejected_generation_count']} rejected generations; "
                f"{quality['rejected_candidate_count']} rejected candidates"
            ),
        }
    )

    for model in MODELS:
        direction_manifest = PULLBACK_ROOT / model / "mean_pool/manifest.json"
        sources.append(direction_manifest)
        direction = load(direction_manifest)
        seconds = float(direction["duration_seconds"])
        rows.append(
            {
                "stage": "canonical pullback construction",
                "model": model,
                "device": "one CUDA process",
                "wall_seconds": seconds,
                "gpu_process_hours": seconds / 3600,
                "generated_tokens": "",
                "work_units": "2 concepts × 32 paired Jacobians",
            }
        )

        null_manifest = (
            PULLBACK_ROOT
            / model
            / "mean_pool/confirmatory_hidden_audit/matched_nulls/manifest.json"
        )
        sources.append(null_manifest)
        null = load(null_manifest)
        seconds = float(null["duration_seconds"])
        rows.append(
            {
                "stage": "100-pipeline matched hidden null",
                "model": model,
                "device": "one CUDA process",
                "wall_seconds": seconds,
                "gpu_process_hours": seconds / 3600,
                "generated_tokens": "",
                "work_units": "2 concepts × 100 nulls × 100 fixed sequences",
            }
        )

        for subdir, label in (
            (
                "causal_pullback_confirmatory_generation",
                "one-seed response matrix and random-axis calibration",
            ),
            (
                "causal_pullback_seed_robustness_generation",
                "three-seed learned-axis replication",
            ),
        ):
            seconds, tokens, paths = generation_seconds(model, subdir)
            sources.extend(paths)
            rows.append(
                {
                    "stage": label,
                    "model": model,
                    "device": "one CUDA process",
                    "wall_seconds": seconds,
                    "gpu_process_hours": seconds / 3600,
                    "generated_tokens": tokens,
                    "work_units": f"{len(paths)} direction files",
                }
            )

    frame = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "compute_summary.csv"
    frame.to_csv(summary_path, index=False)
    total_gpu_hours = float(
        pd.to_numeric(frame["gpu_process_hours"], errors="coerce").sum()
    )
    manifest = {
        "schema": "acl_confirmatory_compute_audit_v1",
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "definition": (
            "GPU-process hours sum recorded elapsed time for one model-owning "
            "CUDA process. Two-model jobs can run concurrently, so elapsed "
            "wall time is lower than the sum."
        ),
        "gpu_hardware": (
            "Hardware identity must be joined from the server environment "
            "snapshot; older stage manifests record CUDA placement but not "
            "the GPU product name."
        ),
        "total_recorded_gpu_process_hours": total_gpu_hours,
        "inputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in dict.fromkeys(sources)
        },
        "output": {
            str(summary_path.relative_to(ROOT)): sha256(summary_path),
        },
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(frame.to_string(index=False))
    print(f"\nTotal recorded GPU-process hours: {total_gpu_hours:.3f}")


if __name__ == "__main__":
    main()
