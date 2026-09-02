#!/usr/bin/env python3
"""Save pinned, deterministic HF row subsets without transforming text."""

import argparse
import json
import random
import re
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path

import yaml
from huggingface_hub import HfApi


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LANGUAGES = ("en", "zh")
SAFE_REPO_ID = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--language",
        choices=LANGUAGES,
        required=True,
        help="Language boundary for both the registry and raw output path.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Default: data/corpora/<language>.yaml",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Registry source names. Default: sources with default_download=true.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve commits, source shards and row quotas without reading rows.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing subset with the same pinned specification.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def repo_slug(repo_id: str) -> str:
    if not SAFE_REPO_ID.fullmatch(repo_id):
        raise ValueError(f"Unsafe or unsupported Hugging Face dataset ID: {repo_id!r}")
    return repo_id.replace("/", "__")


def source_dir(registry: dict, spec: dict) -> Path:
    root = project_path(Path(registry["download_root"]))
    return root / repo_slug(spec["dataset_id"]) / spec["snapshot_commit"]


def subset_path(registry: dict, spec: dict) -> Path:
    return source_dir(registry, spec) / f"rows_{spec['sample_rows']}.jsonl"


def manifest_path(registry: dict, spec: dict) -> Path:
    return source_dir(registry, spec) / f"rows_{spec['sample_rows']}.manifest.json"


def load_registry(path: Path, language: str) -> dict:
    with path.open(encoding="utf-8") as handle:
        registry = yaml.safe_load(handle)
    if registry.get("language") != language:
        raise SystemExit(
            f"Registry language {registry.get('language')!r} does not match "
            f"--language {language!r}"
        )
    expected_root = Path("data") / "raw" / language / "hf"
    if Path(registry.get("download_root", "")) != expected_root:
        raise SystemExit(
            f"Registry download_root must be {expected_root}, got "
            f"{registry.get('download_root')!r}"
        )
    return registry


def selected_sources(registry: dict, requested: list[str] | None) -> list[str]:
    specs = registry["sources"]
    selected = requested or [
        name for name, spec in specs.items() if spec.get("default_download")
    ]
    unknown = sorted(set(selected) - set(specs))
    if unknown:
        raise SystemExit(f"Unknown sources: {', '.join(unknown)}")
    for name in selected:
        if not specs[name].get("enabled"):
            raise SystemExit(
                f"Source {name} is disabled: "
                f"{specs[name].get('exclusion_reason', 'no reason recorded')}"
            )
    return selected


def validate_spec(source_name: str, spec: dict) -> None:
    required = (
        "dataset_id",
        "snapshot_revision",
        "snapshot_commit",
        "prefix_file_patterns",
        "sample_rows",
        "sample_seed",
        "sampling_strategy",
    )
    missing = [key for key in required if spec.get(key) is None]
    if missing:
        raise SystemExit(
            f"Source {source_name} lacks subset metadata: {', '.join(missing)}"
        )
    if spec["sample_rows"] <= 0:
        raise SystemExit(f"Source {source_name} has a non-positive sample setting")
    if spec["sampling_strategy"] != "shard_balanced_contiguous_v1":
        raise SystemExit(
            f"Source {source_name} uses an unsupported sampling strategy: "
            f"{spec['sampling_strategy']}"
        )
    repo_slug(spec["dataset_id"])


def matching_source_files(info, patterns: list[str]) -> list[dict]:
    files = []
    for sibling in info.siblings:
        path = sibling.rfilename
        if any(fnmatch(path, pattern) for pattern in patterns):
            files.append({"path": path, "bytes": sibling.size or 0})
    return sorted(files, key=lambda item: item["path"])


def balanced_quotas(total_rows: int, n_files: int) -> list[int]:
    if n_files <= 0:
        raise ValueError("Cannot allocate rows over zero source files")
    base, remainder = divmod(total_rows, n_files)
    return [base + (index < remainder) for index in range(n_files)]


def hf_url(repo_id: str, commit: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{commit}/{path}"


def write_subset(
    source_name: str,
    language: str,
    spec: dict,
    source_files: list[dict],
    output: Path,
) -> list[dict]:
    import polars as pl

    quotas = balanced_quotas(spec["sample_rows"], len(source_files))
    temporary = output.with_suffix(".jsonl.incomplete")
    inventory = []
    written = 0
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for shard_index, (source_file, quota) in enumerate(
                zip(source_files, quotas)
            ):
                if quota == 0:
                    continue
                url = hf_url(
                    spec["dataset_id"],
                    spec["snapshot_commit"],
                    source_file["path"],
                )
                lazy_frame = pl.scan_parquet(url)
                source_rows = lazy_frame.select(pl.len()).collect().item()
                if quota > source_rows:
                    raise RuntimeError(
                        f"{source_file['path']}: requested {quota} rows from "
                        f"a shard containing only {source_rows}"
                    )
                rng = random.Random(spec["sample_seed"] + shard_index)
                start_row = rng.randrange(source_rows - quota + 1)
                frame = lazy_frame.slice(start_row, quota).collect()
                shard_written = 0
                for native_row in frame.iter_rows(named=True):
                    source_row_index = start_row + shard_written
                    record = {
                        "schema_version": 1,
                        "language": language,
                        "source_name": source_name,
                        "provenance": {
                            "dataset_id": spec["dataset_id"],
                            "source_revision": spec.get("source_revision"),
                            "snapshot_commit": spec["snapshot_commit"],
                            "config": spec.get("config"),
                            "split": spec.get("prefix_split"),
                            "source_file": source_file["path"],
                            "source_row_index": source_row_index,
                        },
                        "native_row": native_row,
                    }
                    handle.write(
                        json.dumps(record, ensure_ascii=False, default=str) + "\n"
                    )
                    shard_written += 1
                    written += 1
                inventory.append(
                    {
                        "source_file": source_file["path"],
                        "source_rows": source_rows,
                        "start_row": start_row,
                        "requested_rows": quota,
                        "written_rows": shard_written,
                    }
                )
        if written != spec["sample_rows"]:
            raise RuntimeError(
                f"{source_name}: requested {spec['sample_rows']} rows, "
                f"but read only {written}"
            )
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return inventory


def write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_subset_records(output: Path, language: str, spec: dict) -> int:
    seen = set()
    count = 0
    with output.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            provenance = record.get("provenance", {})
            identity = (
                provenance.get("source_file"),
                provenance.get("source_row_index"),
            )
            if record.get("language") != language:
                raise RuntimeError(
                    f"{output}:{line_number}: language boundary violation"
                )
            if provenance.get("dataset_id") != spec["dataset_id"]:
                raise RuntimeError(
                    f"{output}:{line_number}: dataset_id mismatch"
                )
            if provenance.get("snapshot_commit") != spec["snapshot_commit"]:
                raise RuntimeError(
                    f"{output}:{line_number}: snapshot_commit mismatch"
                )
            if not isinstance(record.get("native_row"), dict):
                raise RuntimeError(
                    f"{output}:{line_number}: native_row is not an object"
                )
            if identity in seen:
                raise RuntimeError(
                    f"{output}:{line_number}: duplicate source row {identity}"
                )
            seen.add(identity)
            count += 1
    if count != spec["sample_rows"]:
        raise RuntimeError(
            f"{output}: expected {spec['sample_rows']} rows, found {count}"
        )
    return count


def existing_subset_is_valid(
    output: Path,
    manifest_file: Path,
    language: str,
    spec: dict,
) -> bool:
    if not output.is_file() or not manifest_file.is_file():
        return False
    with manifest_file.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    return (
        manifest.get("dataset_id") == spec["dataset_id"]
        and manifest.get("snapshot_commit") == spec["snapshot_commit"]
        and manifest.get("sample_rows") == spec["sample_rows"]
        and manifest.get("sampling_strategy") == spec["sampling_strategy"]
        and validate_subset_records(output, language, spec) == spec["sample_rows"]
    )


def main():
    args = parse_args()
    registry_path = project_path(
        args.registry or Path("data/corpora") / f"{args.language}.yaml"
    )
    registry = load_registry(registry_path, args.language)
    specs = registry["sources"]
    selected = selected_sources(registry, args.sources)

    api = HfApi()
    for source_name in selected:
        spec = specs[source_name]
        validate_spec(source_name, spec)
        info = api.dataset_info(
            spec["dataset_id"],
            revision=spec["snapshot_revision"],
            files_metadata=True,
        )
        if info.sha != spec["snapshot_commit"]:
            raise SystemExit(
                f"Pinned commit mismatch for {spec['dataset_id']}: "
                f"expected {spec['snapshot_commit']}, resolved {info.sha}"
            )

        source_files = matching_source_files(
            info, spec["prefix_file_patterns"]
        )
        if not source_files:
            raise SystemExit(
                f"{source_name}: no files match {spec['prefix_file_patterns']}"
            )
        quotas = balanced_quotas(spec["sample_rows"], len(source_files))
        output = subset_path(registry, spec)
        manifest_file = manifest_path(registry, spec)
        print(
            f"[{args.language}] {source_name}: "
            f"{spec['dataset_id']}@{info.sha}\n"
            f"  subset: {spec['sample_rows']} rows from "
            f"{len(source_files)} train shard(s)\n"
            f"  quota range: {min(quotas)}-{max(quotas)} rows/shard\n"
            f"  target: {output}",
            flush=True,
        )
        if args.dry_run:
            continue
        if not args.force and existing_subset_is_valid(
            output, manifest_file, args.language, spec
        ):
            print("  already verified; skipping", flush=True)
            continue

        output.parent.mkdir(parents=True, exist_ok=True)
        shard_inventory = write_subset(
            source_name,
            args.language,
            spec,
            source_files,
            output,
        )
        validate_subset_records(output, args.language, spec)
        manifest = {
            "schema_version": 1,
            "created_at": utc_now(),
            "mode": "deterministic_hf_parquet_row_subset",
            "sampling_strategy": spec["sampling_strategy"],
            "text_transformation": "none",
            "language": args.language,
            "source_name": source_name,
            "dataset_id": spec["dataset_id"],
            "source_revision": spec.get("source_revision"),
            "snapshot_commit": info.sha,
            "config": spec.get("config"),
            "split": spec.get("prefix_split"),
            "sample_rows": spec["sample_rows"],
            "sample_seed": spec["sample_seed"],
            "source_files": shard_inventory,
            "output": str(output.relative_to(PROJECT_ROOT)),
            "output_bytes": output.stat().st_size,
        }
        write_manifest(manifest_file, manifest)
        print(
            f"  verified: {spec['sample_rows']} rows, "
            f"{output.stat().st_size / 1024**2:.1f} MiB",
            flush=True,
        )


if __name__ == "__main__":
    main()
