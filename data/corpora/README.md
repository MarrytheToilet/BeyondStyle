# Existing-dataset registries

The English and Chinese corpora are registered independently:

- `en.yaml`: English sources and English-only download metadata;
- `zh.yaml`: Chinese sources and Chinese-only download metadata.

A source is never inferred from its text or from a filename. Every download
command must specify `--language en` or `--language zh`, and the registry's
top-level `language` value must match.

## Data boundary

The pipeline uses existing Hugging Face datasets. The raw stage saves a
small deterministic subset of native rows from pinned Parquet shards. It does
not scrape pages, clean text, translate, split sentences, or generate examples.
Each saved row retains repository, commit, config, split, source shard and row
index. Later operations belong in the canonical prefix bank.

```
Hugging Face dataset
        │  repository + commit + shard-balanced fixed-size row subset
        ▼
data/raw/{en,zh}/hf/<owner__dataset>/<commit>/
        │  sentence-boundary window selection only
        ▼
data/prefixes/{en,zh}/balanced_natural/{samples.jsonl,manifest.json}
        │  multiple generator models and the canonical generation protocol
        ▼
data/paired/{en,zh}/proposition_aligned/{pairs.jsonl,manifest.json}
        │  extraction, direction estimation, steering and evaluation
        ▼
outputs/{en,zh}/<stage>/<target_model>/<dataset_tag>/
```

The project does not download complete upstream datasets. Legacy flat
JSON/JSONL caches are not part of this registry or the canonical data tree.

## Selection policy

- **core**: documented human-authored source with usable provenance and license;
- **provisional**: excluded until its provenance or license issue is resolved;
- **stress-test candidate**: potentially useful for robustness, but deliberately
  excluded from estimating the main direction.

Government-report corpora, private correspondence, medical records, legal case
files, and personal-profile datasets are outside the project scope. Ordinary
news is not automatically excluded, but its provenance and license must be
documented.

## Current core domain matrix

| Domain | English | Chinese |
|---|---|---|
| Encyclopedic | Wikipedia (English) | Wikipedia (Chinese) |
| Instructional | WikiLingua (English) | WikiLingua (Chinese) |
| News | XL-Sum (English) | XL-Sum (Simplified Chinese) |
| Literary narrative | PG-19 | — |

The directory structure is symmetric, but the corpus matrix is not forced to
be symmetric. A missing well-documented Chinese counterpart is preferable to
silently introducing a source confound.
