# `data/`

English and Chinese are separate experimental boundaries. The canonical path
is:

> **Dataset-card use warning.** These paired corpora support aggregate
> linguistic and representation research. Their machine--human contrasts do
> not identify authorship and must not be operationalized as thresholds for
> accusations, access control, academic misconduct decisions, or evaluation
> of individual writers. The four human corpora and four generators are
> contingent reference populations, not normative definitions of human and
> machine writing.

```text
corpora/{en,zh}.yaml
    → raw/{en,zh}/hf/
    → prefixes/{en,zh}/balanced_natural/
    → paired/{en,zh}/proposition_aligned/
    → prompts/{en,zh}/formal.json
```

## Raw subsets

`raw/{language}/hf/` contains small deterministic subsets of native rows from
pinned Hugging Face Parquet shards. Rows are not cleaned, translated,
rewritten, or sentence-split at this stage. Each subset retains dataset,
revision, config, shard and source-row provenance.

The project never downloads an entire upstream corpus. Government reports,
private correspondence, medical records, legal case files and personal
profiles are outside scope.

## Prefix banks

`prefixes/{language}/balanced_natural/` contains verbatim
sentence-boundary windows. English boundaries use words; Chinese boundaries
use natural text units. Model tokenizers audit the frozen windows but never
choose or modify them.

## Paired corpora

`paired/{language}/proposition_aligned/` is the only paired-data protocol.
Each item contains a shared prefix, original human continuation, atomic
proposition plan, generator continuation, source provenance, generator
identity and automatic validation record. The final `pairs.jsonl`, manifest,
quality report and generation audit files are retained together.

## Formal prompts

`prompts/{language}/formal.json` contains verbatim prompts drawn from source
documents unused by direction training. Its adjacent manifest freezes the
source-bank hash, exclusion rule and output hash.

No legacy flat raw cache, alternate prompt collection, repaired text or
parallel experimental dataset belongs in the canonical tree.
