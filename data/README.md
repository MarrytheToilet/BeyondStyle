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

The saved `tokenizer_audit.json` files are historical prefix-bank diagnostics,
not the observer-model roster of the current paper. Their model IDs and token
counts are preserved; machine-specific tokenizer paths are omitted from the
portable reports. Rerunning `audit_prefix_tokenizers.py` measures new
tokenizers without changing the frozen text windows.

## Paired corpora

`paired/{language}/proposition_aligned/` is the direction-fitting corpus.
Each item contains a shared prefix, original human continuation, atomic
proposition plan, generator continuation, source provenance, generator
identity and automatic validation record. The final `pairs.jsonl`, manifest,
quality report and generation audit files are retained together.

`paired/en/prefix_only_control/` is the English interface-transfer corpus.
It keeps the matched prefixes, human continuations, generator identities, and
target lengths, but supplies no proposition plan to the generator. It is held
out from surface-axis and hidden-readout fitting.

## Formal prompts

`prompts/{language}/formal.json` contains verbatim prompts drawn from source
documents unused by direction training. Its adjacent manifest freezes the
source-bank hash, exclusion rule and output hash.

The paper's generation split uses prompt indices 0–19 for development and
34–199 for testing; 20–33 are unused in that evaluation. Do not silently
replace the frozen prompt file or use the test prompts to select settings.

## Release checks

The registries retain source-license labels and dataset-card links, but those
metadata do not establish permission for every derived release. Review each
source's terms, required attribution, and downstream distribution conditions
before repackaging text. This repository does not relicense upstream text.

Excluding private-document datasets does not remove names or other identifying
information from public news, books, or encyclopedic text. The existing
construction checks do not constitute a dedicated privacy or offensive-content
screen. Such a review must be documented separately before claiming that
released texts are de-identified or safety-filtered.
