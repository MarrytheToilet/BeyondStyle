# `data/`

```
data/
├── raw/      HuggingFace corpus caches    (gitignored)
├── prompts/  Held-out evaluation prompts  (en/, zh/, JSONL: id + prompt)
└── paired/   Paired (human, AI) datasets  (en/, zh/, JSON, schema below)
```

## Paired sample schema

```json
{
  "id": "...",
  "domain": "news",
  "prefix": "...",                    // human prefix (~100 tokens)
  "human_continuation": "...",        // original human continuation
  "ai_continuation_chat": "...",      // AI continuation under topic+entity sketch
  "ai_continuation_free": "...",      // AI continuation with no sketch
  "metadata": {"topic": "...", "key_entities": [...]}
}
```

`scripts/build_dataset.py` builds `paired/<lang>/paired_<teacher>.json` from a
local corpus + a remote teacher API. `scripts/build_prompts.py` samples the
neutral held-out prompts under `prompts/<lang>/`.
