# `data/prompts/zh/` — 中文评测提示

与英文正式 prompt 同结构（JSON 列表，字段 `id` / `text` /
`domain` / provenance hashes）。

`formal.json` 由 `scripts/build_formal_prompts.py` 从冻结中文 prefix bank
逐字选取，并排除 paired training 中出现过的整篇源文档。提示不经过翻译、
重写或 tokenizer 切分。
