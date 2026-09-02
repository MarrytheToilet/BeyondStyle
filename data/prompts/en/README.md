# `data/prompts/en/` — 英文评测提示

正式文件 `formal.json` 是 JSON 列表，每条字段：

| 字段 | 说明 |
|------|------|
| `id` | 唯一标识 |
| `text` | 从冻结 prefix bank 逐字选取的自然文本 prefix |
| `domain` | journalistic / encyclopedic / instructional / literary_narrative |
| `source_document_sha256` | 仅用于核验 document-level 数据隔离的哈希 |

由 `scripts/build_formal_prompts.py` 从未进入 paired training 的源文档中
确定性选取，不改写文本。选择规则、输入与输出哈希记录在
`formal.manifest.json`。
