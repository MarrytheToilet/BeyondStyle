# `data/prompts/en/` — 英文评测提示

JSONL 格式（一行一条）。每条字段：

| 字段 | 说明 |
|------|------|
| `id` | 唯一标识 |
| `prompt` | 中性写作提示 |
| `domain` (可选) | news / wiki / story 等 |

由 `src/data/prompt_builder.py` 生成或人工撰写。
