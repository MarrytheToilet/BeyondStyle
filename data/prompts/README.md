# `data/prompts/` — 评测提示

存放可解释性、引导生成和人工评测使用的中性提示。按语言分目录：`en/`, `zh/`。

每个 JSONL 文件一行一条提示：

```json
{"id": "p001", "prompt": "Write a paragraph about ..."}
```

被以下脚本读取：

- `scripts/run_analysis.py` — 引导生成的 prompt 集
- `scripts/eval_*.py` — 评测时使用同一 prompt 集计算公平的 baseline
