# `data/paired/zh/` — 中文配对数据集

| 文件 | Teacher 模型 | 样本数 | 语料 |
|------|-------------|--------|------|
| `paired_Qwen2.5-32B-Instruct.json` | Qwen2.5-32B-Instruct (SiliconFlow API) | ~1000 | THUCNews |

> Llama-3.1 中文能力弱，故中文 teacher 选用 Qwen2.5-32B-Instruct。中文学生模型推荐 `Qwen3-4B`。

## 重新生成

```bash
python scripts/build_dataset.py \
    --language zh \
    --corpus chinese_news \
    --num-samples 1000 \
    --max-docs 3000 \
    --workers 8 \
    --model-name 'Qwen/Qwen2.5-32B-Instruct'
```

中文语料 `chinese_news` 自动下载 `dirtycomputer/THUCNews` 到 `data/raw/zh/`。

## 字段

与英文版相同：`id`, `domain`, `prefix`, `human_continuation`, `ai_continuation_chat`, `ai_continuation_free`, `metadata`。
中文文本以 UTF-8 存储；`prefix` ≈ 100 tokens，`continuation` ≈ 200 tokens（按 BPE 计算）。
