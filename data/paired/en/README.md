# `data/paired/en/` — 英文配对数据集

| 文件 | Teacher 模型 | 样本数 | 语料 |
|------|-------------|--------|------|
| `paired_Qwen3.5-397B-A17B.json` | Qwen3.5-397B-A17B (SiliconFlow API) | ~1000 | CNN/DailyMail |

## 重新生成

```bash
python scripts/build_dataset.py \
    --language en \
    --corpus cnn_dailymail \
    --num-samples 1000 \
    --max-docs 3000 \
    --workers 8
```

输出会覆盖现有 `paired_<TeacherID>.json`。Teacher 由 `.env` 中 `BEYONDSTYLE_MODEL_NAME` 控制，或 CLI `--model-name` 覆盖。

## 字段

每条样本包含：`id`, `domain`, `prefix`, `human_continuation`, `ai_continuation_chat`, `ai_continuation_free`, `metadata`。
