# `scripts/` — 入口脚本

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `build_dataset.py` | 构建配对数据集 | 语料 | `data/paired/<lang>/paired_<teacher>.json` |
| `steering_pipeline.py` | 抽取隐藏状态 + 训练 v_src | 配对数据 | `outputs/<lang>/steering/...` |
| `run_analysis.py` | 配对/引导分析图（fig01-fig08） | 隐藏状态 | `outputs/<lang>/figures/...` |
| `run_interpretability.py` | 可解释性深度分析（fig01-fig11） | 隐藏状态 + v_src | `outputs/<lang>/interp/...` |
| `eval_gptzero.py` | GPTZero AI 检测 | 生成文本 | `outputs/<lang>/eval/gptzero/` |
| `eval_lm_benchmark.py` | PPL / MAUVE | 生成文本 | `outputs/<lang>/eval/lm_benchmark/` |
| `eval_human_scoring.py` | 人工/LLM 打分 | 生成文本 | `outputs/<lang>/eval/human_scoring/` |

## 标准流水线

```bash
# 1) 构建数据
python scripts/build_dataset.py --language en -n 1000 -w 8

# 2) 抽取隐藏状态 + 学习 v_src
python scripts/steering_pipeline.py --language en

# 3) 配对分析图
python scripts/run_analysis.py --language en

# 4) 可解释性
python scripts/run_interpretability.py --language en --max_samples 200

# 5) 评测
python scripts/eval_lm_benchmark.py --language en
```

中文流水线把 `--language en` 替换为 `--language zh`，并在第 2-4 步通过 `--model /root/autodl-tmp/models/Qwen3-4B` 指定中文学生模型。

## 通用参数

- `--language {en,zh}` — 选择语言（影响输出目录、spaCy 模型、图表标签）
- `--n_layers N` — 当未指定 `--layers` 时，自动均匀抽取 N 层（默认 4）
- `--skip_sae` / `--skip_deep` / `--skip_vector` — 跳过对应 Part（仅 `run_interpretability.py`）

## 长任务建议

- 数据构建（API 调用）：`tmux new -s build_en "python scripts/build_dataset.py …"`
- SAE 训练 / 可解释性：同样使用 `tmux`，日志可 `tee /tmp/<task>.log`
