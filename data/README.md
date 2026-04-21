# `data/` — 数据目录

存放所有原始语料、提示模板与配对数据集。按语言分目录（`en/` 与 `zh/`），便于双语对齐分析。

## 目录结构

```
data/
├── raw/              # HuggingFace 下载的原始语料缓存（被 .gitignore 忽略）
│   ├── en/           #   CNN/DailyMail, Wikipedia, AG News …
│   └── zh/           #   THUCNews 等
├── prompts/          # 评测/续写所用的中性提示
│   ├── en/
│   └── zh/
└── paired/           # 配对数据集：human + AI guided + AI free
    ├── en/           #   paired_<TeacherModel>.json
    └── zh/
```

## 关键概念

- **prefix**：来自人类语料的前缀（约 100 token）。
- **human_continuation**：原文中真实的人类续写（约 200 token）。
- **ai_continuation_chat**：AI 在「轻约束」（topic + entities）下的续写。
- **ai_continuation_free**：AI 在「无约束」下的自由续写。

成对样本支持学习 **源向量 v_src**：

$$v_{\text{src}}^{(\ell)} = \mathbb{E}_{x\in\mathcal{D}_{\text{AI}}}\!\bigl[h^{(\ell)}(x)\bigr] - \mathbb{E}_{x\in\mathcal{D}_{\text{H}}}\!\bigl[h^{(\ell)}(x)\bigr]$$

其中 $h^{(\ell)}(x)$ 为模型第 $\ell$ 层最后一个 token 的隐藏状态。

## 构建数据

```bash
# 英文（CNN/DailyMail，1000 样本）
python scripts/build_dataset.py --language en --corpus cnn_dailymail -n 1000 -w 8

# 中文（THUCNews，1000 样本）
python scripts/build_dataset.py --language zh --corpus chinese_news -n 1000 -w 8 \
  --model-name 'Qwen/Qwen2.5-32B-Instruct'
```

详细脚本说明见 `scripts/README.md`。
