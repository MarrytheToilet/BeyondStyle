# `data/paired/` — 配对数据集

每个 JSON 文件存放一个 teacher 模型生成的配对样本集合。文件命名为 `paired_<TeacherModelID>.json`。

## 文件格式

```json
{
  "metadata": {
    "model": "Qwen/Qwen3.5-397B-A17B",
    "n_samples": 1000,
    "language": "en"
  },
  "samples": [
    {
      "id": "...",
      "domain": "news",
      "prefix": "...",                       // 人类前缀
      "human_continuation": "...",           // 人类续写
      "ai_continuation_chat": "...",         // AI 轻约束续写
      "ai_continuation_free": "...",         // AI 自由续写
      "metadata": {...}
    },
    ...
  ]
}
```

## 子目录

- `en/` — 英文（CNN/DailyMail），teacher 默认 `Qwen3.5-397B-A17B`。
- `zh/` — 中文（THUCNews），teacher 默认 `Qwen2.5-32B-Instruct`（Llama 不支持中文）。

## 用法

被以下脚本读取：

- `scripts/steering_pipeline.py` — 抽取隐藏状态 + 学习 v_src
- `scripts/run_analysis.py` — 配对嵌入/投影可视化
- `scripts/run_interpretability.py` — 可解释性深度分析

## v_src 估计公式

对 AI 续写 $\mathcal{D}_A$ 与人类续写 $\mathcal{D}_H$，记每条样本第 $\ell$ 层最末 token 隐藏向量为 $h^{(\ell)}$，则：

$$v_{\text{src}}^{(\ell)} = \mu_A^{(\ell)} - \mu_H^{(\ell)},\quad
\mu_C^{(\ell)} = \frac{1}{|\mathcal{D}_C|}\sum_{x\in\mathcal{D}_C} h^{(\ell)}(x)$$

可选变体（在 `src/steering/mean_diff.py` 中实现）：

- **nullspace**：`v_src` 投影到内容子空间正交补，缓解 prefix 主题混淆。
- **fisher**：用类内/类间散度白化，等价于 LDA 方向。
