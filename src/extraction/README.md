# `src/extraction/` — 隐藏状态抽取

| 文件 | 作用 |
|------|------|
| `hidden_states.py` | `HiddenStateExtractor`：批量前向、抽取每层最后 token 隐藏向量 |
| `paired_extractor.py` | `PairedExtractor`：对配对样本（H / AI-chat / AI-free）分别抽取并保存为 `paired_states_layer<NN>.pt` |

## 输出 schema

```python
{
  "human":   Tensor[N, d_model],
  "ai_chat": Tensor[N, d_model],
  "ai_free": Tensor[N, d_model],
  "metadata": {"layer": int, "model": str, ...}
}
```

## 形式化

对样本 $x = [\text{prefix}; \text{continuation}]$，抽取的向量为：

$$h^{(\ell)}(x) = \text{HiddenState}_\ell(x)[T-1]$$

即第 $\ell$ 层、最后一个 token（位置 $T-1$）的隐藏向量。
