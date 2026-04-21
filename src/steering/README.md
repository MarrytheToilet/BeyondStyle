# `src/steering/` — v_src 学习、SAE、可解释性

## 模块概览

| 文件 | 作用 |
|------|------|
| `base.py` | `SteeringMethod` 抽象基类 |
| `mean_diff.py` | 三种 v_src：`mean_diff`、`mean_diff_nullspace`、`mean_diff_fisher` |
| `denoise.py` | 去噪：去除高方差噪声维度 |
| `hooks.py` | forward-hook 工具 |
| `metrics.py` | AUC / Cohen-d / margin 等分离度指标 |
| `sae.py` | 稀疏自编码器（SAE）训练 + 方向分解 |
| `interpretability.py` | 浅层可解释性：logit lens、SAE feature names、深度结构 |
| `vector_interp.py` | 向量级可解释性：层轨迹、词汇语义场、token 归因、注意力头/MLP 神经元 |
| `deep_interp.py` | 工具函数：构式、对比 n-gram、信息密度、词组挖掘 |

## 公式速览

### v_src 三种估计

| 方法 | 公式 |
|------|------|
| mean_diff | $v = \mu_A - \mu_H$ |
| nullspace | $v = (I - P_C)(\mu_A - \mu_H)$，$P_C$ 为内容子空间投影 |
| fisher | $v = \Sigma_W^{-1}(\mu_A - \mu_H)$，$\Sigma_W = \tfrac12(\Sigma_A + \Sigma_H)$ |

### SAE

$$
\begin{aligned}
z &= \text{ReLU}(W_e x + b_e) \\
\hat x &= W_d z + b_d \\
\mathcal{L} &= \|x - \hat x\|_2^2 + \lambda \|z\|_1
\end{aligned}
$$

训练后通过 $\hat v_{\text{src}} = \sum_i z_i d_i$ 解释 v_src。

### Logit lens

$$\text{tokens}^{\text{top-k}} = \text{topk}\bigl(W_U \cdot v_{\text{src}}^{(\ell)}\bigr)$$

### Token 归因（grad·input）

$$a_t = \langle \nabla_{h_t}\, s,\ h_t \rangle,\quad s = \langle h^{(\ell)}_{T-1}, v_{\text{src}}^{(\ell)} \rangle$$

## 语言支持

- `_is_meaningful_token(tok, language)`：英文按 ASCII / 字母数字过滤；中文模式仅保留全 CJK token
- `logit_lens` / `token_position_attribution` 在 `language="zh"` 下加载 `zh_core_web_sm`（spaCy）做分词
- `vocabulary_semantic_fields` 通过 tokenizer name 自动判断语言（qwen/chatglm/baichuan/yi → zh）
