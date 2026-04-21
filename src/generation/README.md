# `src/generation/` — 文本生成

| 文件 | 作用 |
|------|------|
| `backends.py` | 统一接口：`SiliconFlowGenerator`（OpenAI 兼容 API）+ `LocalGenerator`（HF transformers） |
| `generator.py` | 高层封装：批量生成 + 重试 |
| `prompts.py` | 通用 system prompt 模板 |
| `steered.py` | `SteeredGenerator`：在前向中通过 forward hook 注入 $h \mathrel{+}= \alpha\,v_{\text{src}}$ |

## 引导生成公式

$$\tilde h^{(\ell)} = h^{(\ell)} + \alpha\,\frac{v_{\text{src}}^{(\ell)}}{\|v_{\text{src}}^{(\ell)}\|}$$

通过 hook 在指定层把 $h^{(\ell)}$ 替换为 $\tilde h^{(\ell)}$，$\alpha$ 可正可负（正值更 AI-like，负值更 human-like）。

## 用法

```python
from src.generation.steered import SteeredGenerator
gen = SteeredGenerator(model, tokenizer, layer=14, v_src=v, alpha=2.0)
out = gen.generate(prompts)
```
