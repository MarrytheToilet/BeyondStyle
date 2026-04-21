# `src/visualization/` — 全部绘图函数

## 模块

| 文件 | 内容 |
|------|------|
| `palette.py` | 配色（浅粉 `#FFB3D1` H / 浅蓝 `#A6D8FF` AI）、CJK 字体注册、`LABELS` 双语 i18n 字典、`L(language, key)` 帮助函数、`save_fig` |
| `paired_plots.py` | 配对分析：fig01-fig04 |
| `steering_plots.py` | 引导指标：fig05, fig06, fig08 |
| `ablation_plots.py` | 消融对比：fig07-aux 等 |
| `interp_plots.py` | 可解释性：fig01-fig11（interp 文件夹）+ 两个 dispatcher |

## i18n

所有绘图函数均接受 `language: str = "en"`，标签通过 `L(language, key, **fmt)` 查询。新增标签：

```python
# src/visualization/palette.py
LABELS["en"]["my_label"] = "Some label"
LABELS["zh"]["my_label"] = "某标签"
```

## Dispatcher

```python
from src.visualization import plot_all_interpretability, plot_vector_interpretability

plot_all_interpretability(results, output_dir, language="zh")
plot_vector_interpretability(vector_results, output_dir, language="zh")
```

## 颜色与字体

- `PALETTE` 主色：`#FFB3D1`（浅粉，Human）+ `#A6D8FF`（浅蓝，AI）
- 中文字体回退顺序：Noto Sans CJK SC → Source Han Sans SC → WenQuanYi Zen Hei → SimHei
- 保存：`save_fig(fig, path)` 自动 `bbox_inches="tight"`、`dpi=150`
