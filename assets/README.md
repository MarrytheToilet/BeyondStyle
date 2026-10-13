# BeyondStyle visual assets

These are the selected public-facing visuals used by the root README. The
full manuscript remains outside the code repository. PNGs display in GitHub;
the matching PDFs retain the original figure resolution.

## Illustrations

| File | Role | Origin |
| --- | --- | --- |
| [overview.pdf](overview.pdf), [overview.png](overview.png) | The paper's four-stage overview | PDF copied from the current manuscript; PNG rendered from that PDF |
| [pr-comic.png](pr-comic.png) | Comic-style project introduction | AI-generated concept illustration using the built-in image-generation tool |

The comic's characters, dial, and word streams are explanatory metaphors, not
experimental outputs. It contains no measured curves or invented numerical
results. The phrase examples are illustrative rather than paired generation
quotations. The overview is also a schematic, not a plot of measured values.
The comic's `+w` and `−w` denote opposite interventions along the same lexical
direction; pink syntax markers identify a separate measurement group.

## Experimental figures and table

| Preview / PDF | What it shows | Reproduction entry point |
| --- | --- | --- |
| [Surface signature](results/surface-signature.png) / [PDF](results/surface-signature.pdf) | Paired feature effects and cross-environment stability | `scripts/plot_fig_surface_signature.py` |
| [Lexical expressions](results/lexical-expressions.png) / [PDF](results/lexical-expressions.pdf) | Contextual hidden-state projections, not frequency counts | `scripts/plot_projected_lexical_field.py` |
| [Transport null](results/transport-null.png) / [PDF](results/transport-null.pdf) | Fixed-token transport against refitted nulls in Llama and Qwen | `scripts/plot_fig_transport_null.py` |
| [Generation control](results/generation-control.png) / [PDF](results/generation-control.pdf) | Target/off-target responses, prompt distributions, and paired seeds | `scripts/plot_fig_generation.py` |
| [Generation summary CSV](results/generation-summary.csv) | The paper's held-out target/selectivity table and selected settings | Values transcribed from the current paper's generation table and protocol |

The quantitative PDFs are byte-for-byte copies of the current manuscript
figures, not image-model redrawings. The PNGs are rendered previews, with no
changes to data, labels, colors, or layout. The CSV preserves the reported
three-decimal rounding; consult the experiment outputs for full precision.

Generation intervals are 95% prompt-bootstrap intervals over 166 held-out
prompts, after averaging three paired decoding seeds within each prompt.
Effects use the frozen corpus-defined score, not accuracy, Cohen's d, or human
ratings. Model-specific layers and doses preclude ranking models by effect
magnitude. The reproduction commands and data dependencies are in the
[root README](../README.md).

## Updating the assets

When a quantitative figure changes, rerun its retained plotting script with
the matching numerical artifacts, then replace both the PDF and its preview.
For example, from the repository root, render a preview with Poppler:

```bash
pdftoppm -singlefile -scale-to-x 2000 -scale-to-y -1 -png assets/results/generation-control.pdf assets/results/generation-control
```

Do not edit measured curves with an image-generation model. Check the result
table against the paper before updating the CSV or README. Keep AI-generated
promotional art labeled as illustration, and keep manuscript text, author
details, API credentials, and server configuration out of this directory.

中文说明：这里公开主图、漫画宣传图及四张关键结果图。漫画用于讲解研究思路，
不代表真实实验输出；结果图直接复制论文 PDF，再导出 PNG 预览。所有数字、
置信区间和模型设置均保持与当前正文一致。
