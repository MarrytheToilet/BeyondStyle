"""Shared palette, style, and save utility."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as _fm
# Rebuild font cache before pyplot init to pick up system CJK fonts
_fm._load_fontmanager(try_read_cache=False)

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path


def _setup_cjk_font() -> str | None:
    """Register a CJK font family in rcParams if available."""
    for fam in [
        "Noto Sans CJK JP", "Noto Sans CJK SC", "Noto Sans CJK",
        "WenQuanYi Micro Hei", "SimHei", "Source Han Sans SC",
    ]:
        try:
            prop = _fm.FontProperties(family=fam)
            resolved = _fm.findfont(prop, fallback_to_default=False)
            if resolved and "DejaVu" not in resolved:
                plt.rcParams["font.sans-serif"] = [fam] + plt.rcParams.get("font.sans-serif", [])
                plt.rcParams["axes.unicode_minus"] = False
                return fam
        except Exception:
            continue
    return None


CJK_FONT = _setup_cjk_font()

PALETTE = {
    "human":       "#A8D8EA",   # soft sky blue
    "ai":          "#F7C5CC",   # soft blush pink
    "human_dark":  "#4A90B8",   # emphasis blue
    "ai_dark":     "#D45B7A",   # emphasis pink
    "bg":          "#FAFBFD",
    "grid":        "#E6E9ED",
    "accent":      "#6C5CE7",   # purple accent
}

# Short aliases used throughout
H, A = PALETTE["human"], PALETTE["ai"]
H_D, A_D = PALETTE["human_dark"], PALETTE["ai_dark"]
BG = PALETTE["bg"]

# Custom colormaps: blue ↔ pink
BP_CMAP = LinearSegmentedColormap.from_list("bp", [H_D, "#F0F4F8", A_D])
BP_SEQ  = LinearSegmentedColormap.from_list("bp_seq", [H_D, "#C8B8DB", A_D])

DPI = 200


def setup_style():
    """Apply BeyondStyle matplotlib rcParams."""
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "axes.facecolor":    BG,
        "figure.facecolor":  BG,
        "axes.grid":         True,
        "grid.alpha":        0.30,
        "grid.color":        PALETTE["grid"],
        "grid.linewidth":    0.5,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.edgecolor":    "#CCCCCC",
        "axes.labelsize":    13,
        "axes.titlesize":    15,
        "axes.titleweight":  "bold",
        "xtick.labelsize":   12,
        "ytick.labelsize":   12,
        "legend.fontsize":   12,
        "figure.titlesize":  17,
        "figure.titleweight": "bold",
    })


# ─────────────────────────────────────────────────────────
# Bilingual labels (en / zh) for figure text
# ─────────────────────────────────────────────────────────
LABELS = {
    "en": {
        # paired_plots
        "fig01_suptitle": r"Projection onto Steering Direction  $v_{src}$",
        "human": "Human", "ai": "AI", "layer": "Layer", "auc": "AUC",
        "projection": "Projection", "count": "Count",
        "fig02_pca_title": "PCA — Layer {ly}   (Var explained: {var:.1%})",
        "fig02_align_title": r"$v_{src}$  Alignment with PCA Components",
        "cosine_sim": "Cosine Similarity",
        "fig03_title": r"$v_{src}$  Cosine Similarity Across Layers",
        "fig04_left_title": "AI Centroid Trajectory — Layer {ly}",
        "fig04_right_title": "Mean Distance to Human Anchor vs α",
        "fig04_suptitle": "Steering Trajectory: AI Continuation vs Human Anchor",
        "human_anchor": "Human (anchor)",
        "alpha_strength": "Steering strength α",
        "mean_dist": "Mean L2 distance (PCA space)",
        "toward_human": "← toward Human", "away_human": "away from Human →",
        "baseline": "α=0\n(baseline)",
        # steering_plots
        "fig05_suptitle": "Layer-wise Metrics ({method})",
        "norm_hull": "Normalized Hull Centroid Dist  ★",
        "fisher_ratio": "Fisher Discriminant Ratio",
        "auc_ref": "AUC  (reference)",
        "centroid_per_diam": "Centroid Dist / Hull Diameter",
        "j_higher": "J (higher = better)",
        "best": "Best",
        "fig06_title": "ROC Curves — MeanDiff",
        "fpr": "False Positive Rate", "tpr": "True Positive Rate",
        "layer_auc": "Layer (AUC)",
        "fig08_title": "Projection Distribution by Layer  (z-scored per layer)",
        "norm_proj": "Normalized Projection",
        # ablation_plots
        "fig09_suptitle": "Denoising Method Comparison — Performance Maintained, Direction Purified",
        "class_auc": "Classification AUC",
        "best_layer_acc": "Best-Layer Accuracy",
        "dir_sim_raw": "Direction Similarity to Raw",
        "accuracy": "Accuracy",
        "cos_v_raw": "cos(v, v_raw)",
        "fig10_title": "Projection Distributions (Human vs AI)",
        # interp_plots
        "fig01_logit_title": "Logit Lens — L{ly}",
        "fig01_logit_x": "← Human-promoted     AI-promoted →",
        "fig01_tp_title": "Source-Exclusive Token Projection — L{ly}",
        "fig01_tp_x": "← Human-exclusive     AI-exclusive →",
        "fig01_suptitle_interp": "Steering Vector Vocabulary Decomposition",
        "fig01_mean_proj": "Mean proj: AI={ai:+.3f}  H={hu:+.3f}",
        "fig02_interp_title": "Construction Grammar: AI / Human Frequency Ratio",
        "fig02_interp_x": "log₂(ratio)    ← Human-preferred          AI-preferred →",
        "fig02_legend_ai": "AI-preferred (>1.5×)",
        "fig02_legend_neutral": "Neutral",
        "fig02_legend_h": "Human-preferred (<0.67×)",
        "fig03_ai_title": "AI-Distinctive {n}s",
        "fig03_h_title": "Human-Distinctive {n}s",
        "fig03_x": "G² (log-likelihood ratio)",
        "fig03_suptitle": "Contrastive Multi-Word Patterns",
        "fig04_open_title": "Sentence Opening Patterns",
        "fig04_open_x": "% of sentences",
        "fig04_struct_title": "Structural Feature Comparison",
        "fig04_struct_x": "Feature value",
        "fig04_interp_suptitle": "Syntactic Structure Analysis",
        "fig05_interp_title": "Information Density Profile: Human vs AI",
        "value": "Value",
        "fig06_interp_title": "Exclusive {n} Bundles",
        "frequency": "Frequency",
        "fig06_suptitle": "Formulaic Sequence Repertoire",
        "fig06_legend_ai": "AI-exclusive",
        "fig06_legend_h": "Human-exclusive",
        "fig06_bundle_types": "Bundle types: AI={a}  H={h}",
        # vector interp
        "fig07_panelA": "Projection onto v_src",
        "fig07_mean_proj": "Mean Projection",
        "fig07_panelB": "Separation Strength",
        "cohens_d": "Cohen's d",
        "fig07_panelC": "v_src Cross-Layer Similarity",
        "fig07_suptitle": "Layer-wise Projection Trajectory & Direction Evolution",
        "fig08_top_words": "Top {direction} Words",
        "ai_ward": "AI-ward", "human_ward": "Human-ward",
        "fig08_x": "|Projection Score|",
        "fig08_suptitle": "Vocabulary Projection onto v_src {suffix}",
        "fig08_suffix": "(real English words only)",
        "fig08_valid": "Valid words: {v} / {t}",
        "fig09_panelA_title": "Position Differential\n(AI − Human)",
        "fig09_panelA_x": "Text Position",
        "fig09_panelA_y": "Attribution Δ",
        "fig09_qlabels": ["Start\n(Q1)", "Early-Mid\n(Q2)", "Late-Mid\n(Q3)", "End\n(Q4)"],
        "fig09_panelB_title": "POS Tag Differential\n(AI − Human)",
        "fig09_panelC_title": "Top Discriminative Tokens\n(AI − Human)",
        "fig09_attr_x": "Attribution Δ",
        "fig09_suptitle": "Token Attribution Differential (signed grad·input)",
        "fig09_legend": "Pink (→) = higher in AI context  |  Blue (←) = higher in Human context",
        "fig10_panelA": "Attention Head Attribution (Differential)",
        "fig10_xlabel": "Layer", "fig10_ylabel": "Head",
        "fig10_cbar": "AI − Human contribution to v_src",
        "fig10_panelB": "Top Style Heads",
        "fig10_x": "Contribution to v_src",
        "fig10_suptitle": "Attention Head Attribution to Steering Vector",
        "fig11_panelA": "Top Neurons (Layer {ly})",
        "fig11_panelA_x": "Alignment × Activation Δ",
        "fig11_panelB": "Neuron Profiles: Decoded Words + Trigger Words from Actual Text",
        "fig11_suptitle": "MLP Neuron Analysis — v_src Alignment × Activation Difference",
        "fig11_header": "Neuron|Score|Promoted Words|Trigger (Human)|Trigger (AI)",
        "fig12_suptitle": "SAE Feature Cards — Decoder atoms ranked by AI-vs-Human AUROC (L{ly})",
        "fig12_header": "Feature|AUROC|⟨z⟩ H → AI|Promotes (logit lens)|Top context (Human / AI)",
        "fig12_caption": "AUROC > 0.5 = AI-leaning feature. Token tendencies are logit-lens projections of decoder direction (not feature names). Contexts are top-activating excerpts from paired data.",
    },
    "zh": {
        "fig01_suptitle": r"在转向方向 $v_{src}$ 上的投影分布",
        "human": "人类", "ai": "AI", "layer": "层", "auc": "AUC",
        "projection": "投影值", "count": "样本数",
        "fig02_pca_title": "PCA 嵌入 — 第 {ly} 层（解释方差: {var:.1%}）",
        "fig02_align_title": r"$v_{src}$ 与 PCA 主成分的对齐度",
        "cosine_sim": "余弦相似度",
        "fig03_title": r"$v_{src}$ 在不同层间的余弦相似度",
        "fig04_left_title": "AI 质心轨迹 — 第 {ly} 层",
        "fig04_right_title": "AI 与人类锚点的平均距离 vs α",
        "fig04_suptitle": "转向轨迹：AI 续写 vs 人类锚点",
        "human_anchor": "人类（锚点）",
        "alpha_strength": "转向强度 α",
        "mean_dist": "平均 L2 距离（PCA 空间）",
        "toward_human": "← 趋近人类", "away_human": "远离人类 →",
        "baseline": "α=0\n（基线）",
        "fig05_suptitle": "逐层指标（{method}）",
        "norm_hull": "归一化凸包质心距 ★",
        "fisher_ratio": "Fisher 判别比",
        "auc_ref": "AUC（参考）",
        "centroid_per_diam": "质心距 / 凸包直径",
        "j_higher": "J（越高越好）",
        "best": "最优",
        "fig06_title": "ROC 曲线 — MeanDiff",
        "fpr": "假阳率", "tpr": "真阳率",
        "layer_auc": "层（AUC）",
        "fig08_title": "各层投影分布（按层 z-score 归一化）",
        "norm_proj": "归一化投影",
        "fig09_suptitle": "去噪方法对比 — 性能保持，方向纯化",
        "class_auc": "分类 AUC",
        "best_layer_acc": "最佳层准确率",
        "dir_sim_raw": "与原始方向的相似度",
        "accuracy": "准确率",
        "cos_v_raw": "cos(v, v_raw)",
        "fig10_title": "投影分布（人类 vs AI）",
        "fig01_logit_title": "Logit Lens 词表投影 — 第 {ly} 层",
        "fig01_logit_x": "← 人类倾向     AI 倾向 →",
        "fig01_tp_title": "来源专属词投影 — 第 {ly} 层",
        "fig01_tp_x": "← 人类专属     AI 专属 →",
        "fig01_suptitle_interp": "转向向量的词表分解",
        "fig01_mean_proj": "平均投影：AI={ai:+.3f}  人类={hu:+.3f}",
        "fig02_interp_title": "句法构式：AI / 人类 频率比",
        "fig02_interp_x": "log₂(比值)    ← 人类偏好          AI 偏好 →",
        "fig02_legend_ai": "AI 偏好（>1.5×）",
        "fig02_legend_neutral": "中性",
        "fig02_legend_h": "人类偏好（<0.67×）",
        "fig03_ai_title": "AI 特征性 {n}",
        "fig03_h_title": "人类特征性 {n}",
        "fig03_x": "G²（对数似然比）",
        "fig03_suptitle": "对比性多词模式",
        "fig04_open_title": "句首模式分布",
        "fig04_open_x": "句子比例 (%)",
        "fig04_struct_title": "结构特征对比",
        "fig04_struct_x": "特征值",
        "fig04_interp_suptitle": "句法结构分析",
        "fig05_interp_title": "信息密度画像：人类 vs AI",
        "value": "数值",
        "fig06_interp_title": "专属 {n} 束语",
        "frequency": "频次",
        "fig06_suptitle": "套语序列谱",
        "fig06_legend_ai": "AI 专属",
        "fig06_legend_h": "人类专属",
        "fig06_bundle_types": "束语种类数：AI={a}  人类={h}",
        "fig07_panelA": "在 v_src 方向上的投影",
        "fig07_mean_proj": "平均投影",
        "fig07_panelB": "可分性强度",
        "cohens_d": "Cohen's d",
        "fig07_panelC": "v_src 跨层相似度",
        "fig07_suptitle": "逐层投影轨迹与方向演化",
        "fig08_top_words": "Top {direction} 词",
        "ai_ward": "AI 向", "human_ward": "人类向",
        "fig08_x": "|投影得分|",
        "fig08_suptitle": "词表在 v_src 上的投影 {suffix}",
        "fig08_suffix": "（中文词汇）",
        "fig08_valid": "有效词数：{v} / {t}",
        "fig09_panelA_title": "位置差异\n(AI − 人类)",
        "fig09_panelA_x": "文本位置",
        "fig09_panelA_y": "归因差 Δ",
        "fig09_qlabels": ["开头\n(Q1)", "前中\n(Q2)", "后中\n(Q3)", "结尾\n(Q4)"],
        "fig09_panelB_title": "词性差异\n(AI − 人类)",
        "fig09_panelC_title": "Top 区分性词元\n(AI − 人类)",
        "fig09_attr_x": "归因差 Δ",
        "fig09_suptitle": "词元归因差异（带符号 grad·input）",
        "fig09_legend": "粉色（→）= AI 上下文更高  |  蓝色（←）= 人类上下文更高",
        "fig10_panelA": "注意力头归因（差分）",
        "fig10_xlabel": "层", "fig10_ylabel": "头",
        "fig10_cbar": "AI − 人类 对 v_src 的贡献",
        "fig10_panelB": "Top 风格头",
        "fig10_x": "对 v_src 的贡献",
        "fig10_suptitle": "注意力头对转向向量的归因",
        "fig11_panelA": "Top 神经元（第 {ly} 层）",
        "fig11_panelA_x": "对齐度 × 激活差 Δ",
        "fig11_panelB": "神经元画像：解码词 + 来自实际文本的触发词",
        "fig11_suptitle": "MLP 神经元分析 — v_src 对齐度 × 激活差异",
        "fig11_header": "神经元|得分|促进词|触发词(人类)|触发词(AI)",
        "fig12_suptitle": "SAE 特征卡片 — 按 AI vs 人类 AUROC 排序的字典原子（第 {ly} 层）",
        "fig12_header": "特征|AUROC|⟨z⟩ 人类 → AI|倾向促进 (logit lens)|高激活语境（人类 / AI）",
        "fig12_caption": "AUROC > 0.5 表示偏向 AI 的特征。Token 倾向是解码方向通过 unembedding 的投影（非特征命名）。语境为来自配对数据的最高激活片段。",
    },
}


def L(language: str, key: str, **fmt) -> str:
    """Lookup a label string.

    Per project convention, *all* axis labels, titles, legends, and panel
    captions are rendered in English even for ZH figures — only the
    \"core data\" content (token strings, n-grams, vocabulary words, etc.)
    that is loaded directly from JSON results stays in its original
    language. This keeps Chinese figures readable to international
    reviewers while preserving the linguistic content under study.
    """
    # Small whitelist of keys whose value genuinely depends on the data
    # language (e.g. the \"(real English words only)\" suffix needs to
    # become \"(Chinese words only)\" for ZH).
    _ZH_DATA_OVERRIDES = {
        "fig08_suffix": "(Chinese words only)",
    }
    if language == "zh" and key in _ZH_DATA_OVERRIDES:
        s = _ZH_DATA_OVERRIDES[key]
    else:
        # Always look up in EN: ZH → EN fallback for all chart text.
        s = LABELS["en"].get(key, key)
    return s.format(**fmt) if fmt else s


def save_fig(fig, path: Path, name: str) -> Path:
    """Save figure to *path/name* and return the full path."""
    out = Path(path) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
    print(f"  ✓ {out}")
    return out
