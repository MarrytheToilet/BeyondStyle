"""
Interpretability visualisations — paper-aligned, 8-figure version.

Layout follows the 3 paragraphs of `\\subsection{Vector- and token-level
interpretability}` plus four supplementary stylistic readouts and one
vector-geometry figure:

  fig01 — Logit-lens readout (raw unembedding projection)             [paper §1]
  fig02 — Vocabulary semantic fields (frequency-filtered logit lens)  [paper §2]
  fig03 — Contrastive n-grams (Dunning's G²)                          [paper §3]

  fig04 — Construction-grammar patterns      (supplementary stylistic)
  fig05 — Sentence openings + structural features
  fig06 — Lexical bundles
  fig07 — Information-density profile

  fig08 — Layer trajectory (Cohen's d / AUC / cross-layer cosine)     [vector geom]

MLP / SAE / token-position / attention-head analyses have been removed
to keep the readout aligned with the paper.

Palette: light-blue (#B8D4E8) = Human, light-pink (#F5C6D0) = AI.
White background, minimal gridlines.
"""

from __future__ import annotations
import json
import math
import warnings
from pathlib import Path
from typing import Dict, Any

# palette.py handles CJK font init before pyplot — import it first
from .palette import CJK_FONT, L

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_CJK_FONT = CJK_FONT

# ── Palette ──────────────────────────────────────────────────────────
C_HUMAN   = "#B8D4E8"
C_AI      = "#F5C6D0"
C_HUMAN_D = "#6AAFE0"
C_AI_D    = "#E8899A"
C_GREY    = "#999999"
C_BG      = "#FFFFFF"
C_GRID    = "#ECECEC"
C_TEXT    = "#2C2C2C"


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _style(ax, title="", xlabel="", ylabel="", grid_axis="y"):
    ax.set_facecolor(C_BG)
    ax.figure.set_facecolor(C_BG)
    if title:
        ax.set_title(title, fontsize=18, fontweight="bold", color=C_TEXT, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=15.5, color=C_TEXT)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=15.5, color=C_TEXT)
    ax.tick_params(axis="both", labelsize=12)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(C_TEXT)
    if grid_axis:
        ax.grid(axis=grid_axis, color=C_GRID, linewidth=0.8, zorder=0)


def _save(fig, path: Path):
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"    ✓ {path.name}")


def _parse_key(key: str):
    method, layer = key.rsplit("_L", 1)
    return method, int(layer)


def _pick_best_layer(results: dict, section: str = "logit_lens",
                     preferred: int = 16) -> int:
    """Pick a representative layer from cached results, preferring `preferred`."""
    keys = list(results.get(section, {}).keys())
    layers = sorted({_parse_key(k)[1] for k in keys})
    if not layers:
        return preferred
    if preferred in layers:
        return preferred
    return min(layers, key=lambda l: abs(l - preferred))


def _dump_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def _is_cjk(s: str) -> bool:
    if not s:
        return False
    return any(0x4E00 <= ord(c) <= 0x9FFF for c in s)


def _esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ").strip()


def _setup_fonts(language: str = "en"):
    if language == "zh":
        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["axes.unicode_minus"] = False
    else:
        plt.rcParams["font.family"] = "sans-serif"


# =====================================================================
#  Fig 01 — Logit-lens readout (raw)              [paper paragraph 1]
# =====================================================================

def _plot_fig01(results: dict, fig_dir: Path, layer: int = 15,
                method: str = "mean_diff", n_tok: int = 40, language: str = "en"):
    """Source-aware token projection.

    For each paired sample (same prefix, AI vs human continuation), we
    compute the residual-stream hidden state $h_t$ at every continuation
    position and project it onto the source direction:
        proj_t = h_t · v_src.
    Aggregating these per-token projections across the corpus gives,
    for each token type, its mean projection on the AI side and on the
    human side. We report:
      * AI-exclusive tokens — only appear on the AI side, ranked by
        mean projection (most strongly AI-aligned in actual generations).
      * Human-exclusive tokens — only appear on the human side, ranked
        by mean projection (most strongly human-aligned).
      * Contrastive shared tokens — appear on both sides; ranked by
        the AI − Human gap, showing tokens whose *contextual usage*
        differs the most between sources even though the surface form
        is the same.
    """
    tp = results.get("token_projection", {}) or {}
    key = f"{method}_L{layer}"
    block = tp.get(key) or {}
    if not block:
        # Fallback: pick any layer for the requested method.
        cand = [(k, v) for k, v in tp.items() if k.startswith(f"{method}_L")]
        if not cand:
            return
        key, block = cand[0]
        layer = block.get("layer", layer)

    def _filt(rows, key="token"):
        if language != "zh":
            return rows
        return [r for r in rows if _is_cjk(str(r.get(key, "")))]

    ai_excl = _filt(block.get("ai_exclusive_tokens", []))[:n_tok]
    hu_excl = _filt(block.get("human_exclusive_tokens", []))[:n_tok]
    contrast = _filt(block.get("contrastive_tokens", []))[:n_tok]
    if not (ai_excl or hu_excl or contrast):
        return

    g_ai = block.get("global_ai_mean", 0.0)
    g_hu = block.get("global_human_mean", 0.0)
    n_samples = block.get("n_samples", "?")

    lines = [
        f"# Fig 01 — Source-Aware Token Projection (L{layer})",
        "",
        f"_Language: **{language.upper()}**.  Method: `{method}`.  "
        f"Samples: {n_samples}.  Top {n_tok} per panel._",
        "",
        "For every continuation token $t$ in the paired corpus we measure "
        "$\\mathrm{proj}_t = h_t^{(\\ell)} \\cdot \\hat v_{\\mathrm{src}}^{(\\ell)}$ "
        "and aggregate per token type, separately for AI and human "
        "continuations.  A *token* here is the surface string after BPE "
        "decoding, not a vocabulary id, so the same word in two contexts is "
        "merged.  Tokens are kept only if they occur at least 3 times.",
        "",
        f"Global mean projection — **AI side**: {g_ai:+.3f} · "
        f"**Human side**: {g_hu:+.3f}.",
        "",
        "## (A) AI-exclusive tokens",
        "_Tokens that appear only in AI continuations.  Ranked by mean "
        "projection on $\\hat v_{\\mathrm{src}}$._",
        "",
        "| Rank | Token | mean proj. | count |",
        "|---:|:---|---:|---:|",
    ]
    for i, r in enumerate(ai_excl, 1):
        lines.append(
            f"| {i} | {_esc(r['token'])} | {r['mean']:+.3f} | {r['count']} |"
        )

    lines += [
        "",
        "## (B) Human-exclusive tokens",
        "_Tokens that appear only in human continuations.  Ranked by mean "
        "projection (most negative = most human-aligned)._",
        "",
        "| Rank | Token | mean proj. | count |",
        "|---:|:---|---:|---:|",
    ]
    for i, r in enumerate(hu_excl, 1):
        lines.append(
            f"| {i} | {_esc(r['token'])} | {r['mean']:+.3f} | {r['count']} |"
        )

    lines += [
        "",
        "## (C) Contrastive shared tokens",
        "_Same surface token appears on both sides.  $\\Delta = "
        "\\overline{\\mathrm{proj}}_{\\mathrm{AI}} - "
        "\\overline{\\mathrm{proj}}_{\\mathrm{Human}}$.  Large positive $\\Delta$ "
        "means the token sits much further along the AI side when it appears "
        "in AI text than when it appears in human text._",
        "",
        "| Rank | Token | $\\Delta$ | AI mean | Human mean | AI n | Human n |",
        "|---:|:---|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(contrast, 1):
        lines.append(
            f"| {i} | {_esc(r['token'])} | {r['delta']:+.3f} | "
            f"{r['ai_mean']:+.3f} | {r['human_mean']:+.3f} | "
            f"{r['ai_count']} | {r['human_count']} |"
        )
    lines.append("")

    out = fig_dir / "fig01_token_projection.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"    ✓ {out.name}")
    _dump_json(fig_dir / "fig01_token_projection.json", {
        "layer": layer, "method": method, "language": language,
        "global_ai_mean": g_ai, "global_human_mean": g_hu,
        "n_samples": n_samples,
        "ai_exclusive": ai_excl,
        "human_exclusive": hu_excl,
        "contrastive": contrast,
    })


# =====================================================================
#  Fig 02 — Vocabulary semantic fields           [paper paragraph 2]
# =====================================================================

def _plot_fig02(vec_results: dict, fig_dir: Path, language: str = "en"):
    """Frequency-filtered logit lens (no semantic-category labels)."""
    from src.steering.word_filters import filter_word_list
    vf = vec_results.get("vocab_fields", {})
    if not vf:
        return

    ai_words = filter_word_list(
        vf.get("ai_words", []), language=language,
        also_drop_news_quote_words=True, min_len=3, min_freq=1e-6)[:60]
    hu_words = filter_word_list(
        vf.get("human_words", []), language=language,
        also_drop_news_quote_words=True, min_len=3, min_freq=1e-6)[:60]
    if not (ai_words or hu_words):
        return

    lines = [
        "# Fig 02 — Vocabulary Logit-Lens (frequency-filtered)",
        "",
        "_Logit-lens readout $s(w)=W_U[w]^{\\top}\\hat v_{\\mathrm{src}}$ "
        "restricted to the curated valid-word subset (alphabetic words of "
        "length ≥ 3 and unigram frequency above threshold; news-attribution / "
        "quote / markup tokens removed)._",
        "",
        "| Rank | AI-leaning word | score | Human-leaning word | score |",
        "|---:|:---|---:|:---|---:|",
    ]
    n_rows = max(len(ai_words), len(hu_words))
    for i in range(n_rows):
        a = ai_words[i] if i < len(ai_words) else None
        h = hu_words[i] if i < len(hu_words) else None
        a_cell = (f"{_esc(a['word'])} | {a['score']:+.3f}"
                  if a else "— | —")
        h_cell = (f"{_esc(h['word'])} | {h['score']:+.3f}"
                  if h else "— | —")
        lines.append(f"| {i + 1} | {a_cell} | {h_cell} |")
    lines.append("")

    out = fig_dir / "fig02_vocab_lens.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"    ✓ {out.name}")
    _dump_json(fig_dir / "fig02_vocab_lens.json", {
        "ai_words": [{"word": w["word"], "score": w["score"]} for w in ai_words],
        "human_words": [{"word": w["word"], "score": w["score"]} for w in hu_words],
    })


# =====================================================================
#  Fig 03 — Contrastive n-grams                  [paper paragraph 3]
# =====================================================================

def _plot_fig03(deep: dict, fig_dir: Path, top_k: int = 40, language: str = "en"):
    from src.steering.word_filters import filter_ngram_list

    ngrams = deep.get("contrastive_ngrams", {}) or {}
    if not ngrams:
        return

    lines = [
        "# Fig 03 — Contrastive Multi-Word Patterns",
        "",
        "_Top n-grams ranked by Dunning's log-likelihood ratio $G^{2}$. "
        "News-attribution n-grams (said / told / according to / spokesman …) excluded._",
        "",
    ]

    emitted = False
    json_payload = {}
    for n_label in ["2-gram", "3-gram"]:
        data = ngrams.get(n_label, {})
        if not data:
            continue
        ai_top = filter_ngram_list(data.get("ai_distinctive", []),
                                   language=language)[:top_k]
        hu_top = filter_ngram_list(data.get("human_distinctive", []),
                                   language=language)[:top_k]
        if not (ai_top or hu_top):
            continue
        emitted = True
        json_payload[n_label] = {"ai_distinctive": ai_top, "human_distinctive": hu_top}
        lines += [
            f"## {n_label}s",
            "",
            f"| Rank | AI-distinctive {n_label} | $G^{{2}}$ | AI per-1k | "
            f"Human-distinctive {n_label} | $G^{{2}}$ | Human per-1k |",
            "|---:|:---|---:|---:|:---|---:|---:|",
        ]
        n_rows = max(len(ai_top), len(hu_top))
        for i in range(n_rows):
            a = ai_top[i] if i < len(ai_top) else None
            h = hu_top[i] if i < len(hu_top) else None
            a_cell = (f"{_esc(a['ngram'])} | {a['g2']:+.2f} | "
                      f"{a.get('ai_per_1k', 0):.2f}" if a else "— | — | —")
            h_cell = (f"{_esc(h['ngram'])} | {abs(h['g2']):.2f} | "
                      f"{h.get('human_per_1k', 0):.2f}" if h else "— | — | —")
            lines.append(f"| {i + 1} | {a_cell} | {h_cell} |")
        lines.append("")

    if not emitted:
        return
    out = fig_dir / "fig03_contrastive_ngrams.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"    ✓ {out.name}")
    _dump_json(fig_dir / "fig03_contrastive_ngrams.json", json_payload)


# =====================================================================
#  Fig 04 — Construction-grammar patterns        [supplementary]
# =====================================================================

def _plot_fig04(deep: dict, fig_dir: Path, language: str = "en"):
    constr = deep.get("construction_patterns", {}) or {}
    if not constr:
        return

    items = []
    for name, data in constr.items():
        total = data["human_count"] + data["ai_count"]
        if total < 3:
            continue
        items.append((
            data.get("description", "") if language == "zh" else name.replace("_", " "),
            data["ratio"], data["human_per_text"], data["ai_per_text"],
            data["human_count"], data["ai_count"],
        ))
    if not items:
        return
    items.sort(key=lambda x: x[1], reverse=True)

    def _side(r):
        if r > 1.5:
            return "AI →"
        if r < 0.67:
            return "← Human"
        return "neutral"

    lines = [
        "# Fig 04 — Construction-Grammar Patterns",
        "",
        "Frequency of named syntactic constructions in human vs AI text. "
        "Ratio is `(AI-per-text + ε) / (Human-per-text + ε)`. "
        "`AI →` if ratio > 1.5, `← Human` if < 0.67, else neutral.",
        "",
        "| Construction | log₂(ratio) | ratio | H per-text | AI per-text | H count | AI count | Bias |",
        "|:---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for name, ratio, h_pt, a_pt, h_c, a_c in items:
        log_r = math.log2(max(ratio, 0.01))
        lines.append(
            f"| {_esc(name)} | {log_r:+.2f} | ×{ratio:.2f} | "
            f"{h_pt:.2f} | {a_pt:.2f} | {h_c} | {a_c} | {_side(ratio)} |"
        )
    lines.append("")

    out = fig_dir / "fig04_construction_patterns.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"    ✓ {out.name}")
    _dump_json(fig_dir / "fig04_construction_patterns.json", {
        "patterns": [
            {"name": n, "ratio": r, "human_per_text": hp,
             "ai_per_text": ap, "human_count": hc, "ai_count": ac,
             "log2_ratio": math.log2(max(r, 0.01))}
            for (n, r, hp, ap, hc, ac) in items
        ],
    })


# =====================================================================
#  Fig 05 — Layer trajectory (vector geometry)
# =====================================================================

def _plot_fig05(vec_results: dict, fig_dir: Path, language: str = "en"):
    """Layer projection trajectory + Cohen's d/AUC + cross-layer cosine."""
    traj = vec_results.get("layer_trajectory", {})
    if not traj:
        return

    layers = traj["layers"]
    n = len(layers)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5),
                             gridspec_kw={"width_ratios": [3, 2, 2]})

    # Panel A — projection trajectory with error bands
    ax = axes[0]
    h_mean = np.array(traj["human_proj_mean"])
    a_mean = np.array(traj["ai_proj_mean"])
    h_std = np.array(traj["human_proj_std"])
    a_std = np.array(traj["ai_proj_std"])
    ax.fill_between(layers, h_mean - h_std, h_mean + h_std, color=C_HUMAN, alpha=0.3)
    ax.fill_between(layers, a_mean - a_std, a_mean + a_std, color=C_AI, alpha=0.3)
    ax.plot(layers, h_mean, "-o", color=C_HUMAN_D, linewidth=2, markersize=5,
            label=L(language, "human"))
    ax.plot(layers, a_mean, "-s", color=C_AI_D, linewidth=2, markersize=5,
            label=L(language, "ai"))
    if len(layers) >= 3:
        last, prev = layers[-1], layers[-2]
        max_prev = max(abs(h_mean[-2]), abs(a_mean[-2]))
        max_last = max(abs(h_mean[-1]), abs(a_mean[-1]))
        if max_prev > 0 and max_last > 2.5 * max_prev:
            ax.axvspan((prev + last) / 2, last + 0.5, color="#DDDDDD",
                       alpha=0.55, zorder=0)
            ax.text(last, ax.get_ylim()[1] * 0.95, "post-norm\nartifact",
                    ha="right", va="top", fontsize=13.5, color=C_GREY,
                    style="italic")
    ax.legend(fontsize=15)
    _style(ax, L(language, "fig07_panelA"), L(language, "layer"),
           L(language, "fig07_mean_proj"))

    # Panel B — Cohen's d + AUC
    ax = axes[1]
    ax.bar(range(n), traj["cohens_d"], color=C_HUMAN_D, alpha=0.7, width=0.6,
           label=L(language, "cohens_d"))
    ax2 = ax.twinx()
    ax2.plot(range(n), traj["auc_separation"], "-D", color=C_AI_D, linewidth=2,
             markersize=4, label=L(language, "auc"))
    ax2.set_ylabel(L(language, "auc"), fontsize=15.5, color=C_AI_D)
    ax2.set_ylim(0.8, 1.02)
    ax2.tick_params(axis="y", labelcolor=C_AI_D, labelsize=11.5)
    ax2.spines["top"].set_visible(False)
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(l) for l in layers], fontsize=13.5)
    _style(ax, L(language, "fig07_panelB"), L(language, "layer"),
           L(language, "cohens_d"), grid_axis="y")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=14, loc="lower right")

    # Panel C — v_src cross-layer cosine
    ax = axes[2]
    cos = np.array(traj["v_src_cosine_matrix"])
    im = ax.imshow(cos, cmap="RdBu_r", vmin=-0.2, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(l) for l in layers], fontsize=13.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([str(l) for l in layers], fontsize=13.5)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label=L(language, "cosine_sim"))
    cbar.ax.tick_params(labelsize=10.5)
    cbar.set_label(L(language, "cosine_sim"), fontsize=15)
    _style(ax, L(language, "fig07_panelC"), L(language, "layer"),
           L(language, "layer"), grid_axis="")

    fig.suptitle(L(language, "fig07_suptitle"), fontsize=20,
                 fontweight="bold", color=C_TEXT, y=1.02)
    fig.tight_layout()
    _save(fig, fig_dir / "fig05_layer_trajectory.png")


# =====================================================================
#  Dispatchers
# =====================================================================

def plot_all_interpretability(results: dict, fig_dir: Path, language: str = "en"):
    """Generate fig01 (token projection) + fig03–fig04 (corpus stylistics).
    fig02 (vocab lens) and fig05 (layer trajectory) come from
    `plot_vector_interpretability`."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=UserWarning)
    _setup_fonts(language)

    layer = _pick_best_layer(results, "logit_lens", preferred=16)
    _plot_fig01(results, fig_dir, layer=layer, language=language)

    deep = results.get("deep", {})
    if not deep:
        print("    ⚠ No deep analysis results — only fig01 generated")
        return

    _plot_fig03(deep, fig_dir, language=language)
    _plot_fig04(deep, fig_dir, language=language)

    print("  ✓ Corpus-level artefacts generated (fig01, fig03–fig04)")


def plot_vector_interpretability(vec_results: dict, fig_dir: Path,
                                 language: str = "en"):
    """Generate vector-based figures: fig02 (vocab lens) + fig05 (trajectory)."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=UserWarning)
    _setup_fonts(language)
    plt.rcParams.update({
        "figure.facecolor": C_BG,
        "axes.facecolor": C_BG,
        "savefig.facecolor": C_BG,
    })

    _plot_fig02(vec_results, fig_dir, language=language)
    _plot_fig05(vec_results, fig_dir, language=language)

    print("  ✓ Vector-based figures generated (fig02, fig05)")
