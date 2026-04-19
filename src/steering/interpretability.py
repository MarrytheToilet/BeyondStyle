"""
Interpretability analysis of steering vectors v_src.

Three complementary lenses:
  1. **Logit Lens** — project v_src through unembedding to decode vocabulary meaning.
     Filters to complete, meaningful tokens (no BPE fragments).
  2. **Contrastive Token Projection** — per-token projection onto v_src, split by
     source (Human vs AI). Reports Δ = proj_in_AI_context − proj_in_Human_context
     for tokens appearing in both → reveals *contextual* source signal.
  3. **Structural Fingerprint** — discourse markers, hedging, passive voice, sentence
     patterns. Within-source regression isolates genuine structural effects from
     trivial source-label confounds.
"""

import re
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import spacy


# ═══════════════════════════════════════════════════════════
#  Token filtering helpers
# ═══════════════════════════════════════════════════════════

# Discourse markers and hedging words — linguistically motivated feature sets
DISCOURSE_MARKERS = {
    "additive": {"furthermore", "moreover", "additionally", "also", "indeed",
                 "likewise", "similarly", "notably"},
    "contrastive": {"however", "nevertheless", "nonetheless", "although",
                    "whereas", "despite", "yet", "conversely", "instead"},
    "causal": {"therefore", "consequently", "thus", "hence", "accordingly",
               "because", "since", "as"},
    "temporal": {"meanwhile", "subsequently", "previously", "finally",
                 "ultimately", "initially", "eventually"},
    "summative": {"overall", "essentially", "fundamentally", "broadly",
                  "collectively", "ultimately"},
}
ALL_DISCOURSE_MARKERS = set()
for v in DISCOURSE_MARKERS.values():
    ALL_DISCOURSE_MARKERS |= v

HEDGING_WORDS = {
    "may", "might", "could", "perhaps", "possibly", "potentially",
    "likely", "unlikely", "somewhat", "arguably", "seemingly",
    "approximately", "roughly", "tends", "suggests", "appears",
    "relatively", "generally", "typically", "often", "usually",
}

INTENSIFIERS = {
    "very", "extremely", "highly", "incredibly", "remarkably",
    "significantly", "substantially", "absolutely", "utterly",
    "deeply", "profoundly", "critically", "fundamentally",
}

FILLER_ABSTRACT = {
    "landscape", "narrative", "paradigm", "framework", "dynamics",
    "implications", "multifaceted", "comprehensive", "nuanced",
    "robust", "pivotal", "crucial", "underscore", "underscores",
    "highlight", "highlights", "navigate", "foster", "facilitates",
    "realm", "sphere", "spectrum", "trajectory", "intersection",
}


def _is_meaningful_token(tok_str: str) -> bool:
    """Filter for complete, human-readable tokens (not BPE fragments)."""
    s = tok_str.strip()
    if len(s) < 2:
        return False
    if not s.isascii():
        return False
    if all(c in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' for c in s):
        return False
    if re.match(r'^[^a-zA-Z]*$', s):
        return False
    # Reject fragments: must start with space (word-initial) or be capitalized
    if not tok_str.startswith(" ") and not s[0].isupper():
        return False
    return True


# ═══════════════════════════════════════════════════════════
#  Logit Lens (filtered)
# ═══════════════════════════════════════════════════════════

@dataclass
class LogitLensResult:
    layer: int
    method: str
    top_positive: List[Tuple[str, float]]
    top_negative: List[Tuple[str, float]]
    logit_stats: Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "method": self.method,
            "top_positive_ai": [{"token": t.strip(), "logit": round(v, 4)}
                                for t, v in self.top_positive],
            "top_negative_human": [{"token": t.strip(), "logit": round(v, 4)}
                                   for t, v in self.top_negative],
            "logit_stats": {k: round(v, 6) for k, v in self.logit_stats.items()},
        }


def logit_lens(
    v_src: torch.Tensor, model, tokenizer,
    top_k: int = 50, apply_ln: bool = True,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]], Dict[str, float]]:
    """
    Project v_src through unembedding → token-level meaning.
    Filters to meaningful complete tokens and uses z-scored logits.
    """
    v = v_src.float().to(model.device)

    if apply_ln:
        ln = model.model.norm
        v = ln(v.unsqueeze(0)).squeeze(0)

    W = model.lm_head.weight.float()
    logits = W @ v
    logits_np = logits.detach().cpu().numpy()

    # Z-score for fairer ranking
    mean, std = logits_np.mean(), logits_np.std()
    z_scores = (logits_np - mean) / (std + 1e-8)

    stats = {
        "mean": float(mean), "std": float(std),
        "min": float(logits_np.min()), "max": float(logits_np.max()),
    }

    # Build filtered token list
    filtered = []
    for idx in range(len(z_scores)):
        tok_str = tokenizer.decode([idx])
        if _is_meaningful_token(tok_str):
            filtered.append((idx, tok_str, float(z_scores[idx]), float(logits_np[idx])))

    # Sort by z-score
    filtered.sort(key=lambda x: x[2], reverse=True)

    top_positive = [(t, logit) for _, t, _, logit in filtered[:top_k]]
    top_negative = [(t, logit) for _, t, _, logit in filtered[-top_k:][::-1]]

    return top_positive, top_negative, stats


# ═══════════════════════════════════════════════════════════
#  Contrastive Token Projection
# ═══════════════════════════════════════════════════════════

@dataclass
class TokenProjectionResult:
    layer: int
    method: str
    n_samples: int
    # Contrastive: tokens appearing in BOTH sources
    contrastive_tokens: List[Dict]  # [{token, delta, ai_mean, human_mean, ai_n, human_n}]
    # Source-exclusive high-signal tokens
    ai_exclusive: List[Dict]
    human_exclusive: List[Dict]
    # Global stats
    global_ai_mean: float
    global_human_mean: float

    def to_dict(self, top_k: int = 50) -> dict:
        return {
            "layer": self.layer,
            "method": self.method,
            "n_samples": self.n_samples,
            "contrastive_tokens": self.contrastive_tokens[:top_k],
            "ai_exclusive_tokens": self.ai_exclusive[:top_k],
            "human_exclusive_tokens": self.human_exclusive[:top_k],
            "global_ai_mean": round(self.global_ai_mean, 4),
            "global_human_mean": round(self.global_human_mean, 4),
        }


@torch.no_grad()
def token_level_projection(
    v_src: torch.Tensor, model, tokenizer,
    samples: List[dict], layer: int,
    method: str = "mean_diff", max_samples: int = 100,
) -> TokenProjectionResult:
    """
    Source-aware token projection: track projections separately for H vs A context.
    """
    v = v_src.float().to(model.device)

    # Separate aggregation per source
    ai_agg = defaultdict(lambda: {"sum": 0.0, "sum_sq": 0.0, "count": 0})
    human_agg = defaultdict(lambda: {"sum": 0.0, "sum_sq": 0.0, "count": 0})
    all_ai_projs = []
    all_human_projs = []

    for i, sample in enumerate(samples[:max_samples]):
        prefix = sample["prefix"]
        for source, cont_key, agg in [
            ("human", "human_continuation", human_agg),
            ("ai", "ai_continuation_chat", ai_agg),
        ]:
            continuation = sample.get(cont_key, "")
            if not continuation:
                continue
            full_text = prefix + continuation

            inputs = tokenizer(full_text, return_tensors="pt",
                               truncation=True, max_length=2048)
            inputs = {k: v_.to(model.device) for k, v_ in inputs.items()}
            token_ids = inputs["input_ids"][0]

            outputs = model(**inputs, output_hidden_states=True)
            hs = outputs.hidden_states[layer][0].float()

            prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
            cont_start = len(prefix_ids) - 1

            projs = (hs[cont_start:] @ v).cpu().numpy()
            cont_ids = token_ids[cont_start:].cpu().tolist()

            if source == "ai":
                all_ai_projs.extend(projs.tolist())
            else:
                all_human_projs.extend(projs.tolist())

            for tid, proj in zip(cont_ids, projs):
                tok_str = tokenizer.decode([tid]).strip()
                if not tok_str or len(tok_str) < 2:
                    continue
                agg[tok_str]["sum"] += proj
                agg[tok_str]["sum_sq"] += proj ** 2
                agg[tok_str]["count"] += 1

    # Compute per-token stats
    def _stats(agg):
        out = {}
        for tok, d in agg.items():
            n = d["count"]
            if n < 3:
                continue
            mean = d["sum"] / n
            var = d["sum_sq"] / n - mean ** 2
            out[tok] = {"mean": mean, "std": max(var, 0) ** 0.5, "count": n}
        return out

    ai_stats = _stats(ai_agg)
    human_stats = _stats(human_agg)

    # Contrastive: tokens in both sources
    shared = set(ai_stats.keys()) & set(human_stats.keys())
    contrastive = []
    for tok in shared:
        delta = ai_stats[tok]["mean"] - human_stats[tok]["mean"]
        contrastive.append({
            "token": tok,
            "delta": round(float(delta), 4),
            "ai_mean": round(float(ai_stats[tok]["mean"]), 4),
            "human_mean": round(float(human_stats[tok]["mean"]), 4),
            "ai_count": ai_stats[tok]["count"],
            "human_count": human_stats[tok]["count"],
            "effect_size": round(float(delta) / (ai_stats[tok]["std"] + human_stats[tok]["std"] + 1e-8) * 2, 4),
        })
    contrastive.sort(key=lambda x: abs(x["delta"]), reverse=True)

    # Source-exclusive tokens
    ai_only = set(ai_stats.keys()) - set(human_stats.keys())
    human_only = set(human_stats.keys()) - set(ai_stats.keys())

    ai_exclusive = sorted(
        [{"token": t, "mean": round(ai_stats[t]["mean"], 4), "count": ai_stats[t]["count"]}
         for t in ai_only],
        key=lambda x: x["mean"], reverse=True,
    )
    human_exclusive = sorted(
        [{"token": t, "mean": round(human_stats[t]["mean"], 4), "count": human_stats[t]["count"]}
         for t in human_only],
        key=lambda x: x["mean"],
    )

    return TokenProjectionResult(
        layer=layer, method=method,
        n_samples=min(len(samples), max_samples),
        contrastive_tokens=contrastive,
        ai_exclusive=ai_exclusive,
        human_exclusive=human_exclusive,
        global_ai_mean=float(np.mean(all_ai_projs)) if all_ai_projs else 0.0,
        global_human_mean=float(np.mean(all_human_projs)) if all_human_projs else 0.0,
    )


# ═══════════════════════════════════════════════════════════
#  Structural Fingerprint Analysis
# ═══════════════════════════════════════════════════════════

@dataclass
class StructuralAnalysisResult:
    layer: int
    method: str
    # Discourse marker frequencies per source
    discourse_freq: Dict[str, Dict[str, float]]  # {category: {human: f, ai: f, ratio: f}}
    # Hedging / intensifier / filler ratios per source
    lexical_markers: Dict[str, Dict[str, float]]  # {marker_type: {human: f, ai: f}}
    # POS stats split by source
    pos_by_source: Dict[str, Dict[str, float]]  # {POS: {human_mean_proj, ai_mean_proj, delta}}
    # Dep stats split by source
    dep_by_source: Dict[str, Dict[str, float]]
    # Within-source R² (the real test)
    within_human_r2: float
    within_ai_r2: float
    between_r2: float  # for comparison (expected high)
    # Feature importance from within-source regression
    feature_importance: Dict[str, float]
    # Sentence-level structural features per source
    structural_comparison: Dict[str, Dict[str, float]]
    # Passive voice ratio per source
    passive_voice: Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "method": self.method,
            "discourse_frequency": self.discourse_freq,
            "lexical_markers": self.lexical_markers,
            "pos_by_source": dict(sorted(self.pos_by_source.items(),
                                         key=lambda x: abs(x[1].get("delta", 0)),
                                         reverse=True)[:20]),
            "dep_by_source": dict(sorted(self.dep_by_source.items(),
                                         key=lambda x: abs(x[1].get("delta", 0)),
                                         reverse=True)[:20]),
            "within_human_r2": round(self.within_human_r2, 4),
            "within_ai_r2": round(self.within_ai_r2, 4),
            "between_r2": round(self.between_r2, 4),
            "feature_importance": {k: round(v, 4) for k, v in
                                   sorted(self.feature_importance.items(),
                                          key=lambda x: abs(x[1]), reverse=True)[:15]},
            "structural_comparison": self.structural_comparison,
            "passive_voice": self.passive_voice,
        }


def _detect_passive(doc) -> int:
    """Count passive voice constructions in a spaCy doc."""
    count = 0
    for tok in doc:
        if tok.dep_ == "nsubjpass" or tok.dep_ == "auxpass":
            count += 1
        if tok.dep_ == "nsubj:pass":
            count += 1
    return count


def _count_word_set(text_lower: str, word_set: set) -> int:
    words = text_lower.split()
    return sum(1 for w in words if w.strip(".,;:!?\"'()") in word_set)


@torch.no_grad()
def structural_analysis(
    v_src: torch.Tensor, model, tokenizer,
    samples: List[dict], layer: int,
    method: str = "mean_diff", max_samples: int = 100, nlp=None,
) -> StructuralAnalysisResult:
    """
    Structural fingerprint: discourse markers, hedging, passive voice,
    within-source regression to isolate genuine structural effects.
    """
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")

    v = v_src.float().to(model.device)

    # Per-source aggregation for POS / dep
    pos_agg = {"human": defaultdict(lambda: {"sum": 0.0, "sum_sq": 0.0, "count": 0}),
               "ai":    defaultdict(lambda: {"sum": 0.0, "sum_sq": 0.0, "count": 0})}
    dep_agg = {"human": defaultdict(lambda: {"sum": 0.0, "sum_sq": 0.0, "count": 0}),
               "ai":    defaultdict(lambda: {"sum": 0.0, "sum_sq": 0.0, "count": 0})}

    # Discourse/lexical marker counts
    discourse_counts = {"human": defaultdict(int), "ai": defaultdict(int)}
    marker_counts = {"human": {"hedging": 0, "intensifier": 0, "filler_abstract": 0, "total_words": 0},
                     "ai": {"hedging": 0, "intensifier": 0, "filler_abstract": 0, "total_words": 0}}
    passive_counts = {"human": 0, "ai": 0}
    sent_counts = {"human": 0, "ai": 0}

    # Sentence-level features for regression
    sent_features = []  # (features_dict, mean_proj, source)

    for i, sample in enumerate(samples[:max_samples]):
        prefix = sample["prefix"]
        for source, cont_key in [("human", "human_continuation"),
                                  ("ai", "ai_continuation_chat")]:
            continuation = sample.get(cont_key, "")
            if not continuation:
                continue

            full_text = prefix + continuation
            inputs = tokenizer(full_text, return_tensors="pt",
                               truncation=True, max_length=2048)
            inputs = {k: v_.to(model.device) for k, v_ in inputs.items()}
            token_ids = inputs["input_ids"][0]

            outputs = model(**inputs, output_hidden_states=True)
            hs = outputs.hidden_states[layer][0].float()

            prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
            cont_start = len(prefix_ids) - 1
            cont_projs = (hs[cont_start:] @ v).cpu().numpy()
            cont_token_ids = token_ids[cont_start:].cpu().tolist()

            # spaCy analysis
            doc = nlp(continuation)

            # POS/dep mapping
            char_to_syntax = {}
            for token in doc:
                for ci in range(token.idx, token.idx + len(token.text)):
                    char_to_syntax[ci] = (token.pos_, token.dep_)

            cont_text_from_tokens = ""
            token_spans = []
            for tid in cont_token_ids:
                tok_str = tokenizer.decode([tid])
                start = len(cont_text_from_tokens)
                cont_text_from_tokens += tok_str
                end = len(cont_text_from_tokens)
                token_spans.append((start, end))

            for (start, end), proj in zip(token_spans, cont_projs):
                mid = (start + end) // 2
                pos_tag, dep_label = char_to_syntax.get(mid, ("X", "unknown"))
                pos_agg[source][pos_tag]["sum"] += proj
                pos_agg[source][pos_tag]["sum_sq"] += proj ** 2
                pos_agg[source][pos_tag]["count"] += 1
                dep_agg[source][dep_label]["sum"] += proj
                dep_agg[source][dep_label]["sum_sq"] += proj ** 2
                dep_agg[source][dep_label]["count"] += 1

            # Discourse markers
            text_lower = continuation.lower()
            words_lower = text_lower.split()
            for cat, markers in DISCOURSE_MARKERS.items():
                cnt = sum(1 for w in words_lower if w.strip(".,;:!?\"'()") in markers)
                discourse_counts[source][cat] += cnt

            # Lexical markers
            n_words = len(words_lower)
            marker_counts[source]["total_words"] += n_words
            marker_counts[source]["hedging"] += _count_word_set(text_lower, HEDGING_WORDS)
            marker_counts[source]["intensifier"] += _count_word_set(text_lower, INTENSIFIERS)
            marker_counts[source]["filler_abstract"] += _count_word_set(text_lower, FILLER_ABSTRACT)

            # Passive voice
            passive_counts[source] += _detect_passive(doc)
            sent_counts[source] += len(list(doc.sents))

            # Sentence-level features (richer set)
            n_sents = max(len(list(doc.sents)), 1)
            tree_depths = []
            for token in doc:
                depth = 0
                t = token
                while t.head != t:
                    depth += 1
                    t = t.head
                tree_depths.append(depth)

            n_commas = sum(1 for t in doc if t.text == ",")
            avg_word_len = np.mean([len(t.text) for t in doc if t.is_alpha]) if len(doc) > 0 else 0

            # Sentence-initial word patterns
            sents = list(doc.sents)
            n_discourse_initial = sum(
                1 for s in sents
                if len(s) > 0 and s[0].text.lower().strip(".,") in ALL_DISCOURSE_MARKERS
            )

            feat = {
                "mean_sent_len": len(doc) / n_sents,
                "mean_tree_depth": float(np.mean(tree_depths)) if tree_depths else 0,
                "max_tree_depth": float(max(tree_depths)) if tree_depths else 0,
                "noun_ratio": sum(1 for t in doc if t.pos_ == "NOUN") / max(len(doc), 1),
                "verb_ratio": sum(1 for t in doc if t.pos_ == "VERB") / max(len(doc), 1),
                "adj_ratio": sum(1 for t in doc if t.pos_ == "ADJ") / max(len(doc), 1),
                "adv_ratio": sum(1 for t in doc if t.pos_ == "ADV") / max(len(doc), 1),
                "comma_density": n_commas / max(len(doc), 1),
                "avg_word_length": float(avg_word_len),
                "discourse_marker_ratio": _count_word_set(text_lower, ALL_DISCOURSE_MARKERS) / max(n_words, 1),
                "hedging_ratio": _count_word_set(text_lower, HEDGING_WORDS) / max(n_words, 1),
                "intensifier_ratio": _count_word_set(text_lower, INTENSIFIERS) / max(n_words, 1),
                "filler_abstract_ratio": _count_word_set(text_lower, FILLER_ABSTRACT) / max(n_words, 1),
                "passive_ratio": _detect_passive(doc) / max(n_sents, 1),
                "discourse_initial_ratio": n_discourse_initial / max(n_sents, 1),
                "unique_token_ratio": len(set(t.lower_ for t in doc)) / max(len(doc), 1),
            }
            sent_features.append((feat, float(cont_projs.mean()), source))

    # ── Aggregate POS/dep with source split ──
    def _agg_stats(agg_dict):
        out = {}
        for key, d in agg_dict.items():
            n = d["count"]
            if n < 5:
                continue
            mean = d["sum"] / n
            out[key] = {"mean": round(float(mean), 4), "count": int(n)}
        return out

    pos_by_source = {}
    all_pos = set(list(pos_agg["human"].keys()) + list(pos_agg["ai"].keys()))
    for pos in all_pos:
        h_stats = _agg_stats({"x": pos_agg["human"][pos]}).get("x", {})
        a_stats = _agg_stats({"x": pos_agg["ai"][pos]}).get("x", {})
        if h_stats and a_stats:
            pos_by_source[pos] = {
                "human_mean": h_stats["mean"],
                "ai_mean": a_stats["mean"],
                "delta": round(a_stats["mean"] - h_stats["mean"], 4),
                "human_count": h_stats["count"],
                "ai_count": a_stats["count"],
            }

    dep_by_source = {}
    all_dep = set(list(dep_agg["human"].keys()) + list(dep_agg["ai"].keys()))
    for dep in all_dep:
        h_stats = _agg_stats({"x": dep_agg["human"][dep]}).get("x", {})
        a_stats = _agg_stats({"x": dep_agg["ai"][dep]}).get("x", {})
        if h_stats and a_stats:
            dep_by_source[dep] = {
                "human_mean": h_stats["mean"],
                "ai_mean": a_stats["mean"],
                "delta": round(a_stats["mean"] - h_stats["mean"], 4),
                "human_count": h_stats["count"],
                "ai_count": a_stats["count"],
            }

    # ── Discourse frequency ──
    discourse_freq = {}
    for cat in DISCOURSE_MARKERS:
        h_total = marker_counts["human"]["total_words"]
        a_total = marker_counts["ai"]["total_words"]
        h_freq = discourse_counts["human"][cat] / max(h_total, 1) * 1000  # per 1000 words
        a_freq = discourse_counts["ai"][cat] / max(a_total, 1) * 1000
        discourse_freq[cat] = {
            "human_per_1k": round(h_freq, 2),
            "ai_per_1k": round(a_freq, 2),
            "ratio": round(a_freq / max(h_freq, 0.01), 2),
        }

    # ── Lexical markers ──
    lexical_markers = {}
    for mk in ["hedging", "intensifier", "filler_abstract"]:
        h_total = marker_counts["human"]["total_words"]
        a_total = marker_counts["ai"]["total_words"]
        h_rate = marker_counts["human"][mk] / max(h_total, 1) * 1000
        a_rate = marker_counts["ai"][mk] / max(a_total, 1) * 1000
        lexical_markers[mk] = {
            "human_per_1k": round(h_rate, 2),
            "ai_per_1k": round(a_rate, 2),
            "ratio": round(a_rate / max(h_rate, 0.01), 2),
        }

    # Passive voice
    passive_voice = {
        "human_per_sent": round(passive_counts["human"] / max(sent_counts["human"], 1), 3),
        "ai_per_sent": round(passive_counts["ai"] / max(sent_counts["ai"], 1), 3),
    }

    # ── Structural comparison ──
    structural_comparison = {}
    feat_names = list(sent_features[0][0].keys()) if sent_features else []
    for fn in feat_names:
        h_vals = [sf[0][fn] for sf in sent_features if sf[2] == "human"]
        a_vals = [sf[0][fn] for sf in sent_features if sf[2] == "ai"]
        if h_vals and a_vals:
            structural_comparison[fn] = {
                "human_mean": round(float(np.mean(h_vals)), 4),
                "ai_mean": round(float(np.mean(a_vals)), 4),
                "delta": round(float(np.mean(a_vals) - np.mean(h_vals)), 4),
            }

    # ── Within-source regression ──
    from sklearn.linear_model import Ridge
    from scipy.stats import pearsonr

    within_human_r2 = 0.0
    within_ai_r2 = 0.0
    between_r2 = 0.0
    feature_importance = {}

    if len(sent_features) >= 20:
        h_feats = [(sf[0], sf[1]) for sf in sent_features if sf[2] == "human"]
        a_feats = [(sf[0], sf[1]) for sf in sent_features if sf[2] == "ai"]

        feat_names_reg = [k for k in feat_names]

        def _do_regression(feats_list):
            if len(feats_list) < 10:
                return 0.0, {}
            X = np.array([[f[fn] for fn in feat_names_reg] for f, _ in feats_list])
            y = np.array([p for _, p in feats_list])
            # Standardize
            X_mean, X_std = X.mean(0), X.std(0)
            X_std[X_std < 1e-8] = 1.0
            X_norm = (X - X_mean) / X_std
            reg = Ridge(alpha=1.0).fit(X_norm, y)
            r2 = float(reg.score(X_norm, y))
            imp = {fn: float(abs(c)) for fn, c in zip(feat_names_reg, reg.coef_)}
            return r2, imp

        within_human_r2, h_imp = _do_regression(h_feats)
        within_ai_r2, a_imp = _do_regression(a_feats)

        # Between-source (expected high — this is the "trivial" baseline)
        all_X = np.array([[sf[0][fn] for fn in feat_names_reg] for sf in sent_features])
        all_y = np.array([sf[1] for sf in sent_features])
        X_mean, X_std = all_X.mean(0), all_X.std(0)
        X_std[X_std < 1e-8] = 1.0
        reg_all = Ridge(alpha=1.0).fit((all_X - X_mean) / X_std, all_y)
        between_r2 = float(reg_all.score((all_X - X_mean) / X_std, all_y))

        # Average importance across sources
        for fn in feat_names_reg:
            feature_importance[fn] = (h_imp.get(fn, 0) + a_imp.get(fn, 0)) / 2

    return StructuralAnalysisResult(
        layer=layer, method=method,
        discourse_freq=discourse_freq,
        lexical_markers=lexical_markers,
        pos_by_source=pos_by_source,
        dep_by_source=dep_by_source,
        within_human_r2=within_human_r2,
        within_ai_r2=within_ai_r2,
        between_r2=between_r2,
        feature_importance=feature_importance,
        structural_comparison=structural_comparison,
        passive_voice=passive_voice,
    )


# ═══════════════════════════════════════════════════════════
#  Full analysis pipeline
# ═══════════════════════════════════════════════════════════

def run_interpretability(
    model, tokenizer, models_dir: Path,
    samples: List[dict], layers: List[int], methods: List[str],
    device: str = "cuda", top_k: int = 50, max_samples: int = 100,
    keep_raw: bool = False,
) -> dict:
    nlp = spacy.load("en_core_web_sm")
    results = {
        "logit_lens": {},
        "token_projection": {},
        "structural": {},
    }

    for layer in layers:
        for method in methods:
            model_path = models_dir / f"{method}_layer{layer:02d}.pt"
            if not model_path.exists():
                print(f"  ⚠ Skipping {method} L{layer} (not found)")
                continue

            data = torch.load(model_path, map_location="cpu", weights_only=False)
            v_src = data["state_dict"]["v_src"].float()

            key = f"{method}_L{layer}"
            print(f"\n  ── {key} ──")

            # 1) Logit Lens (filtered)
            print(f"    Logit Lens (filtered)...")
            top_pos, top_neg, stats = logit_lens(v_src, model, tokenizer, top_k=top_k)
            results["logit_lens"][key] = LogitLensResult(
                layer=layer, method=method,
                top_positive=top_pos, top_negative=top_neg,
                logit_stats=stats,
            ).to_dict()
            print(f"      AI-ward:    {' | '.join(f'{t.strip()}({v:.2f})' for t,v in top_pos[:5])}")
            print(f"      Human-ward: {' | '.join(f'{t.strip()}({v:.2f})' for t,v in top_neg[:5])}")

            # 2) Contrastive token projection
            print(f"    Contrastive token projection ({max_samples} samples)...")
            tp = token_level_projection(
                v_src, model, tokenizer, samples, layer, method,
                max_samples=max_samples,
            )
            results["token_projection"][key] = tp.to_dict(top_k=top_k)
            top_delta = tp.contrastive_tokens[:5]
            delta_strs = [f'{d["token"]}(Δ={d["delta"]:.3f})' for d in top_delta]
            print(f"      Top Δ(AI−Human): {' | '.join(delta_strs)}")
            print(f"      Global mean: AI={tp.global_ai_mean:.3f}, Human={tp.global_human_mean:.3f}")

            # 3) Structural fingerprint
            print(f"    Structural analysis...")
            sa = structural_analysis(
                v_src, model, tokenizer, samples, layer, method,
                max_samples=max_samples, nlp=nlp,
            )
            results["structural"][key] = sa.to_dict()
            print(f"      R²: within_H={sa.within_human_r2:.4f}, within_AI={sa.within_ai_r2:.4f}, between={sa.between_r2:.4f}")
            top_imp = sorted(sa.feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            print(f"      Top features: {', '.join(f'{k}={v:.3f}' for k,v in top_imp)}")
            # Discourse markers
            for cat in ["additive", "contrastive", "causal"]:
                d = sa.discourse_freq.get(cat, {})
                print(f"      {cat}: H={d.get('human_per_1k',0):.1f}/1k, AI={d.get('ai_per_1k',0):.1f}/1k, ratio={d.get('ratio',0):.2f}x")

    return results
