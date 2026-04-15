"""
Interpretability analysis of steering vectors v_src.

Two complementary lenses:
  1. **Lexical (词汇)** — What vocabulary does v_src encode?
     - Logit Lens: project v_src through the unembedding matrix → top tokens
       that the direction promotes (AI-ward) or suppresses (Human-ward).
     - Token-level projection: on real samples, compute per-token projection
       onto v_src and aggregate by token identity → which words carry the
       strongest source signal?

  2. **Syntactic (句法)** — What structural features correlate with v_src?
     - POS-tag analysis: group token projections by POS tag → bar chart
     - Dependency-role analysis: group by dependency label
     - Feature regression: how much variance in v_src projection can
       structural features (POS distribution, tree depth, …) explain?
"""

import torch
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import spacy

# ═══════════════════════════════════════════════════════════
#  Logit Lens
# ═══════════════════════════════════════════════════════════

@dataclass
class LogitLensResult:
    """Result of projecting v_src through unembedding."""
    layer: int
    method: str
    top_positive: List[Tuple[str, float]]   # tokens promoted (→ AI)
    top_negative: List[Tuple[str, float]]   # tokens suppressed (→ Human)
    logit_stats: Dict[str, float]           # mean, std, skew of full logit vector

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "method": self.method,
            "top_positive_ai": [{"token": t, "logit": round(v, 4)} for t, v in self.top_positive],
            "top_negative_human": [{"token": t, "logit": round(v, 4)} for t, v in self.top_negative],
            "logit_stats": {k: round(v, 6) for k, v in self.logit_stats.items()},
        }


def logit_lens(
    v_src: torch.Tensor,
    model,
    tokenizer,
    top_k: int = 50,
    apply_ln: bool = True,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]], Dict[str, float]]:
    """
    Project v_src through the model's unembedding to get token-level meaning.

    Args:
        v_src:  (hidden_size,) unit steering direction
        model:  HuggingFace CausalLM
        tokenizer: corresponding tokenizer
        top_k:  how many tokens to return per side
        apply_ln: whether to apply final LayerNorm before unembedding

    Returns:
        (top_positive, top_negative, stats)
        positive = high logit = tokens the AI-direction promotes
        negative = low logit  = tokens the Human-direction promotes
    """
    v = v_src.float().to(model.device)

    # Apply final layer norm if requested (important for accurate logit lens)
    if apply_ln:
        ln = model.model.norm  # Llama-style: model.model.norm
        v = ln(v.unsqueeze(0)).squeeze(0)

    # Unembedding: logits = v @ W_unembed^T  (lm_head weight)
    W = model.lm_head.weight.float()  # (vocab_size, hidden_size)
    logits = W @ v  # (vocab_size,)

    logits_np = logits.cpu().numpy()
    stats = {
        "mean": float(logits_np.mean()),
        "std": float(logits_np.std()),
        "min": float(logits_np.min()),
        "max": float(logits_np.max()),
    }

    # Top positive (AI-ward tokens)
    top_pos_idx = np.argsort(logits_np)[::-1][:top_k]
    top_positive = []
    for idx in top_pos_idx:
        token = tokenizer.decode([int(idx)])
        top_positive.append((token, float(logits_np[idx])))

    # Top negative (Human-ward tokens)
    top_neg_idx = np.argsort(logits_np)[:top_k]
    top_negative = []
    for idx in top_neg_idx:
        token = tokenizer.decode([int(idx)])
        top_negative.append((token, float(logits_np[idx])))

    return top_positive, top_negative, stats


# ═══════════════════════════════════════════════════════════
#  Token-level projection on real samples
# ═══════════════════════════════════════════════════════════

@dataclass
class TokenProjectionResult:
    """Aggregated token-projection statistics."""
    layer: int
    method: str
    n_samples: int
    # per-token-type aggregates
    token_stats: Dict[str, Dict[str, float]]  # token_str → {mean, std, count, ...}
    # raw per-position data for visualization
    all_projections: Optional[List[dict]] = None  # [{token, proj, pos, sample_id, source}, ...]

    def top_tokens(self, k: int = 30, by: str = "mean",
                   source: str = "both") -> List[Tuple[str, float, int]]:
        """Return top-k tokens by mean projection (positive = AI-like)."""
        items = []
        for tok, s in self.token_stats.items():
            if source != "both" and s.get("primary_source", "both") != source:
                continue
            items.append((tok, s[by], int(s["count"])))
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:k]

    def to_dict(self, top_k: int = 50) -> dict:
        top_ai = self.top_tokens(top_k, by="mean")
        top_human = list(reversed(self.top_tokens(top_k * 2, by="mean")))[:top_k]
        return {
            "layer": self.layer,
            "method": self.method,
            "n_samples": self.n_samples,
            "top_ai_tokens": [{"token": t, "mean_proj": round(v, 4), "count": c}
                              for t, v, c in top_ai],
            "top_human_tokens": [{"token": t, "mean_proj": round(v, 4), "count": c}
                                 for t, v, c in top_human],
        }


@torch.no_grad()
def token_level_projection(
    v_src: torch.Tensor,
    model,
    tokenizer,
    samples: List[dict],
    layer: int,
    method: str = "mean_diff",
    max_samples: int = 100,
    keep_raw: bool = False,
) -> TokenProjectionResult:
    """
    For each sample, run a forward pass, extract per-token hidden states
    at the target layer, project onto v_src, and aggregate by token type.

    Args:
        v_src: (hidden_size,) unit steering direction
        model: HuggingFace CausalLM
        tokenizer: tokenizer
        samples: list of dataset samples (must have prefix, human_continuation,
                 ai_continuation_chat)
        layer: which layer to extract
        method: label for this method
        max_samples: cap on number of samples
        keep_raw: if True, keep all per-token projection data
    """
    v = v_src.float().to(model.device)
    token_agg = defaultdict(lambda: {"sum": 0.0, "sum_sq": 0.0, "count": 0,
                                      "human_count": 0, "ai_count": 0})
    raw_data = [] if keep_raw else None

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
            hs = outputs.hidden_states[layer][0].float()  # (seq, hidden)

            # Find continuation boundary
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
            cont_start = len(prefix_ids) - 1

            # Project continuation tokens onto v_src
            projs = (hs[cont_start:] @ v).cpu().numpy()
            cont_ids = token_ids[cont_start:].cpu().tolist()

            for pos, (tid, proj) in enumerate(zip(cont_ids, projs)):
                tok_str = tokenizer.decode([tid]).strip()
                if not tok_str:
                    continue
                entry = token_agg[tok_str]
                entry["sum"] += proj
                entry["sum_sq"] += proj ** 2
                entry["count"] += 1
                entry[f"{source}_count"] += 1

                if raw_data is not None:
                    raw_data.append({
                        "token": tok_str, "proj": float(proj),
                        "pos": pos, "sample_id": i, "source": source,
                    })

    # Compute per-token stats
    token_stats = {}
    for tok, agg in token_agg.items():
        n = agg["count"]
        if n < 3:  # require at least 3 occurrences
            continue
        mean = agg["sum"] / n
        var = agg["sum_sq"] / n - mean ** 2
        token_stats[tok] = {
            "mean": mean,
            "std": max(var, 0) ** 0.5,
            "count": n,
            "human_count": agg["human_count"],
            "ai_count": agg["ai_count"],
            "primary_source": "human" if agg["human_count"] > agg["ai_count"] else "ai",
        }

    return TokenProjectionResult(
        layer=layer, method=method,
        n_samples=min(len(samples), max_samples),
        token_stats=token_stats,
        all_projections=raw_data,
    )


# ═══════════════════════════════════════════════════════════
#  Syntactic analysis
# ═══════════════════════════════════════════════════════════

@dataclass
class SyntacticAnalysisResult:
    """Syntactic decomposition of v_src projection."""
    layer: int
    method: str
    # POS-level: {POS_tag: {mean_proj, std, count, ...}}
    pos_stats: Dict[str, Dict[str, float]]
    # Dependency-level: {dep_label: {mean_proj, std, count, ...}}
    dep_stats: Dict[str, Dict[str, float]]
    # Sentence-level features → projection correlation
    sentence_correlations: Dict[str, float]  # {feature_name: pearson_r}
    # R² from syntactic feature regression
    syntactic_r_squared: float
    # per-source POS distribution comparison
    pos_distribution: Dict[str, Dict[str, float]]  # {source: {POS: fraction}}

    def to_dict(self) -> dict:
        def top_by_mean(d, k=20):
            items = sorted(d.items(), key=lambda x: abs(x[1]["mean"]), reverse=True)
            return {tag: {k_: round(v_, 4) for k_, v_ in stats.items()}
                    for tag, stats in items[:k]}
        return {
            "layer": self.layer,
            "method": self.method,
            "pos_stats": top_by_mean(self.pos_stats),
            "dep_stats": top_by_mean(self.dep_stats),
            "sentence_correlations": {k: round(v, 4) for k, v in self.sentence_correlations.items()},
            "syntactic_r_squared": round(self.syntactic_r_squared, 4),
            "pos_distribution": self.pos_distribution,
        }


def _aggregate_stats(agg_dict: dict) -> Dict[str, Dict[str, float]]:
    """Convert running sums to mean/std stats."""
    result = {}
    for key, agg in agg_dict.items():
        n = agg["count"]
        if n < 3:
            continue
        mean = agg["sum"] / n
        var = agg["sum_sq"] / n - mean ** 2
        result[key] = {
            "mean": mean,
            "std": max(var, 0) ** 0.5,
            "count": n,
        }
    return result


@torch.no_grad()
def syntactic_analysis(
    v_src: torch.Tensor,
    model,
    tokenizer,
    samples: List[dict],
    layer: int,
    method: str = "mean_diff",
    max_samples: int = 100,
    nlp=None,
) -> SyntacticAnalysisResult:
    """
    Analyze how v_src projection correlates with syntactic features.

    For each continuation token, we:
      1. Get its projection onto v_src
      2. Map it to spaCy POS tag and dependency label
      3. Aggregate projections by POS / dep
      4. Compute sentence-level feature correlations

    Args:
        v_src: (hidden_size,) unit steering direction
        model: HuggingFace CausalLM
        tokenizer: tokenizer
        samples: dataset samples
        layer: target layer
        method: label
        max_samples: cap
        nlp: spaCy language model (loaded if None)
    """
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")

    v = v_src.float().to(model.device)

    pos_agg = defaultdict(lambda: {"sum": 0.0, "sum_sq": 0.0, "count": 0})
    dep_agg = defaultdict(lambda: {"sum": 0.0, "sum_sq": 0.0, "count": 0})
    pos_dist = {"human": defaultdict(int), "ai": defaultdict(int)}

    # Sentence-level features
    sent_features = []  # list of (feature_dict, mean_proj, source)

    for i, sample in enumerate(samples[:max_samples]):
        prefix = sample["prefix"]
        for source, cont_key in [("human", "human_continuation"),
                                  ("ai", "ai_continuation_chat")]:
            continuation = sample.get(cont_key, "")
            if not continuation:
                continue
            full_text = prefix + continuation

            # ── Model forward pass ──
            inputs = tokenizer(full_text, return_tensors="pt",
                               truncation=True, max_length=2048)
            inputs = {k: v_.to(model.device) for k, v_ in inputs.items()}
            token_ids = inputs["input_ids"][0]

            outputs = model(**inputs, output_hidden_states=True)
            hs = outputs.hidden_states[layer][0].float()

            prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
            cont_start = len(prefix_ids) - 1
            cont_token_ids = token_ids[cont_start:].cpu().tolist()
            cont_projs = (hs[cont_start:] @ v).cpu().numpy()

            # ── spaCy analysis on continuation text ──
            doc = nlp(continuation)

            # Build a mapping: character offset → (POS, dep)
            char_to_syntax = {}
            for token in doc:
                for ci in range(token.idx, token.idx + len(token.text)):
                    char_to_syntax[ci] = (token.pos_, token.dep_)

            # Map each subword token to its spaCy POS/dep
            # Reconstruct character positions from tokenizer
            cont_text_from_tokens = ""
            token_spans = []
            for tid in cont_token_ids:
                tok_str = tokenizer.decode([tid])
                start = len(cont_text_from_tokens)
                cont_text_from_tokens += tok_str
                end = len(cont_text_from_tokens)
                token_spans.append((start, end))

            for j, ((start, end), proj) in enumerate(zip(token_spans, cont_projs)):
                # Find the dominant POS/dep for this subword span
                mid = (start + end) // 2
                pos_tag, dep_label = char_to_syntax.get(mid, ("X", "unknown"))

                pos_agg[pos_tag]["sum"] += proj
                pos_agg[pos_tag]["sum_sq"] += proj ** 2
                pos_agg[pos_tag]["count"] += 1
                dep_agg[dep_label]["sum"] += proj
                dep_agg[dep_label]["sum_sq"] += proj ** 2
                dep_agg[dep_label]["count"] += 1

                pos_dist[source][pos_tag] += 1

            # Sentence-level features
            n_sents = len(list(doc.sents))
            tree_depths = []
            for token in doc:
                depth = 0
                t = token
                while t.head != t:
                    depth += 1
                    t = t.head
                tree_depths.append(depth)

            feat = {
                "n_tokens": len(doc),
                "n_sents": n_sents,
                "mean_sent_len": len(doc) / max(n_sents, 1),
                "mean_tree_depth": np.mean(tree_depths) if tree_depths else 0,
                "max_tree_depth": max(tree_depths) if tree_depths else 0,
                "noun_ratio": sum(1 for t in doc if t.pos_ == "NOUN") / max(len(doc), 1),
                "verb_ratio": sum(1 for t in doc if t.pos_ == "VERB") / max(len(doc), 1),
                "adj_ratio": sum(1 for t in doc if t.pos_ == "ADJ") / max(len(doc), 1),
                "adv_ratio": sum(1 for t in doc if t.pos_ == "ADV") / max(len(doc), 1),
                "punct_ratio": sum(1 for t in doc if t.pos_ == "PUNCT") / max(len(doc), 1),
                "unique_token_ratio": len(set(t.lower_ for t in doc)) / max(len(doc), 1),
                "source_label": 1.0 if source == "ai" else 0.0,
            }
            sent_features.append((feat, float(cont_projs.mean()), source))

    # ── Aggregate stats ──
    pos_stats = _aggregate_stats(pos_agg)
    dep_stats = _aggregate_stats(dep_agg)

    # Normalize POS distributions per source
    pos_distribution = {}
    for source in ["human", "ai"]:
        total = sum(pos_dist[source].values())
        pos_distribution[source] = {
            pos: round(count / max(total, 1), 4)
            for pos, count in sorted(pos_dist[source].items(),
                                     key=lambda x: -x[1])
        }

    # ── Sentence-level correlations ──
    from scipy.stats import pearsonr
    from sklearn.linear_model import LinearRegression

    if len(sent_features) >= 10:
        feat_names = [k for k in sent_features[0][0].keys() if k != "source_label"]
        projs = np.array([sf[1] for sf in sent_features])
        X = np.array([[sf[0][fn] for fn in feat_names] for sf in sent_features])

        correlations = {}
        for fi, fn in enumerate(feat_names):
            r, p = pearsonr(X[:, fi], projs)
            correlations[fn] = float(r)

        # R² from linear regression
        reg = LinearRegression().fit(X, projs)
        r_sq = float(reg.score(X, projs))
    else:
        correlations = {}
        r_sq = 0.0

    return SyntacticAnalysisResult(
        layer=layer,
        method=method,
        pos_stats=pos_stats,
        dep_stats=dep_stats,
        sentence_correlations=correlations,
        syntactic_r_squared=r_sq,
        pos_distribution=pos_distribution,
    )


# ═══════════════════════════════════════════════════════════
#  Full analysis pipeline
# ═══════════════════════════════════════════════════════════

def run_interpretability(
    model,
    tokenizer,
    models_dir: Path,
    samples: List[dict],
    layers: List[int],
    methods: List[str],
    device: str = "cuda",
    top_k: int = 50,
    max_samples: int = 100,
    keep_raw: bool = False,
) -> dict:
    """
    Run full lexical + syntactic interpretability analysis.

    Returns a dict with all results, ready for JSON serialization and plotting.
    """
    nlp = spacy.load("en_core_web_sm")
    results = {
        "logit_lens": {},
        "token_projection": {},
        "syntactic": {},
    }

    for layer in layers:
        for method in methods:
            model_path = models_dir / f"{method}_layer{layer:02d}.pt"
            if not model_path.exists():
                print(f"  ⚠ Skipping {method} L{layer} (not found)")
                continue

            data = torch.load(model_path, map_location="cpu")
            v_src = data["state_dict"]["v_src"].float()

            key = f"{method}_L{layer}"
            print(f"\n  ── {key} ──")

            # 1) Logit Lens
            print(f"    Logit Lens...")
            top_pos, top_neg, stats = logit_lens(v_src, model, tokenizer, top_k=top_k)
            results["logit_lens"][key] = LogitLensResult(
                layer=layer, method=method,
                top_positive=top_pos, top_negative=top_neg,
                logit_stats=stats,
            ).to_dict()
            print(f"      AI-ward:    {' | '.join(f'{t}({v:.2f})' for t,v in top_pos[:5])}")
            print(f"      Human-ward: {' | '.join(f'{t}({v:.2f})' for t,v in top_neg[:5])}")

            # 2) Token-level projection
            print(f"    Token projection ({max_samples} samples)...")
            tp = token_level_projection(
                v_src, model, tokenizer, samples, layer, method,
                max_samples=max_samples, keep_raw=keep_raw,
            )
            results["token_projection"][key] = tp.to_dict(top_k=top_k)
            top_ai = tp.top_tokens(5, by="mean")
            top_hum = list(reversed(tp.top_tokens(100, by="mean")))[:5]
            print(f"      AI tokens:    {' | '.join(f'{t}({v:.3f},n={c})' for t,v,c in top_ai)}")
            print(f"      Human tokens: {' | '.join(f'{t}({v:.3f},n={c})' for t,v,c in top_hum)}")

            # 3) Syntactic analysis
            print(f"    Syntactic analysis...")
            sa = syntactic_analysis(
                v_src, model, tokenizer, samples, layer, method,
                max_samples=max_samples, nlp=nlp,
            )
            results["syntactic"][key] = sa.to_dict()
            print(f"      R²(syntax→proj) = {sa.syntactic_r_squared:.4f}")
            top_corr = sorted(sa.sentence_correlations.items(),
                              key=lambda x: abs(x[1]), reverse=True)[:3]
            print(f"      Top correlations: {', '.join(f'{k}={v:.3f}' for k,v in top_corr)}")

    return results
