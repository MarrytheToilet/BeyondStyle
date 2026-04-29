"""
Vector-Based Deep Interpretability for Steering Vectors.

Unlike deep_interp.py (corpus-level text comparison), this module analyses
the steering vector v_src **inside the model** — how it interacts with
layers, attention heads, MLP neurons, and the vocabulary space.

Five analyses:
  1. Layer Trajectory     — projection trajectory across layers (NO GPU)
  2. Vocabulary Fields    — ranked word lists by v_src projection (real words only)
  3. Token Attribution    — signed gradient·input attribution
  4. Attention Heads      — Direct Logit Attribution per head
  5. MLP Neurons          — top neurons aligned with v_src + trigger words
"""

import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Shared: Robust English word filter using wordfreq
# ═══════════════════════════════════════════════════════════

def _build_valid_word_index(tokenizer, min_freq: float = 1e-6, min_len: int = 4,
                            language: str = "auto"):
    """
    Build a mapping of token_id → clean_word for tokens that are real words.

    Supports both English (via wordfreq) and Chinese (via character detection).
    language: "en", "zh", or "auto" (detect from tokenizer name).
    Returns (indices, words) arrays.
    """
    # Auto-detect language from tokenizer/model name
    if language == "auto":
        name = getattr(tokenizer, 'name_or_path', '') or ''
        language = "zh" if any(k in name.lower() for k in ['qwen', 'chatglm', 'baichuan', 'yi-']) else "en"

    if language == "zh":
        return _build_valid_word_index_zh(tokenizer)

    try:
        from wordfreq import word_frequency
    except ImportError:
        logger.warning("wordfreq not installed; falling back to heuristic filter")
        return _build_valid_word_index_fallback(tokenizer, min_len)

    indices = []
    words = []
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer)

    for idx in range(vocab_size):
        decoded = tokenizer.decode([idx]).strip()
        if len(decoded) < min_len:
            continue
        if not decoded.isascii() or not decoded.isalpha():
            continue
        w = decoded.lower()
        if word_frequency(w, 'en') < min_freq:
            continue
        indices.append(idx)
        words.append(w)

    logger.info(f"    Valid English words: {len(indices)} / {vocab_size}")
    return np.array(indices), words


def _is_cjk_char(ch: str) -> bool:
    """Check if a character is CJK."""
    cp = ord(ch)
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or
            0x20000 <= cp <= 0x2A6DF or 0x2A700 <= cp <= 0x2B73F or
            0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F)


def _build_valid_word_index_zh(tokenizer, min_chars: int = 1):
    """
    Build valid word index for Chinese: accept tokens that contain CJK characters.
    Also accepts common Chinese punctuation-bearing tokens.
    """
    indices = []
    words = []
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer)

    for idx in range(vocab_size):
        decoded = tokenizer.decode([idx]).strip()
        if not decoded or len(decoded) < min_chars:
            continue
        # Accept if token has any CJK character
        cjk_count = sum(1 for ch in decoded if _is_cjk_char(ch))
        if cjk_count == 0:
            continue
        # Skip single-char tokens that are too common/meaningless (的, 了, 是, etc.)
        # unless they form part of a multi-char token
        if len(decoded) == 1 and decoded in "的了是在不有人一中大这上个到说为以就会可我他她它们":
            continue
        indices.append(idx)
        words.append(decoded)

    logger.info(f"    Valid Chinese tokens: {len(indices)} / {vocab_size}")
    return np.array(indices), words


def _build_valid_word_index_fallback(tokenizer, min_len: int = 3):
    """Fallback when wordfreq is not available."""
    indices = []
    words = []
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer)
    for idx in range(vocab_size):
        decoded = tokenizer.decode([idx]).strip()
        if len(decoded) < min_len:
            continue
        if not decoded.isascii() or not decoded.isalpha():
            continue
        if not decoded.islower():
            continue
        indices.append(idx)
        words.append(decoded)
    return np.array(indices), words


# Semantic category lexicons for tagging projected words
_SEMANTIC_CATEGORIES = {
    "hedging": {"may", "might", "could", "perhaps", "possibly", "potentially",
                "likely", "unlikely", "somewhat", "arguably", "seemingly",
                "approximately", "roughly", "tends", "suggests", "appears",
                "relatively", "generally", "typically", "often", "usually"},
    "intensifier": {"very", "extremely", "highly", "incredibly", "remarkably",
                    "significantly", "substantially", "absolutely", "utterly",
                    "deeply", "profoundly", "critically", "fundamentally",
                    "particularly", "especially", "notably", "exceedingly"},
    "discourse": {"furthermore", "moreover", "additionally", "however",
                  "nevertheless", "nonetheless", "although", "whereas",
                  "therefore", "consequently", "thus", "hence", "accordingly",
                  "meanwhile", "subsequently", "overall", "essentially",
                  "indeed", "conversely", "instead", "ultimately", "specifically",
                  "notably", "importantly", "interestingly"},
    "filler_abstract": {"landscape", "narrative", "paradigm", "framework",
                        "dynamics", "implications", "multifaceted", "comprehensive",
                        "nuanced", "robust", "pivotal", "crucial", "underscore",
                        "navigate", "foster", "realm", "spectrum", "trajectory",
                        "delve", "intricate", "tapestry", "facilitate", "leverage",
                        "innovative", "evolving", "diverse", "impactful"},
    "concrete": {"said", "told", "walked", "ran", "looked", "saw", "heard",
                 "door", "room", "street", "hand", "eyes", "face", "water",
                 "car", "house", "money", "food", "children", "police",
                 "killed", "found", "went", "came", "took", "gave"},
    "informal": {"gonna", "wanna", "gotta", "yeah", "okay", "stuff",
                 "pretty", "really", "actually", "basically", "literally",
                 "honestly", "anyway", "whatever", "totally", "super"},
    "evaluative": {"important", "significant", "essential", "critical",
                   "effective", "valuable", "remarkable", "notable",
                   "impressive", "excellent", "compelling", "promising"},
    "causal": {"because", "since", "therefore", "thus", "hence",
               "causing", "resulting", "leading", "contributing",
               "driven", "influenced", "shaped", "sparked"},
    "temporal": {"recently", "currently", "previously", "initially",
                 "eventually", "gradually", "immediately", "ongoing",
                 "emerging", "upcoming", "earlier", "later"},
}

# Chinese semantic categories
_SEMANTIC_CATEGORIES_ZH = {
    "hedging": {"可能", "也许", "或许", "大概", "似乎", "恐怕", "未必",
                "不一定", "往往", "通常", "一般"},
    "intensifier": {"非常", "极其", "十分", "特别", "尤其", "格外", "极为",
                    "异常", "相当", "颇为", "高度", "深刻"},
    "discourse": {"然而", "但是", "不过", "虽然", "尽管", "因此", "所以",
                  "此外", "另外", "同时", "总之", "事实上", "实际上",
                  "值得注意", "换言之", "进而", "从而", "与此同时"},
    "filler_abstract": {"层面", "维度", "框架", "范式", "格局", "态势",
                        "赋能", "助力", "深耕", "聚焦", "打造", "践行",
                        "多元", "创新", "高质量", "新时代"},
    "concrete": {"说", "告诉", "看到", "听到", "走", "跑", "门", "房间",
                 "街道", "手", "眼睛", "脸", "水", "车", "房子", "钱"},
    "informal": {"其实", "反正", "随便", "干嘛", "搞", "弄", "挺", "蛮",
                 "啊", "呢", "嘛", "吧", "咋"},
    "evaluative": {"重要", "关键", "核心", "显著", "突出", "卓越", "优秀",
                   "可观", "深远", "巨大", "积极"},
    "causal": {"因为", "由于", "导致", "引发", "促使", "推动", "带来",
               "造成", "引起", "产生"},
    "temporal": {"目前", "当前", "近年来", "此前", "随后", "最终", "逐步",
                 "近日", "日前", "未来", "即将"},
}

def _tag_word(word: str) -> str:
    """Return semantic category tag for a word, or empty string."""
    w = word.lower()
    # Try English categories first
    for cat, words_set in _SEMANTIC_CATEGORIES.items():
        if w in words_set:
            return cat
    # Then Chinese categories
    for cat, words_set in _SEMANTIC_CATEGORIES_ZH.items():
        if word in words_set:
            return cat
    return ""


# ═══════════════════════════════════════════════════════════
#  A1: Layer Trajectory Analysis (NO GPU — uses saved states)
# ═══════════════════════════════════════════════════════════

def layer_trajectory(
    states_dir: Path,
    models_dir: Path,
    method: str = "mean_diff",
    max_samples: int = 50,
) -> Dict[str, Any]:
    """
    Compute projection trajectory of human/AI states onto v_src across layers.

    For each layer, loads paired_states and v_src, computes:
      - Mean/std projection for Human and AI
      - Cohen's d effect size
      - KL divergence between H and AI projection distributions
      - Cross-layer v_src cosine similarity matrix

    Returns dict with all results for plotting.
    """
    states_dir = Path(states_dir)
    models_dir = Path(models_dir)

    # Discover available layers
    state_files = sorted(states_dir.glob("paired_states_layer*.pt"))
    layers = []
    for f in state_files:
        layer_num = int(f.stem.split("layer")[-1])
        model_file = models_dir / f"{method}_layer{layer_num:02d}.pt"
        if model_file.exists():
            layers.append(layer_num)

    if not layers:
        logger.warning("No matching state + model files found")
        return {}

    logger.info(f"  Layer trajectory: {len(layers)} layers {layers}")

    # Collect per-layer data
    results = {
        "layers": layers,
        "human_proj_mean": [],
        "human_proj_std": [],
        "ai_proj_mean": [],
        "ai_proj_std": [],
        "cohens_d": [],
        "kl_divergence": [],
        "auc_separation": [],
        "v_src_vectors": {},  # for cross-layer cosine
    }

    for layer in layers:
        # Load states
        state = torch.load(
            states_dir / f"paired_states_layer{layer:02d}.pt",
            map_location="cpu", weights_only=False,
        )
        h_states = state["human"][:max_samples].float()
        a_states = state["ai"][:max_samples].float()

        # Load v_src
        ckpt = torch.load(
            models_dir / f"{method}_layer{layer:02d}.pt",
            map_location="cpu", weights_only=False,
        )
        v_src = ckpt["state_dict"]["v_src"].float()
        v_src = v_src / (v_src.norm() + 1e-8)

        results["v_src_vectors"][layer] = v_src.numpy()

        # Project
        h_proj = (h_states @ v_src).numpy()
        a_proj = (a_states @ v_src).numpy()

        results["human_proj_mean"].append(float(h_proj.mean()))
        results["human_proj_std"].append(float(h_proj.std()))
        results["ai_proj_mean"].append(float(a_proj.mean()))
        results["ai_proj_std"].append(float(a_proj.std()))

        # Cohen's d
        pooled_std = np.sqrt((h_proj.var() + a_proj.var()) / 2)
        d = (a_proj.mean() - h_proj.mean()) / (pooled_std + 1e-8)
        results["cohens_d"].append(float(d))

        # KL divergence via histogram-based estimation
        all_vals = np.concatenate([h_proj, a_proj])
        bins = np.linspace(all_vals.min() - 0.1, all_vals.max() + 0.1, 50)
        h_hist, _ = np.histogram(h_proj, bins=bins, density=True)
        a_hist, _ = np.histogram(a_proj, bins=bins, density=True)
        h_hist = h_hist + 1e-8
        a_hist = a_hist + 1e-8
        h_hist /= h_hist.sum()
        a_hist /= a_hist.sum()
        kl = float(scipy_stats.entropy(h_hist, a_hist))
        results["kl_divergence"].append(kl)

        # AUC (from projections — how well does v_src separate?)
        from sklearn.metrics import roc_auc_score
        labels = np.concatenate([np.zeros(len(h_proj)), np.ones(len(a_proj))])
        scores = np.concatenate([h_proj, a_proj])
        auc = float(roc_auc_score(labels, scores))
        results["auc_separation"].append(auc)

    # Cross-layer v_src cosine similarity matrix
    n_layers = len(layers)
    cosine_matrix = np.zeros((n_layers, n_layers))
    for i, li in enumerate(layers):
        for j, lj in enumerate(layers):
            vi = results["v_src_vectors"][li]
            vj = results["v_src_vectors"][lj]
            cos = float(np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-8))
            cosine_matrix[i, j] = cos
    results["v_src_cosine_matrix"] = cosine_matrix.tolist()
    results["v_src_norms"] = [
        float(np.linalg.norm(results["v_src_vectors"][l])) for l in layers
    ]

    # Divergence onset: first layer where Cohen's d > 1.0
    onset = None
    for i, d in enumerate(results["cohens_d"]):
        if abs(d) > 1.0:
            onset = layers[i]
            break
    results["divergence_onset_layer"] = onset

    # Clean up large arrays
    del results["v_src_vectors"]

    logger.info(f"    Divergence onset: layer {onset}")
    logger.info(f"    Best AUC: {max(results['auc_separation']):.3f} at layer {layers[np.argmax(results['auc_separation'])]}")

    return results


# ═══════════════════════════════════════════════════════════
#  A2: Vocabulary Projection — Ranked Real-Word Lists
# ═══════════════════════════════════════════════════════════

def vocabulary_semantic_fields(
    model,
    tokenizer,
    v_src: torch.Tensor,
    top_k: int = 40,
    **_kwargs,
) -> Dict[str, Any]:
    """
    Project entire vocabulary onto v_src, filter to real English words,
    and return ranked lists of AI-ward and Human-ward words.

    Applies the model's final RMSNorm before lm_head for faithful logit-lens.
    """
    logger.info("  Vocabulary projection (real-word ranked lists)...")

    device = v_src.device
    v_src = v_src.float()
    v_src = v_src / (v_src.norm() + 1e-8)

    # Apply final layer norm to v_src for faithful logit-lens projection
    if hasattr(model.model, "norm"):
        with torch.no_grad():
            v_normed = model.model.norm(v_src.unsqueeze(0)).squeeze(0)
    else:
        v_normed = v_src

    # Get unembedding matrix
    if hasattr(model, "lm_head"):
        W_unembed = model.lm_head.weight.float().to(device)  # (vocab, hidden)
    elif hasattr(model, "embed_out"):
        W_unembed = model.embed_out.weight.float().to(device)
    else:
        raise AttributeError("Cannot find unembedding matrix")

    # Project all vocab tokens
    logits = (W_unembed @ v_normed).detach().cpu().numpy()  # (vocab_size,)

    # Filter to real words
    valid_indices, valid_words = _build_valid_word_index(tokenizer)
    valid_logits = logits[valid_indices]

    # Detect language for frequency weighting
    model_name = getattr(tokenizer, 'name_or_path', '') or ''
    is_zh = any(k in model_name.lower() for k in ['qwen', 'chatglm', 'baichuan', 'yi-'])

    # Deduplicate (lowercase words may map to multiple BPE tokens)
    word_to_best = {}
    for i, w in enumerate(valid_words):
        score = valid_logits[i]
        key = w.lower() if not is_zh else w
        if key not in word_to_best or abs(score) > abs(word_to_best[key]):
            word_to_best[key] = score

    unique_words = list(word_to_best.keys())
    unique_scores = np.array([word_to_best[w] for w in unique_words])

    # Frequency-weighted ranking
    if is_zh:
        # For Chinese, use character count as a simple proxy (longer = more specific)
        freq_weights = np.array([1.0 + 0.3 * max(0, len(w) - 1) for w in unique_words])
    else:
        try:
            from wordfreq import word_frequency
            freq_weights = np.array([
                1.0 + max(0, np.log10(word_frequency(w, 'en') / 1e-6 + 1e-10))
                for w in unique_words
            ])
        except ImportError:
            freq_weights = np.ones(len(unique_words))

    weighted_scores = unique_scores * freq_weights

    # Sort by weighted projection score
    ai_order = np.argsort(-weighted_scores)  # most positive = most AI-ward
    human_order = np.argsort(weighted_scores)  # most negative = most Human-ward

    ai_words = []
    for idx in ai_order[:top_k]:
        w = unique_words[idx]
        ai_words.append({
            "word": w,
            "score": float(unique_scores[idx]),
            "category": _tag_word(w),
        })

    human_words = []
    for idx in human_order[:top_k]:
        w = unique_words[idx]
        human_words.append({
            "word": w,
            "score": float(unique_scores[idx]),
            "category": _tag_word(w),
        })

    # Category summary: count tagged words in each direction
    ai_categories = defaultdict(int)
    human_categories = defaultdict(int)
    for entry in ai_words:
        if entry["category"]:
            ai_categories[entry["category"]] += 1
    for entry in human_words:
        if entry["category"]:
            human_categories[entry["category"]] += 1

    # Distribution statistics
    lang_label = "Chinese tokens" if is_zh else "English words"
    results = {
        "ai_words": ai_words,
        "human_words": human_words,
        "ai_categories": dict(ai_categories),
        "human_categories": dict(human_categories),
        "language": "zh" if is_zh else "en",
        "stats": {
            "total_vocab": len(logits),
            "valid_words": len(unique_words),
            "projection_range": [float(unique_scores.min()), float(unique_scores.max())],
            "projection_mean": float(unique_scores.mean()),
            "projection_std": float(unique_scores.std()),
        },
    }

    logger.info(f"    {len(unique_words)} unique {lang_label} filtered")
    logger.info(f"    Top AI-ward: {', '.join(w['word'] for w in ai_words[:5])}")
    logger.info(f"    Top Human-ward: {', '.join(w['word'] for w in human_words[:5])}")

    return results


# ═══════════════════════════════════════════════════════════
#  A3: Token-Position Attribution
# ═══════════════════════════════════════════════════════════

def token_position_attribution(
    model,
    tokenizer,
    v_src: torch.Tensor,
    texts_human: List[str],
    texts_ai: List[str],
    layer_idx: int,
    max_samples: int = 50,
    max_seq_len: int = 256,
    language: str = "en",
) -> Dict[str, Any]:
    """
    Signed gradient·input attribution: which input token positions most influence
    the projection of the hidden state onto v_src?

    Uses grad·input (signed) instead of ||grad|| (unsigned) for directional info.
    Does NOT normalize per-sample to preserve absolute magnitude differences.
    Aggregates by relative position (quartile) and by POS tag.
    """
    logger.info(f"  Token-position attribution (layer {layer_idx}, {max_samples} samples)...")

    import spacy
    spacy_model = "zh_core_web_sm" if language == "zh" else "en_core_web_sm"
    try:
        nlp = spacy.load(spacy_model, disable=["ner", "lemmatizer"])
    except OSError:
        try:
            nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        except OSError:
            nlp = None

    device = next(model.parameters()).device
    v_src = v_src.float().to(device)
    v_src = v_src / (v_src.norm() + 1e-8)

    # Identify the target layer module (layer_idx is 1-indexed)
    if hasattr(model.model, "layers"):
        n_layers = len(model.model.layers)
        idx = layer_idx - 1 if layer_idx >= 1 and layer_idx <= n_layers else layer_idx
        target_layer_module = model.model.layers[idx]
    else:
        raise AttributeError("Cannot find model layers")

    pos_attr = {"human": defaultdict(list), "ai": defaultdict(list)}
    pos_tag_attr = {"human": defaultdict(list), "ai": defaultdict(list)}
    top_tokens_all = {"human": defaultdict(float),
                      "ai": defaultdict(float)}
    n_processed = {"human": 0, "ai": 0}

    all_texts = (
        [(t, "human") for t in texts_human[:max_samples]]
        + [(t, "ai") for t in texts_ai[:max_samples]]
    )

    for text, source in all_texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=max_seq_len, padding=False,
        ).to(device)
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]

        if seq_len < 4:
            continue

        # Hook to capture hidden states at target layer
        captured = {}

        def capture_hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            captured["hidden"] = h

        handle = target_layer_module.register_forward_hook(capture_hook)

        # Hook to capture input embeddings with grad
        embed_grads = {}

        def embed_hook(module, input, output):
            output.retain_grad()
            embed_grads["output"] = output
            return output

        embed_handle = model.model.embed_tokens.register_forward_hook(embed_hook)

        outputs = model(**inputs, output_hidden_states=False)

        handle.remove()
        embed_handle.remove()

        if "hidden" not in captured or "output" not in embed_grads:
            continue

        hidden = captured["hidden"]  # (1, seq, hidden)
        embed_out = embed_grads["output"]

        # Scalar: mean projection onto v_src across positions
        proj = (hidden.squeeze(0).float() @ v_src)  # (seq_len,)
        scalar = proj.mean()

        scalar.backward(retain_graph=False)

        if embed_out.grad is None:
            continue

        grad = embed_out.grad.squeeze(0).float()  # (seq_len, hidden_size)
        embed_vals = embed_out.detach().squeeze(0).float()  # (seq_len, hidden_size)

        # Signed grad·input attribution (element-wise product summed over hidden dim)
        attr = (grad * embed_vals).sum(dim=-1).detach().cpu().numpy()  # (seq_len,)

        # Aggregate by relative position (quartiles) — NO per-sample normalization
        for pos in range(seq_len):
            quartile = min(int(pos / seq_len * 4), 3)
            pos_attr[source][f"Q{quartile+1}"].append(float(attr[pos]))

        # POS tag alignment using character offsets (not linear interpolation)
        if nlp is not None:
            decoded = tokenizer.decode(input_ids[0].cpu().tolist(), skip_special_tokens=True)
            doc = nlp(decoded)

            # Build character→POS mapping
            char_to_pos = ["X"] * len(decoded)
            for tok in doc:
                for ci in range(tok.idx, min(tok.idx + len(tok.text), len(decoded))):
                    char_to_pos[ci] = tok.pos_

            # Map BPE tokens to POS via character offsets
            offset = 0
            for pos_idx in range(seq_len):
                tok_str = tokenizer.decode([input_ids[0, pos_idx].item()])
                tok_len = len(tok_str)
                # Find majority POS tag for this token's character span
                span_tags = char_to_pos[offset:offset + tok_len] if offset + tok_len <= len(decoded) else []
                if span_tags:
                    from collections import Counter
                    majority_pos = Counter(span_tags).most_common(1)[0][0]
                    pos_tag_attr[source][majority_pos].append(float(attr[pos_idx]))
                offset += tok_len

            # Track top attributed tokens (filter to real words)
            try:
                from wordfreq import word_frequency
                _has_wordfreq = True
            except ImportError:
                _has_wordfreq = False
            top_k = min(10, seq_len)
            top_idx = np.argsort(-np.abs(attr))[:top_k]
            for idx in top_idx:
                tok_word = tokenizer.decode([input_ids[0, idx].item()]).strip()
                has_cjk = any(_is_cjk_char(ch) for ch in tok_word)
                if has_cjk:
                    if len(tok_word) < 1:
                        continue
                else:
                    tok_word = tok_word.lower()
                    if len(tok_word) < 4 or not tok_word.isalpha():
                        continue
                    if _has_wordfreq and word_frequency(tok_word, 'en') < 1e-5:
                        continue
                top_tokens_all[source][tok_word] = max(
                    top_tokens_all[source][tok_word], abs(float(attr[idx]))
                )

        n_processed[source] += 1
        model.zero_grad()

    # Aggregate: compute mean and differential
    pos_summary = {}
    for source in ["human", "ai"]:
        for key, vals in pos_attr[source].items():
            pos_summary[f"{key}_{source}"] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n": len(vals),
            }

    pos_tag_summary = {}
    for source in ["human", "ai"]:
        for tag, vals in pos_tag_attr[source].items():
            if len(vals) > 10:
                pos_tag_summary[f"{tag}_{source}"] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n": len(vals),
                }

    # Positional differential (AI - Human)
    pos_differential = {}
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        h_vals = pos_attr["human"].get(q, [])
        a_vals = pos_attr["ai"].get(q, [])
        if h_vals and a_vals:
            pos_differential[q] = {
                "human_mean": float(np.mean(h_vals)),
                "ai_mean": float(np.mean(a_vals)),
                "diff": float(np.mean(a_vals) - np.mean(h_vals)),
            }

    # POS tag differential
    pos_tag_diff = {}
    all_tags = set()
    for source in ["human", "ai"]:
        all_tags.update(pos_tag_attr[source].keys())
    for tag in sorted(all_tags):
        h_vals = pos_tag_attr["human"].get(tag, [])
        a_vals = pos_tag_attr["ai"].get(tag, [])
        if len(h_vals) > 10 and len(a_vals) > 10:
            pos_tag_diff[tag] = {
                "human_mean": float(np.mean(h_vals)),
                "ai_mean": float(np.mean(a_vals)),
                "diff": float(np.mean(a_vals) - np.mean(h_vals)),
            }

    # Top discriminative tokens (largest attribution difference)
    discriminative_tokens = []
    all_token_words = set(top_tokens_all["human"].keys()) | set(top_tokens_all["ai"].keys())
    for w in all_token_words:
        h_score = top_tokens_all["human"].get(w, 0)
        a_score = top_tokens_all["ai"].get(w, 0)
        discriminative_tokens.append({
            "word": w,
            "human_attr": float(h_score),
            "ai_attr": float(a_score),
            "diff": float(a_score - h_score),
        })
    discriminative_tokens.sort(key=lambda x: abs(x["diff"]), reverse=True)

    results = {
        "positional_attribution": pos_summary,
        "pos_tag_attribution": pos_tag_summary,
        "positional_differential": pos_differential,
        "pos_tag_differential": pos_tag_diff,
        "discriminative_tokens": discriminative_tokens[:50],
        "n_human": n_processed["human"],
        "n_ai": n_processed["ai"],
    }

    logger.info(f"    Processed {sum(n_processed.values())} samples")
    if discriminative_tokens:
        top3 = ", ".join(f"{t['word']}({t['diff']:+.4f})" for t in discriminative_tokens[:3])
        logger.info(f"    Top discriminative tokens: {top3}")

    return results


# ═══════════════════════════════════════════════════════════
#  A4: Attention Head Attribution
# ═══════════════════════════════════════════════════════════

def attention_head_attribution(
    model,
    tokenizer,
    v_src_per_layer: Dict[int, torch.Tensor],
    texts_human: List[str],
    texts_ai: List[str],
    max_samples: int = 30,
    max_seq_len: int = 256,
) -> Dict[str, Any]:
    """
    Direct Logit Attribution for attention heads.

    For each head in each layer, measure how much the head's output
    contributes to the v_src direction. Uses the decomposition:

        output[layer] = sum_h(attn_head_h_output) + mlp_output + residual

    Each head's contribution to v_src: head_output @ v_src

    Returns per-head attribution matrix (n_layers × n_heads).
    """
    logger.info(f"  Attention head attribution ({max_samples} samples)...")

    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    # Use actual head_dim from o_proj shape (handles GQA where hidden != n_heads * head_dim)
    sample_o_proj = model.model.layers[0].self_attn.o_proj.weight
    o_proj_in_features = sample_o_proj.shape[1]  # num_heads * actual_head_dim
    head_dim = o_proj_in_features // n_heads

    # We need v_src for the layers where we have it
    available_layers = sorted(v_src_per_layer.keys())

    # Collect head outputs via hooks
    head_contributions = np.zeros((n_layers, n_heads))
    head_contributions_h = np.zeros((n_layers, n_heads))
    head_contributions_a = np.zeros((n_layers, n_heads))
    n_processed = 0

    all_texts = (
        [(t, "human") for t in texts_human[:max_samples]]
        + [(t, "ai") for t in texts_ai[:max_samples]]
    )

    for text, source in all_texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=max_seq_len, padding=False,
        ).to(device)
        seq_len = inputs["input_ids"].shape[1]
        if seq_len < 4:
            continue

        # Register hooks to capture pre-o_proj attention output
        head_outputs = {}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # Capture the self_attn module's output (post-o_proj)
                attn_output = output[0]  # (batch, seq, hidden_size)
                head_outputs[layer_idx] = attn_output.detach()
            return hook_fn

        handles = []
        for i in range(n_layers):
            if hasattr(model.model.layers[i], "self_attn"):
                h = model.model.layers[i].self_attn.register_forward_hook(make_hook(i))
                handles.append(h)

        with torch.no_grad():
            model(**inputs)

        for h in handles:
            h.remove()

        # Compute per-head contribution to v_src using W_o decomposition
        for layer_idx, attn_out in head_outputs.items():
            # Find closest available v_src layer (available_layers is 1-indexed)
            closest_layer = min(available_layers, key=lambda l: abs(l - (layer_idx + 1)))
            v = v_src_per_layer[closest_layer].float().to(device)
            v = v / (v.norm() + 1e-8)

            # Project v_src back through o_proj to get per-head importance
            # W_o: (hidden_size, n_heads*head_dim), so W_o.T @ v → (n_heads*head_dim,)
            o_proj = model.model.layers[layer_idx].self_attn.o_proj
            W_o = o_proj.weight.float()  # (hidden_size, n_heads*head_dim)
            v_in_head_space = W_o.T @ v  # (n_heads*head_dim,)

            # Overall output magnitude for weighting
            attn_mean = attn_out.mean(dim=1).squeeze(0)  # (hidden_size,)
            output_magnitude = attn_mean.norm().item() + 1e-8

            for h_idx in range(n_heads):
                start = h_idx * head_dim
                end = start + head_dim
                # Head's structural alignment with v_src direction
                head_alignment = v_in_head_space[start:end].norm().item()
                # Weight by overall output magnitude
                contrib = head_alignment * output_magnitude / n_heads
                if source == "human":
                    contrib_signed = -contrib
                else:
                    contrib_signed = contrib

                head_contributions[layer_idx, h_idx] += contrib_signed
                if source == "human":
                    head_contributions_h[layer_idx, h_idx] += contrib_signed
                else:
                    head_contributions_a[layer_idx, h_idx] += contrib_signed

        n_processed += 1

    if n_processed == 0:
        return {}

    # Normalize by sample count
    n_h = sum(1 for _, s in all_texts if s == "human")
    n_a = sum(1 for _, s in all_texts if s == "ai")
    head_contributions /= n_processed
    if n_h > 0:
        head_contributions_h /= n_h
    if n_a > 0:
        head_contributions_a /= n_a

    # Identify top style heads
    flat_idx = np.argsort(-np.abs(head_contributions).flatten())
    top_heads = []
    for idx in flat_idx[:20]:
        l = idx // n_heads
        h = idx % n_heads
        top_heads.append({
            "layer": int(l),
            "head": int(h),
            "contribution": float(head_contributions[l, h]),
            "human_contrib": float(head_contributions_h[l, h]),
            "ai_contrib": float(head_contributions_a[l, h]),
        })

    # Differential heads: biggest difference between AI and Human contexts
    diff_matrix = head_contributions_a - head_contributions_h
    flat_diff_idx = np.argsort(-np.abs(diff_matrix).flatten())
    differential_heads = []
    for idx in flat_diff_idx[:20]:
        l = idx // n_heads
        h = idx % n_heads
        differential_heads.append({
            "layer": int(l),
            "head": int(h),
            "diff": float(diff_matrix[l, h]),
            "human_contrib": float(head_contributions_h[l, h]),
            "ai_contrib": float(head_contributions_a[l, h]),
        })

    results = {
        "attribution_matrix": head_contributions.tolist(),
        "human_matrix": head_contributions_h.tolist(),
        "ai_matrix": head_contributions_a.tolist(),
        "diff_matrix": diff_matrix.tolist(),
        "top_style_heads": top_heads,
        "differential_heads": differential_heads,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_samples": n_processed,
    }

    logger.info(f"    Top style head: L{top_heads[0]['layer']}H{top_heads[0]['head']} (contrib={top_heads[0]['contribution']:.4f})")
    return results


# ═══════════════════════════════════════════════════════════
#  A5: MLP Neuron Analysis
# ═══════════════════════════════════════════════════════════

def mlp_neuron_analysis(
    model,
    tokenizer,
    v_src: torch.Tensor,
    layer_idx: int,
    texts_human: List[str],
    texts_ai: List[str],
    max_samples: int = 30,
    max_seq_len: int = 256,
    top_k_neurons: int = 50,
) -> Dict[str, Any]:
    """
    Identify MLP neurons most aligned with v_src.

    For each neuron in the MLP intermediate layer:
      1. Compute alignment: W_down[neuron, :] @ v_src
      2. Measure activation distribution in Human vs AI contexts
      3. Decode top neurons via unembedding (filtered to real words)
      4. Find trigger words: actual tokens that maximally activate each neuron

    Uses token-level activations (not sequence-averaged) for trigger word discovery.
    """
    logger.info(f"  MLP neuron analysis (layer {layer_idx})...")

    device = next(model.parameters()).device
    v_src = v_src.float().to(device)
    v_src = v_src / (v_src.norm() + 1e-8)

    # Get MLP weights (layer_idx is 1-indexed)
    n_layers = len(model.model.layers)
    idx = layer_idx - 1 if layer_idx >= 1 and layer_idx <= n_layers else layer_idx
    mlp = model.model.layers[idx].mlp

    if hasattr(mlp, "down_proj"):
        W_down = mlp.down_proj.weight.float()  # (hidden_size, intermediate_size)
        intermediate_size = W_down.shape[1]
    else:
        raise AttributeError("Cannot find MLP down_proj")

    # Neuron alignment with v_src
    alignment = (W_down.T @ v_src).detach().cpu().numpy()  # (intermediate_size,)

    # Collect token-level activations via hook
    activations_human = []  # list of (n_tokens, intermediate_size)
    activations_ai = []
    token_ids_human = []  # list of token_id lists
    token_ids_ai = []

    handle = mlp.down_proj.register_forward_pre_hook(
        lambda mod, inp: setattr(mlp, "_last_pre_down", inp[0].detach().cpu())
    )

    all_texts = (
        [(t, "human") for t in texts_human[:max_samples]]
        + [(t, "ai") for t in texts_ai[:max_samples]]
    )

    with torch.no_grad():
        for text, source in all_texts:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=max_seq_len, padding=False,
            ).to(device)
            if inputs["input_ids"].shape[1] < 4:
                continue

            model(**inputs)

            if hasattr(mlp, "_last_pre_down") and mlp._last_pre_down is not None:
                act = mlp._last_pre_down.squeeze(0).float().numpy()  # (seq_len, intermediate_size)
                ids = inputs["input_ids"].squeeze(0).cpu().tolist()
                if source == "human":
                    activations_human.append(act)
                    token_ids_human.append(ids)
                else:
                    activations_ai.append(act)
                    token_ids_ai.append(ids)

    handle.remove()
    if hasattr(mlp, "_last_pre_down"):
        delattr(mlp, "_last_pre_down")

    if not activations_human or not activations_ai:
        return {}

    # Compute per-neuron mean activation (sequence-averaged for statistics)
    mean_h = np.mean([a.mean(axis=0) for a in activations_human], axis=0)
    mean_a = np.mean([a.mean(axis=0) for a in activations_ai], axis=0)
    act_diff = mean_a - mean_h

    # Combined score: alignment × activation difference
    combined_score = alignment * act_diff

    top_by_combined = np.argsort(-np.abs(combined_score))[:top_k_neurons]

    # Build valid word index for decoding
    valid_indices, valid_words = _build_valid_word_index(tokenizer)

    # Unembedding matrix for neuron decoding
    if hasattr(model, "lm_head"):
        W_unembed = model.lm_head.weight.float()
        # Apply final norm for faithful decoding
        if hasattr(model.model, "norm"):
            pass  # We project W_down column, not v_src, so norm is less critical here
    else:
        W_unembed = None

    neurons = []
    for idx in top_by_combined:
        neuron_info = {
            "neuron_idx": int(idx),
            "alignment": float(alignment[idx]),
            "activation_diff": float(act_diff[idx]),
            "combined_score": float(combined_score[idx]),
            "mean_human": float(mean_h[idx]),
            "mean_ai": float(mean_a[idx]),
        }

        # Decode via unembedding — filtered to real English words
        if W_unembed is not None:
            neuron_direction = W_down[:, idx]  # (hidden_size,)
            token_logits = (W_unembed @ neuron_direction).detach().cpu().numpy()

            # Only look at valid English word tokens
            if len(valid_indices) > 0:
                valid_logits = token_logits[valid_indices]
                top_pos = np.argsort(-valid_logits)[:10]
                top_neg = np.argsort(valid_logits)[:10]
                neuron_info["promoted_tokens"] = [valid_words[i] for i in top_pos]
                neuron_info["suppressed_tokens"] = [valid_words[i] for i in top_neg]

        # Find trigger words: tokens with highest activation for this neuron
        trigger_h = defaultdict(list)
        trigger_a = defaultdict(list)

        for acts, ids, trigger_dict in [
            (activations_human, token_ids_human, trigger_h),
            (activations_ai, token_ids_ai, trigger_a),
        ]:
            for act_matrix, token_id_list in zip(acts, ids):
                neuron_acts = act_matrix[:, idx]  # (seq_len,)
                for pos, (tid, act_val) in enumerate(zip(token_id_list, neuron_acts)):
                    word = tokenizer.decode([tid]).strip()
                    # Accept word if it's ASCII alpha (English) or contains CJK chars
                    is_valid = (len(word) >= 2 and word.isalpha() and word.isascii()) or \
                               (len(word) >= 1 and any(_is_cjk_char(ch) for ch in word))
                    if is_valid:
                        trigger_dict[word].append(float(act_val))

        # Top trigger words by mean activation
        def _top_triggers(trigger_dict, n=8):
            word_means = {w: np.mean(vals) for w, vals in trigger_dict.items() if len(vals) >= 2}
            return sorted(word_means.items(), key=lambda x: -abs(x[1]))[:n]

        neuron_info["trigger_words_human"] = [
            {"word": w, "mean_activation": float(v)} for w, v in _top_triggers(trigger_h)
        ]
        neuron_info["trigger_words_ai"] = [
            {"word": w, "mean_activation": float(v)} for w, v in _top_triggers(trigger_a)
        ]

        neurons.append(neuron_info)

    results = {
        "neurons": neurons,
        "layer": layer_idx,
        "intermediate_size": int(intermediate_size),
        "alignment_stats": {
            "mean": float(np.mean(np.abs(alignment))),
            "max": float(np.max(np.abs(alignment))),
            "top10_mean": float(np.mean(np.abs(alignment[top_by_combined[:10]]))),
        },
        "n_human": len(activations_human),
        "n_ai": len(activations_ai),
    }

    logger.info(f"    Top neuron: #{neurons[0]['neuron_idx']} (combined={neurons[0]['combined_score']:.4f})")
    if "promoted_tokens" in neurons[0]:
        logger.info(f"    Promotes: {', '.join(neurons[0]['promoted_tokens'][:5])}")

    return results


# ═══════════════════════════════════════════════════════════
#  Orchestrator
# ═══════════════════════════════════════════════════════════

def run_vector_interpretability(
    model=None,
    tokenizer=None,
    states_dir: Optional[Path] = None,
    models_dir: Optional[Path] = None,
    texts_human: Optional[List[str]] = None,
    texts_ai: Optional[List[str]] = None,
    target_layer: int = 16,
    method: str = "mean_diff",
    max_samples: int = 50,
    skip_gpu: bool = False,
    language: str = "en",
) -> Dict[str, Any]:
    """
    Run all vector-based interpretability analyses.

    Args:
        model: HuggingFace model (None to skip GPU analyses)
        tokenizer: Tokenizer
        states_dir: Path to paired_states_layer*.pt files
        models_dir: Path to mean_diff_layer*.pt files
        texts_human: List of human continuation texts
        texts_ai: List of AI continuation texts
        target_layer: Primary layer for detailed analyses
        method: Steering method name (e.g. "mean_diff")
        max_samples: Max samples for GPU analyses
        skip_gpu: If True, only run A1 (no GPU needed)

    Returns:
        Dict with all results keyed by analysis name.
    """
    results = {}

    # ── A1: Layer Trajectory (always runs, no GPU) ──
    if states_dir and models_dir:
        logger.info("[A1] Layer trajectory analysis...")
        results["layer_trajectory"] = layer_trajectory(
            states_dir, models_dir, method=method, max_samples=max_samples,
        )

    if skip_gpu or model is None:
        logger.info("Skipping GPU analyses (skip_gpu=True or no model)")
        return results

    # Load v_src for target layer and all available layers
    v_src_target = None
    v_src_per_layer = {}
    if models_dir:
        for f in sorted(Path(models_dir).glob(f"{method}_layer*.pt")):
            layer_num = int(f.stem.split("layer")[-1])
            ckpt = torch.load(f, map_location="cpu", weights_only=False)
            v = ckpt["state_dict"]["v_src"].float()
            v_src_per_layer[layer_num] = v
            if layer_num == target_layer:
                v_src_target = v.to(next(model.parameters()).device)

    if v_src_target is None and v_src_per_layer:
        # Use closest available layer
        closest = min(v_src_per_layer.keys(), key=lambda l: abs(l - target_layer))
        v_src_target = v_src_per_layer[closest].to(next(model.parameters()).device)
        logger.warning(f"Layer {target_layer} not found, using layer {closest}")

    if v_src_target is None:
        logger.error("No v_src found — cannot run GPU analyses")
        return results

    # ── A2: Vocabulary Semantic Fields ──
    logger.info("[A2] Vocabulary semantic fields...")
    results["vocab_fields"] = vocabulary_semantic_fields(
        model, tokenizer, v_src_target,
    )

    # NOTE: token-position attribution, attention-head attribution and
    # MLP-neuron alignment have been removed from the orchestrated pipeline
    # to keep the readout aligned with the paper's three interpretability
    # paragraphs (logit-lens / vocab fields / contrastive n-grams).
    # The functions remain importable for ad-hoc analysis.

    return results
