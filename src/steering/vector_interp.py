"""
Vector-Based Deep Interpretability for Steering Vectors.

Unlike deep_interp.py (corpus-level text comparison), this module analyses
the steering vector v_src **inside the model** — how it interacts with
layers, attention heads, MLP neurons, and the vocabulary space.

Five analyses:
  1. Layer Trajectory     — projection trajectory across layers (NO GPU)
  2. Vocabulary Fields    — semantic clustering of vocab by v_src projection
  3. Token Attribution    — gradient-based per-position importance
  4. Attention Heads      — Direct Logit Attribution per head
  5. MLP Neurons          — top neurons aligned with v_src
"""

import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from scipy import stats as scipy_stats
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


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
#  A2: Vocabulary Semantic Fields
# ═══════════════════════════════════════════════════════════

def vocabulary_semantic_fields(
    model,
    tokenizer,
    v_src: torch.Tensor,
    n_clusters: int = 8,
    top_k_per_cluster: int = 15,
    min_token_len: int = 2,
) -> Dict[str, Any]:
    """
    Project entire vocabulary onto v_src, then cluster tokens by their
    embedding similarity to reveal semantic fields.

    Returns clusters of AI-ward and Human-ward tokens with semantic labels.
    """
    logger.info("  Vocabulary semantic fields...")

    device = v_src.device
    v_src = v_src.float()
    v_src = v_src / (v_src.norm() + 1e-8)

    # Get unembedding matrix
    if hasattr(model, "lm_head"):
        W_unembed = model.lm_head.weight.float().to(device)  # (vocab, hidden)
    elif hasattr(model, "embed_out"):
        W_unembed = model.embed_out.weight.float().to(device)
    else:
        raise AttributeError("Cannot find unembedding matrix")

    # Project all vocab tokens
    logits = (W_unembed @ v_src).detach().cpu().numpy()  # (vocab_size,)

    # Filter to complete English words only (strict)
    # Llama-3/GPT-2 BPE uses "Ġ" prefix for word boundary (space before token)
    vocab_size = len(logits)
    valid_indices = []
    valid_tokens = []
    for idx in range(vocab_size):
        raw = tokenizer.convert_ids_to_tokens(idx)
        if raw is None:
            continue
        # Must start with Ġ (BPE word boundary = space-prefixed)
        if not raw.startswith("Ġ"):
            continue
        tok_clean = raw[1:]  # strip the Ġ
        if len(tok_clean) < max(min_token_len, 4):
            continue
        if not tok_clean.isalpha():
            continue
        if not tok_clean.isascii():
            continue
        # Only fully lowercase words (skip camelCase, proper nouns, acronyms)
        if not tok_clean.islower():
            continue
        valid_indices.append(idx)
        valid_tokens.append(tok_clean)

    valid_indices = np.array(valid_indices)
    valid_logits = logits[valid_indices]

    # Separate into AI-ward (positive) and Human-ward (negative)
    ai_mask = valid_logits > 0
    human_mask = valid_logits <= 0

    # Get input embeddings for clustering
    if hasattr(model.model, "embed_tokens"):
        embeddings = model.model.embed_tokens.weight[valid_indices].float().cpu().detach().numpy()
    elif hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings().weight[valid_indices].float().cpu().detach().numpy()
    else:
        embeddings = W_unembed[valid_indices].cpu().detach().numpy()

    results = {"ai_clusters": [], "human_clusters": [], "stats": {}}

    for direction, mask, label in [
        ("ai", ai_mask, "AI-ward"),
        ("human", human_mask, "Human-ward"),
    ]:
        dir_indices = np.where(mask)[0]
        if len(dir_indices) < n_clusters * 3:
            continue

        dir_logits = valid_logits[dir_indices]

        # Focus on strongly-projected tokens (top 30% by magnitude)
        magnitude_threshold = np.percentile(np.abs(dir_logits), 70)
        strong_mask = np.abs(dir_logits) >= magnitude_threshold
        strong_indices = np.where(strong_mask)[0]

        if len(strong_indices) < n_clusters * 3:
            strong_indices = np.arange(len(dir_indices))

        s_logits = dir_logits[strong_indices]
        s_embeddings = embeddings[dir_indices[strong_indices]]
        s_tokens = [valid_tokens[dir_indices[i]] for i in strong_indices]

        # Normalize embeddings for cosine-based clustering
        norms = np.linalg.norm(s_embeddings, axis=1, keepdims=True) + 1e-8
        s_embeddings_norm = s_embeddings / norms

        # K-means clustering on embeddings
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = km.fit_predict(s_embeddings_norm)

        clusters = []
        for c in range(n_clusters):
            c_mask = cluster_labels == c
            c_logits = s_logits[c_mask]
            c_tokens = [s_tokens[i] for i in np.where(c_mask)[0]]

            # Sort by magnitude
            sort_idx = np.argsort(-np.abs(c_logits))
            top_tokens = [c_tokens[i] for i in sort_idx[:top_k_per_cluster]]
            top_scores = [float(c_logits[i]) for i in sort_idx[:top_k_per_cluster]]

            clusters.append({
                "tokens": top_tokens,
                "scores": top_scores,
                "size": int(c_mask.sum()),
                "mean_projection": float(c_logits.mean()),
                "max_projection": float(np.abs(c_logits).max()),
            })

        # Sort clusters by mean projection magnitude
        clusters.sort(key=lambda x: abs(x["mean_projection"]), reverse=True)
        results[f"{direction}_clusters"] = clusters

    results["stats"] = {
        "total_vocab": int(vocab_size),
        "valid_tokens": len(valid_tokens),
        "ai_ward_count": int(ai_mask.sum()),
        "human_ward_count": int(human_mask.sum()),
        "projection_range": [float(valid_logits.min()), float(valid_logits.max())],
    }

    logger.info(f"    {results['stats']['ai_ward_count']} AI-ward, {results['stats']['human_ward_count']} Human-ward tokens")

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
) -> Dict[str, Any]:
    """
    Gradient-based attribution: which input token positions most influence
    the projection of the hidden state onto v_src?

    For each text, computes:
      scalar = mean_over_positions(hidden_state[layer] @ v_src)
      grad = d(scalar) / d(input_embeddings)
      attribution[pos] = ||grad[pos]||

    Then aggregates by relative position (quartile) and by POS tag.
    """
    logger.info(f"  Token-position attribution (layer {layer_idx}, {max_samples} samples)...")

    import spacy
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    except OSError:
        nlp = None

    device = next(model.parameters()).device
    v_src = v_src.float().to(device)
    v_src = v_src / (v_src.norm() + 1e-8)

    # Identify the target layer module
    if hasattr(model.model, "layers"):
        target_layer_module = model.model.layers[layer_idx]
    else:
        raise AttributeError("Cannot find model layers")

    results = {
        "positional_attribution": defaultdict(list),  # quartile → [attr_values]
        "pos_tag_attribution": defaultdict(list),     # POS → [attr_values]
        "token_examples": [],  # top-attributed tokens per sample
    }

    all_texts = (
        [(t, "human") for t in texts_human[:max_samples]]
        + [(t, "ai") for t in texts_ai[:max_samples]]
    )

    for text, source in all_texts:
        # Tokenize
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=max_seq_len, padding=False,
        ).to(device)
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]

        if seq_len < 4:
            continue

        # Use a hook to capture hidden states at target layer
        captured = {}

        def capture_hook(module, input, output):
            # output is (hidden_states, ...) or hidden_states
            h = output[0] if isinstance(output, tuple) else output
            captured["hidden"] = h

        handle = target_layer_module.register_forward_hook(capture_hook)

        # Enable gradient on input embeddings via hook
        embed_grads = {}

        def embed_hook(module, input, output):
            output.retain_grad()
            embed_grads["output"] = output
            return output

        embed_handle = model.model.embed_tokens.register_forward_hook(embed_hook)

        # Forward pass (normal — model handles RoPE etc.)
        outputs = model(**inputs, output_hidden_states=False)

        handle.remove()
        embed_handle.remove()

        if "hidden" not in captured or "output" not in embed_grads:
            continue

        hidden = captured["hidden"]  # (1, seq, hidden)
        embed_out = embed_grads["output"]

        # Projection scalar: mean projection onto v_src across positions
        proj = (hidden.squeeze(0).float() @ v_src)  # (seq_len,)
        scalar = proj.mean()

        # Backprop
        scalar.backward(retain_graph=False)

        if embed_out.grad is None:
            continue

        grad = embed_out.grad.squeeze(0)  # (seq_len, hidden_size)
        attr = grad.float().norm(dim=-1).detach().cpu().numpy()  # (seq_len,)

        # Normalize attribution to sum to 1
        attr_sum = attr.sum()
        if attr_sum > 0:
            attr_norm = attr / attr_sum
        else:
            continue

        # Aggregate by relative position (quartiles)
        for pos in range(seq_len):
            quartile = int(pos / seq_len * 4)
            quartile = min(quartile, 3)
            results["positional_attribution"][f"Q{quartile+1}_{source}"].append(float(attr_norm[pos]))

        # Aggregate by POS tag (if spacy available)
        if nlp is not None:
            tokens_text = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
            decoded = tokenizer.decode(input_ids[0].cpu().tolist(), skip_special_tokens=True)
            doc = nlp(decoded)

            # Simple alignment: map spacy tokens to BPE positions
            spacy_pos = [tok.pos_ for tok in doc]
            # Approximate: distribute POS tags across BPE tokens
            if len(spacy_pos) > 0:
                for pos_idx in range(seq_len):
                    approx_spacy_idx = min(int(pos_idx / seq_len * len(spacy_pos)), len(spacy_pos) - 1)
                    pos_tag = spacy_pos[approx_spacy_idx]
                    results["pos_tag_attribution"][f"{pos_tag}_{source}"].append(float(attr_norm[pos_idx]))

        # Top-attributed tokens for this sample
        top_k = min(5, seq_len)
        top_idx = np.argsort(-attr_norm)[:top_k]
        top_tokens = []
        for idx in top_idx:
            tok = tokenizer.decode([input_ids[0, idx].item()]).strip()
            top_tokens.append({"token": tok, "position": int(idx), "attribution": float(attr_norm[idx])})
        results["token_examples"].append({
            "source": source,
            "top_tokens": top_tokens,
            "text_preview": text[:80],
        })

        # Clear grads
        model.zero_grad()

    # Aggregate positional attribution
    pos_summary = {}
    for key, vals in results["positional_attribution"].items():
        pos_summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    results["positional_attribution"] = pos_summary

    # Aggregate POS attribution
    pos_tag_summary = {}
    for key, vals in results["pos_tag_attribution"].items():
        if len(vals) > 10:
            pos_tag_summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
    results["pos_tag_attribution"] = pos_tag_summary

    # Keep only first 20 examples
    results["token_examples"] = results["token_examples"][:20]

    logger.info(f"    Processed {len(all_texts)} samples")
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
    head_dim = model.config.hidden_size // n_heads

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

        # Register hooks to capture attention output per head
        head_outputs = {}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # output is the attention output before o_proj
                # We need to capture it after o_proj but before residual add
                # In LlamaAttention, output[0] is the projected output
                attn_output = output[0]  # (batch, seq, hidden)
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

        # Compute per-head contribution to v_src
        for layer_idx, attn_out in head_outputs.items():
            # Find closest available v_src layer
            closest_layer = min(available_layers, key=lambda l: abs(l - layer_idx))
            v = v_src_per_layer[closest_layer].float().to(device)
            v = v / (v.norm() + 1e-8)

            # Reshape to get per-head contribution
            # attn_out: (1, seq, hidden) → split into heads
            attn_mean = attn_out.mean(dim=1).squeeze(0)  # (hidden,)

            # Split into head contributions via o_proj
            # Each head contributes head_dim dimensions, projected through o_proj
            o_proj = model.model.layers[layer_idx].self_attn.o_proj
            W_o = o_proj.weight.float()  # (hidden, hidden)

            for h_idx in range(n_heads):
                start = h_idx * head_dim
                end = start + head_dim
                # Head's contribution through o_proj
                head_slice = torch.zeros(model.config.hidden_size, device=device)
                head_slice[start:end] = attn_mean[start:end]
                head_out = W_o @ head_slice  # (hidden,)
                contrib = float((head_out @ v).detach().cpu())

                head_contributions[layer_idx, h_idx] += contrib
                if source == "human":
                    head_contributions_h[layer_idx, h_idx] += contrib
                else:
                    head_contributions_a[layer_idx, h_idx] += contrib

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
      3. Decode top neurons via unembedding to understand what they represent

    Returns top-k neurons with their properties.
    """
    logger.info(f"  MLP neuron analysis (layer {layer_idx})...")

    device = next(model.parameters()).device
    v_src = v_src.float().to(device)
    v_src = v_src / (v_src.norm() + 1e-8)

    # Get MLP weights
    mlp = model.model.layers[layer_idx].mlp

    # For Llama: gate_proj (up) and down_proj
    if hasattr(mlp, "down_proj"):
        W_down = mlp.down_proj.weight.float()  # (hidden_size, intermediate_size)
        intermediate_size = W_down.shape[1]
    else:
        raise AttributeError("Cannot find MLP down_proj")

    # Neuron alignment: how much does each neuron's output direction align with v_src?
    # Each column of W_down is the output direction for one neuron
    alignment = (W_down.T @ v_src).detach().cpu().numpy()  # (intermediate_size,)

    # Hook to capture MLP intermediate activations
    activations_human = []
    activations_ai = []

    def capture_hook(module, input, output):
        # For Llama, the activation after gate: SiLU(gate_proj(x)) * up_proj(x)
        # We capture the result before down_proj
        capture_hook.last_activation = output.detach().mean(dim=1).squeeze(0).cpu()

    capture_hook.last_activation = None

    # Hook on the activation function (or gate proj output)
    # In LlamaMLPActivate: act = silu(gate(x)) * up(x), then down(act)
    # We need to hook inside the MLP forward to get intermediate activations
    # Alternative: hook on down_proj input
    handle = mlp.down_proj.register_forward_pre_hook(
        lambda mod, inp: setattr(capture_hook, "last_activation",
                                  inp[0].detach().mean(dim=1).squeeze(0).cpu())
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

            if capture_hook.last_activation is not None:
                act = capture_hook.last_activation.numpy()
                if source == "human":
                    activations_human.append(act)
                else:
                    activations_ai.append(act)

    handle.remove()

    if not activations_human or not activations_ai:
        return {}

    act_human = np.stack(activations_human)  # (n_h, intermediate_size)
    act_ai = np.stack(activations_ai)        # (n_a, intermediate_size)

    # Compute per-neuron statistics
    mean_h = act_human.mean(axis=0)
    mean_a = act_ai.mean(axis=0)
    std_h = act_human.std(axis=0) + 1e-8
    std_a = act_ai.std(axis=0) + 1e-8

    # Activation difference
    act_diff = mean_a - mean_h

    # Combined score: alignment × activation difference
    combined_score = alignment * act_diff

    # Top neurons by alignment
    top_by_alignment = np.argsort(-np.abs(alignment))[:top_k_neurons]
    # Top neurons by activation difference
    top_by_diff = np.argsort(-np.abs(act_diff))[:top_k_neurons]
    # Top neurons by combined score
    top_by_combined = np.argsort(-np.abs(combined_score))[:top_k_neurons]

    # Decode top neurons: what vocabulary does each neuron activate?
    if hasattr(model, "lm_head"):
        W_unembed = model.lm_head.weight.float()  # (vocab, hidden)
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

        # Decode: what tokens does this neuron's output direction promote?
        if W_unembed is not None:
            neuron_direction = W_down[:, idx]  # (hidden_size,)
            token_logits = (W_unembed @ neuron_direction).detach().cpu().numpy()

            # Filter to clean English words only
            def _decode_top(logit_arr, top_n=10, ascending=False):
                sorted_idx = np.argsort(logit_arr) if ascending else np.argsort(-logit_arr)
                words = []
                for i in sorted_idx:
                    raw = tokenizer.convert_ids_to_tokens(int(i))
                    if raw and raw.startswith("Ġ"):
                        w = raw[1:]
                        if w.isalpha() and w.isascii() and len(w) >= 4 and w.islower():
                            words.append(w)
                            if len(words) >= top_n:
                                break
                return words

            neuron_info["promoted_tokens"] = _decode_top(token_logits, 10, ascending=False)
            neuron_info["suppressed_tokens"] = _decode_top(token_logits, 10, ascending=True)

        neurons.append(neuron_info)

    results = {
        "neurons": neurons,
        "layer": layer_idx,
        "intermediate_size": int(intermediate_size),
        "alignment_stats": {
            "mean": float(np.mean(np.abs(alignment))),
            "max": float(np.max(np.abs(alignment))),
            "top10_mean": float(np.mean(np.abs(alignment[top_by_alignment[:10]]))),
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

    # ── A3: Token-Position Attribution ──
    if texts_human and texts_ai:
        logger.info("[A3] Token-position attribution...")
        results["token_attribution"] = token_position_attribution(
            model, tokenizer, v_src_target,
            texts_human, texts_ai,
            layer_idx=target_layer,
            max_samples=max_samples,
        )

    # ── A4: Attention Head Attribution ──
    if texts_human and texts_ai:
        logger.info("[A4] Attention head attribution...")
        results["attention_heads"] = attention_head_attribution(
            model, tokenizer, v_src_per_layer,
            texts_human, texts_ai,
            max_samples=min(max_samples, 30),
        )

    # ── A5: MLP Neuron Analysis ──
    if texts_human and texts_ai:
        logger.info("[A5] MLP neuron analysis...")
        results["mlp_neurons"] = mlp_neuron_analysis(
            model, tokenizer, v_src_target,
            layer_idx=target_layer,
            texts_human=texts_human,
            texts_ai=texts_ai,
            max_samples=min(max_samples, 30),
        )

    return results
