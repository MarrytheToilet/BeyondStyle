"""
Lightweight Sparse Autoencoder for steering-vector interpretability.

Trains a simple ReLU SAE on the paired hidden states, then decomposes
v_src into sparse dictionary features to understand what concepts the
Human-vs-AI direction encodes.

Architecture:
    encoder:  z = ReLU(W_enc @ (x - b_dec) + b_enc)    [d_model → d_sae]
    decoder:  x̂ = W_dec @ z + b_dec                     [d_sae → d_model]
    loss:     ||x - x̂||² + λ ||z||₁

The decoder columns W_dec[:, j] are unit-normalized after each step,
so each dictionary atom is a direction in activation space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from tqdm import trange


# ═══════════════════════════════════════════════════════════
#  SAE Model
# ═══════════════════════════════════════════════════════════

class SparseAutoencoder(nn.Module):
    """
    Standard ReLU sparse autoencoder.

    Args:
        d_model:    input dimension (e.g. 4096 for Llama-3.1-8B)
        d_sae:      dictionary size (typically 4×–16× d_model)
        l1_coeff:   sparsity penalty weight
    """

    def __init__(self, d_model: int, d_sae: int, l1_coeff: float = 5e-3):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.l1_coeff = l1_coeff

        self.W_enc = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # Kaiming init
        nn.init.kaiming_uniform_(self.W_enc, nonlinearity="relu")
        # Tie decoder init to encoder transpose (then normalize)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T.clone())
            self._normalize_decoder()

    def _normalize_decoder(self):
        """Unit-normalize each decoder column (row of W_dec)."""
        with torch.no_grad():
            norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
            self.W_dec.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to sparse features: z = ReLU(W_enc @ (x - b_dec) + b_enc)."""
        return F.relu(
            (x - self.b_dec) @ self.W_enc + self.b_enc
        )

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from features: x̂ = W_dec @ z + b_dec."""
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Returns:
            x_recon: reconstructed input
            z:       sparse feature activations
            loss:    reconstruction + sparsity loss
        """
        z = self.encode(x)
        x_recon = self.decode(z)

        recon_loss = F.mse_loss(x_recon, x)
        l1_loss = z.abs().mean()
        loss = recon_loss + self.l1_coeff * l1_loss

        return x_recon, z, loss

    def save(self, path: Path, metadata: Optional[dict] = None):
        torch.save({
            "state_dict": self.state_dict(),
            "d_model": self.d_model,
            "d_sae": self.d_sae,
            "l1_coeff": self.l1_coeff,
            "metadata": metadata or {},
        }, path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "SparseAutoencoder":
        data = torch.load(path, map_location=device, weights_only=False)
        sae = cls(data["d_model"], data["d_sae"], data["l1_coeff"])
        sae.load_state_dict(data["state_dict"])
        return sae


# ═══════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════

@dataclass
class SAETrainConfig:
    """Training hyperparameters."""
    d_sae_multiplier: int = 4         # d_sae = d_model × multiplier
    l1_coeff: float = 5e-3            # sparsity penalty
    lr: float = 3e-4
    epochs: int = 200
    batch_size: int = 64
    normalize_input: bool = True      # center + unit-var the inputs
    seed: int = 42


def train_sae(
    activations: torch.Tensor,
    config: SAETrainConfig = SAETrainConfig(),
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[SparseAutoencoder, dict]:
    """
    Train a sparse autoencoder on activation vectors.

    Args:
        activations: (N, d_model) hidden state vectors
        config:      training config
        device:      training device
        verbose:     print progress

    Returns:
        (trained_sae, training_metrics)
    """
    torch.manual_seed(config.seed)

    X = activations.float().to(device)
    N, d_model = X.shape
    d_sae = d_model * config.d_sae_multiplier

    # Optional normalization
    x_mean = X.mean(dim=0)
    x_std = X.std(dim=0).clamp(min=1e-8)
    if config.normalize_input:
        X = (X - x_mean) / x_std

    sae = SparseAutoencoder(d_model, d_sae, config.l1_coeff).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.lr * 0.01
    )

    history = {"loss": [], "recon": [], "l1": [], "l0": []}

    iterator = trange(config.epochs, desc="SAE Training") if verbose else range(config.epochs)
    for epoch in iterator:
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_l1 = 0.0
        epoch_l0 = 0.0
        n_batches = 0

        for i in range(0, N, config.batch_size):
            batch = X[perm[i:i + config.batch_size]]
            x_recon, z, loss = sae(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae._normalize_decoder()

            with torch.no_grad():
                epoch_loss += loss.item()
                epoch_recon += F.mse_loss(x_recon, batch).item()
                epoch_l1 += z.abs().mean().item()
                epoch_l0 += (z > 0).float().mean().item() * z.shape[1]
                n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon / n_batches
        avg_l1 = epoch_l1 / n_batches
        avg_l0 = epoch_l0 / n_batches

        history["loss"].append(avg_loss)
        history["recon"].append(avg_recon)
        history["l1"].append(avg_l1)
        history["l0"].append(avg_l0)

        if verbose and (epoch % 50 == 0 or epoch == config.epochs - 1):
            iterator.write(
                f"  epoch {epoch:4d}  loss={avg_loss:.6f}  "
                f"recon={avg_recon:.6f}  L0={avg_l0:.1f}")

    # Final evaluation
    with torch.no_grad():
        z_all = sae.encode(X)
        x_recon_all = sae.decode(z_all)
        final_recon_cos = F.cosine_similarity(X, x_recon_all, dim=1).mean().item()
        final_mse = F.mse_loss(x_recon_all, X).item()
        final_l0 = (z_all > 0).float().sum(dim=1).mean().item()
        alive_features = (z_all.sum(dim=0) > 0).sum().item()

    metrics = {
        "d_model": d_model,
        "d_sae": d_sae,
        "n_samples": N,
        "epochs": config.epochs,
        "l1_coeff": config.l1_coeff,
        "final_mse": final_mse,
        "final_recon_cosine": final_recon_cos,
        "final_l0": final_l0,
        "alive_features": alive_features,
        "alive_ratio": alive_features / d_sae,
        "history": history,
    }

    # Store normalization params so we can transform v_src the same way
    sae._norm_mean = x_mean.cpu()
    sae._norm_std = x_std.cpu()
    sae._normalize_input = config.normalize_input

    if verbose:
        print(f"\n  SAE trained: d_sae={d_sae}, L0={final_l0:.1f}, "
              f"alive={alive_features}/{d_sae} ({alive_features/d_sae:.1%}), "
              f"recon_cos={final_recon_cos:.4f}")

    return sae, metrics


# ═══════════════════════════════════════════════════════════
#  Decomposition
# ═══════════════════════════════════════════════════════════

@dataclass
class SAEFeature:
    """A single SAE feature's relation to v_src."""
    index: int
    activation: float           # encoder output
    cosine_with_dec: float      # cos(W_dec[i], v_src)
    decoder_direction: Optional[np.ndarray] = None  # the actual direction (optional)

@dataclass
class SAEDecomposition:
    """Full decomposition result for one (layer, method) pair."""
    layer: int
    method: str
    d_sae: int
    l0: int
    reconstruction_cosine: float
    explained_variance: float
    top_features: List[SAEFeature]
    # All feature activations (for distribution plots)
    all_activations: Optional[np.ndarray] = None

    def to_dict(self, include_activations: bool = False) -> dict:
        d = {
            "layer": self.layer,
            "method": self.method,
            "d_sae": self.d_sae,
            "l0": self.l0,
            "reconstruction_cosine": round(self.reconstruction_cosine, 6),
            "explained_variance": round(self.explained_variance, 6),
            "top_features": [
                {
                    "index": f.index,
                    "activation": round(f.activation, 6),
                    "cosine_dec": round(f.cosine_with_dec, 6),
                }
                for f in self.top_features
            ],
        }
        if include_activations and self.all_activations is not None:
            d["activation_histogram"] = {
                "nonzero_count": int((self.all_activations > 0).sum()),
                "nonzero_mean": float(self.all_activations[self.all_activations > 0].mean())
                    if (self.all_activations > 0).any() else 0,
                "max": float(self.all_activations.max()),
            }
        return d


def decompose_direction(
    v_src: torch.Tensor,
    sae: SparseAutoencoder,
    layer: int,
    method: str,
    top_k: int = 50,
) -> SAEDecomposition:
    """
    Decompose v_src into SAE features.

    Three views:
      1. Encoder activations: which features fire for v_src?
      2. Decoder cosines: which dictionary atoms align with v_src?
      3. Reconstruction quality: how well does SAE span v_src?
    """
    device = next(sae.parameters()).device
    v = v_src.float().to(device)

    # Apply same normalization as training data
    if hasattr(sae, '_normalize_input') and sae._normalize_input:
        v_normed = (v - sae._norm_mean.to(device)) / sae._norm_std.to(device)
    else:
        v_normed = v

    with torch.no_grad():
        z = sae.encode(v_normed.unsqueeze(0)).squeeze(0)  # (d_sae,)
        v_recon = sae.decode(z.unsqueeze(0)).squeeze(0)    # (d_model,)

    # Reconstruction quality (in normalized space)
    recon_cos = F.cosine_similarity(v_normed.unsqueeze(0), v_recon.unsqueeze(0)).item()
    residual_sq = (v_normed - v_recon).pow(2).sum().item()
    v_sq = v_normed.pow(2).sum().item()
    explained_var = 1.0 - residual_sq / (v_sq + 1e-8)

    # Decoder cosines: cos(W_dec[i], v_src)
    W_dec = sae.W_dec.data.float()  # (d_sae, d_model)
    dec_norms = W_dec.norm(dim=1)
    v_norm = v_normed.norm()
    dec_cosines = (W_dec @ v_normed) / (dec_norms * v_norm + 1e-8)

    z_np = z.cpu().numpy()
    cos_np = dec_cosines.cpu().numpy()
    l0 = int((z_np > 0).sum())

    # Top-k by activation
    top_idx = np.argsort(z_np)[::-1][:top_k]
    features = [
        SAEFeature(
            index=int(i),
            activation=float(z_np[i]),
            cosine_with_dec=float(cos_np[i]),
        )
        for i in top_idx
    ]

    return SAEDecomposition(
        layer=layer, method=method, d_sae=sae.d_sae,
        l0=l0, reconstruction_cosine=recon_cos,
        explained_variance=explained_var,
        top_features=features,
        all_activations=z_np,
    )


# ═══════════════════════════════════════════════════════════
#  Full pipeline: train + decompose across layers/methods
# ═══════════════════════════════════════════════════════════

def train_and_decompose(
    states_dir: Path,
    models_dir: Path,
    layers: List[int],
    methods: List[str],
    train_config: SAETrainConfig = SAETrainConfig(),
    device: str = "cpu",
    top_k: int = 50,
    sae_out_dir: Optional[Path] = None,
) -> Tuple[Dict[str, SAEDecomposition], Dict[int, dict]]:
    """
    For each layer: train SAE on paired states, then decompose each method's v_src.

    Args:
        states_dir:   directory with paired_states_layerXX.pt
        models_dir:   directory with mean_diff*_layerXX.pt
        layers:       which layers to process
        methods:      steering method keys
        train_config: SAE training config
        device:       training device
        top_k:        top features to report
        sae_out_dir:  where to save trained SAEs (optional)

    Returns:
        (decompositions, train_metrics)
        decompositions: keyed by "{method}_L{layer}"
        train_metrics:  keyed by layer
    """
    from src.extraction import PairedHiddenStates

    decompositions = {}
    train_metrics_all = {}

    for layer in layers:
        print(f"\n{'─'*40}")
        print(f"  Layer {layer}")
        print(f"{'─'*40}")

        # Load paired states
        states_path = states_dir / f"paired_states_layer{layer:02d}.pt"
        if not states_path.exists():
            print(f"  ⚠ States not found: {states_path}")
            continue

        ps = PairedHiddenStates.load(states_path)
        # Combine human + AI as training data for SAE
        activations = torch.cat([ps.human, ps.ai], dim=0).float()
        print(f"  Training data: {activations.shape} ({ps.n_samples} pairs)")

        # Train SAE
        sae, metrics = train_sae(activations, train_config, device=device)
        train_metrics_all[layer] = metrics

        # Save SAE
        if sae_out_dir:
            sae_path = sae_out_dir / f"sae_layer{layer:02d}.pt"
            sae_out_dir.mkdir(parents=True, exist_ok=True)
            sae.save(sae_path, metadata={
                "layer": layer, "n_samples": activations.shape[0],
                "train_config": {
                    "d_sae_multiplier": train_config.d_sae_multiplier,
                    "l1_coeff": train_config.l1_coeff,
                    "epochs": train_config.epochs,
                },
            })
            print(f"  Saved SAE → {sae_path}")

        # Decompose each method's v_src
        for method in methods:
            model_path = models_dir / f"{method}_layer{layer:02d}.pt"
            if not model_path.exists():
                print(f"  ⚠ Skipping {method} (not found)")
                continue

            data = torch.load(model_path, map_location="cpu")
            v_src = data["state_dict"]["v_src"].float()

            decomp = decompose_direction(v_src, sae, layer, method, top_k=top_k)
            key = f"{method}_L{layer}"
            decompositions[key] = decomp

            top3 = decomp.top_features[:3]
            print(f"  {method:30s}  L0={decomp.l0:4d}  "
                  f"recon_cos={decomp.reconstruction_cosine:.4f}  "
                  f"top3=[{', '.join(f'#{f.index}({f.activation:.3f})' for f in top3)}]")

    return decompositions, train_metrics_all


# ═══════════════════════════════════════════════════════════
#  Cross-method feature overlap
# ═══════════════════════════════════════════════════════════

def compute_feature_overlap(
    decompositions: Dict[str, SAEDecomposition],
    top_k: int = 30,
) -> dict:
    """Jaccard overlap of top-k features across methods (within same layer)."""
    from collections import defaultdict

    by_layer = defaultdict(dict)
    for key, decomp in decompositions.items():
        by_layer[decomp.layer][key] = decomp

    overlap = {}
    for layer, methods in by_layer.items():
        keys = sorted(methods.keys())
        for i, k1 in enumerate(keys):
            top1 = set(f.index for f in methods[k1].top_features[:top_k])
            for k2 in keys[i+1:]:
                top2 = set(f.index for f in methods[k2].top_features[:top_k])
                union = len(top1 | top2)
                jaccard = len(top1 & top2) / union if union else 0
                overlap[f"{k1} vs {k2}"] = {
                    "jaccard": round(jaccard, 4),
                    "intersection": len(top1 & top2),
                    "union": union,
                }

    return overlap
