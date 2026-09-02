"""
Layer selection metrics based on geometric separation in representation space.

Provides normalized hull-centroid distance: Euclidean distance between
class hull centroids, normalized by mean within-class hull diameter.
"""

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from typing import Dict, Any, Tuple


def _hull_diameter(points: np.ndarray, hull: ConvexHull) -> float:
    """Max pairwise distance among convex-hull vertices."""
    verts = points[hull.vertices]
    if len(verts) < 2:
        return 1e-10
    return float(pdist(verts).max())


def pca_hull_centroid_distance(
    human_states: np.ndarray,
    ai_states: np.ndarray,
    n_components: int = 2,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Euclidean distance between PCA convex-hull centroids, plus a
    normalized version that divides by mean within-class hull diameter.

    Steps:
      1. Fit PCA on the combined (human + AI) data.
      2. Project both classes into PCA space.
      3. Compute convex hull for each class.
      4. Hull centroid = mean of hull vertices.
      5. Hull diameter = max pairwise distance among hull vertices.
      6. Normalized distance = centroid_dist / mean(diameter_h, diameter_a).

    Returns:
        raw_distance:  Euclidean distance (unnormalized).
        norm_distance: Normalized by mean hull diameter (scale-invariant).
        info:          Dict with centroids, diameters, PCA variance, etc.
    """
    h = np.asarray(human_states, dtype=np.float32)
    a = np.asarray(ai_states, dtype=np.float32)
    combined = np.vstack([h, a])

    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(combined)

    h_proj = projected[: len(h)]
    a_proj = projected[len(h):]

    try:
        h_hull = ConvexHull(h_proj)
        a_hull = ConvexHull(a_proj)
        h_centroid = h_proj[h_hull.vertices].mean(axis=0)
        a_centroid = a_proj[a_hull.vertices].mean(axis=0)
        h_diam = _hull_diameter(h_proj, h_hull)
        a_diam = _hull_diameter(a_proj, a_hull)
        h_hull_size = len(h_hull.vertices)
        a_hull_size = len(a_hull.vertices)
    except Exception:
        h_centroid = h_proj.mean(axis=0)
        a_centroid = a_proj.mean(axis=0)
        h_diam = float(pdist(h_proj).max()) if len(h_proj) > 1 else 1e-10
        a_diam = float(pdist(a_proj).max()) if len(a_proj) > 1 else 1e-10
        h_hull_size = 0
        a_hull_size = 0

    raw_dist = float(np.linalg.norm(h_centroid - a_centroid))
    mean_diam = (h_diam + a_diam) / 2.0
    norm_dist = raw_dist / mean_diam if mean_diam > 1e-10 else 0.0

    info = {
        "h_centroid": h_centroid.tolist(),
        "a_centroid": a_centroid.tolist(),
        "h_hull_diameter": float(h_diam),
        "a_hull_diameter": float(a_diam),
        "h_hull_vertices": h_hull_size,
        "a_hull_vertices": a_hull_size,
        "pca_variance_explained": pca.explained_variance_ratio_.tolist(),
    }
    return raw_dist, norm_dist, info


def compute_layer_geometry(
    human_states: np.ndarray,
    ai_states: np.ndarray,
    pca_components: int = 2,
) -> Dict[str, Any]:
    """
    Convenience wrapper: compute all geometric metrics for one layer.

    Returns dict with keys:
      - hull_centroid_dist       (raw, unnormalized)
      - hull_centroid_dist_norm  (★ primary, scale-invariant)
      - pca_variance_explained
      - h_centroid / a_centroid / hull diameters
    """
    raw, norm, info = pca_hull_centroid_distance(
        human_states, ai_states, n_components=pca_components
    )
    return {
        "hull_centroid_dist": raw,
        "hull_centroid_dist_norm": norm,
        **info,
    }
