"""Duplicate-group detection, shared by the analysis script and the split script.

Why this exists: the dataset contains many crops of the SAME physical tile. If crops of one tile land
in both train and test, the test score is inflated (the model has effectively seen the test image).
So we find groups of near-identical images and later keep every group inside ONE split.

How: shrink each image to 32x32 grayscale, standardise it, and compare images with Pearson correlation.
Re-crops of the same tile correlate at ~0.98-1.0; different tiles are lower. Images linked by
'correlation > threshold' form a connected component = one group.
"""
import numpy as np
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

GROUP_THRESHOLD = 0.98  # chosen in Phase 1 (see docs/dataset_analysis.md, sensitivity table)


def thumbnail_matrix(paths, size: int = 32) -> np.ndarray:
    """Each image -> size x size grayscale -> zero-mean, unit-norm vector (rows of the result)."""
    T = np.stack([
        np.asarray(Image.open(p).convert("L").resize((size, size), Image.LANCZOS), dtype=np.float32).ravel()
        for p in paths
    ])
    T -= T.mean(1, keepdims=True)
    return T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-6)


def duplicate_groups(T: np.ndarray, thr: float = GROUP_THRESHOLD) -> np.ndarray:
    """Group id per image (connected components of the graph 'correlation > thr')."""
    C = T @ T.T  # dot product of unit vectors = Pearson correlation
    np.fill_diagonal(C, 0)
    return connected_components(csr_matrix(C > thr), directed=False)[1]
