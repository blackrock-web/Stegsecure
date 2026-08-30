"""Histogram and residual statistics for steganalysis features."""

from __future__ import annotations
import numpy as np
from typing import Dict


def histogram_features(image: np.ndarray) -> Dict[str, float]:
    if image.ndim == 3:
        gray = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype(np.float64)
    else:
        gray = image.astype(np.float64)

    hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    # Adjacent bin differences (sensitive to LSB embedding)
    adj_diff = np.abs(np.diff(hist)).mean()
    residual = np.abs(np.diff(gray, axis=1)).mean() if gray.shape[1] > 1 else 0.0
    return {
        "hist_adj_diff": float(adj_diff),
        "mean_residual": float(residual),
        "pixel_variance": float(np.var(gray)),
        "entropy_approx": float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0] + 1e-12))),
    }
