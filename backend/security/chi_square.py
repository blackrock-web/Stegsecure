"""Chi-square attack on LSB pairs (Westfeld & Pfitzmann style, simplified)."""

from __future__ import annotations
import numpy as np
from typing import Dict


def chi_square_analysis(image: np.ndarray) -> Dict[str, float]:
    if image.ndim == 3:
        gray = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype(np.uint8)
    else:
        gray = image.astype(np.uint8)

    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    chi = 0.0
    pairs = 0
    for i in range(0, 256, 2):
        n1, n2 = hist[i], hist[i + 1]
        expected = (n1 + n2) / 2.0
        if expected > 0:
            chi += ((n1 - expected) ** 2 + (n2 - expected) ** 2) / expected
            pairs += 1
    # p-value approximation via survival function of chi2 (rough)
    # Higher chi → more likely natural (uneven pairs); low chi → possible LSB embedding
    dof = max(pairs - 1, 1)
    # normalized score in [0,1]: 1 = highly suspicious of embedding
    score = float(np.exp(-chi / (2 * dof)))
    return {
        "chi_square": float(chi),
        "pairs": pairs,
        "suspicion": float(np.clip(score, 0, 1)),
    }
