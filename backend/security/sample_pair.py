"""Sample Pair Analysis (SPA) style feature — simplified educational version."""

from __future__ import annotations
import numpy as np
from typing import Dict


def sample_pair_analysis(image: np.ndarray) -> Dict[str, float]:
    if image.ndim == 3:
        gray = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype(np.int32)
    else:
        gray = image.astype(np.int32)

    # Horizontal pairs
    x = gray[:, :-1].ravel()
    y = gray[:, 1:].ravel()
    d = y - x
    # Count transitions useful for SPA
    even_even = np.sum((x % 2 == 0) & (y % 2 == 0))
    odd_odd = np.sum((x % 2 == 1) & (y % 2 == 1))
    even_odd = np.sum((x % 2 == 0) & (y % 2 == 1))
    odd_even = np.sum((x % 2 == 1) & (y % 2 == 0))
    total = max(len(x), 1)
    # Simple imbalance score
    imbalance = abs(even_odd - odd_even) / total
    return {
        "even_even": float(even_even / total),
        "odd_odd": float(odd_odd / total),
        "even_odd": float(even_odd / total),
        "odd_even": float(odd_even / total),
        "imbalance": float(imbalance),
        "suspicion": float(np.clip(imbalance * 4, 0, 1)),
    }
