"""RS (Regular-Singular) analysis — Fridrich et al. approximate LSB detection."""

from __future__ import annotations
import numpy as np
from typing import Dict


def _flip_lsb_group(block: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = block.astype(np.int32).copy()
    for i, m in enumerate(mask):
        if m == 1:
            out[i] = int(out[i]) ^ 1
        elif m == -1:
            out[i] = (int(out[i]) & ~1) | (1 - (int(out[i]) & 1))
    return out.astype(np.uint8)


def _smoothness(block: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(block.astype(np.float64)))))


def rs_analysis(image: np.ndarray, group_size: int = 4) -> Dict[str, float]:
    """
    Approximate RS analysis on grayscale (or luminance of RGB).
    Returns estimated LSB embedding rate and discrimination scores.
    This is a simplified educational implementation, not a production forensic tool.
    """
    if image.ndim == 3:
        gray = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype(np.uint8)
    else:
        gray = image.astype(np.uint8)

    flat = gray.ravel()
    n = len(flat) - (len(flat) % group_size)
    if n < group_size * 10:
        return {"estimated_rate": 0.0, "R_M": 0.5, "S_M": 0.5, "R_m": 0.5, "S_m": 0.5}

    flat = flat[:n]
    mask_p = np.array([1, 0, 1, 0][:group_size])
    mask_n = -mask_p

    R_M = S_M = R_m = S_m = 0
    total = n // group_size

    for i in range(0, n, group_size):
        g = flat[i : i + group_size]
        f0 = _smoothness(g)
        fp = _smoothness(_flip_lsb_group(g, mask_p))
        fn = _smoothness(_flip_lsb_group(g, mask_n))
        if fp > f0:
            S_M += 1
        elif fp < f0:
            R_M += 1
        if fn > f0:
            S_m += 1
        elif fn < f0:
            R_m += 1

    R_M /= total
    S_M /= total
    R_m /= total
    S_m /= total

    # Rough rate estimate (simplified)
    d0 = R_M - S_M
    d1 = R_m - S_m
    rate = 0.0
    if abs(d0 - d1) > 1e-6:
        rate = max(0.0, min(1.0, abs(d0) / (abs(d0) + abs(d1) + 1e-9)))

    return {
        "estimated_rate": float(rate),
        "R_M": float(R_M),
        "S_M": float(S_M),
        "R_m": float(R_m),
        "S_m": float(S_m),
        "discrimination": float(abs(R_M - S_M) + abs(R_m - S_m)),
    }
