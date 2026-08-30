"""
Adaptive Zoning for SecureStegVault v3

Zones are derived from cost-map percentiles (not fixed 0.35/0.65 thresholds),
with optional fixed-threshold baseline for ablation.

Zone A (low cost / smooth)  → EMD
Zone B (medium cost)        → OPAP k_B
Zone C (high cost / textured) → OPAP k_C

Quantization + morphological median keep zone labels stable under ±1 embedding noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


@dataclass
class ZoningConfig:
    # Percentile mode (default, adaptive)
    percentile_a: float = 35.0   # bottom % → Zone A
    percentile_b: float = 65.0   # up to this % → Zone B; rest Zone C
    # Fixed-threshold baseline (ablation)
    use_fixed_thresholds: bool = False
    thresh_a: float = 0.35
    thresh_b: float = 0.65
    # Embedding params
    emd_group_size: int = 2
    kb_bits: int = 2
    kc_bits: int = 3
    # Stability
    quantize_buckets: int = 20
    median_ksize: int = 5

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = ZoningConfig()


def _quantize(cost: np.ndarray, buckets: int = 20) -> np.ndarray:
    return np.round(cost.astype(np.float64) * buckets) / buckets


def classify_zones(
    cost_map: np.ndarray,
    config: ZoningConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """
    Returns uint8 zone labels same shape as cost_map (or H×W if cost is 2D).
    0 = Zone A, 1 = Zone B, 2 = Zone C.
    """
    if cost_map.ndim == 3:
        cost_2d = cost_map[:, :, 0]
        expand = True
        C = cost_map.shape[2]
    else:
        cost_2d = cost_map
        expand = False
        C = 1

    q = _quantize(cost_2d, config.quantize_buckets)

    if config.use_fixed_thresholds:
        t_a, t_b = config.thresh_a, config.thresh_b
    else:
        flat = q.ravel()
        t_a = float(np.percentile(flat, config.percentile_a))
        t_b = float(np.percentile(flat, config.percentile_b))
        # ensure strict ordering
        if t_b <= t_a:
            t_b = min(1.0, t_a + 0.05)

    z2 = np.zeros(q.shape, dtype=np.uint8)
    z2[(q >= t_a) & (q < t_b)] = 1
    z2[q >= t_b] = 2

    if HAS_CV2 and config.median_ksize >= 3:
        k = config.median_ksize if config.median_ksize % 2 == 1 else config.median_ksize + 1
        z2 = cv2.medianBlur(z2, k)

    if expand:
        return np.repeat(z2[:, :, np.newaxis], C, axis=2)
    return z2


def zone_boundaries(cost_map: np.ndarray, config: ZoningConfig) -> Dict[str, float]:
    """Return the actual cost thresholds used for this image."""
    if cost_map.ndim == 3:
        flat = cost_map[:, :, 0].ravel()
    else:
        flat = cost_map.ravel()
    q = _quantize(flat, config.quantize_buckets)
    if config.use_fixed_thresholds:
        return {"thresh_a": config.thresh_a, "thresh_b": config.thresh_b, "mode": "fixed"}
    t_a = float(np.percentile(q, config.percentile_a))
    t_b = float(np.percentile(q, config.percentile_b))
    return {
        "thresh_a": t_a,
        "thresh_b": t_b,
        "percentile_a": config.percentile_a,
        "percentile_b": config.percentile_b,
        "mode": "percentile",
    }


def calculate_capacity(
    image_shape: Tuple[int, int, int],
    cost_map: np.ndarray,
    config: ZoningConfig = DEFAULT_CONFIG,
) -> Dict[str, Any]:
    H, W, C = image_shape
    total_pixels = H * W * C
    if cost_map.ndim == 2 and C > 1:
        cost_3d = np.repeat(cost_map[:, :, np.newaxis], C, axis=2)
    else:
        cost_3d = cost_map
    zones = classify_zones(cost_3d, config)
    count_a = int(np.sum(zones == 0))
    count_b = int(np.sum(zones == 1))
    count_c = int(np.sum(zones == 2))

    emd_groups = count_a // config.emd_group_size
    bits_a = emd_groups * math.log2(2 * config.emd_group_size + 1)
    bits_b = count_b * config.kb_bits
    bits_c = count_c * config.kc_bits
    total_bits = bits_a + bits_b + bits_c
    total_bytes = int(total_bits // 8)
    crypto_overhead = 48  # conservative (v3 header is larger)
    max_pt = max(0, total_bytes - crypto_overhead)

    return {
        "total_pixels": total_pixels,
        "count_zone_a": count_a,
        "count_zone_b": count_b,
        "count_zone_c": count_c,
        "max_bits": float(total_bits),
        "max_bytes": total_bytes,
        "max_plaintext_bytes": max_pt,
        "overall_bpp": float(total_bits) / float(total_pixels) if total_pixels else 0.0,
        "zone_a_bpp": float(bits_a) / float(count_a) if count_a else 0.0,
        "zone_b_bpp": float(config.kb_bits),
        "zone_c_bpp": float(config.kc_bits),
        "boundaries": zone_boundaries(cost_map, config),
    }
