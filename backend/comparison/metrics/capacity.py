"""bpp / total bits / % of theoretical max — adapter-aware."""

from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np


def compute_capacity(
    cover: np.ndarray,
    bits_embedded: int,
    *,
    adapter_used: str = "bitstream",
    tile_capacity_bits: Optional[int] = None,
) -> Dict[str, Any]:
    H, W = cover.shape[:2]
    C = cover.shape[2] if cover.ndim == 3 else 1
    total_pixels = H * W * C
    theoretical_max_1lsb = total_pixels  # 1 bit per channel sample

    if adapter_used == "image_tile" and tile_capacity_bits is not None:
        bpp = bits_embedded / max(1, tile_capacity_bits)
        return {
            "bits_embedded": bits_embedded,
            "bpp": round(bpp, 6),
            "theoretical_max_bits": tile_capacity_bits,
            "pct_of_max": round(100.0 * bits_embedded / max(1, tile_capacity_bits), 4),
            "unit": "bits-per-tile",
        }

    bpp = bits_embedded / max(1, total_pixels)
    return {
        "bits_embedded": bits_embedded,
        "bpp": round(bpp, 6),
        "theoretical_max_bits": theoretical_max_1lsb,
        "pct_of_max": round(100.0 * bits_embedded / max(1, theoretical_max_1lsb), 4),
        "unit": "bits-per-pixel-channel",
    }
