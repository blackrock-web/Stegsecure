"""Per-pixel modification cost for candidate deltas {-2,-1,0,+1,+2}."""

from __future__ import annotations
import numpy as np
from typing import Optional


def pixel_modification_cost(
    cover_val: int,
    delta: int,
    cnn_cost: float,
    gradient_mag: float = 0.0,
    channel_weight: float = 1.0,
) -> float:
    """
    Combined distortion for changing cover_val by delta.

    cost = |delta| * cnn_cost * channel_weight
         + 0.25 * |delta| * (1 - gradient_mag)   # prefer textured
         + 0.1 * (delta ** 2)                    # quadratic magnitude penalty
    """
    if delta == 0:
        return 0.0
    new_val = cover_val + delta
    if new_val < 0 or new_val > 255:
        return 1e9
    mag = abs(delta)
    return (
        mag * float(cnn_cost) * channel_weight
        + 0.25 * mag * (1.0 - float(gradient_mag))
        + 0.1 * (delta ** 2)
    )


def candidate_deltas(k: int = 1) -> list:
    """Allowed modification magnitudes for LSB/OPAP-style (default ±1)."""
    return list(range(-k, k + 1))
