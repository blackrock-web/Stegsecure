"""
Experimental cost-ordered syndrome coding approximation.

Algorithm (research approximation, NOT classical STC):
  1. Sort candidate pixel positions by ascending embedding cost.
  2. For each message bit, scan the ordered list and choose the lowest-cost
     valid ±1 modification that produces the required parity / syndrome bit.
  3. Record modifications for later extraction (parity on LSB of selected pixels).

This approximates the "use cheapest positions first" behaviour of STC while
remaining simple and CPU-friendly. A full Viterbi trellis is left as future work.
"""

from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np

from .cost import pixel_modification_cost


def stc_embed_bits(
    image_flat: np.ndarray,
    positions: np.ndarray,
    message_bits: List[int],
    costs: np.ndarray,
    *,
    gradient: Optional[np.ndarray] = None,
    adversarial_signs: Optional[np.ndarray] = None,
    adversarial_weight: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """
    Embed message_bits into image_flat at the given positions using a
    cost-minimizing ±1 modification strategy.

    Returns (modified_image_flat, bits_embedded).
    """
    if len(positions) == 0 or len(message_bits) == 0:
        return image_flat, 0

    # Order positions by cost (cheapest first)
    pos_costs = costs[positions] if costs is not None else np.zeros(len(positions))
    order = np.argsort(pos_costs)
    ordered_pos = positions[order]

    img = image_flat.copy()
    bit_idx = 0
    used = 0

    for p in ordered_pos:
        if bit_idx >= len(message_bits):
            break
        target_bit = message_bits[bit_idx] & 1
        cur = int(img[p])
        cur_lsb = cur & 1

        if cur_lsb == target_bit:
            # already matches — no change
            bit_idx += 1
            used += 1
            continue

        # Need to flip LSB: prefer direction that matches adversarial sign if available
        candidates = []
        for d in (-1, 1):
            nv = cur + d
            if 0 <= nv <= 255 and (nv & 1) == target_bit:
                cnn_c = float(pos_costs[order[used]] if used < len(order) else 0.5)
                gmag = 0.0
                if gradient is not None and p < len(gradient):
                    gmag = float(abs(gradient[p])) if np.ndim(gradient) == 1 else 0.0
                base = pixel_modification_cost(cur, d, cnn_c, gmag)
                if adversarial_signs is not None and adversarial_weight > 0 and p < len(adversarial_signs):
                    # lower cost if delta aligns with desired adversarial direction
                    align = d * int(adversarial_signs[p])
                    base -= adversarial_weight * 0.5 * max(0, align)
                candidates.append((base, d, nv))

        if not candidates:
            continue
        candidates.sort(key=lambda x: x[0])
        _, _, best_val = candidates[0]
        img[p] = best_val
        bit_idx += 1
        used += 1

    return img, bit_idx


def stc_extract_bits(
    image_flat: np.ndarray,
    positions: np.ndarray,
    num_bits: int,
    costs: Optional[np.ndarray] = None,
) -> List[int]:
    """Extract LSBs from the same cost-ordered positions used at embed time."""
    if len(positions) == 0 or num_bits <= 0:
        return []
    if costs is not None:
        order = np.argsort(costs[positions])
        ordered = positions[order]
    else:
        ordered = np.sort(positions)

    bits = []
    for p in ordered:
        if len(bits) >= num_bits:
            break
        bits.append(int(image_flat[p]) & 1)
    return bits
