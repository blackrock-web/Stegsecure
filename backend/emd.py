"""
EMD (Exploiting Modification Direction) Module for SecureStegVault
Formula reference: Zhang & Wang (2006) "Exploiting modification direction for data hiding"

Algorithm (for Zone A pixels):
1. Partition Zone A pixels into non-overlapping groups of n pixels (default n = 2, optional n = 3).
2. Base: 2*n + 1 (for n=2, base is 5; for n=3, base is 7).
3. Extraction function: f(g1, ..., gn) = [ sum_{i=1}^{n} (gi * i) ] mod (2n + 1)
   For n=2: f(g1, g2) = (g1 * 1 + g2 * 2) mod 5.
   For n=3: f(g1, g2, g3) = (g1 * 1 + g2 * 2 + g3 * 3) mod 7.
4. To embed base-5/base-7 digit d:
   if d == f: no change
   else: s = (d - f) mod (2n + 1)
         if s <= n: increment pixel g_s by 1
         if s > n: decrement pixel g_(2n+1-s) by 1
5. Maximum modification per group: exactly +-1 on at most one pixel (with boundary protection).
6. Adversarial Guidance: When multiple candidate modifications exist, guidance selects direction
   matching the surrogate steganalyzer's adversarial sign gradient to reduce detection risk.
"""

import math
import random
import numpy as np
from typing import List, Tuple, Optional


def emd_embed_group_n2(
    g1: int,
    g2: int,
    digit: int,
    adv_signs: Optional[Tuple[int, int]] = None,
    adv_strength: float = 0.0,
) -> Tuple[int, int]:
    """
    Embeds a base-5 digit (0..4) into a group of 2 pixels (g1, g2) using EMD (Zhang & Wang 2006).
    Optionally guided by adversarial signs to minimize steganalyzer detection.
    """
    f = (int(g1) + 2 * int(g2)) % 5
    if digit == f:
        return g1, g2
    
    # Check if adversarial selection should trigger
    if adv_signs is not None and adv_strength > 0.0 and random.random() < adv_strength:
        candidates = []
        for dg1 in range(-2, 3):
            for dg2 in range(-2, 3):
                p1, p2 = g1 + dg1, g2 + dg2
                if 0 <= p1 <= 255 and 0 <= p2 <= 255:
                    if (p1 + 2 * p2) % 5 == digit:
                        cost = dg1 * dg1 + dg2 * dg2
                        if cost <= 2:  # Low distortion threshold
                            score = dg1 * adv_signs[0] + dg2 * adv_signs[1]
                            candidates.append((cost, -score, p1, p2))
        if candidates:
            candidates.sort(key=lambda x: (x[0], x[1]))
            return candidates[0][2], candidates[0][3]

    # Standard EMD modification
    s = (digit - f) % 5
    target_dg1, target_dg2 = 0, 0
    if s == 1:
        target_dg1 = 1
    elif s == 2:
        target_dg2 = 1
    elif s == 3:
        target_dg2 = -1
    elif s == 4:
        target_dg1 = -1

    cand1, cand2 = g1 + target_dg1, g2 + target_dg2

    # Check bounds [0, 255]
    if 0 <= cand1 <= 255 and 0 <= cand2 <= 255:
        return cand1, cand2

    # Boundary fallback
    best_cost = 999
    best_pair = (g1, g2)
    for dg1 in range(-2, 3):
        for dg2 in range(-2, 3):
            p1, p2 = g1 + dg1, g2 + dg2
            if 0 <= p1 <= 255 and 0 <= p2 <= 255:
                if (p1 + 2 * p2) % 5 == digit:
                    cost = dg1 * dg1 + dg2 * dg2
                    if cost < best_cost:
                        best_cost = cost
                        best_pair = (p1, p2)

    return best_pair


def emd_extract_group_n2(g1: int, g2: int) -> int:
    """Extracts base-5 digit from a group of 2 pixels (g1, g2)."""
    return (int(g1) + 2 * int(g2)) % 5


def emd_embed_group_n3(
    g1: int,
    g2: int,
    g3: int,
    digit: int,
    adv_signs: Optional[Tuple[int, int, int]] = None,
    adv_strength: float = 0.0,
) -> Tuple[int, int, int]:
    """
    Embeds a base-7 digit (0..6) into a group of 3 pixels (g1, g2, g3) using n=3 EMD.
    f(g1, g2, g3) = (g1 + 2*g2 + 3*g3) mod 7.
    """
    f = (int(g1) + 2 * int(g2) + 3 * int(g3)) % 7
    if digit == f:
        return g1, g2, g3

    if adv_signs is not None and adv_strength > 0.0 and random.random() < adv_strength:
        candidates = []
        for dg1 in range(-2, 3):
            for dg2 in range(-2, 3):
                for dg3 in range(-2, 3):
                    p1, p2, p3 = g1 + dg1, g2 + dg2, g3 + dg3
                    if 0 <= p1 <= 255 and 0 <= p2 <= 255 and 0 <= p3 <= 255:
                        if (p1 + 2 * p2 + 3 * p3) % 7 == digit:
                            cost = dg1 * dg1 + dg2 * dg2 + dg3 * dg3
                            if cost <= 2:
                                score = dg1 * adv_signs[0] + dg2 * adv_signs[1] + dg3 * adv_signs[2]
                                candidates.append((cost, -score, p1, p2, p3))
        if candidates:
            candidates.sort(key=lambda x: (x[0], x[1]))
            return candidates[0][2], candidates[0][3], candidates[0][4]

    s = (digit - f) % 7
    target = [0, 0, 0]
    if s == 1:
        target[0] = 1
    elif s == 2:
        target[1] = 1
    elif s == 3:
        target[2] = 1
    elif s == 4:
        target[2] = -1
    elif s == 5:
        target[1] = -1
    elif s == 6:
        target[0] = -1

    cand1, cand2, cand3 = g1 + target[0], g2 + target[1], g3 + target[2]
    if 0 <= cand1 <= 255 and 0 <= cand2 <= 255 and 0 <= cand3 <= 255:
        return cand1, cand2, cand3

    # Boundary fallback
    best_cost = 999
    best_triplet = (g1, g2, g3)
    for dg1 in range(-2, 3):
        for dg2 in range(-2, 3):
            for dg3 in range(-2, 3):
                p1, p2, p3 = g1 + dg1, g2 + dg2, g3 + dg3
                if 0 <= p1 <= 255 and 0 <= p2 <= 255 and 0 <= p3 <= 255:
                    if (p1 + 2 * p2 + 3 * p3) % 7 == digit:
                        cost = dg1 * dg1 + dg2 * dg2 + dg3 * dg3
                        if cost < best_cost:
                            best_cost = cost
                            best_triplet = (p1, p2, p3)
    return best_triplet


def emd_extract_group_n3(g1: int, g2: int, g3: int) -> int:
    """Extracts base-7 digit from a group of 3 pixels (g1, g2, g3)."""
    return (int(g1) + 2 * int(g2) + 3 * int(g3)) % 7


def bytes_to_base5_digits(data_bytes: bytes) -> List[int]:
    """Converts bytes to base-5 digits (4 digits per byte)."""
    digits = []
    for b in data_bytes:
        val = int(b)
        d3 = val // 125
        val %= 125
        d2 = val // 25
        val %= 25
        d1 = val // 5
        d0 = val % 5
        digits.extend([d3, d2, d1, d0])
    return digits


def base5_digits_to_bytes(digits: List[int]) -> bytes:
    """Converts base-5 digits back to bytes."""
    out = bytearray()
    num_groups = len(digits) // 4
    for i in range(num_groups):
        d3, d2, d1, d0 = digits[i * 4 : (i + 1) * 4]
        val = d3 * 125 + d2 * 25 + d1 * 5 + d0
        if val > 255:
            val = 255
        out.append(val)
    return bytes(out)


def bytes_to_base7_digits(data_bytes: bytes) -> List[int]:
    """Converts bytes to base-7 digits (3 digits per byte, 7^3 = 343 > 256)."""
    digits = []
    for b in data_bytes:
        val = int(b)
        d2 = val // 49
        val %= 49
        d1 = val // 7
        d0 = val % 7
        digits.extend([d2, d1, d0])
    return digits


def base7_digits_to_bytes(digits: List[int]) -> bytes:
    """Converts base-7 digits back to bytes."""
    out = bytearray()
    num_groups = len(digits) // 3
    for i in range(num_groups):
        d2, d1, d0 = digits[i * 3 : (i + 1) * 3]
        val = d2 * 49 + d1 * 7 + d0
        if val > 255:
            val = 255
        out.append(val)
    return bytes(out)


def embed_emd_zone_a(
    image_flat: np.ndarray,
    zone_a_indices: np.ndarray,
    digits: List[int],
    emd_n: int = 2,
    adversarial_signs: Optional[np.ndarray] = None,
    adversarial_strength: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """
    Embeds digits into Zone A pixel indices using EMD (n=2 or n=3).
    Works on a temporary int16 view so ±1 modifications never overflow uint8.
    """
    # Work in int16 to avoid uint8 wrap-around / out-of-bounds assignment
    work = image_flat.astype(np.int16)

    if emd_n == 3:
        n_groups = len(zone_a_indices) // 3
        digits_to_embed = min(len(digits), n_groups)
        for i in range(digits_to_embed):
            idx1 = zone_a_indices[i * 3]
            idx2 = zone_a_indices[i * 3 + 1]
            idx3 = zone_a_indices[i * 3 + 2]
            digit = digits[i]

            adv_signs = None
            if adversarial_signs is not None:
                adv_signs = (
                    int(adversarial_signs[idx1]),
                    int(adversarial_signs[idx2]),
                    int(adversarial_signs[idx3]),
                )

            p1, p2, p3 = emd_embed_group_n3(
                int(work[idx1]), int(work[idx2]), int(work[idx3]),
                digit, adv_signs, adversarial_strength,
            )
            work[idx1] = int(np.clip(p1, 0, 255))
            work[idx2] = int(np.clip(p2, 0, 255))
            work[idx3] = int(np.clip(p3, 0, 255))
        image_flat[:] = work.astype(image_flat.dtype)
        return image_flat, digits_to_embed
    else:
        n_groups = len(zone_a_indices) // 2
        digits_to_embed = min(len(digits), n_groups)
        for i in range(digits_to_embed):
            idx1 = zone_a_indices[i * 2]
            idx2 = zone_a_indices[i * 2 + 1]
            digit = digits[i]

            adv_signs = None
            if adversarial_signs is not None:
                adv_signs = (
                    int(adversarial_signs[idx1]),
                    int(adversarial_signs[idx2]),
                )

            p1, p2 = emd_embed_group_n2(
                int(work[idx1]), int(work[idx2]),
                digit, adv_signs, adversarial_strength,
            )
            work[idx1] = int(np.clip(p1, 0, 255))
            work[idx2] = int(np.clip(p2, 0, 255))
        image_flat[:] = work.astype(image_flat.dtype)
        return image_flat, digits_to_embed


def extract_emd_zone_a(
    image_flat: np.ndarray,
    zone_a_indices: np.ndarray,
    num_digits: int,
    emd_n: int = 2,
) -> List[int]:
    """
    Extracts digits from Zone A pixel indices using EMD (n=2 or n=3).
    """
    extracted_digits = []
    if emd_n == 3:
        n_groups = min(num_digits, len(zone_a_indices) // 3)
        for i in range(n_groups):
            idx1 = zone_a_indices[i * 3]
            idx2 = zone_a_indices[i * 3 + 1]
            idx3 = zone_a_indices[i * 3 + 2]
            digit = emd_extract_group_n3(image_flat[idx1], image_flat[idx2], image_flat[idx3])
            extracted_digits.append(digit)
    else:
        n_groups = min(num_digits, len(zone_a_indices) // 2)
        for i in range(n_groups):
            idx1 = zone_a_indices[i * 2]
            idx2 = zone_a_indices[i * 2 + 1]
            digit = emd_extract_group_n2(image_flat[idx1], image_flat[idx2])
            extracted_digits.append(digit)
    return extracted_digits
