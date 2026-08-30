"""
OPAP (Optimal Pixel Adjustment Process) Module for SecureStegVault
Formula reference: Chan & Cheng (2004) "Efficient data hiding scheme based on multidimensional LSB substitution and optimal pixel adjustment process"

Algorithm (for Zone B and Zone C pixels):
1. Direct k-bit LSB substitution:
   p'_i = (p_i & ~((1 << k) - 1)) | secret_k_bits
   delta = p'_i - p_i

2. Candidates for final stego pixel p''_i:
   Candidate 1: p'_i
   Candidate 2: p'_i + 2^k
   Candidate 3: p'_i - 2^k

3. Selection:
   Choose candidate p''_i that minimizes |p''_i - p_i| subject to 0 <= p''_i <= 255.
   When candidates have equal or near-equal distortion (within 1), select candidate matching
   the adversarial gradient sign when adversarial_strength > 0.
"""

import random
import numpy as np
from typing import List, Tuple, Optional


def opap_embed_pixel(
    orig_pixel: int,
    secret_k_bits: int,
    k: int,
    adv_sign: int = 0,
    adv_strength: float = 0.0,
) -> int:
    """
    Embeds k secret bits into orig_pixel using k-bit LSB substitution followed by OPAP correction.
    Optionally guided by adversarial sign direction.
    Returns optimal stego pixel in [0, 255].
    """
    mask = (1 << k) - 1
    # Step 1: k-bit LSB substitution
    p_prime = (orig_pixel & ~mask) | (secret_k_bits & mask)
    
    step = 1 << k  # 2^k
    
    # Step 2: Evaluate candidates
    candidates = [p_prime, p_prime + step, p_prime - step]
    valid_cands = [c for c in candidates if 0 <= c <= 255]
    
    if not valid_cands:
        return orig_pixel

    # Find minimum difference
    diffs = [abs(c - orig_pixel) for c in valid_cands]
    min_diff = min(diffs)

    # Candidates within 1 unit of min_diff
    tolerated_cands = [c for c in valid_cands if abs(c - orig_pixel) <= min_diff + 1]

    if adv_sign != 0 and adv_strength > 0.0 and random.random() < adv_strength and len(tolerated_cands) > 1:
        # Score candidates by alignment with adv_sign direction
        best_cand = min_diff
        best_score = -999
        best_cand_val = valid_cands[diffs.index(min_diff)]
        for cand in tolerated_cands:
            delta = cand - orig_pixel
            score = delta * adv_sign
            cand_diff = abs(delta)
            # Prefer higher alignment score, breaking ties by lower diff
            if score > best_score or (score == best_score and cand_diff < abs(best_cand_val - orig_pixel)):
                best_score = score
                best_cand_val = cand
        return best_cand_val

    # Default OPAP selection (minimal distortion)
    best_pixel = valid_cands[diffs.index(min_diff)]
    return best_pixel


def opap_extract_pixel(stego_pixel: int, k: int) -> int:
    """Extracts k secret bits from stego_pixel."""
    mask = (1 << k) - 1
    return stego_pixel & mask


def embed_opap_zone(
    image_flat: np.ndarray,
    zone_indices: np.ndarray,
    bitstream: List[int],
    k: int,
    adversarial_signs: Optional[np.ndarray] = None,
    adversarial_strength: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """
    Embeds a stream of bits (0 or 1) into specified zone pixel indices using k-bit OPAP.
    Modifies image_flat in-place and returns (image_flat, num_bits_embedded).
    """
    num_pixels = len(zone_indices)
    max_bits = num_pixels * k
    bits_to_embed = min(len(bitstream), max_bits)
    
    pixels_needed = (bits_to_embed + k - 1) // k
    
    bit_idx = 0
    for i in range(pixels_needed):
        idx = zone_indices[i]
        orig_p = image_flat[idx]
        
        # Read next k bits from bitstream
        chunk_bits = 0
        bits_in_chunk = min(k, bits_to_embed - bit_idx)
        for b in range(bits_in_chunk):
            chunk_bits = (chunk_bits << 1) | bitstream[bit_idx + b]
            
        # If fewer than k bits available, pad lower bits or align
        if bits_in_chunk < k:
            chunk_bits = chunk_bits << (k - bits_in_chunk)

        adv_sign = 0
        if adversarial_signs is not None:
            adv_sign = int(adversarial_signs[idx])

        stego_p = opap_embed_pixel(int(orig_p), chunk_bits, k, adv_sign, adversarial_strength)
        image_flat[idx] = int(np.clip(stego_p, 0, 255))

        bit_idx += bits_in_chunk

    return image_flat, bit_idx


def extract_opap_zone(
    image_flat: np.ndarray,
    zone_indices: np.ndarray,
    num_bits: int,
    k: int,
) -> List[int]:
    """Extracts bits from specified zone pixel indices using k-bit OPAP."""
    extracted_bits = []
    bit_idx = 0
    i = 0
    
    while bit_idx < num_bits and i < len(zone_indices):
        idx = zone_indices[i]
        stego_p = image_flat[idx]
        extracted_k = opap_extract_pixel(stego_p, k)
        
        bits_to_take = min(k, num_bits - bit_idx)
        # Extract bits from MSB to LSB of the k-bit chunk
        for b in range(bits_to_take):
            shift = (k - 1 - b)
            bit_val = (extracted_k >> shift) & 1
            extracted_bits.append(bit_val)
            bit_idx += 1
            
        i += 1
        
    return extracted_bits
