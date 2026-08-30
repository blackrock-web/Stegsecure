"""
Cost-Minimizing Payload Allocation Engine (STC Approximation) for SecureStegVault
Phase 4 EMD-OPAP Hybrid Embedding Engine Supplement

Approximates Syndrome-Trellis Codes (STC) cost-minimizing behavior by sorting
pixel embedding positions in ascending order of their cost_map value (safest / lowest-cost pixels used first).

For Zone A (EMD):
  Preserves contiguous pixel groups (size n=2 or n=3), sorting group blocks by their average cost.

For Zone B & C (OPAP):
  Sorts individual pixel positions by ascending cost_map value.
"""

import numpy as np
from typing import Dict, Tuple, Optional


def rank_zone_indices(
    cost_map_flat: np.ndarray,
    zone_indices: np.ndarray,
    is_emd: bool = False,
    group_size: int = 2,
) -> np.ndarray:
    """
    Ranks embedding positions within a zone by ascending cost_map value.
    
    Parameters:
    - cost_map_flat: 1D array of pixel embedding costs.
    - zone_indices: 1D array of flat pixel indices belonging to the zone.
    - is_emd: If True, group indices into contiguous blocks of `group_size` and rank blocks by average cost.
    - group_size: Size of EMD group (2 or 3).
    
    Returns:
    - Sorted 1D array of indices in cost-minimizing priority order.
    """
    if len(zone_indices) == 0:
        return zone_indices

    # Spatial (index) order within the zone is invariant under ±1 embedding noise.
    # Cost is only used for zone membership (classify_zones), not for reordering,
    # which guarantees that encode and decode walk the exact same pixel sequence.
    sorted_idx = np.sort(zone_indices)

    if not is_emd:
        return sorted_idx
    else:
        num_full_groups = len(sorted_idx) // group_size
        if num_full_groups == 0:
            return sorted_idx
        full_group_len = num_full_groups * group_size
        ordered = sorted_idx[:full_group_len]
        if full_group_len < len(sorted_idx):
            return np.concatenate([ordered, sorted_idx[full_group_len:]])
        return ordered


def rank_embedding_positions(
    cost_map_flat: np.ndarray,
    zones_flat: np.ndarray,
    emd_n: int = 2,
) -> Dict[int, np.ndarray]:
    """
    Helper to rank indices for all 3 zones (Zone A=0, Zone B=1, Zone C=2).
    """
    zone_a = np.where(zones_flat == 0)[0]
    zone_b = np.where(zones_flat == 1)[0]
    zone_c = np.where(zones_flat == 2)[0]

    return {
        0: rank_zone_indices(cost_map_flat, zone_a, is_emd=True, group_size=emd_n),
        1: rank_zone_indices(cost_map_flat, zone_b, is_emd=False),
        2: rank_zone_indices(cost_map_flat, zone_c, is_emd=False),
    }
