"""
Multi-objective scoring for candidate pixel modifications.

J = λ1·Distortion + λ2·DetectionProbability + λ3·EmbeddingError + λ4·ModificationCount

All λ coefficients are configurable. Randomness is controlled via seed only when
explicitly requested (e.g. tie-breaking).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class ObjectiveWeights:
    lambda_distortion: float = 1.0
    lambda_detection: float = 0.5
    lambda_error: float = 10.0   # must satisfy message bit — high penalty if not
    lambda_modcount: float = 0.1
    adversarial_weight: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def score_candidate(
    distortion: float,
    detection_prob: float,
    embeds_correctly: bool,
    mod_count: int,
    adversarial_gain: float = 0.0,
    weights: Optional[ObjectiveWeights] = None,
) -> float:
    """
    Lower score is better.
    detection_prob in [0,1] (higher = worse for security).
    adversarial_gain: positive when modification moves against detector gradient.
    """
    w = weights or ObjectiveWeights()
    err = 0.0 if embeds_correctly else 1.0
    j = (
        w.lambda_distortion * distortion
        + w.lambda_detection * detection_prob
        + w.lambda_error * err
        + w.lambda_modcount * mod_count
        - w.adversarial_weight * adversarial_gain
    )
    return float(j)


def select_best_modification(
    candidates: List[Tuple[float, int, int]],
    *,
    weights: Optional[ObjectiveWeights] = None,
    detection_prob: float = 0.5,
) -> Tuple[int, float]:
    """
    candidates: list of (distortion, delta, new_value) that already satisfy the
    embedding constraint (embeds_correctly=True).
    Returns (best_new_value, best_score).
    """
    if not candidates:
        raise ValueError("No valid candidates")
    best_score = 1e18
    best_val = candidates[0][2]
    for dist, delta, nv in candidates:
        s = score_candidate(
            distortion=dist,
            detection_prob=detection_prob,
            embeds_correctly=True,
            mod_count=0 if delta == 0 else 1,
            adversarial_gain=0.0,
            weights=weights,
        )
        if s < best_score:
            best_score = s
            best_val = nv
    return best_val, best_score
