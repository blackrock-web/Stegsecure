"""Scoring: normalize, weights, Pareto, statistics."""

from .weights import WeightConfig, DEFAULT_WEIGHTS
from .normalize import normalize_value, normalize_table

__all__ = ["WeightConfig", "DEFAULT_WEIGHTS", "normalize_value", "normalize_table"]
