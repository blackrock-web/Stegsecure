"""Friedman test + Nemenyi post-hoc across methods per metric."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def summarise_repetitions(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.array([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": int(arr.size),
    }


def friedman_test(
    method_metric_matrix: np.ndarray,
    method_names: Sequence[str],
    *,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    method_metric_matrix: shape (n_blocks, n_methods) — e.g. images × methods.
    Missing values should already be dropped so every row is complete.
    """
    if not HAS_SCIPY:
        return {
            "test": "Friedman",
            "statistic": None,
            "p_value": None,
            "significant": None,
            "status": "FAILED",
            "reason": "scipy not available",
        }
    if method_metric_matrix.ndim != 2 or method_metric_matrix.shape[0] < 2 or method_metric_matrix.shape[1] < 2:
        return {
            "test": "Friedman",
            "statistic": None,
            "p_value": None,
            "significant": False,
            "status": "N/A",
            "reason": "need ≥2 blocks and ≥2 methods",
        }

    try:
        stat, p = stats.friedmanchisquare(*[method_metric_matrix[:, j] for j in range(method_metric_matrix.shape[1])])
    except Exception as e:
        return {
            "test": "Friedman",
            "statistic": None,
            "p_value": None,
            "significant": None,
            "status": "FAILED",
            "reason": str(e),
        }

    significant = bool(p < alpha)
    result: Dict[str, Any] = {
        "test": "Friedman",
        "statistic": float(stat),
        "p_value": float(p),
        "significant": significant,
        "alpha": alpha,
        "methods": list(method_names),
        "status": "ok",
        "interpretation": (
            "statistically significant difference among methods"
            if significant
            else "not statistically significant"
        ),
    }

    if significant:
        result["nemenyi"] = nemenyi_posthoc(method_metric_matrix, method_names, alpha=alpha)
    else:
        result["nemenyi"] = None
    return result


def nemenyi_posthoc(
    matrix: np.ndarray,
    method_names: Sequence[str],
    *,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Critical-difference Nemenyi post-hoc on average ranks.
    Simplified CD table for common k; pairwise declare significant if rank-diff > CD.
    """
    n, k = matrix.shape
    ranks = np.zeros_like(matrix, dtype=float)
    for i in range(n):
        ranks[i] = stats.rankdata(-matrix[i]) if False else stats.rankdata(matrix[i])
        # For benefit metrics caller should pass already-oriented values;
        # we rank ascending (lower rank = better for cost-style, higher raw = higher rank).
        # Use average rank of values as-is.
    avg_ranks = ranks.mean(axis=0)
    # Critical difference (studentized range q_alpha approx for alpha=0.05)
    q = {2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031}.get(k, 3.0)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n))

    pairwise = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(avg_ranks[i] - avg_ranks[j])
            pairwise.append({
                "method_a": method_names[i],
                "method_b": method_names[j],
                "rank_diff": float(diff),
                "critical_difference": float(cd),
                "significant": bool(diff > cd),
            })
    return {
        "average_ranks": {method_names[i]: float(avg_ranks[i]) for i in range(k)},
        "critical_difference": float(cd),
        "pairwise": pairwise,
        "alpha": alpha,
    }
