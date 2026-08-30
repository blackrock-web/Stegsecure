"""Pareto dominance analysis across methods."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence


def _better(a: float, b: float, higher_is_better: bool) -> bool:
    return a > b if higher_is_better else a < b


def _not_worse(a: float, b: float, higher_is_better: bool) -> bool:
    return a >= b if higher_is_better else a <= b


def dominates(
    a: Dict[str, Optional[float]],
    b: Dict[str, Optional[float]],
    directions: Dict[str, bool],
) -> bool:
    """
    A dominates B if A is not worse on every applicable metric and strictly
    better on at least one. N/A metrics are excluded from that pairwise check.
    directions: metric -> higher_is_better
    """
    any_strict = False
    compared = 0
    for metric, hib in directions.items():
        va, vb = a.get(metric), b.get(metric)
        if va is None or vb is None:
            continue
        compared += 1
        if not _not_worse(va, vb, hib):
            return False
        if _better(va, vb, hib):
            any_strict = True
    return compared > 0 and any_strict


def pareto_analysis(
    methods: List[Dict[str, Any]],
    directions: Dict[str, bool],
    *,
    id_key: str = "method_id",
) -> Dict[str, Any]:
    """
    methods: list of dicts with id_key and metric values.
    Returns pareto_optimal, dominated, dominance_map, per-method status.
    """
    ids = [m[id_key] for m in methods]
    metrics_by_id = {m[id_key]: m for m in methods}
    dominance_map: Dict[str, List[str]] = {i: [] for i in ids}
    dominated_by: Dict[str, List[str]] = {i: [] for i in ids}

    for a_id in ids:
        for b_id in ids:
            if a_id == b_id:
                continue
            if dominates(metrics_by_id[a_id], metrics_by_id[b_id], directions):
                dominance_map[a_id].append(b_id)
                dominated_by[b_id].append(a_id)

    pareto_optimal = [i for i in ids if not dominated_by[i]]
    dominated = [i for i in ids if dominated_by[i]]

    per_method = []
    for i in ids:
        per_method.append({
            "method_id": i,
            "pareto_status": "optimal" if i in pareto_optimal else "dominated",
            "dominated_by": dominated_by[i],
            "dominates": dominance_map[i],
        })

    return {
        "pareto_optimal_methods": pareto_optimal,
        "dominated_methods": dominated,
        "dominance_map": dominance_map,
        "per_method": per_method,
    }


# Default metric directions for live comparison
DEFAULT_DIRECTIONS = {
    "psnr": True,
    "ssim": True,
    "mse": False,
    "bpp": True,
    "extraction_accuracy": True,
    "ber": False,
    "embed_time_s": False,
    "detection_accuracy": False,  # lower detection = more secure
    "roc_auc": False,
}
