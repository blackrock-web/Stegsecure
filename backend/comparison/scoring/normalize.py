"""Benefit / cost normalization. N/A excluded; ties -> 1.0."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence


def normalize_value(
    values: Sequence[Optional[float]],
    *,
    higher_is_better: bool = True,
) -> List[Optional[float]]:
    """Min-max normalize a column. N/A (None) stays None and is excluded from range."""
    present = [v for v in values if v is not None]
    if not present:
        return [None] * len(values)
    lo, hi = min(present), max(present)
    if abs(hi - lo) < 1e-12:
        # ties -> 1.0 for all present
        return [1.0 if v is not None else None for v in values]
    out: List[Optional[float]] = []
    for v in values:
        if v is None:
            out.append(None)
        elif higher_is_better:
            out.append((v - lo) / (hi - lo))
        else:
            out.append((hi - v) / (hi - lo))
    return out


def normalize_table(
    rows: List[Dict[str, Any]],
    columns: Dict[str, bool],
) -> List[Dict[str, Any]]:
    """
    columns: metric_name -> higher_is_better
    Adds metric_name_norm to each row.
    """
    if not rows:
        return rows
    for col, hib in columns.items():
        vals = [r.get(col) for r in rows]
        norms = normalize_value(vals, higher_is_better=hib)
        for r, n in zip(rows, norms):
            r[f"{col}_norm"] = n
    return rows
