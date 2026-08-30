"""Aggregate per-image metrics into batch summaries — no fabricated numbers."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


def _nums(results: List[Dict], key_path: str) -> List[float]:
    out = []
    for r in results:
        cur: Any = r
        for part in key_path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                cur = None
                break
            cur = cur[part]
        if isinstance(cur, (int, float)) and not (isinstance(cur, float) and math.isnan(cur)):
            out.append(float(cur))
    return out


def _stats(vals: List[float]) -> Dict[str, Optional[float]]:
    if not vals:
        return {"mean": None, "median": None, "min": None, "max": None, "std": None, "n": 0}
    vals = sorted(vals)
    n = len(vals)
    mean = sum(vals) / n
    mid = n // 2
    median = vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2
    var = sum((v - mean) ** 2 for v in vals) / n
    return {
        "mean": round(mean, 6),
        "median": round(median, 6),
        "min": round(vals[0], 6),
        "max": round(vals[-1], 6),
        "std": round(math.sqrt(var), 6),
        "n": n,
    }


def aggregate_job(job_dict: Dict[str, Any]) -> Dict[str, Any]:
    items = job_dict.get("items") or []
    completed = [i for i in items if i.get("status") == "completed" and i.get("result")]
    failed = [i for i in items if i.get("status") == "failed"]
    results = [i["result"] for i in completed]

    metric_keys = [
        ("metrics.psnr_db", "psnr_db"),
        ("metrics.ssim", "ssim"),
        ("metrics.mse", "mse"),
        ("metrics.achieved_bpp", "achieved_bpp"),
        ("metrics.modified_pixel_percentage", "modified_pct"),
        ("security.composite_suspicion", "suspicion"),
        ("security.cnn_stego_probability", "cnn_stego_prob"),
    ]

    metrics_agg = {}
    for path, name in metric_keys:
        metrics_agg[name] = _stats(_nums(results, path))

    times = [
        float(i["processing_time_s"])
        for i in completed
        if isinstance(i.get("processing_time_s"), (int, float))
    ]
    metrics_agg["processing_time_s"] = _stats(times)

    return {
        "job_id": job_dict.get("job_id"),
        "type": job_dict.get("type"),
        "status": job_dict.get("status"),
        "total": job_dict.get("total", len(items)),
        "successful": len(completed),
        "failed": len(failed),
        "cancelled": sum(1 for i in items if i.get("status") == "cancelled"),
        "metrics": metrics_agg,
        "failed_items": [
            {"filename": i.get("filename"), "error": i.get("error")} for i in failed
        ],
    }
