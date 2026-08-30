"""Export comparison results as JSON / CSV using existing patterns."""

from __future__ import annotations
from typing import Any, Dict
import csv
import io
import json


def export_json(result: Dict[str, Any]) -> str:
    return json.dumps(result, indent=2, default=str)


def export_csv(result: Dict[str, Any]) -> str:
    rows = result.get("results") or []
    if not rows:
        return "method_id,status\n"
    fieldnames = [
        "method_id", "status", "psnr", "ssim", "mse", "bpp", "ber",
        "extraction_accuracy", "embed_time_s", "overall_score", "rank",
        "pareto_status", "model_status", "adapter_used", "failure_reason",
    ]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()
