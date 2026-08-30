"""PSNR / SSIM / MSE — reuses backend/metrics.py."""

from __future__ import annotations
from typing import Any, Dict
import numpy as np


def compute_quality(cover: np.ndarray, stego: np.ndarray) -> Dict[str, Any]:
    from backend.metrics import calculate_metrics

    m = calculate_metrics(cover, stego, 0, {})
    psnr = m.get("psnr_db", m.get("psnr"))
    return {
        "psnr": psnr,
        "psnr_db": psnr,
        "ssim": m.get("ssim"),
        "mse": m.get("mse"),
    }
