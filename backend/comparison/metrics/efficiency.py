"""Timing and memory efficiency metrics."""

from __future__ import annotations
from typing import Any, Dict, Optional
import time

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def device_label() -> str:
    if HAS_TORCH and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def peak_rss_mb() -> Optional[float]:
    if not HAS_PSUTIL:
        return None
    try:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def peak_cuda_mb() -> Optional[float]:
    if not (HAS_TORCH and torch.cuda.is_available()):
        return None
    try:
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        return None


class Timer:
    def __init__(self):
        self.t0 = None
        self.elapsed = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.t0


def compute_efficiency(
    embed_s: Optional[float] = None,
    extract_s: Optional[float] = None,
    preprocess_s: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "embed_time_s": embed_s,
        "extract_time_s": extract_s,
        "preprocess_time_s": preprocess_s,
        "device": device_label(),
        "peak_rss_mb": peak_rss_mb(),
        "peak_cuda_mb": peak_cuda_mb(),
    }
