"""
SecureStegVault v3.2 — Batch processing orchestration layer.

Reuses the existing single-image pipeline (strategies / encode / decode).
Does NOT reimplement EMD, OPAP, STC, CNN, crypto, or metrics.
"""

from .models import (
    BatchJob,
    BatchItem,
    BatchConfig,
    JobStatus,
    ItemStatus,
    JobType,
)
from .manager import BatchManager, get_batch_manager

__all__ = [
    "BatchJob",
    "BatchItem",
    "BatchConfig",
    "JobStatus",
    "ItemStatus",
    "JobType",
    "BatchManager",
    "get_batch_manager",
]
