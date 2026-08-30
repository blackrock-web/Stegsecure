"""Batch job / item data models for SecureStegVault v3.2."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_job_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"batch_{ts}_{uuid.uuid4().hex[:6]}"


def new_item_id() -> str:
    return f"item_{uuid.uuid4().hex[:8]}"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ItemStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    ENCODE = "encode"
    DECODE = "decode"
    EXPERIMENT = "experiment"


@dataclass
class BatchConfig:
    """Shared configuration applied to every item in a batch."""

    engine: str = "python"  # python | typescript | auto
    strategy: str = "cnn_emd_opap"  # strategy registry name
    message_mode: str = "same"  # same | per_image
    secret_text: str = ""
    passphrase: str = ""
    thresh_a: float = 0.35
    thresh_b: float = 0.65
    gamma: float = 0.7
    kb_bits: int = 2
    kc_bits: int = 3
    cost_map_mode: str = "cnn"
    adversarial_strength: float = 0.0
    emd_n: int = 2
    target_bpp: Optional[float] = None
    workers: int = 2
    seed: int = 42
    # experiment matrix (optional)
    strategies: List[str] = field(default_factory=list)
    bpp_list: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Never export secrets in public metadata
        d.pop("passphrase", None)
        d.pop("secret_text", None)
        return d

    def to_dict_full(self) -> Dict[str, Any]:
        """Internal only — includes secrets for worker use."""
        return asdict(self)


@dataclass
class BatchItem:
    id: str = field(default_factory=new_item_id)
    filename: str = ""
    safe_filename: str = ""
    status: str = ItemStatus.QUEUED.value
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    processing_time_s: Optional[float] = None
    # per-image message (message_mode=per_image)
    secret_text: Optional[str] = None
    # paths relative to job temp dir
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    # experiment tags
    strategy: Optional[str] = None
    bpp_target: Optional[float] = None

    def to_dict(self, include_result: bool = True) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("secret_text", None)  # never leak
        if not include_result:
            d.pop("result", None)
        elif d.get("result") and isinstance(d["result"], dict):
            # strip large base64 blobs from list views
            r = dict(d["result"])
            if "visuals" in r:
                r["visuals"] = {k: ("<omitted>" if isinstance(v, str) and len(v) > 64 else v)
                                for k, v in r["visuals"].items()}
            d["result"] = r
        return d


@dataclass
class BatchJob:
    job_id: str = field(default_factory=new_job_id)
    type: str = JobType.ENCODE.value
    status: str = JobStatus.QUEUED.value
    created_at: str = field(default_factory=_now)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    total: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    configuration: Dict[str, Any] = field(default_factory=dict)
    items: List[BatchItem] = field(default_factory=list)
    experiment_id: Optional[str] = None
    error: Optional[str] = None
    workers: int = 2
    # internal
    _cancel_requested: bool = field(default=False, repr=False)
    _pause_requested: bool = field(default=False, repr=False)

    def recompute_counts(self) -> None:
        self.completed = sum(1 for i in self.items if i.status == ItemStatus.COMPLETED.value)
        self.failed = sum(1 for i in self.items if i.status == ItemStatus.FAILED.value)
        self.cancelled = sum(1 for i in self.items if i.status == ItemStatus.CANCELLED.value)
        self.total = len(self.items)

    def finalize_status(self) -> None:
        self.recompute_counts()
        if self._cancel_requested and self.completed + self.failed + self.cancelled >= self.total:
            self.status = JobStatus.CANCELLED.value
        elif self.failed == self.total and self.total > 0:
            self.status = JobStatus.FAILED.value
        elif self.failed > 0:
            self.status = JobStatus.COMPLETED_WITH_ERRORS.value
        else:
            self.status = JobStatus.COMPLETED.value
        self.completed_at = _now()

    def to_dict(self, include_item_results: bool = False) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "type": self.type,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "cancelled": self.cancelled,
            "configuration": self.configuration,
            "experiment_id": self.experiment_id,
            "error": self.error,
            "workers": self.workers,
            "items": [i.to_dict(include_result=include_item_results) for i in self.items],
        }
