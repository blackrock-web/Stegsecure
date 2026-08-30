"""Isolated temporary directories and persistent job storage for batch jobs."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
TMP_ROOT = ROOT / "tmp"
EXPERIMENTS_ROOT = ROOT / "experiments"
JOBS_ROOT = ROOT / "experiments" / "batch_jobs"

SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._\-]+")


def sanitize_filename(name: str, max_len: int = 120) -> str:
    base = os.path.basename(name or "image.png")
    base = SAFE_NAME_RE.sub("_", base).strip("._") or "image.png"
    if len(base) > max_len:
        stem, ext = os.path.splitext(base)
        base = stem[: max_len - len(ext) - 1] + ext
    return base


def ensure_dirs() -> None:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_ROOT.mkdir(parents=True, exist_ok=True)
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)


def job_tmp_dir(job_id: str) -> Path:
    ensure_dirs()
    d = TMP_ROOT / job_id
    for sub in ("input", "processing", "output"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def job_meta_path(job_id: str) -> Path:
    ensure_dirs()
    return JOBS_ROOT / f"{job_id}.json"


def save_job_meta(job_dict: Dict[str, Any]) -> Path:
    ensure_dirs()
    p = job_meta_path(job_dict["job_id"])
    # strip secrets if present
    cfg = dict(job_dict.get("configuration") or {})
    cfg.pop("passphrase", None)
    cfg.pop("secret_text", None)
    payload = {**job_dict, "configuration": cfg}
    with open(p, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return p


def load_job_meta(job_id: str) -> Optional[Dict[str, Any]]:
    p = job_meta_path(job_id)
    if not p.is_file():
        return None
    with open(p) as f:
        return json.load(f)


def list_job_ids() -> List[str]:
    ensure_dirs()
    return sorted(
        [p.stem for p in JOBS_ROOT.glob("batch_*.json")],
        reverse=True,
    )


def cleanup_job_tmp(job_id: str) -> None:
    d = TMP_ROOT / job_id
    if d.is_dir():
        shutil.rmtree(d, ignore_errors=True)


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def experiment_dir(experiment_id: str) -> Path:
    ensure_dirs()
    d = EXPERIMENTS_ROOT / experiment_id
    d.mkdir(parents=True, exist_ok=True)
    return d
