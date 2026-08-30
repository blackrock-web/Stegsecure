"""Centralized checkpoint discovery and load verification.

Never auto-downloads unknown models. Only loads files already present under
models/paper{N}/official/ or explicitly supplied paths.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_ROOT = PROJECT_ROOT / "models"

WEIGHT_EXTS = {".pth", ".pt", ".ckpt", ".bin", ".safetensors"}


@dataclass
class CheckpointInfo:
    paper_id: str
    path: Optional[str] = None
    exists: bool = False
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    source_url: Optional[str] = None
    verification_status: str = "not_found"
    notes: str = ""
    discovered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


# Authoritative findings (updated after web + repo audit 2026-08-17)
OFFICIAL_STATUS = {
    "paper1_joint_cnn": {
        "official_repo": "https://github.com/Ayshcoder987/Joint-Encryption-Steganography-CNN",
        "paper_claims_weights": True,
        "repo_contains_weights": False,
        "releases": 0,
        "notes": (
            "Official repo ships architecture + training script only. "
            "No .pth/.pt in tree; GitHub Releases empty. "
            "Paper text claims pretrained models at that URL but none are published. "
            "Place a compatible checkpoint under models/paper1/official/ to upgrade status."
        ),
    },
    "paper2_cyclegan_steg": {
        "official_repo": None,
        "paper_claims_weights": False,
        "repo_contains_weights": False,
        "releases": 0,
        "notes": (
            "NO OFFICIAL COMPATIBLE CHECKPOINT FOUND. "
            "No author GitHub / Zenodo / HF release located for Abdollahi et al. JISA 2023. "
            "Do not substitute unrelated CycleGAN or SteganoGAN weights."
        ),
    },
    "paper3_block_prep_net": {
        "official_repo": None,
        "paper_claims_weights": False,
        "repo_contains_weights": False,
        "releases": 0,
        "notes": (
            "NO OFFICIAL COMPATIBLE CHECKPOINT FOUND. "
            "No author GitHub / IIIT Lucknow code release with checkpoints located "
            "for Dabhade, Chakraborty & Sen MTA 2026."
        ),
    },
    "paper4_lsb_magicmatrix": {
        "official_repo": None,
        "paper_claims_weights": False,
        "repo_contains_weights": False,
        "notes": "Classical deterministic algorithm — no neural checkpoint required.",
    },
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_checkpoints(paper_id: str) -> CheckpointInfo:
    """Scan models/<paper>/official/ for weight files."""
    meta = OFFICIAL_STATUS.get(paper_id, {})
    # map method id to folder
    folder_map = {
        "paper1_joint_cnn": "paper1",
        "paper2_cyclegan_steg": "paper2",
        "paper3_block_prep_net": "paper3",
        "paper4_lsb_magicmatrix": "paper4",
    }
    folder = MODELS_ROOT / folder_map.get(paper_id, paper_id) / "official"
    info = CheckpointInfo(
        paper_id=paper_id,
        source_url=meta.get("official_repo"),
        notes=meta.get("notes", ""),
    )
    if not folder.is_dir():
        info.verification_status = "directory_missing"
        return info

    candidates = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in WEIGHT_EXTS:
            candidates.append(p)
    if not candidates:
        info.verification_status = "not_found"
        return info

    # Prefer best_model / largest
    candidates.sort(key=lambda p: (0 if "best" in p.name.lower() else 1, -p.stat().st_size))
    chosen = candidates[0]
    info.path = str(chosen)
    info.exists = True
    info.size_bytes = chosen.stat().st_size
    info.sha256 = _sha256_file(chosen)
    info.verification_status = "file_present_unverified"
    return info


class CheckpointManager:
    def status_report(self) -> Dict[str, Any]:
        report = {}
        for pid in OFFICIAL_STATUS:
            info = discover_checkpoints(pid)
            report[pid] = {
                **info.to_dict(),
                **{k: OFFICIAL_STATUS[pid].get(k) for k in (
                    "official_repo", "paper_claims_weights", "repo_contains_weights", "releases"
                ) if k in OFFICIAL_STATUS[pid]},
            }
        return report

    def try_load_torch_state(self, path: str) -> Dict[str, Any]:
        """Load torch checkpoint; return keys/meta without claiming model compatibility."""
        try:
            import torch
        except ImportError:
            return {"ok": False, "reason": "torch not available"}
        p = Path(path)
        if not p.is_file():
            return {"ok": False, "reason": "file not found"}
        try:
            obj = torch.load(str(p), map_location="cpu")
        except Exception as e:
            return {"ok": False, "reason": f"torch.load failed: {e}"}
        if isinstance(obj, dict):
            keys = list(obj.keys())
            return {"ok": True, "type": "dict", "keys": keys[:30], "n_keys": len(keys)}
        return {"ok": True, "type": type(obj).__name__}
