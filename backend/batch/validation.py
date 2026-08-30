"""Pre-flight validation for batch uploads and configuration."""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

ALLOWED_EXTS = {".png", ".bmp"}
MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_DIMENSION = 8192
MAX_BATCH_IMAGES = 200


def validate_image_bytes(data: bytes, filename: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """Return (ok, error, meta)."""
    if not data:
        return False, "Empty file", None
    if len(data) > MAX_FILE_BYTES:
        return False, f"File exceeds {MAX_FILE_BYTES // (1024*1024)} MB limit", None
    lower = (filename or "").lower()
    if not any(lower.endswith(ext) for ext in ALLOWED_EXTS):
        return False, "Unsupported format (PNG or BMP required)", None
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        w, h = img.size
        if w < 8 or h < 8:
            return False, "Image too small (min 8×8)", None
        if w > MAX_DIMENSION or h > MAX_DIMENSION:
            return False, f"Image exceeds max dimension {MAX_DIMENSION}", None
        fmt = (img.format or "").upper()
        if fmt not in ("PNG", "BMP"):
            return False, f"Unsupported format '{fmt}'", None
        return True, None, {
            "width": w,
            "height": h,
            "mode": img.mode,
            "format": fmt,
            "size_bytes": len(data),
        }
    except Exception as e:
        return False, f"Corrupted or unreadable image: {e}", None


def validate_batch_config(cfg: Dict[str, Any], n_images: int) -> List[str]:
    errors: List[str] = []
    if n_images < 1:
        errors.append("At least one image is required")
    if n_images > MAX_BATCH_IMAGES:
        errors.append(f"Batch limited to {MAX_BATCH_IMAGES} images")
    job_type = cfg.get("type") or "encode"
    if job_type == "encode":
        mode = cfg.get("message_mode") or "same"
        if mode == "same" and not (cfg.get("secret_text") or "").strip():
            errors.append("Secret text is required for encode batches")
        if not cfg.get("passphrase"):
            errors.append("Passphrase is required")
    elif job_type == "decode":
        if not cfg.get("passphrase"):
            errors.append("Passphrase is required for decode batches")
    workers = int(cfg.get("workers") or 2)
    if workers < 1 or workers > 16:
        errors.append("Workers must be between 1 and 16")
    return errors


def detect_duplicates(filenames: List[str]) -> List[str]:
    seen = set()
    dups = []
    for f in filenames:
        key = f.lower()
        if key in seen:
            dups.append(f)
        seen.add(key)
    return dups
