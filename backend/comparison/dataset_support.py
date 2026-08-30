"""Dataset import helpers — user-supplied only; no automatic downloads.

Tiers:
  A. Local folder / datasets/covers/
  B. ZIP upload
  C. Explicit public fetch (caller must pass confirm=True)
  D. Synthetic smoke test (excluded from ranking)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def index_image_bytes(data: bytes, name: str = "image") -> Optional[Dict[str, Any]]:
    try:
        pil = Image.open(io.BytesIO(data))
        fmt = (pil.format or "").upper()
        arr = np.array(pil.convert("RGB"), dtype=np.uint8)
        return {
            "name": name,
            "content_hash": _hash_bytes(data),
            "width": arr.shape[1],
            "height": arr.shape[0],
            "format": fmt,
            "channels": 3,
        }
    except Exception:
        return None


def build_manifest(
    images: List[Dict[str, Any]],
    *,
    dataset_name: str,
    source: str,
    dataset_type: str = "user",
) -> Dict[str, Any]:
    return {
        "dataset_id": _hash_bytes(json.dumps(images, sort_keys=True).encode())[:12],
        "dataset_name": dataset_name,
        "source": source,
        "dataset_type": dataset_type,
        "image_count": len(images),
        "images": images,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "content_hash": _hash_bytes(json.dumps([i.get("content_hash") for i in images]).encode()),
    }


def import_from_zip(zip_bytes: bytes, dest: Path) -> Dict[str, Any]:
    dest.mkdir(parents=True, exist_ok=True)
    images = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            ext = Path(info.filename).suffix.lower()
            if ext not in ALLOWED_EXTS:
                continue
            data = zf.read(info)
            meta = index_image_bytes(data, name=Path(info.filename).name)
            if meta is None:
                continue
            out = dest / f"{meta['content_hash'][:12]}_{Path(info.filename).name}"
            out.write_bytes(data)
            meta["path"] = str(out)
            images.append(meta)
    return build_manifest(images, dataset_name="zip_upload", source="user_zip", dataset_type="user")


def synthetic_cover(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
