"""
Modular dataset loader for SecureStegVault v3.

Expected layout (user-provided, never auto-downloaded):
  datasets/
    covers/          # raw cover images (PNG/BMP preferred)
    generated/       # optional cache of stego pairs
    train/
    validation/
    test/

Supports BOSSBase / BOWS-2 / ALASKA2 / DIV2K / COCO subsets once placed locally.
"""

from __future__ import annotations

import json
import hashlib
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from PIL import Image


@dataclass
class DatasetConfig:
    root: str = "datasets"
    covers_subdir: str = "covers"
    generated_subdir: str = "generated"
    seed: int = 42
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    max_images: Optional[int] = None
    target_size: Optional[Tuple[int, int]] = None  # (H, W) or None = keep original

    def covers_path(self) -> Path:
        return Path(self.root) / self.covers_subdir

    def generated_path(self) -> Path:
        return Path(self.root) / self.generated_subdir


def list_cover_images(cfg: DatasetConfig) -> List[Path]:
    root = cfg.covers_path()
    if not root.is_dir():
        return []
    exts = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}
    files = sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])
    if cfg.max_images:
        files = files[: cfg.max_images]
    return files


def load_image(path: Path, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if target_size:
        img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def split_dataset(paths: List[Path], cfg: DatasetConfig) -> Dict[str, List[Path]]:
    rng = random.Random(cfg.seed)
    items = list(paths)
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)
    return {
        "train": items[:n_train],
        "validation": items[n_train : n_train + n_val],
        "test": items[n_train + n_val :],
    }


def generate_stego_pair(
    cover: np.ndarray,
    message: str,
    passphrase: str,
    *,
    cost_map_mode: str = "cnn",
    emd_n: int = 2,
    adversarial_strength: float = 0.0,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Produce a stego image using the real SecureStegVault embedding pipeline.
    Used for training the steganalyzer on authentic cover/stego pairs.
    """
    from backend.crypto import encrypt_payload
    from backend.cnn_costmap import compute_cnn_costmap
    from backend.zoning import ZoningConfig, classify_zones, calculate_capacity
    from backend.emd import (
        bytes_to_base5_digits, bytes_to_base7_digits, embed_emd_zone_a,
    )
    from backend.opap import embed_opap_zone
    from backend.cost_optimizer import rank_zone_indices
    import math

    rng = np.random.RandomState(seed)
    config = ZoningConfig(emd_group_size=emd_n)
    encrypted = encrypt_payload(message, passphrase)
    cost_map = compute_cnn_costmap(cover, cost_map_mode=cost_map_mode)
    H, W, C = cover.shape
    cost_3d = np.repeat(cost_map[:, :, np.newaxis], C, axis=2)
    zones_3d = classify_zones(cost_3d, config)

    stego = cover.copy()
    flat = stego.flatten()
    zflat = zones_3d.flatten()
    cflat = cost_3d.flatten()

    za = rank_zone_indices(cflat, np.where(zflat == 0)[0], is_emd=True, group_size=emd_n)
    zb = rank_zone_indices(cflat, np.where(zflat == 1)[0], is_emd=False)
    zc = rank_zone_indices(cflat, np.where(zflat == 2)[0], is_emd=False)

    bits = []
    for b in encrypted:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    total = len(bits)
    rem = total
    idx = 0

    if emd_n == 2:
        groups = len(za) // 2
        max_b = int(groups * math.log2(5))
        if max_b > 0 and rem > 0:
            nbits = min(rem, max_b)
            nbytes = (nbits + 7) // 8
            digs = bytes_to_base5_digits(encrypted[:nbytes])
            _, used = embed_emd_zone_a(flat, za, digs, emd_n=2)
            idx = (used // 4) * 8
            rem = max(0, total - idx)
    else:
        groups = len(za) // 3
        max_b = int(groups * math.log2(7))
        if max_b > 0 and rem > 0:
            nbits = min(rem, max_b)
            nbytes = (nbits + 7) // 8
            digs = bytes_to_base7_digits(encrypted[:nbytes])
            _, used = embed_emd_zone_a(flat, za, digs, emd_n=3)
            idx = (used // 3) * 8
            rem = max(0, total - idx)

    if rem > 0 and len(zb):
        stream = bits[idx:idx + rem]
        _, n = embed_opap_zone(flat, zb, stream, k=config.kb_bits)
        idx += n
        rem = max(0, total - idx)
    if rem > 0 and len(zc):
        stream = bits[idx:idx + rem]
        _, n = embed_opap_zone(flat, zc, stream, k=config.kc_bits)
        idx += n
        rem = max(0, total - idx)

    stego = flat.reshape(cover.shape)
    meta = {
        "payload_bytes": len(encrypted),
        "bits_embedded": idx,
        "bits_remaining": rem,
        "message_len": len(message),
        "seed": seed,
        "cost_map_mode": cost_map_mode,
    }
    return stego, meta


def dataset_stats(cfg: DatasetConfig) -> Dict[str, Any]:
    paths = list_cover_images(cfg)
    splits = split_dataset(paths, cfg) if paths else {"train": [], "validation": [], "test": []}
    return {
        "config": asdict(cfg),
        "n_covers": len(paths),
        "n_train": len(splits["train"]),
        "n_val": len(splits["validation"]),
        "n_test": len(splits["test"]),
        "covers_path": str(cfg.covers_path()),
        "exists": cfg.covers_path().is_dir(),
    }
