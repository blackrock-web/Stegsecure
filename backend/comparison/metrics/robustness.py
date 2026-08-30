"""Attack suite for robustness evaluation.

Pipeline: stego -> attack -> extract -> BER / accuracy.
Failures return N/A (never zero).
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional
import io
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


def _to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(img.astype(np.uint8))


def _from_pil(pil: Image.Image, shape: tuple) -> np.ndarray:
    arr = np.array(pil.convert("RGB"), dtype=np.uint8)
    if arr.shape[:2] != shape[:2]:
        arr = np.array(pil.resize((shape[1], shape[0]), Image.BILINEAR).convert("RGB"), dtype=np.uint8)
    return arr


def attack_jpeg(stego: np.ndarray, quality: int) -> np.ndarray:
    pil = _to_pil(stego)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return _from_pil(Image.open(buf), stego.shape)


def attack_gaussian_noise(stego: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0, sigma, stego.shape)
    return np.clip(stego.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def attack_gaussian_blur(stego: np.ndarray, radius: float = 1.5) -> np.ndarray:
    return _from_pil(_to_pil(stego).filter(ImageFilter.GaussianBlur(radius=radius)), stego.shape)


def attack_median_filter(stego: np.ndarray, size: int = 3) -> np.ndarray:
    return _from_pil(_to_pil(stego).filter(ImageFilter.MedianFilter(size=size)), stego.shape)


def attack_resize(stego: np.ndarray, scale: float = 0.75) -> np.ndarray:
    H, W = stego.shape[:2]
    pil = _to_pil(stego).resize((max(1, int(W * scale)), max(1, int(H * scale))), Image.BILINEAR)
    return _from_pil(pil.resize((W, H), Image.BILINEAR), stego.shape)


def attack_crop_10pct(stego: np.ndarray) -> np.ndarray:
    H, W = stego.shape[:2]
    dy, dx = int(H * 0.05), int(W * 0.05)
    cropped = stego[dy : H - dy, dx : W - dx]
    # paste back into full canvas (zeros outside) — geometry-breaking for some methods
    out = np.zeros_like(stego)
    out[dy : dy + cropped.shape[0], dx : dx + cropped.shape[1]] = cropped
    return out


def attack_brightness(stego: np.ndarray, factor: float) -> np.ndarray:
    return _from_pil(ImageEnhance.Brightness(_to_pil(stego)).enhance(factor), stego.shape)


def attack_contrast(stego: np.ndarray, factor: float) -> np.ndarray:
    return _from_pil(ImageEnhance.Contrast(_to_pil(stego)).enhance(factor), stego.shape)


ATTACKS: List[Dict[str, Any]] = [
    {"attack_id": "jpeg_q70", "attack_name": "JPEG Q70", "parameters": {"quality": 70}, "fn": lambda s: attack_jpeg(s, 70)},
    {"attack_id": "jpeg_q50", "attack_name": "JPEG Q50", "parameters": {"quality": 50}, "fn": lambda s: attack_jpeg(s, 50)},
    {"attack_id": "jpeg_q30", "attack_name": "JPEG Q30", "parameters": {"quality": 30}, "fn": lambda s: attack_jpeg(s, 30)},
    {"attack_id": "gauss_noise_5", "attack_name": "Gaussian noise σ=5", "parameters": {"sigma": 5}, "fn": lambda s: attack_gaussian_noise(s, 5)},
    {"attack_id": "gauss_noise_15", "attack_name": "Gaussian noise σ=15", "parameters": {"sigma": 15}, "fn": lambda s: attack_gaussian_noise(s, 15)},
    {"attack_id": "gauss_blur", "attack_name": "Gaussian blur", "parameters": {"radius": 1.5}, "fn": lambda s: attack_gaussian_blur(s)},
    {"attack_id": "median", "attack_name": "Median filter", "parameters": {"size": 3}, "fn": lambda s: attack_median_filter(s)},
    {"attack_id": "resize_0_75", "attack_name": "Resize 0.75×", "parameters": {"scale": 0.75}, "fn": lambda s: attack_resize(s, 0.75)},
    {"attack_id": "crop_10", "attack_name": "Crop 10%", "parameters": {"pct": 10}, "fn": lambda s: attack_crop_10pct(s)},
    {"attack_id": "bright_p15", "attack_name": "Brightness +15%", "parameters": {"factor": 1.15}, "fn": lambda s: attack_brightness(s, 1.15)},
    {"attack_id": "bright_m15", "attack_name": "Brightness −15%", "parameters": {"factor": 0.85}, "fn": lambda s: attack_brightness(s, 0.85)},
    {"attack_id": "contrast_p15", "attack_name": "Contrast +15%", "parameters": {"factor": 1.15}, "fn": lambda s: attack_contrast(s, 1.15)},
    {"attack_id": "contrast_m15", "attack_name": "Contrast −15%", "parameters": {"factor": 0.85}, "fn": lambda s: attack_contrast(s, 0.85)},
]


def run_robustness_suite(
    stego: np.ndarray,
    original_message: str,
    passphrase: str,
    extract_fn: Callable[..., str],
    *,
    paper_claimed: Optional[Dict[str, bool]] = None,
    tile_alignment_sensitive: bool = False,
) -> List[Dict[str, Any]]:
    """Apply each attack, re-extract, record BER/accuracy or N/A."""
    from backend.comparison.metrics.reliability import compute_reliability

    paper_claimed = paper_claimed or {}
    results = []
    for atk in ATTACKS:
        row: Dict[str, Any] = {
            "attack_id": atk["attack_id"],
            "attack_name": atk["attack_name"],
            "parameters": atk["parameters"],
            "paper_claimed": bool(paper_claimed.get(atk["attack_id"], False)),
            "native_supported": True,
            "status": "ok",
            "ber": None,
            "accuracy": None,
            "failure_reason": None,
        }
        # Geometry-breaking attacks on tile methods
        if tile_alignment_sensitive and atk["attack_id"] in ("crop_10", "resize_0_75"):
            row["status"] = "N/A"
            row["native_supported"] = False
            row["failure_reason"] = "attack breaks tile alignment"
            results.append(row)
            continue
        try:
            attacked = atk["fn"](stego)
            recovered = extract_fn(attacked, passphrase)
            rel = compute_reliability(original_message, recovered)
            if rel.get("status") == "FAILED":
                row["status"] = "N/A"
                row["failure_reason"] = rel.get("reason", "extraction failed after attack")
            else:
                row["ber"] = rel.get("ber")
                row["accuracy"] = rel.get("extraction_accuracy")
                row["status"] = "ok"
        except Exception as e:
            row["status"] = "N/A"
            row["failure_reason"] = str(e)
        results.append(row)
    return results
