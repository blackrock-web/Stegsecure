"""BER and Extraction Accuracy."""

from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np


def hamming_ber(original: bytes, recovered: bytes) -> float:
    n = min(len(original), len(recovered))
    if n == 0:
        return 1.0 if len(original) != len(recovered) else 0.0
    orig_bits = np.unpackbits(np.frombuffer(original[:n], dtype=np.uint8))
    rec_bits = np.unpackbits(np.frombuffer(recovered[:n], dtype=np.uint8))
    # Pad length difference as errors
    extra = abs(len(original) - len(recovered)) * 8
    diffs = int(np.sum(orig_bits != rec_bits)) + extra
    total = max(len(original), len(recovered)) * 8
    return diffs / max(1, total)


def compute_reliability(
    original_message: str,
    recovered_message: Optional[str],
    *,
    payload_type: str = "bitstream",
    original_tile: Optional[np.ndarray] = None,
    recovered_tile: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    if recovered_message is None:
        return {
            "ber": None,
            "extraction_accuracy": None,
            "status": "FAILED",
            "reason": "extraction returned no message",
            "payload_type": payload_type,
        }

    if payload_type == "image_tile" and original_tile is not None and recovered_tile is not None:
        from backend.metrics import calculate_metrics
        m = calculate_metrics(original_tile, recovered_tile, 0, {})
        return {
            "ber": None,
            "extraction_accuracy": None,
            "recovered_secret_psnr": m.get("psnr"),
            "recovered_secret_ssim": m.get("ssim"),
            "status": "ok",
            "payload_type": "image_tile",
            "note": "image-tile accuracy reported as secret PSNR/SSIM, not BER",
        }

    # Bitstream: compare UTF-8 bytes of plaintext
    orig_b = original_message.encode("utf-8")
    rec_b = recovered_message.encode("utf-8")
    ber = hamming_ber(orig_b, rec_b)
    acc = 1.0 - ber
    exact = original_message == recovered_message
    return {
        "ber": round(ber, 8),
        "extraction_accuracy": round(acc, 8),
        "exact_match": exact,
        "status": "ok",
        "payload_type": "bitstream",
    }
