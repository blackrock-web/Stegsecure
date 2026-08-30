"""Synchronous single-input comparison across all methods."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import base64
import io
import time
import traceback

import numpy as np
from PIL import Image

from backend.strategies.registry import get_strategy, list_strategies
from backend.comparison.metrics.quality import compute_quality
from backend.comparison.metrics.capacity import compute_capacity
from backend.comparison.metrics.reliability import compute_reliability
from backend.comparison.metrics.efficiency import compute_efficiency, Timer


DEFAULT_STRATEGIES = [
    "paper1_joint_cnn",
    "paper2_cyclegan_steg",
    "paper3_block_prep_net",
    "paper4_lsb_magicmatrix",
    "cnn_emd_opap",
]


def _stego_to_b64(stego: np.ndarray) -> str:
    img = Image.fromarray(stego.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def compare_one(
    cover: np.ndarray,
    secret_text: str,
    passphrase: str,
    strategies: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run the same cover + secret + passphrase through each requested strategy.
    Returns per-method stego (b64), metrics, reliability, timing.
    Never fabricates metrics — FAILED/N/A with reason when something cannot run.
    """
    names = strategies or DEFAULT_STRATEGIES
    available = set(list_strategies())
    results = []

    for name in names:
        row: Dict[str, Any] = {
            "strategy": name,
            "status": "ok",
            "adapter_used": "bitstream",
            "capacity_limited": False,
            "native_operating_point": True,
            "device": compute_efficiency()["device"],
        }
        if name not in available:
            row["status"] = "FAILED"
            row["reason"] = f"strategy '{name}' not registered"
            results.append(row)
            continue

        try:
            strat = get_strategy(name)
            with Timer() as t_embed:
                sr = strat.embed(cover, secret_text, passphrase)
            row["embed_time_s"] = t_embed.elapsed

            if sr.meta.get("capacity_limited"):
                row["status"] = "N/A"
                row["reason"] = sr.metrics.get("reason", "capacity_limited")
                row["capacity_limited"] = True
                row["adapter_used"] = sr.meta.get("adapter_used", "unknown")
                row["metrics"] = sr.metrics
                row["meta"] = {k: v for k, v in sr.meta.items() if k != "gray_cover"}
                results.append(row)
                continue

            if sr.metrics.get("status") in ("FAILED", "N/A"):
                row["status"] = sr.metrics.get("status", "FAILED")
                row["reason"] = sr.metrics.get("reason", "unknown")
                row["metrics"] = sr.metrics
                row["meta"] = sr.meta
                results.append(row)
                continue

            stego = sr.stego
            quality = compute_quality(cover, stego)
            bits = int(sr.metrics.get("bits_embedded") or sr.meta.get("bits_embedded") or 0)
            if bits == 0:
                # estimate from payload
                from backend.crypto import encrypt_payload
                bits = len(encrypt_payload(secret_text, passphrase)) * 8
            capacity = compute_capacity(
                cover, bits,
                adapter_used=sr.meta.get("adapter_used", "bitstream"),
            )

            recovered = None
            extract_err = None
            with Timer() as t_ext:
                try:
                    recovered = strat.extract(stego, passphrase)
                except Exception as e:
                    extract_err = str(e)
            row["extract_time_s"] = t_ext.elapsed

            reliability = compute_reliability(secret_text, recovered)
            if extract_err and recovered is None:
                reliability = {
                    "ber": None,
                    "extraction_accuracy": None,
                    "status": "FAILED",
                    "reason": extract_err,
                    "payload_type": "bitstream",
                }

            row.update({
                "stego_b64": _stego_to_b64(stego),
                "quality": quality,
                "capacity": capacity,
                "reliability": reliability,
                "efficiency": compute_efficiency(
                    embed_s=t_embed.elapsed, extract_s=t_ext.elapsed,
                ),
                "security": sr.security or {},
                "metrics_raw": sr.metrics,
                "meta": {k: v for k, v in (sr.meta or {}).items() if not isinstance(v, np.ndarray)},
                "adapter_used": sr.meta.get("adapter_used", "bitstream"),
                "capacity_limited": bool(sr.meta.get("capacity_limited", False)),
                "native_operating_point": bool(sr.meta.get("native_operating_point", True)),
            })
        except Exception as e:
            traceback.print_exc()
            row["status"] = "FAILED"
            row["reason"] = str(e)
        results.append(row)

    return {
        "success": True,
        "cover_shape": list(cover.shape),
        "secret_len": len(secret_text),
        "strategies_requested": names,
        "results": results,
    }
