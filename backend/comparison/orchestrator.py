"""Benchmark orchestrator — coordinates adapters, strategies, metrics, scoring.

Does not contain algorithm-specific embedding code; calls registered strategies.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import hashlib
import json
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from backend.strategies.registry import get_strategy, list_strategies
from backend.comparison.single_input_compare import compare_one, DEFAULT_STRATEGIES
from backend.comparison.scoring.weights import DEFAULT_WEIGHTS, WeightConfig
from backend.comparison.scoring.normalize import normalize_table
from backend.comparison.scoring.pareto import pareto_analysis, DEFAULT_DIRECTIONS
from backend.comparison.metrics.robustness import run_robustness_suite
from backend.comparison.metrics.security import classical_and_cnn_report, shared_detector_scores


def _content_hash(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _payload_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def run_single_orchestrated(
    cover: np.ndarray,
    secret_text: str,
    passphrase: str,
    strategies: Optional[List[str]] = None,
    *,
    run_robustness: bool = False,
    run_security: bool = True,
    weights: Optional[WeightConfig] = None,
) -> Dict[str, Any]:
    """Full single-input pipeline with optional robustness and scoring."""
    weights = weights or DEFAULT_WEIGHTS
    weights.validate()
    base = compare_one(cover, secret_text, passphrase, strategies=strategies)

    cover_h = _content_hash(cover)
    payload_h = _payload_hash(secret_text)
    enriched = []
    pairs_for_detector = []

    for row in base.get("results", []):
        method_id = row.get("strategy", "unknown")
        status = row.get("status", "FAILED")
        entry: Dict[str, Any] = {
            "method_id": method_id,
            "method_name": method_id,
            "status": status,
            "cover_hash": cover_h,
            "payload_hash": payload_h,
            "payload_type": "text",
            "payload_size": len(secret_text),
            "adapter_used": row.get("adapter_used"),
            "capacity_limited": row.get("capacity_limited"),
            "native_operating_point": row.get("native_operating_point"),
            "device": row.get("device") or (row.get("efficiency") or {}).get("device"),
            "failure_reason": row.get("reason"),
            "model_status": (row.get("meta") or {}).get("training_mode", "unknown"),
            "meta": row.get("meta") or {},
        }

        q = row.get("quality") or {}
        entry["psnr"] = q.get("psnr_db") if q.get("psnr_db") is not None else q.get("psnr")
        entry["ssim"] = q.get("ssim")
        entry["mse"] = q.get("mse")
        cap = row.get("capacity") or {}
        entry["bpp"] = cap.get("bpp")
        entry["bits_embedded"] = cap.get("bits_embedded")
        rel = row.get("reliability") or {}
        entry["ber"] = rel.get("ber")
        entry["extraction_accuracy"] = rel.get("extraction_accuracy")
        entry["extraction_metric_type"] = rel.get("payload_type", "bitstream")
        eff = row.get("efficiency") or {}
        entry["embed_time_s"] = eff.get("embed_time_s") or row.get("embed_time_s")
        entry["extract_time_s"] = eff.get("extract_time_s") or row.get("extract_time_s")
        entry["stego_b64"] = row.get("stego_b64")

        # Security (per-method classical + CNN)
        if run_security and status == "ok" and row.get("stego_b64"):
            try:
                import base64, io
                from PIL import Image
                stego = np.array(Image.open(io.BytesIO(base64.b64decode(row["stego_b64"]))).convert("RGB"), dtype=np.uint8)
                entry["steganalysis"] = classical_and_cnn_report(cover, stego)
                pairs_for_detector.append((cover, stego))
            except Exception as e:
                entry["steganalysis"] = {"status": "FAILED", "reason": str(e)}

        # Robustness (optional, expensive)
        if run_robustness and status == "ok":
            try:
                strat = get_strategy(method_id)
                import base64, io
                from PIL import Image
                stego = np.array(Image.open(io.BytesIO(base64.b64decode(row["stego_b64"]))).convert("RGB"), dtype=np.uint8)
                tile_sensitive = method_id in ("paper1_joint_cnn", "paper3_block_prep_net")
                entry["robustness"] = run_robustness_suite(
                    stego, secret_text, passphrase,
                    extract_fn=lambda img, pw: strat.extract(img, pw),
                    tile_alignment_sensitive=tile_sensitive,
                )
            except Exception as e:
                entry["robustness"] = [{"status": "FAILED", "failure_reason": str(e)}]

        enriched.append(entry)

    # Shared detector across methods
    shared_sec = shared_detector_scores(pairs_for_detector) if pairs_for_detector else None

    # Normalize + score
    norm_rows = normalize_table(
        [{**e} for e in enriched],
        {
            "psnr": True, "ssim": True, "mse": False, "bpp": True,
            "extraction_accuracy": True, "ber": False, "embed_time_s": False,
        },
    )
    for e, n in zip(enriched, norm_rows):
        e["normalized_metrics"] = {
            k: n.get(k) for k in (
                "psnr_norm", "ssim_norm", "mse_norm", "bpp_norm",
                "extraction_accuracy_norm", "ber_norm", "embed_time_s_norm",
            ) if k in n
        }
        e["overall_score"] = _weighted_score(e, weights)

    # Rank
    scorable = [e for e in enriched if e.get("overall_score") is not None]
    scorable.sort(key=lambda x: x["overall_score"], reverse=True)
    for rank, e in enumerate(scorable, 1):
        e["rank"] = rank
    for e in enriched:
        if "rank" not in e:
            e["rank"] = None

    # Pareto
    pareto_input = []
    for e in enriched:
        if e.get("status") != "ok":
            continue
        pareto_input.append({
            "method_id": e["method_id"],
            "psnr": e.get("psnr"),
            "ssim": e.get("ssim"),
            "mse": e.get("mse"),
            "bpp": e.get("bpp"),
            "extraction_accuracy": e.get("extraction_accuracy"),
            "ber": e.get("ber"),
            "embed_time_s": e.get("embed_time_s"),
        })
    pareto = pareto_analysis(pareto_input, DEFAULT_DIRECTIONS) if pareto_input else {}
    for e in enriched:
        for pm in pareto.get("per_method", []):
            if pm["method_id"] == e["method_id"]:
                e["pareto_status"] = pm["pareto_status"]
                e["dominated_by"] = pm["dominated_by"]
                e["dominates"] = pm["dominates"]

    winner = scorable[0] if scorable else None
    explanation = None
    if winner:
        explanation = (
            f"{winner['method_id']} achieved the highest overall score "
            f"({winner['overall_score']:.4f}) based on live measurements only "
            f"(weights={weights.to_dict()})."
        )

    return {
        "success": True,
        "cover_hash": cover_h,
        "payload_hash": payload_h,
        "weights": weights.to_dict(),
        "results": enriched,
        "shared_steganalysis": shared_sec,
        "pareto": pareto,
        "winner": winner["method_id"] if winner else None,
        "winner_explanation": explanation,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _weighted_score(entry: Dict[str, Any], weights: WeightConfig) -> Optional[float]:
    """N/A-safe weighted score; renormalize over present buckets."""
    buckets = {}
    # Imperceptibility
    psnr_n = entry.get("normalized_metrics", {}).get("psnr_norm")
    ssim_n = entry.get("normalized_metrics", {}).get("ssim_norm")
    mse_n = entry.get("normalized_metrics", {}).get("mse_norm")
    imper_parts = []
    if psnr_n is not None:
        imper_parts.append(weights.psnr_sub * psnr_n)
    if ssim_n is not None:
        imper_parts.append(weights.ssim_sub * ssim_n)
    if mse_n is not None:
        imper_parts.append(weights.mse_sub * mse_n)
    if imper_parts:
        buckets["imperceptibility"] = sum(imper_parts) / max(
            1e-9,
            (weights.psnr_sub if psnr_n is not None else 0)
            + (weights.ssim_sub if ssim_n is not None else 0)
            + (weights.mse_sub if mse_n is not None else 0),
        ) * (weights.psnr_sub + weights.ssim_sub + weights.mse_sub)  # keep sub-scale

    bpp_n = entry.get("normalized_metrics", {}).get("bpp_norm")
    if bpp_n is not None:
        buckets["payload_capacity"] = bpp_n
    acc_n = entry.get("normalized_metrics", {}).get("extraction_accuracy_norm")
    if acc_n is not None:
        buckets["reliability"] = acc_n
    time_n = entry.get("normalized_metrics", {}).get("embed_time_s_norm")
    if time_n is not None:
        buckets["efficiency"] = time_n

    weight_map = {
        "imperceptibility": weights.imperceptibility,
        "payload_capacity": weights.payload_capacity,
        "reliability": weights.reliability,
        "robustness": weights.robustness,
        "security": weights.security,
        "efficiency": weights.efficiency,
    }
    present_w = {k: weight_map[k] for k in buckets if k in weight_map}
    if not present_w:
        return None
    total_w = sum(present_w.values())
    score = sum(buckets[k] * (present_w[k] / total_w) for k in present_w)
    entry["bucket_scores"] = buckets
    return float(score)


def persist_experiment(result: Dict[str, Any], root: Optional[Path] = None) -> Path:
    root = root or Path(__file__).resolve().parents[2] / "experiments"
    run_id = datetime.now(timezone.utc).strftime("comparison_%Y%m%dT%H%M%SZ")
    folder = root / run_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "experiment.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    # CSV summary
    lines = ["method_id,status,psnr,ssim,mse,bpp,ber,extraction_accuracy,overall_score,rank,pareto_status"]
    for e in result.get("results", []):
        lines.append(",".join(str(x) if x is not None else "" for x in [
            e.get("method_id"), e.get("status"), e.get("psnr"), e.get("ssim"), e.get("mse"),
            e.get("bpp"), e.get("ber"), e.get("extraction_accuracy"),
            e.get("overall_score"), e.get("rank"), e.get("pareto_status"),
        ]))
    (folder / "results.csv").write_text("\n".join(lines), encoding="utf-8")
    return folder
