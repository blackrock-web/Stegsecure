"""Steganalysis resistance metrics — shared detector across methods.

Reuses classical RS / chi-square / sample-pair / histogram from
backend/security/evaluator.py. Shared CNN steganalyzer (existing
SteganalyzerNet) is applied identically to every method's stego.

Security score rewards LOW detection accuracy and LOW ROC-AUC.
Published Paper-2 steganalyzer figures stay in paper_reference_steganalysis.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from backend.security.evaluator import evaluate_security


def classical_and_cnn_report(cover: Optional[np.ndarray], stego: np.ndarray) -> Dict[str, Any]:
    rep = evaluate_security(cover, stego, run_cnn=True)
    return {
        "cnn_stego_probability": rep.cnn_stego_probability,
        "rs": rep.rs,
        "chi_square": rep.chi_square,
        "sample_pair": rep.sample_pair,
        "histogram": rep.histogram,
        "composite_suspicion": rep.composite_suspicion,
        "notes": rep.notes,
        "label": "Steganalysis Resistance (not cryptographic security)",
    }


def shared_detector_scores(
    cover_stego_pairs: List[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    """
    Apply the same CNN steganalyzer to a list of (cover, stego) pairs.
    Treat cover as class 0 and stego as class 1 for simple accuracy / FPR / FNR.

    With few samples this is an indicative score only — not a calibrated ROC.
    """
    try:
        from backend.adversarial import evaluate_stego_confidence
    except Exception as e:
        return {
            "detection_accuracy": None,
            "roc_auc": None,
            "fpr": None,
            "fnr": None,
            "status": "FAILED",
            "reason": f"steganalyzer unavailable: {e}",
        }

    y_true = []
    y_score = []
    for cover, stego in cover_stego_pairs:
        try:
            p_cover = float(evaluate_stego_confidence(cover))
            p_stego = float(evaluate_stego_confidence(stego))
        except Exception:
            continue
        y_true.extend([0, 1])
        y_score.extend([p_cover, p_stego])

    if len(y_true) < 4:
        return {
            "detection_accuracy": None,
            "roc_auc": None,
            "fpr": None,
            "fnr": None,
            "status": "N/A",
            "reason": "insufficient samples for shared detector",
            "n_pairs": len(cover_stego_pairs),
        }

    y_true_arr = np.array(y_true)
    y_score_arr = np.array(y_score)
    y_pred = (y_score_arr >= 0.5).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true_arr == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true_arr == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true_arr == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true_arr == 1)))
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, fn + tp)

    # Mann-Whitney style AUC approximation
    pos = y_score_arr[y_true_arr == 1]
    neg = y_score_arr[y_true_arr == 0]
    if len(pos) and len(neg):
        auc = float(np.mean([1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg]))
    else:
        auc = None

    return {
        "detection_accuracy": round(acc, 6),
        "roc_auc": round(auc, 6) if auc is not None else None,
        "fpr": round(fpr, 6),
        "fnr": round(fnr, 6),
        "status": "ok",
        "n_pairs": len(cover_stego_pairs),
        "label": "Steganalysis Resistance — shared detector",
    }


# Paper 2 published steganalyzer error rates (reference only — higher = more secure)
PAPER2_REFERENCE_STEGANALYSIS = {
    "0.2_bpp": {"YeNet": 0.34, "YedroudjNet": 0.31, "ZhuNet": 0.26},
    "1.0_bpp": {"YeNet": 0.18, "YedroudjNet": 0.13, "ZhuNet": 0.10},
    "note": "Published paper figures (reference only). Higher error rate = more secure.",
}
