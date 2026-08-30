"""Unified security evaluation combining classical + CNN steganalyzer."""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional
import numpy as np

from .rs_analysis import rs_analysis
from .chi_square import chi_square_analysis
from .sample_pair import sample_pair_analysis
from .histogram import histogram_features


@dataclass
class SecurityReport:
    cnn_stego_probability: float = 0.0
    rs: Dict[str, float] = field(default_factory=dict)
    chi_square: Dict[str, float] = field(default_factory=dict)
    sample_pair: Dict[str, float] = field(default_factory=dict)
    histogram: Dict[str, float] = field(default_factory=dict)
    composite_suspicion: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_security(
    cover: Optional[np.ndarray],
    stego: np.ndarray,
    *,
    run_cnn: bool = True,
) -> SecurityReport:
    report = SecurityReport()
    report.rs = rs_analysis(stego)
    report.chi_square = chi_square_analysis(stego)
    report.sample_pair = sample_pair_analysis(stego)
    report.histogram = histogram_features(stego)

    if run_cnn:
        try:
            from backend.adversarial import evaluate_stego_confidence
            report.cnn_stego_probability = float(evaluate_stego_confidence(stego))
        except Exception as e:
            report.notes += f"CNN eval skipped: {e}; "
            report.cnn_stego_probability = 0.5

    # Weighted composite (not claimed as calibrated detection accuracy)
    scores = [
        report.cnn_stego_probability,
        report.rs.get("estimated_rate", 0),
        report.chi_square.get("suspicion", 0),
        report.sample_pair.get("suspicion", 0),
    ]
    report.composite_suspicion = float(np.mean(scores))
    report.notes += (
        "Composite suspicion is an uncalibrated average of CNN probability and "
        "classical heuristics; it is NOT claimed as true detection accuracy."
    )
    return report
