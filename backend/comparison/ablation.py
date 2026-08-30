"""SecureStegVault internal ablation ladder (not paper competitors).

Rungs:
  A. Basic EMD          -> emd_opap with cost_map_mode=fast, minimal zones
  B. EMD + OPAP         -> emd_opap
  C. Adaptive EMD+OPAP  -> cnn_emd_opap with percentile zoning (default cnn)
  D. CNN-Assisted       -> cnn_emd_opap
  E. + Distortion/Adv   -> cnn_emd_opap_adv
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np

from backend.strategies.base import EmbeddingStrategy, StrategyResult
from backend.strategies.registry import register
from backend.strategies.hybrid import _run_pipeline, _run_extract_pipeline


@register
class AblationA_BasicEMD(EmbeddingStrategy):
    name = "ablation_a_basic_emd"
    is_ablation = True

    def embed(self, cover, message, passphrase, **kw):
        stego, metrics, sec, _, rem = _run_pipeline(
            cover, message, passphrase, cost_map_mode="fast",
            adversarial_strength=0.0, use_stc_approx=False, **kw
        )
        return StrategyResult(
            stego=stego, metrics=metrics, security=sec,
            meta={"strategy": self.name, "is_ablation": True, "rung": "A", "label": "Basic EMD"},
        )

    def extract(self, stego, passphrase, **kw):
        return _run_extract_pipeline(stego, passphrase, cost_map_mode="fast", **kw)


@register
class AblationB_EMD_OPAP(EmbeddingStrategy):
    name = "ablation_b_emd_opap"
    is_ablation = True

    def embed(self, cover, message, passphrase, **kw):
        stego, metrics, sec, _, rem = _run_pipeline(
            cover, message, passphrase, cost_map_mode="fast",
            adversarial_strength=0.0, **kw
        )
        return StrategyResult(
            stego=stego, metrics=metrics, security=sec,
            meta={"strategy": self.name, "is_ablation": True, "rung": "B", "label": "EMD + OPAP"},
        )

    def extract(self, stego, passphrase, **kw):
        return _run_extract_pipeline(stego, passphrase, cost_map_mode="fast", **kw)


@register
class AblationC_Adaptive(EmbeddingStrategy):
    name = "ablation_c_adaptive"
    is_ablation = True

    def embed(self, cover, message, passphrase, **kw):
        stego, metrics, sec, _, rem = _run_pipeline(
            cover, message, passphrase, cost_map_mode="cnn",
            adversarial_strength=0.0, **kw
        )
        return StrategyResult(
            stego=stego, metrics=metrics, security=sec,
            meta={"strategy": self.name, "is_ablation": True, "rung": "C", "label": "Adaptive EMD+OPAP"},
        )

    def extract(self, stego, passphrase, **kw):
        return _run_extract_pipeline(stego, passphrase, cost_map_mode="cnn", **kw)


@register
class AblationD_CNN(EmbeddingStrategy):
    name = "ablation_d_cnn"
    is_ablation = True

    def embed(self, cover, message, passphrase, **kw):
        stego, metrics, sec, _, rem = _run_pipeline(
            cover, message, passphrase, cost_map_mode="cnn",
            adversarial_strength=0.0, **kw
        )
        return StrategyResult(
            stego=stego, metrics=metrics, security=sec,
            meta={"strategy": self.name, "is_ablation": True, "rung": "D", "label": "CNN-Assisted Adaptive"},
        )

    def extract(self, stego, passphrase, **kw):
        return _run_extract_pipeline(stego, passphrase, cost_map_mode="cnn", **kw)


@register
class AblationE_CNN_Adv(EmbeddingStrategy):
    name = "ablation_e_cnn_adv"
    is_ablation = True

    def embed(self, cover, message, passphrase, **kw):
        stego, metrics, sec, _, rem = _run_pipeline(
            cover, message, passphrase, cost_map_mode="advanced",
            adversarial_strength=0.7, **kw
        )
        return StrategyResult(
            stego=stego, metrics=metrics, security=sec,
            meta={"strategy": self.name, "is_ablation": True, "rung": "E",
                  "label": "CNN-Assisted + Distortion Opt / Adv"},
        )

    def extract(self, stego, passphrase, **kw):
        return _run_extract_pipeline(stego, passphrase, cost_map_mode="advanced", **kw)


ABLATION_RUNGS = [
    "ablation_a_basic_emd",
    "ablation_b_emd_opap",
    "ablation_c_adaptive",
    "ablation_d_cnn",
    "ablation_e_cnn_adv",
]


def run_ablation_ladder(
    cover: np.ndarray,
    secret_text: str,
    passphrase: str,
) -> Dict[str, Any]:
    """Run SecureStegVault ablation only — clearly not paper competition."""
    from backend.comparison.orchestrator import run_single_orchestrated
    result = run_single_orchestrated(
        cover, secret_text, passphrase,
        strategies=ABLATION_RUNGS,
        run_robustness=False,
        run_security=True,
    )
    result["section"] = "ABLATION OF SECURESTEGVAULT"
    result["note"] = (
        "These are internal SecureStegVault variants, not research-paper competitors."
    )
    return result
