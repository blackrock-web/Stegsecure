"""Single WeightConfig dataclass — never hard-code weights elsewhere."""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Dict


@dataclass
class WeightConfig:
    imperceptibility: float = 0.25
    payload_capacity: float = 0.15
    reliability: float = 0.15
    robustness: float = 0.15
    security: float = 0.20
    efficiency: float = 0.10

    # Sub-weights inside imperceptibility (must sum to 1)
    psnr_sub: float = 0.5
    ssim_sub: float = 0.3
    mse_sub: float = 0.2

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        total = (
            self.imperceptibility + self.payload_capacity + self.reliability
            + self.robustness + self.security + self.efficiency
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        sub = self.psnr_sub + self.ssim_sub + self.mse_sub
        if abs(sub - 1.0) > 1e-6:
            raise ValueError(f"Imperceptibility sub-weights must sum to 1.0, got {sub}")


DEFAULT_WEIGHTS = WeightConfig()
