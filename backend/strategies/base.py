from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class StrategyResult:
    stego: np.ndarray
    metrics: Dict[str, Any] = field(default_factory=dict)
    security: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("stego", None)  # too large for JSON
        return d


class EmbeddingStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def embed(
        self,
        cover: np.ndarray,
        message: str,
        passphrase: str,
        **kwargs,
    ) -> StrategyResult:
        ...

    @abstractmethod
    def extract(
        self,
        stego: np.ndarray,
        passphrase: str,
        **kwargs,
    ) -> str:
        ...
