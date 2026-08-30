"""Base payload adapter interface."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import numpy as np


@dataclass
class AdapterResult:
    """Result of adapting a payload for a specific embedding method."""

    payload: Any  # bytes | np.ndarray depending on adapter
    adapter_used: str
    capacity_limited: bool = False
    native_operating_point: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "adapter_used": self.adapter_used,
            "capacity_limited": self.capacity_limited,
            "native_operating_point": self.native_operating_point,
            "meta": self.meta,
        }
        if isinstance(self.payload, (bytes, bytearray)):
            d["payload_len_bytes"] = len(self.payload)
        elif isinstance(self.payload, np.ndarray):
            d["payload_shape"] = list(self.payload.shape)
        return d


class PayloadAdapter(ABC):
    name: str = "base"

    @abstractmethod
    def adapt_for_embed(
        self,
        ciphertext: bytes,
        cover: np.ndarray,
        **kwargs,
    ) -> AdapterResult:
        """Convert AES-GCM ciphertext into the form the target method expects."""
        ...

    @abstractmethod
    def recover_from_extract(
        self,
        extracted: Any,
        **kwargs,
    ) -> bytes:
        """Convert method-native extracted payload back to ciphertext bytes."""
        ...
