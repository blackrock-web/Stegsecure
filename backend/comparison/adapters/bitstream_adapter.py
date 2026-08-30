"""Passthrough adapter for methods that accept arbitrary-length bit streams."""

from __future__ import annotations
import numpy as np

from .base_adapter import PayloadAdapter, AdapterResult


class BitstreamAdapter(PayloadAdapter):
    name = "bitstream"

    def adapt_for_embed(self, ciphertext: bytes, cover: np.ndarray, **kwargs) -> AdapterResult:
        return AdapterResult(
            payload=ciphertext,
            adapter_used=self.name,
            capacity_limited=False,
            native_operating_point=True,
            meta={"payload_len_bytes": len(ciphertext)},
        )

    def recover_from_extract(self, extracted: Any, **kwargs) -> bytes:
        if isinstance(extracted, (bytes, bytearray)):
            return bytes(extracted)
        if isinstance(extracted, str):
            return extracted.encode("utf-8")
        raise TypeError(f"BitstreamAdapter expected bytes, got {type(extracted)}")
