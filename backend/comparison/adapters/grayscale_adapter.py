"""Grayscale cover adapter for Paper 2 (grayscale-only architecture)."""

from __future__ import annotations
import numpy as np

from .base_adapter import PayloadAdapter, AdapterResult


class GrayscaleCoverAdapter(PayloadAdapter):
    name = "grayscale"

    def adapt_for_embed(self, ciphertext: bytes, cover: np.ndarray, **kwargs) -> AdapterResult:
        """Convert RGB cover to grayscale; payload remains bitstream.

        The cover conversion is recorded so results are never presented as
        apples-to-apples RGB comparisons.
        """
        if cover.ndim == 2:
            gray = cover.astype(np.uint8)
        elif cover.ndim == 3 and cover.shape[2] >= 3:
            # ITU-R BT.601 luma
            r, g, b = cover[:, :, 0].astype(np.float64), cover[:, :, 1].astype(np.float64), cover[:, :, 2].astype(np.float64)
            gray = (0.299 * r + 0.587 * g + 0.114 * b).round().clip(0, 255).astype(np.uint8)
        else:
            gray = cover.astype(np.uint8).squeeze()

        return AdapterResult(
            payload=ciphertext,
            adapter_used=self.name,
            capacity_limited=False,
            native_operating_point=True,
            meta={
                "cover_mode": "grayscale",
                "gray_shape": list(gray.shape),
                "original_cover_shape": list(cover.shape),
                "gray_cover": gray,  # consumers must pop this for embedding
            },
        )

    def recover_from_extract(self, extracted: Any, **kwargs) -> bytes:
        if isinstance(extracted, (bytes, bytearray)):
            return bytes(extracted)
        if isinstance(extracted, str):
            return extracted.encode("utf-8")
        raise TypeError(f"GrayscaleCoverAdapter expected bytes, got {type(extracted)}")
