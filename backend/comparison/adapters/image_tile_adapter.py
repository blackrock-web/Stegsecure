"""Image-tile adapter for Paper 1 (96x96x3) and Paper 3 (64x64x3).

Converts AES-GCM ciphertext bytes into a fixed-size binary "image" tile
(UTF-8 style: binary reshape) matching each paper's text-preprocessing.
If ciphertext exceeds one tile, capacity_limited=True and result is N/A —
never silently split across multiple tiles.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np

from .base_adapter import PayloadAdapter, AdapterResult


class ImageTileAdapter(PayloadAdapter):
    name = "image_tile"

    def __init__(self, tile_h: int = 64, tile_w: int = 64, channels: int = 3):
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.channels = channels
        self.tile_bytes = tile_h * tile_w * channels  # 8-bit values

    def adapt_for_embed(self, ciphertext: bytes, cover: np.ndarray, **kwargs) -> AdapterResult:
        capacity = self.tile_bytes
        limited = len(ciphertext) > capacity
        if limited:
            # Do not embed partial; mark capacity_limited
            tile = np.zeros((self.tile_h, self.tile_w, self.channels), dtype=np.uint8)
            return AdapterResult(
                payload=tile,
                adapter_used=self.name,
                capacity_limited=True,
                native_operating_point=False,
                meta={
                    "tile_shape": [self.tile_h, self.tile_w, self.channels],
                    "ciphertext_len": len(ciphertext),
                    "tile_capacity_bytes": capacity,
                    "reason": "ciphertext exceeds single-tile capacity",
                },
            )

        # Pad to exact tile size with zeros
        buf = bytearray(capacity)
        buf[: len(ciphertext)] = ciphertext
        tile = np.frombuffer(bytes(buf), dtype=np.uint8).reshape(
            (self.tile_h, self.tile_w, self.channels)
        )
        return AdapterResult(
            payload=tile.copy(),
            adapter_used=self.name,
            capacity_limited=False,
            native_operating_point=True,
            meta={
                "tile_shape": [self.tile_h, self.tile_w, self.channels],
                "ciphertext_len": len(ciphertext),
                "tile_capacity_bytes": capacity,
            },
        )

    def recover_from_extract(self, extracted: Any, **kwargs) -> bytes:
        """Flatten recovered tile and strip trailing zero padding conservatively.

        Caller should still attempt progressive AES-GCM decrypt if needed.
        """
        if isinstance(extracted, np.ndarray):
            flat = extracted.astype(np.uint8).reshape(-1)
            raw = flat.tobytes()
            # Strip trailing zeros but keep at least header-sized payload
            end = len(raw)
            while end > 48 and raw[end - 1] == 0:
                end -= 1
            return raw[:end]
        if isinstance(extracted, (bytes, bytearray)):
            return bytes(extracted)
        raise TypeError(f"ImageTileAdapter expected ndarray/bytes, got {type(extracted)}")
