"""Payload adapters for cross-method interoperability."""

from .base_adapter import PayloadAdapter, AdapterResult
from .bitstream_adapter import BitstreamAdapter
from .image_tile_adapter import ImageTileAdapter
from .grayscale_adapter import GrayscaleCoverAdapter

__all__ = [
    "PayloadAdapter",
    "AdapterResult",
    "BitstreamAdapter",
    "ImageTileAdapter",
    "GrayscaleCoverAdapter",
]
