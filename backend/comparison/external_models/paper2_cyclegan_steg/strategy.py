"""Paper 2 strategy — Abdollahi et al. (2023). GrayscaleCoverAdapter.

Metadata clearly labels: reduced-training approximation, no adversarial co-training.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, List
import numpy as np

from backend.strategies.base import EmbeddingStrategy, StrategyResult
from backend.strategies.registry import register
from backend.crypto import encrypt_payload, decrypt_payload
from backend.metrics import calculate_metrics
from backend.comparison.adapters.grayscale_adapter import GrayscaleCoverAdapter
from .models import HAS_TORCH, load_paper2_model
from backend.comparison.checkpoint.manager import discover_checkpoints

if HAS_TORCH:
    import torch


def _message_to_z(cipher: bytes, h: int, w: int) -> np.ndarray:
    """Map binary message to a message image Z (grayscale), tile/pad."""
    bits = []
    for b in cipher:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    need = h * w
    if len(bits) < need:
        bits = bits + [0] * (need - len(bits))
    else:
        bits = bits[:need]
    arr = np.array(bits, dtype=np.float32).reshape(h, w)
    return arr


def _z_to_bytes(z: np.ndarray, max_bytes: int) -> bytes:
    bits = (z.flatten() > 0.5).astype(np.uint8).tolist()
    out = bytearray()
    for i in range(0, min(len(bits), max_bytes * 8) - 7, 8):
        b = 0
        for j in range(8):
            b = (b << 1) | int(bits[i + j])
        out.append(b)
    return bytes(out)


@register
class Paper2CycleGANSteg(EmbeddingStrategy):
    name = "paper2_cyclegan_steg"

    def __init__(self, checkpoint: Optional[str] = None):
        self.adapter = GrayscaleCoverAdapter()
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self._model = None
        self._mode = "reduced_training_approximation"
        self._ckpt_info = discover_checkpoints("paper2_cyclegan_steg")

    def _ensure_model(self):
        if self._model is None:
            self._model, self._mode = load_paper2_model(self.checkpoint)

    def embed(self, cover: np.ndarray, message: str, passphrase: str, **kwargs) -> StrategyResult:
        encrypted = encrypt_payload(message, passphrase)
        adapted = self.adapter.adapt_for_embed(encrypted, cover)
        gray = adapted.meta.pop("gray_cover")
        meta = {
            "strategy": self.name,
            "source_type": "paper",
            "paper_id": "paper2_cyclegan_steg",
            "adapter_used": adapted.adapter_used,
            "capacity_limited": False,
            "native_operating_point": True,
            "ml_dl": True,
            "training_mode": self._mode,
            "model_status": self._mode,
            "checkpoint_status": self._ckpt_info.verification_status,
            "checkpoint_path": self._ckpt_info.path,
            "training_status": "untrained",
            "benchmark_status": "ARCHITECTURE_TEST",
            "native_payload_type": "binary_message_image_grayscale",
            "note": "NO OFFICIAL COMPATIBLE CHECKPOINT FOUND. grayscale-only; not apples-to-apples RGB",
        }
        if not HAS_TORCH:
            return StrategyResult(
                stego=cover.copy(),
                metrics={"status": "FAILED", "reason": "torch not available"},
                security={},
                meta={**meta, "training_mode": "no_torch"},
            )

        self._ensure_model()
        meta["training_mode"] = self._mode
        H, W = gray.shape[:2]
        # Downscale to 256 max for memory if needed
        max_side = 256
        scale = 1.0
        if max(H, W) > max_side:
            scale = max_side / max(H, W)
            from PIL import Image
            gray_s = np.array(
                Image.fromarray(gray).resize((int(W * scale), int(H * scale)), Image.BILINEAR),
                dtype=np.uint8,
            )
        else:
            gray_s = gray
        h, w = gray_s.shape
        z = _message_to_z(encrypted, h, w)

        cover_t = torch.from_numpy(gray_s.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        z_t = torch.from_numpy(z).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            stego_s, _ = self._model.embed(cover_t, z_t)
        stego_gray = (stego_s.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        # Upscale back and broadcast to RGB for consistent API
        if scale < 1.0:
            from PIL import Image
            stego_gray = np.array(Image.fromarray(stego_gray).resize((W, H), Image.BILINEAR), dtype=np.uint8)
        stego = np.stack([stego_gray, stego_gray, stego_gray], axis=2)

        metrics = calculate_metrics(cover, stego, len(encrypted) * 8, {})
        metrics["training_mode"] = self._mode
        metrics["adapter_used"] = "grayscale"
        return StrategyResult(stego=stego, metrics=metrics, security={}, meta=meta)

    def extract(self, stego: np.ndarray, passphrase: str, **kwargs) -> str:
        if not HAS_TORCH:
            raise RuntimeError("torch not available")
        self._ensure_model()
        if stego.ndim == 3:
            gray = (0.299 * stego[:, :, 0] + 0.587 * stego[:, :, 1] + 0.114 * stego[:, :, 2]).astype(np.uint8)
        else:
            gray = stego.astype(np.uint8)
        H, W = gray.shape
        max_side = 256
        scale = 1.0
        if max(H, W) > max_side:
            scale = max_side / max(H, W)
            from PIL import Image
            gray_s = np.array(
                Image.fromarray(gray).resize((int(W * scale), int(H * scale)), Image.BILINEAR),
                dtype=np.uint8,
            )
        else:
            gray_s = gray
        t = torch.from_numpy(gray_s.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            z_tilde, _ = self._model.extract(t)
        z = z_tilde.squeeze().cpu().numpy()
        raw = _z_to_bytes(z, max_bytes=len(z.flatten()) // 8)
        min_len = 16 + 16 + 12 + 16
        last_err = None
        for end in range(len(raw), min_len - 1, -1):
            try:
                return decrypt_payload(raw[:end], passphrase)
            except Exception as e:
                last_err = e
        if last_err:
            raise last_err
        raise ValueError("Paper2 extract failed")
