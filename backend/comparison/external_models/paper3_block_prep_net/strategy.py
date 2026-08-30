"""Paper 3 strategy — Dabhade et al. (2026), MTA 85:482. ImageTileAdapter 64x64x3."""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np

from backend.strategies.base import EmbeddingStrategy, StrategyResult
from backend.strategies.registry import register
from backend.crypto import encrypt_payload, decrypt_payload
from backend.metrics import calculate_metrics
from backend.comparison.adapters.image_tile_adapter import ImageTileAdapter
from .models import HAS_TORCH, load_paper3_model, TILE
from backend.comparison.checkpoint.manager import discover_checkpoints

if HAS_TORCH:
    import torch


@register
class Paper3BlockPrepNet(EmbeddingStrategy):
    name = "paper3_block_prep_net"

    def __init__(self, checkpoint: Optional[str] = None):
        self.adapter = ImageTileAdapter(tile_h=TILE, tile_w=TILE, channels=3)
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self._model = None
        self._mode = "untrained_reference_architecture"
        self._ckpt_info = discover_checkpoints("paper3_block_prep_net")

    def _ensure_model(self):
        if self._model is None:
            self._model, self._mode = load_paper3_model(self.checkpoint)

    def _resize(self, img: np.ndarray) -> np.ndarray:
        from PIL import Image
        return np.array(Image.fromarray(img.astype(np.uint8)).resize((TILE, TILE), Image.BILINEAR), dtype=np.uint8)

    def embed(self, cover: np.ndarray, message: str, passphrase: str, **kwargs) -> StrategyResult:
        encrypted = encrypt_payload(message, passphrase)
        adapted = self.adapter.adapt_for_embed(encrypted, cover)
        meta = {
            "strategy": self.name,
            "source_type": "paper",
            "paper_id": "paper3_block_prep_net",
            "adapter_used": adapted.adapter_used,
            "capacity_limited": adapted.capacity_limited,
            "native_operating_point": adapted.native_operating_point,
            "ml_dl": True,
            "training_mode": self._mode,
            "model_status": self._mode,
            "checkpoint_status": self._ckpt_info.verification_status,
            "checkpoint_path": self._ckpt_info.path,
            "training_status": "untrained",
            "benchmark_status": "ARCHITECTURE_TEST",
            "native_payload_type": "secret_image_tile_64x64x3",
            "note": "NO OFFICIAL COMPATIBLE CHECKPOINT FOUND",
        }
        if adapted.capacity_limited:
            return StrategyResult(
                stego=cover.copy(),
                metrics={"status": "N/A", "reason": "ciphertext exceeds 64x64x3 tile"},
                security={},
                meta=meta,
            )
        if not HAS_TORCH:
            return StrategyResult(
                stego=cover.copy(),
                metrics={"status": "FAILED", "reason": "torch not available"},
                security={},
                meta={**meta, "training_mode": "no_torch"},
            )

        self._ensure_model()
        meta["training_mode"] = self._mode
        cover_t = self._resize(cover)
        secret_t = adapted.payload

        ct = torch.from_numpy(cover_t.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        st = torch.from_numpy(secret_t.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            container = self._model.embed(ct, st)
        stego_tile = (container.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        stego = cover.copy()
        H, W = stego.shape[:2]
        y0, x0 = max(0, (H - TILE) // 2), max(0, (W - TILE) // 2)
        y1, x1 = min(H, y0 + TILE), min(W, x0 + TILE)
        stego[y0:y1, x0:x1] = stego_tile[: y1 - y0, : x1 - x0]

        metrics = calculate_metrics(cover, stego, adapted.meta.get("ciphertext_len", 0) * 8, {})
        metrics["training_mode"] = self._mode
        metrics["tile_psnr_note"] = "untrained weights — not paper-reported figures"
        return StrategyResult(stego=stego, metrics=metrics, security={}, meta=meta)

    def extract(self, stego: np.ndarray, passphrase: str, **kwargs) -> str:
        if not HAS_TORCH:
            raise RuntimeError("torch not available")
        self._ensure_model()
        H, W = stego.shape[:2]
        y0, x0 = max(0, (H - TILE) // 2), max(0, (W - TILE) // 2)
        tile = stego[y0 : y0 + TILE, x0 : x0 + TILE]
        if tile.shape[0] != TILE or tile.shape[1] != TILE:
            tile = self._resize(tile)
        t = torch.from_numpy(tile.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            rec = self._model.extract(t)
        arr = (rec.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        raw = self.adapter.recover_from_extract(arr)
        min_len = 16 + 16 + 12 + 16
        last_err = None
        for end in range(len(raw), min_len - 1, -1):
            try:
                return decrypt_payload(raw[:end], passphrase)
            except Exception as e:
                last_err = e
        if last_err:
            raise last_err
        raise ValueError("Paper3 extract failed")
