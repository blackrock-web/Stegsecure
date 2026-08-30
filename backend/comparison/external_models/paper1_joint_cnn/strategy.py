"""Paper 1 strategy — Iqbal et al. (2026).

Uses official architecture from:
  https://github.com/Ayshcoder987/Joint-Encryption-Steganography-CNN
when available. NO official pretrained checkpoint is published in that repo
(Releases: none; tree has code only). Status remains architecture-test until
the user places a compatible checkpoint under models/paper1/official/.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np

from backend.strategies.base import EmbeddingStrategy, StrategyResult
from backend.strategies.registry import register
from backend.crypto import encrypt_payload, decrypt_payload
from backend.metrics import calculate_metrics
from backend.comparison.adapters.image_tile_adapter import ImageTileAdapter
from backend.comparison.checkpoint.manager import discover_checkpoints

HAS_TORCH = False
try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass

TILE = 96
OFFICIAL_REPO = "https://github.com/Ayshcoder987/Joint-Encryption-Steganography-CNN"


def _build_model(checkpoint_path: Optional[Path] = None):
    """Prefer official architecture; fall back to simplified reference nets."""
    mode = "official_architecture_no_checkpoint"
    model = None
    if not HAS_TORCH:
        return None, "no_torch"

    try:
        from .official_arch import OfficialPaper1Model, HAS_TORCH as OT
        if OT:
            model = OfficialPaper1Model()
            mode = "official_architecture_no_checkpoint"
    except Exception:
        from .models import load_paper1_model
        model, mode = load_paper1_model(checkpoint_path)
        return model, mode

    if checkpoint_path is not None and Path(checkpoint_path).is_file():
        try:
            state = torch.load(str(checkpoint_path), map_location="cpu")
            if isinstance(state, dict):
                # Official training script saves encoder/decoder/mixer state dicts
                if "encoder_state_dict" in state:
                    model.encoder.load_state_dict(state["encoder_state_dict"], strict=False)
                    model.decoder.load_state_dict(state["decoder_state_dict"], strict=False)
                    model.mixer.load_state_dict(state["mixer_state_dict"], strict=False)
                    mode = "authors_official_checkpoint"
                else:
                    model.load_state_dict(state, strict=False)
                    mode = "authors_official_checkpoint"
            model.eval()
            return model, mode
        except Exception as e:
            mode = f"official_architecture_checkpoint_load_failed:{e}"
    model.eval()
    return model, mode


@register
class Paper1JointCNN(EmbeddingStrategy):
    name = "paper1_joint_cnn"

    def __init__(self, checkpoint: Optional[str] = None):
        self.adapter = ImageTileAdapter(tile_h=TILE, tile_w=TILE, channels=3)
        ckpt_info = discover_checkpoints("paper1_joint_cnn")
        if checkpoint:
            self.checkpoint = Path(checkpoint)
        elif ckpt_info.exists and ckpt_info.path:
            self.checkpoint = Path(ckpt_info.path)
        else:
            self.checkpoint = None
        self._model = None
        self._mode = "official_architecture_no_checkpoint"
        self._ckpt_info = ckpt_info

    def _ensure_model(self):
        if self._model is None:
            self._model, self._mode = _build_model(self.checkpoint)

    def _model_meta(self) -> dict:
        return {
            "strategy": self.name,
            "source_type": "paper",
            "paper_id": "paper1_joint_cnn",
            "method_type": "joint_cnn",
            "model_status": self._mode,
            "checkpoint_status": self._ckpt_info.verification_status,
            "checkpoint_path": self._ckpt_info.path,
            "checkpoint_source": OFFICIAL_REPO,
            "official_source_url": OFFICIAL_REPO,
            "training_status": (
                "trained" if self._mode == "authors_official_checkpoint" else "untrained"
            ),
            "benchmark_status": (
                "LIVE_VALIDATED" if self._mode == "authors_official_checkpoint"
                else "ARCHITECTURE_TEST"
            ),
            "native_payload_type": "secret_image_tile_96x96x3",
            "ml_dl": True,
            "training_mode": self._mode,
        }

    def _resize_cover(self, cover: np.ndarray) -> np.ndarray:
        from PIL import Image
        img = Image.fromarray(cover.astype(np.uint8))
        img = img.resize((TILE, TILE), Image.BILINEAR)
        return np.array(img, dtype=np.uint8)

    def embed(self, cover: np.ndarray, message: str, passphrase: str, **kwargs) -> StrategyResult:
        encrypted = encrypt_payload(message, passphrase)
        adapted = self.adapter.adapt_for_embed(encrypted, cover)
        meta = self._model_meta()
        meta.update({
            "adapter_used": adapted.adapter_used,
            "capacity_limited": adapted.capacity_limited,
            "native_operating_point": adapted.native_operating_point,
        })
        if adapted.capacity_limited:
            return StrategyResult(
                stego=cover.copy(),
                metrics={"status": "N/A", "reason": "ciphertext exceeds 96x96x3 tile"},
                security={},
                meta=meta,
            )
        if not HAS_TORCH:
            meta["model_status"] = "no_torch"
            meta["benchmark_status"] = "FAILED"
            return StrategyResult(
                stego=cover.copy(),
                metrics={"status": "FAILED", "reason": "torch not available"},
                security={},
                meta=meta,
            )

        self._ensure_model()
        meta = self._model_meta()
        meta.update({
            "adapter_used": adapted.adapter_used,
            "capacity_limited": False,
            "native_operating_point": adapted.native_operating_point,
        })

        cover_tile = self._resize_cover(cover)
        secret_tile = adapted.payload
        # Official nets use Tanh → map to [-1,1]
        cover_t = torch.from_numpy(cover_tile.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
        secret_t = torch.from_numpy(secret_tile.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            container = self._model.embed(cover_t, secret_t)
        # Tanh → [0,255]
        stego_tile = ((container.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

        stego = cover.copy()
        H, W = stego.shape[:2]
        y0 = max(0, (H - TILE) // 2)
        x0 = max(0, (W - TILE) // 2)
        y1, x1 = min(H, y0 + TILE), min(W, x0 + TILE)
        stego[y0:y1, x0:x1] = stego_tile[: y1 - y0, : x1 - x0]

        metrics = calculate_metrics(cover, stego, adapted.meta.get("ciphertext_len", 0) * 8, {})
        metrics["training_mode"] = self._mode
        metrics["benchmark_status"] = meta["benchmark_status"]
        if meta["benchmark_status"] == "ARCHITECTURE_TEST":
            metrics["tile_psnr_note"] = (
                "Official architecture, NO published checkpoint — architecture/pipeline test only"
            )
        return StrategyResult(stego=stego, metrics=metrics, security={}, meta=meta)

    def extract(self, stego: np.ndarray, passphrase: str, **kwargs) -> str:
        if not HAS_TORCH:
            raise RuntimeError("torch not available for Paper1 extract")
        self._ensure_model()
        H, W = stego.shape[:2]
        y0 = max(0, (H - TILE) // 2)
        x0 = max(0, (W - TILE) // 2)
        tile = stego[y0 : y0 + TILE, x0 : x0 + TILE]
        if tile.shape[0] != TILE or tile.shape[1] != TILE:
            from PIL import Image
            tile = np.array(Image.fromarray(tile).resize((TILE, TILE), Image.BILINEAR), dtype=np.uint8)

        t = torch.from_numpy(tile.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            recovered = self._model.extract(t)
        rec = ((recovered.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        raw = self.adapter.recover_from_extract(rec)
        min_len = 16 + 16 + 12 + 16
        last_err = None
        for end in range(len(raw), min_len - 1, -1):
            try:
                return decrypt_payload(raw[:end], passphrase)
            except Exception as e:
                last_err = e
        if last_err:
            raise last_err
        raise ValueError("Paper1 extract failed")
