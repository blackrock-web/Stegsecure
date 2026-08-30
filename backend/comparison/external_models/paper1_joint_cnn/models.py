"""KeyMixer + Encoder + Decoder architectures per Iqbal et al. (2026).

Untrained/reference-architecture mode by default. User-supplied checkpoints
can be loaded later; never claim paper-reported PSNR/SSIM as locally reproduced
unless the model was actually trained here.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    pass

TILE = 96


if HAS_TORCH:

    class KeyMixer(nn.Module):
        """S_trans = clip(sigma(W2*ReLU(W1*S+B1)+b2) + K, 0, 1), K ~ N(0, 0.01)."""

        def __init__(self, channels: int = 3):
            super().__init__()
            self.fc1 = nn.Linear(channels, 32)
            self.fc2 = nn.Linear(32, channels)
            self.register_buffer("K", torch.randn(1, channels, 1, 1) * 0.01)

        def forward(self, s: "torch.Tensor") -> "torch.Tensor":
            # s: B,C,H,W in [0,1]
            b, c, h, w = s.shape
            x = s.permute(0, 2, 3, 1).reshape(-1, c)
            x = F.relu(self.fc1(x))
            x = torch.sigmoid(self.fc2(x))
            x = x.view(b, h, w, c).permute(0, 3, 1, 2)
            return torch.clamp(x + self.K, 0.0, 1.0)

    class Encoder(nn.Module):
        """Encoder(concat(Cover, S_trans)) -> Container."""

        def __init__(self, in_ch: int = 6, out_ch: int = 3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, out_ch, 3, padding=1),
                nn.Sigmoid(),
            )

        def forward(self, cover: "torch.Tensor", s_trans: "torch.Tensor") -> "torch.Tensor":
            x = torch.cat([cover, s_trans], dim=1)
            return self.net(x)

    class Decoder(nn.Module):
        """Decoder(Container) -> Recovered_Secret (no key/aux input)."""

        def __init__(self, in_ch: int = 3, out_ch: int = 3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, out_ch, 3, padding=1),
                nn.Sigmoid(),
            )

        def forward(self, container: "torch.Tensor") -> "torch.Tensor":
            return self.net(container)

    class Paper1Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.key_mixer = KeyMixer()
            self.encoder = Encoder()
            self.decoder = Decoder()

        def embed(self, cover: "torch.Tensor", secret: "torch.Tensor") -> "torch.Tensor":
            s_trans = self.key_mixer(secret)
            return self.encoder(cover, s_trans)

        def extract(self, container: "torch.Tensor") -> "torch.Tensor":
            return self.decoder(container)

else:
    KeyMixer = Encoder = Decoder = Paper1Model = None  # type: ignore


def load_paper1_model(checkpoint: Optional[Path] = None) -> Tuple[Optional[object], str]:
    """Load model; returns (model_or_None, mode_label)."""
    if not HAS_TORCH:
        return None, "no_torch"
    model = Paper1Model()
    model.eval()
    if checkpoint is not None and Path(checkpoint).is_file():
        try:
            state = torch.load(str(checkpoint), map_location="cpu")
            model.load_state_dict(state, strict=False)
            return model, "checkpoint"
        except Exception:
            return model, "untrained/reference-architecture"
    return model, "untrained/reference-architecture"
