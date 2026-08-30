"""Preparation / Hiding / Reveal networks per Dabhade et al. (2026).

Multi-filter parallel branches (3x3/4x4/5x5), 2 iterations, concatenated.
Untrained/reference-architecture by default.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    pass

TILE = 64


if HAS_TORCH:

    class MultiFilterBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.b3 = nn.Conv2d(in_ch, out_ch // 3, 3, padding=1)
            self.b4 = nn.Conv2d(in_ch, out_ch // 3, 4, padding=1)
            self.b5 = nn.Conv2d(in_ch, out_ch - 2 * (out_ch // 3), 5, padding=2)

        def forward(self, x):
            a = F.relu(self.b3(x))
            b = F.relu(self.b4(x))
            # pad b4 output spatially if needed
            if b.shape[-2:] != a.shape[-2:]:
                b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
            c = F.relu(self.b5(x))
            if c.shape[-2:] != a.shape[-2:]:
                c = F.interpolate(c, size=a.shape[-2:], mode="bilinear", align_corners=False)
            return torch.cat([a, b, c], dim=1)

    class PrepNet(nn.Module):
        def __init__(self, in_ch=3, feat=64):
            super().__init__()
            self.iter1 = MultiFilterBlock(in_ch, feat)
            self.iter2 = MultiFilterBlock(feat, feat)

        def forward(self, secret):
            h = self.iter1(secret)
            h = self.iter2(h)
            return h

    class HidingNet(nn.Module):
        def __init__(self, cover_ch=3, feat=64, out_ch=3):
            super().__init__()
            self.fuse = MultiFilterBlock(cover_ch + feat, 64)
            self.out = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, out_ch, 3, padding=1),
                nn.Sigmoid(),
            )

        def forward(self, cover, h_prime):
            if h_prime.shape[-2:] != cover.shape[-2:]:
                h_prime = F.interpolate(h_prime, size=cover.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([cover, h_prime], dim=1)
            return self.out(self.fuse(x))

    class RevealNet(nn.Module):
        def __init__(self, in_ch=3, out_ch=3):
            super().__init__()
            self.net = nn.Sequential(
                MultiFilterBlock(in_ch, 64),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, out_ch, 3, padding=1),
                nn.Sigmoid(),
            )

        def forward(self, stego, noise_sigma: float = 0.01):
            if self.training and noise_sigma > 0:
                stego = stego + noise_sigma * torch.randn_like(stego)
            return self.net(stego)

    class Paper3Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.prep = PrepNet()
            self.hide = HidingNet()
            self.reveal = RevealNet()

        def embed(self, cover, secret):
            h = self.prep(secret)
            return self.hide(cover, h)

        def extract(self, stego):
            return self.reveal(stego, noise_sigma=0.0)

else:
    Paper3Model = None  # type: ignore


def load_paper3_model(checkpoint: Optional[Path] = None) -> Tuple[Optional[object], str]:
    if not HAS_TORCH:
        return None, "no_torch"
    model = Paper3Model()
    model.eval()
    if checkpoint is not None and Path(checkpoint).is_file():
        try:
            state = torch.load(str(checkpoint), map_location="cpu")
            model.load_state_dict(state, strict=False)
            return model, "checkpoint"
        except Exception:
            return model, "untrained/reference-architecture"
    return model, "untrained/reference-architecture"
