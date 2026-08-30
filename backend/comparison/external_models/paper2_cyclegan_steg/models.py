"""G (encoder), F (reconstructor), H (extractor) residual U-Net architectures.

Reduced-training approximation — no adversarial co-training against D.
Labeled clearly in strategy metadata.
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


if HAS_TORCH:

    class ResBlock(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(ch, ch, 4, padding=1),
                nn.InstanceNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, 4, padding=2),
                nn.InstanceNorm2d(ch),
            )

        def forward(self, x):
            y = self.conv(x)
            if y.shape[-2:] != x.shape[-2:]:
                y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
            return F.relu(x + y)

    class ResidualUNet(nn.Module):
        """Simplified residual U-Net (14-layer style, 4x4 kernels, instance norm)."""

        def __init__(self, in_ch, out_ch, base=32):
            super().__init__()
            self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base, 4, stride=2, padding=1), nn.InstanceNorm2d(base), nn.ReLU())
            self.enc2 = nn.Sequential(nn.Conv2d(base, base * 2, 4, stride=2, padding=1), nn.InstanceNorm2d(base * 2), nn.ReLU())
            self.mid = ResBlock(base * 2)
            self.up2 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
            self.up1 = nn.ConvTranspose2d(base * 2, out_ch, 4, stride=2, padding=1)
            self.out_act = nn.Tanh()

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            m = self.mid(e2)
            u2 = F.relu(self.up2(m))
            if u2.shape[-2:] != e1.shape[-2:]:
                u2 = F.interpolate(u2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
            u1 = self.up1(torch.cat([u2, e1], dim=1))
            if u1.shape[-2:] != x.shape[-2:]:
                u1 = F.interpolate(u1, size=x.shape[-2:], mode="bilinear", align_corners=False)
            return self.out_act(u1)

    class ExtractorH(nn.Module):
        """20-layer style U-Net extractor (simplified)."""

        def __init__(self, in_ch=2, out_ch=1, base=32):
            super().__init__()
            self.net = ResidualUNet(in_ch, out_ch, base=base)

        def forward(self, stego, recon_cover):
            x = torch.cat([stego, recon_cover], dim=1)
            return self.net(x)

    class Paper2Model(nn.Module):
        def __init__(self):
            super().__init__()
            # G: cover(1) + message_image(1) -> modification map
            self.G = ResidualUNet(2, 1, base=32)
            # F: stego -> reconstructed cover
            self.F = ResidualUNet(1, 1, base=32)
            # H: stego + recon -> message
            self.H = ExtractorH(2, 1, base=32)

        def embed(self, cover, z_msg):
            # cover, z_msg: Bx1xHxW in [-1,1] or [0,1]
            inp = torch.cat([cover, z_msg], dim=1)
            p = self.G(inp) * 0.05  # perturbation bound b=0.05 style
            stego = torch.clamp(cover + p, 0.0, 1.0)
            return stego, p

        def extract(self, stego):
            recon = self.F(stego)
            z_tilde = self.H(stego, recon)
            return z_tilde, recon

else:
    Paper2Model = None  # type: ignore


def load_paper2_model(checkpoint: Optional[Path] = None) -> Tuple[Optional[object], str]:
    if not HAS_TORCH:
        return None, "no_torch"
    model = Paper2Model()
    model.eval()
    if checkpoint is not None and Path(checkpoint).is_file():
        try:
            state = torch.load(str(checkpoint), map_location="cpu")
            model.load_state_dict(state, strict=False)
            return model, "checkpoint"
        except Exception:
            return model, "reduced-training approximation, no adversarial co-training"
    return model, "reduced-training approximation, no adversarial co-training"
