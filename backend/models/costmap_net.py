"""
CostMapCNN — Learnable multi-scale residual + attention cost predictor.

Architecture (lightweight, CPU-friendly):
  Input  : 1×H×W grayscale (or 3-channel RGB projected)
  Stem   : 5×5 high-pass residual (fixed KV-style) + 3×3 learnable
  Blocks : 3 residual stages with channel attention (SE-style)
  Head   : 1×1 conv → sigmoid cost map in [0, 1]

Higher cost values = safer (more textured / complex) embedding locations.
Trained to approximate a combination of HILL residual energy + multi-scale
texture energy on synthetic textured images.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    F = None

# Default weight location (relative to project root)
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
COSTMAP_WEIGHTS = MODELS_DIR / "costmap_cnn.pth"


if HAS_TORCH:

    class SEBlock(nn.Module):
        """Squeeze-and-Excitation channel attention."""
        def __init__(self, channels: int, reduction: int = 4):
            super().__init__()
            self.fc = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, max(channels // reduction, 4)),
                nn.ReLU(inplace=True),
                nn.Linear(max(channels // reduction, 4), channels),
                nn.Sigmoid(),
            )

        def forward(self, x):
            w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
            return x * w

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)
            self.se = SEBlock(channels)
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            residual = x
            out = self.act(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = self.se(out)
            return self.act(out + residual)

    class CostMapCNN(nn.Module):
        """
        Multi-scale residual cost map network.
        Output is a single-channel map in [0, 1] (higher = safer for embedding).
        """
        def __init__(self, in_channels: int = 1, base_ch: int = 16):
            super().__init__()
            # Fixed high-pass residual extractor (KV kernel style)
            self.register_buffer(
                "hp_kernel",
                torch.tensor(
                    [
                        [-1, 2, -2, 2, -1],
                        [2, -6, 8, -6, 2],
                        [-2, 8, -12, 8, -2],
                        [2, -6, 8, -6, 2],
                        [-1, 2, -2, 2, -1],
                    ],
                    dtype=torch.float32,
                ).view(1, 1, 5, 5)
                / 12.0,
            )

            self.stem = nn.Sequential(
                nn.Conv2d(in_channels + 1, base_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(base_ch),
                nn.ReLU(inplace=True),
            )
            self.block1 = ResidualBlock(base_ch)
            self.down1 = nn.Sequential(
                nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_ch * 2),
                nn.ReLU(inplace=True),
            )
            self.block2 = ResidualBlock(base_ch * 2)
            self.down2 = nn.Sequential(
                nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_ch * 4),
                nn.ReLU(inplace=True),
            )
            self.block3 = ResidualBlock(base_ch * 4)

            # Multi-scale fusion head
            self.head = nn.Sequential(
                nn.Conv2d(base_ch + base_ch * 2 + base_ch * 4, base_ch * 2, 1, bias=False),
                nn.BatchNorm2d(base_ch * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_ch * 2, 1, 1),
                nn.Sigmoid(),
            )

            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: B×C×H×W, expected C=1 (gray) or already normalized
            if x.shape[1] > 1:
                # luminance
                x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

            # high-pass residual
            hp = F.conv2d(x, self.hp_kernel, padding=2)
            inp = torch.cat([x, torch.abs(hp)], dim=1)

            f1 = self.block1(self.stem(inp))
            f2 = self.block2(self.down1(f1))
            f3 = self.block3(self.down2(f2))

            # upsample to original resolution and fuse
            H, W = x.shape[2], x.shape[3]
            f2u = F.interpolate(f2, size=(H, W), mode="bilinear", align_corners=False)
            f3u = F.interpolate(f3, size=(H, W), mode="bilinear", align_corners=False)
            fused = torch.cat([f1, f2u, f3u], dim=1)
            cost = self.head(fused)
            return cost  # B×1×H×W


_COSTMAP_MODEL: Optional["CostMapCNN"] = None
_COSTMAP_DEVICE = "cpu"


def get_costmap_model(
    weights_path: Optional[str | Path] = None,
    force_reload: bool = False,
) -> Optional["CostMapCNN"]:
    """Load (or return cached) CostMapCNN. Prefers trained weights if present."""
    global _COSTMAP_MODEL
    if not HAS_TORCH:
        return None
    if _COSTMAP_MODEL is not None and not force_reload:
        return _COSTMAP_MODEL

    path = Path(weights_path) if weights_path else COSTMAP_WEIGHTS
    model = CostMapCNN(in_channels=1, base_ch=16)
    model.eval()

    if path.is_file():
        try:
            state = torch.load(str(path), map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=False)
            print(f"[CostMapCNN] Loaded trained weights from {path}")
        except Exception as e:
            print(f"[CostMapCNN] Could not load weights ({e}); using random init")
    else:
        print("[CostMapCNN] No trained weights found — using randomly initialized network")

    for p in model.parameters():
        p.requires_grad = False
    _COSTMAP_MODEL = model
    return model


def train_costmap_model(
    num_samples: int = 64,
    epochs: int = 8,
    img_size: int = 128,
    lr: float = 1e-3,
    save_path: Optional[str | Path] = None,
    verbose: bool = True,
) -> Path:
    """
    Quick synthetic training of CostMapCNN.
    Generates textured patches + HILL-like target cost maps and regresses.
    Runs in seconds on CPU for the default sizes.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for training")

    import cv2

    save_path = Path(save_path) if save_path else COSTMAP_WEIGHTS
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model = CostMapCNN(in_channels=1, base_ch=16)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    def make_batch(bs: int = 8):
        imgs = []
        targets = []
        for _ in range(bs):
            # synthetic textured image
            base = np.random.randn(img_size, img_size).astype(np.float32) * 0.15
            # add low-freq structure
            base += cv2.GaussianBlur(np.random.randn(img_size, img_size).astype(np.float32), (31, 31), 5) * 0.4
            # add edges / textures
            noise = np.random.randn(img_size, img_size).astype(np.float32)
            texture = cv2.filter2D(noise, -1, np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], np.float32) / 4)
            img = np.clip(base + 0.3 * texture, -1, 1)
            gray = ((img + 1) * 127.5).astype(np.uint8)

            # HILL-style target
            hp = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], np.float32) / 12
            res = np.abs(cv2.filter2D(gray.astype(np.float32), -1, hp))
            l1 = cv2.GaussianBlur(res, (3, 3), 0.5)
            l2 = cv2.GaussianBlur(l1, (15, 15), 2.0)
            tmin, tmax = l2.min(), l2.max()
            target = (l2 - tmin) / (tmax - tmin + 1e-8)

            imgs.append(gray.astype(np.float32) / 255.0)
            targets.append(target.astype(np.float32))

        x = torch.from_numpy(np.stack(imgs)[:, None, ...])
        y = torch.from_numpy(np.stack(targets)[:, None, ...])
        return x, y

    if verbose:
        print(f"[CostMapCNN] Training on {num_samples} synthetic samples, {epochs} epochs …")

    steps = max(1, num_samples // 8)
    for ep in range(epochs):
        ep_loss = 0.0
        for _ in range(steps):
            x, y = make_batch(8)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        if verbose:
            print(f"  epoch {ep+1}/{epochs}  loss={ep_loss/steps:.5f}")

    model.eval()
    torch.save(model.state_dict(), str(save_path))
    if verbose:
        print(f"[CostMapCNN] Weights saved → {save_path}")
    return save_path
