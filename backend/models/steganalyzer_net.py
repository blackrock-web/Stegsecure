"""
SteganalyzerNet — Improved SRNet-style residual steganalyzer.

Trained binary classifier (Cover=0 / Stego=1).
Used for:
  - Adversarial gradient guidance (ADV-EMB)
  - Security report / detection probability

Architecture:
  High-pass residual → residual blocks with progressive downsampling →
  global average pool → 2-class logits.
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

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
STEGA_WEIGHTS = MODELS_DIR / "steganalyzer_net.pth"


if HAS_TORCH:

    class HighPassFilter(nn.Module):
        def __init__(self):
            super().__init__()
            kv = np.array(
                [
                    [-1, 2, -2, 2, -1],
                    [2, -6, 8, -6, 2],
                    [-2, 8, -12, 8, -2],
                    [2, -6, 8, -6, 2],
                    [-1, 2, -2, 2, -1],
                ],
                dtype=np.float32,
            ) / 12.0
            w = torch.tensor(kv).view(1, 1, 5, 5)
            self.conv = nn.Conv2d(1, 1, 5, padding=2, bias=False)
            self.conv.weight = nn.Parameter(w, requires_grad=False)

        def forward(self, x):
            return self.conv(x)

    class ResidualBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_ch)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_ch)
            self.act = nn.ReLU(inplace=True)
            self.down = None
            if stride != 1 or in_ch != out_ch:
                self.down = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_ch),
                )

        def forward(self, x):
            residual = x if self.down is None else self.down(x)
            out = self.act(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return self.act(out + residual)

    class SteganalyzerNet(nn.Module):
        """
        Lightweight residual steganalyzer (SRNet-inspired).
        Input: B×1×H×W grayscale normalized to [0,1]
        Output: B×2 logits (cover / stego)
        """
        def __init__(self, base_ch: int = 16):
            super().__init__()
            self.hpf = HighPassFilter()
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, base_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(base_ch),
                nn.ReLU(inplace=True),
                ResidualBlock(base_ch, base_ch),
            )
            self.layer2 = ResidualBlock(base_ch, base_ch * 2, stride=2)
            self.layer3 = ResidualBlock(base_ch * 2, base_ch * 4, stride=2)
            self.layer4 = ResidualBlock(base_ch * 4, base_ch * 8, stride=2)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(base_ch * 8, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(64, 2),
            )
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d) and m is not self.hpf.conv:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        def forward(self, x):
            if x.shape[1] > 1:
                x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
            r = self.hpf(x)
            out = self.layer1(r)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.pool(out)
            return self.fc(out)


_STEGA_MODEL: Optional["SteganalyzerNet"] = None


def get_steganalyzer_model(
    weights_path: Optional[str | Path] = None,
    force_reload: bool = False,
) -> Optional["SteganalyzerNet"]:
    global _STEGA_MODEL
    if not HAS_TORCH:
        return None
    if _STEGA_MODEL is not None and not force_reload:
        return _STEGA_MODEL

    path = Path(weights_path) if weights_path else STEGA_WEIGHTS
    model = SteganalyzerNet(base_ch=16)
    model.eval()

    if path.is_file():
        try:
            state = torch.load(str(path), map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=False)
            print(f"[SteganalyzerNet] Loaded trained weights from {path}")
        except Exception as e:
            print(f"[SteganalyzerNet] Could not load weights ({e}); using random init")
    else:
        print("[SteganalyzerNet] No trained weights found — using randomly initialized network")

    for p in model.parameters():
        p.requires_grad = False
    _STEGA_MODEL = model
    return model


def train_steganalyzer_model(
    num_samples: int = 128,
    epochs: int = 12,
    img_size: int = 96,
    lr: float = 1e-3,
    save_path: Optional[str | Path] = None,
    verbose: bool = True,
) -> Path:
    """
    Train SteganalyzerNet on synthetic Cover / Stego pairs.
    Stego pairs are created by ±1 LSB flips on a random subset of pixels
    (simulates the distortion profile of EMD/OPAP).
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for training")

    import cv2

    save_path = Path(save_path) if save_path else STEGA_WEIGHTS
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model = SteganalyzerNet(base_ch=16)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    def make_cover_stego(bs: int = 8):
        covers = []
        stegos = []
        for _ in range(bs):
            # textured cover
            base = np.random.randn(img_size, img_size).astype(np.float32) * 20 + 128
            base = cv2.GaussianBlur(base, (5, 5), 1.0)
            noise = np.random.randn(img_size, img_size).astype(np.float32) * 8
            cover = np.clip(base + noise, 0, 255).astype(np.uint8)

            # create stego by sparse ±1 modifications (EMD/OPAP-like)
            stego = cover.copy()
            n_mod = int(img_size * img_size * 0.15)  # ~15% payload density
            ys = np.random.randint(0, img_size, n_mod)
            xs = np.random.randint(0, img_size, n_mod)
            deltas = np.random.choice([-1, 1], n_mod)
            for y, x, d in zip(ys, xs, deltas):
                v = int(stego[y, x]) + int(d)
                stego[y, x] = np.clip(v, 0, 255)

            covers.append(cover.astype(np.float32) / 255.0)
            stegos.append(stego.astype(np.float32) / 255.0)

        x_c = torch.from_numpy(np.stack(covers)[:, None, ...])
        x_s = torch.from_numpy(np.stack(stegos)[:, None, ...])
        y_c = torch.zeros(bs, dtype=torch.long)
        y_s = torch.ones(bs, dtype=torch.long)
        x = torch.cat([x_c, x_s], dim=0)
        y = torch.cat([y_c, y_s], dim=0)
        # shuffle
        perm = torch.randperm(x.shape[0])
        return x[perm], y[perm]

    if verbose:
        print(f"[SteganalyzerNet] Training on {num_samples} pairs, {epochs} epochs …")

    steps = max(1, num_samples // 8)
    for ep in range(epochs):
        ep_loss = 0.0
        correct = 0
        total = 0
        for _ in range(steps):
            x, y = make_cover_stego(8)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
        if verbose:
            acc = 100.0 * correct / max(total, 1)
            print(f"  epoch {ep+1}/{epochs}  loss={ep_loss/steps:.4f}  acc={acc:.1f}%")

    model.eval()
    torch.save(model.state_dict(), str(save_path))
    if verbose:
        print(f"[SteganalyzerNet] Weights saved → {save_path}")
    return save_path
