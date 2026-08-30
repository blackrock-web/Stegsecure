"""
Adversarial Gradient Post-Processing & Steganalyzer Module
Phase 4/5: ADV-EMB guidance + security evaluation

Uses the trained SteganalyzerNet when available; falls back to the original
lightweight randomly-initialized SRNet-style network or classical residual variance.
"""

from __future__ import annotations

from typing import Optional
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

    class FallbackSteganalyzer(nn.Module):
        def __init__(self):
            super().__init__()
            self.hpf = HighPassFilter()
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.AvgPool2d(2),
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
            torch.manual_seed(42)
            for m in self.modules():
                if isinstance(m, nn.Conv2d) and m is not self.hpf.conv:
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)

        def forward(self, x):
            r = self.hpf(x)
            out = self.layer1(r)
            out = self.layer2(out)
            out = self.layer3(out)
            return self.fc(torch.flatten(out, 1))


_MODEL_CACHE = None


def _get_model():
    """Prefer trained SteganalyzerNet; fall back to lightweight network."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    if not HAS_TORCH:
        return None
    try:
        from backend.models.steganalyzer_net import get_steganalyzer_model
        m = get_steganalyzer_model()
        if m is not None:
            _MODEL_CACHE = m
            return m
    except Exception as e:
        print("Trained steganalyzer load error:", e)
    m = FallbackSteganalyzer()
    m.eval()
    _MODEL_CACHE = m
    return m


def evaluate_stego_confidence(image_np: np.ndarray) -> float:
    """Surrogate stego probability in [0, 1]. Higher = more suspicious."""
    if image_np.ndim == 3:
        gray = 0.299 * image_np[:, :, 0] + 0.587 * image_np[:, :, 1] + 0.114 * image_np[:, :, 2]
    else:
        gray = image_np.astype(np.float32)
    gray_norm = gray.astype(np.float32) / 255.0

    if HAS_TORCH:
        try:
            H, W = gray_norm.shape
            MAX_DIM = 256
            if max(H, W) > MAX_DIM:
                import cv2
                scale = MAX_DIM / float(max(H, W))
                nH, nW = int(H * scale), int(W * scale)
                gray_norm = cv2.resize(gray_norm, (nW, nH), interpolation=cv2.INTER_AREA)

            tensor = torch.tensor(gray_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            model = _get_model()
            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1)
                return float(probs[0, 1].item())
        except Exception as e:
            print("Adversarial eval fallback:", e)

    import cv2
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
    res = cv2.filter2D(gray_norm, -1, kv)
    var = float(np.var(res))
    prob = float(1.0 / (1.0 + np.exp(-100.0 * (var - 0.005))))
    return max(0.01, min(0.99, prob))


def compute_adversarial_gradient_map(image_np: np.ndarray) -> np.ndarray:
    """
    dL_stego / dImage. Negative gradient direction reduces stego confidence.
    Returns array shaped like the input image.
    """
    H, W = image_np.shape[:2]
    C = image_np.shape[2] if image_np.ndim == 3 else 1

    if C == 3:
        gray = 0.299 * image_np[:, :, 0] + 0.587 * image_np[:, :, 1] + 0.114 * image_np[:, :, 2]
    else:
        gray = image_np.astype(np.float32)
    gray_norm = gray.astype(np.float32) / 255.0

    if HAS_TORCH:
        try:
            MAX_DIM = 256
            if max(H, W) > MAX_DIM:
                import cv2
                scale = MAX_DIM / float(max(H, W))
                nH, nW = int(H * scale), int(W * scale)
                proc = cv2.resize(gray_norm, (nW, nH), interpolation=cv2.INTER_AREA)
            else:
                proc = gray_norm

            tensor = torch.tensor(proc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            tensor.requires_grad_(True)
            model = _get_model()
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)

            logits = model(tensor)
            log_probs = F.log_softmax(logits, dim=1)
            stego_loss = log_probs[0, 1]
            stego_loss.backward()

            grad_2d = tensor.grad.squeeze().cpu().numpy()
            if max(H, W) > MAX_DIM:
                import cv2
                grad_2d = cv2.resize(grad_2d, (W, H), interpolation=cv2.INTER_LINEAR)

            if C == 3:
                return np.repeat(grad_2d[:, :, np.newaxis], 3, axis=2)
            return grad_2d
        except Exception as e:
            print("Adversarial grad map fallback:", e)

    import cv2
    gx = cv2.Sobel(gray_norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_norm, cv2.CV_32F, 0, 1, ksize=3)
    g = (gx + gy) * 0.1
    if C == 3:
        return np.repeat(g[:, :, np.newaxis], 3, axis=2)
    return g


compute_adversarial_gradient = compute_adversarial_gradient_map


def adversarial_sign_map(gradient: np.ndarray) -> np.ndarray:
    """-sign(∇) — direction that most reduces stego confidence."""
    return -np.sign(gradient).astype(np.int8)
