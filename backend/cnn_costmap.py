"""
CNN Cost Map Generator for SecureStegVault
Phase 2: CNN-driven image profiling and embedding cost map generation.

Supports three modes:
  - "fast"     : classical multi-scale residual + edge fusion (no Torch needed)
  - "cnn"      : trained CostMapCNN (or VGG16 fallback) + edge fusion
  - "advanced" : trained CostMapCNN + HILL residual + adversarial sensitivity

Formula (cnn / advanced):
  final_map = gamma * rho_cnn + (1 - gamma) * H_edge
  where rho_cnn comes from CostMapCNN (preferred) or VGG16 block2_conv2.
"""

from __future__ import annotations

import numpy as np
import cv2
from PIL import Image
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as T
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    models = None
    T = None

# Global caches
_VGG_MODEL = None
_COSTMAP_CNN = None


def get_vgg_feature_extractor():
    global _VGG_MODEL
    if not HAS_TORCH:
        return None
    if _VGG_MODEL is None:
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            _VGG_MODEL = vgg.features[:7]  # up to block2_conv2
            _VGG_MODEL.eval()
            for p in _VGG_MODEL.parameters():
                p.requires_grad = False
        except Exception as e:
            print("Warning: Could not load VGG16:", e)
            _VGG_MODEL = None
    return _VGG_MODEL


def _load_costmap_cnn():
    global _COSTMAP_CNN
    if _COSTMAP_CNN is not None:
        return _COSTMAP_CNN
    try:
        from backend.models.costmap_net import get_costmap_model
        _COSTMAP_CNN = get_costmap_model()
    except Exception as e:
        print("CostMapCNN load error:", e)
        _COSTMAP_CNN = None
    return _COSTMAP_CNN


def _classical_rho(image_np: np.ndarray) -> np.ndarray:
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    lmin, lmax = lap.min(), lap.max()
    return (lap - lmin) / (lmax - lmin + 1e-8)


def _edge_fusion(gray: np.ndarray, alpha: float = 0.5, beta: float = 0.5) -> np.ndarray:
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.hypot(sobelx, sobely)
    smin, smax = sobel_mag.min(), sobel_mag.max()
    sobel_norm = (sobel_mag - smin) / (smax - smin + 1e-8)
    canny = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
    H = alpha * canny + beta * sobel_norm
    hmin, hmax = H.min(), H.max()
    return (H - hmin) / (hmax - hmin + 1e-8)


def _cnn_rho_from_trained(image_np: np.ndarray) -> Optional[np.ndarray]:
    """Run trained CostMapCNN."""
    model = _load_costmap_cnn()
    if model is None or not HAS_TORCH:
        return None
    try:
        H, W = image_np.shape[:2]
        MAX_DIM = 384
        if max(H, W) > MAX_DIM:
            scale = MAX_DIM / float(max(H, W))
            nH, nW = int(H * scale), int(W * scale)
            proc = cv2.resize(image_np, (nW, nH), interpolation=cv2.INTER_AREA)
        else:
            proc = image_np

        if len(proc.shape) == 3:
            gray = cv2.cvtColor(proc, cv2.COLOR_RGB2GRAY)
        else:
            gray = proc
        tensor = torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            out = model(tensor)
            rho = out.squeeze().cpu().numpy()

        if max(H, W) > MAX_DIM:
            rho = cv2.resize(rho, (W, H), interpolation=cv2.INTER_LINEAR)
        rmin, rmax = rho.min(), rho.max()
        if rmax > rmin:
            rho = (rho - rmin) / (rmax - rmin)
        return rho.astype(np.float32)
    except Exception as e:
        print("Trained CostMapCNN inference error:", e)
        return None


def _cnn_rho_from_vgg(image_np: np.ndarray) -> Optional[np.ndarray]:
    """Fallback: VGG16 early features."""
    if not HAS_TORCH:
        return None
    try:
        H, W = image_np.shape[:2]
        MAX_DIM = 384
        if max(H, W) > MAX_DIM:
            scale = MAX_DIM / float(max(H, W))
            nH, nW = int(H * scale), int(W * scale)
            proc = cv2.resize(image_np, (nW, nH), interpolation=cv2.INTER_AREA)
        else:
            proc = image_np

        pil = Image.fromarray(proc)
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        inp = transform(pil).unsqueeze(0)
        model = get_vgg_feature_extractor()
        if model is None:
            return None
        with torch.no_grad():
            feats = model(inp)
            raw = torch.mean(feats, dim=1, keepdim=True)
            up = nn.functional.interpolate(raw, size=(H, W), mode="bilinear", align_corners=False)
            rho = up.squeeze().cpu().numpy()
        rmin, rmax = rho.min(), rho.max()
        if rmax > rmin:
            rho = (rho - rmin) / (rmax - rmin)
        return rho.astype(np.float32)
    except Exception as e:
        print("VGG costmap error:", e)
        return None


def compute_cnn_costmap(
    image_np: np.ndarray,
    gamma: float = 0.7,
    alpha: float = 0.5,
    beta: float = 0.5,
    cost_map_mode: str = "cnn",
) -> np.ndarray:
    """
    Dense per-pixel embedding cost map (H, W) in [0, 1].
    Higher values → safer / more textured regions.

    cost_map_mode:
      "fast"     – classical Laplacian + edge fusion
      "cnn"      – trained CostMapCNN (preferred) or VGG16 + edge fusion
      "advanced" – cnn + HILL residual + steganalyzer sensitivity penalty
    """
    H_orig, W_orig = image_np.shape[:2]
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np

    # 1. Obtain rho (CNN or classical)
    rho = None
    if cost_map_mode in ("cnn", "advanced"):
        rho = _cnn_rho_from_trained(image_np)
        if rho is None:
            rho = _cnn_rho_from_vgg(image_np)
    if rho is None:
        rho = _classical_rho(image_np)

    # 2. Edge fusion signal
    H_edge = _edge_fusion(gray, alpha=alpha, beta=beta)

    # 3. Advanced post-processing
    if cost_map_mode == "advanced":
        try:
            hp_kernel = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32) / 12.0
            residual = np.abs(cv2.filter2D(gray.astype(np.float32), -1, hp_kernel))
            l1 = cv2.GaussianBlur(residual, (3, 3), 0.5)
            l2 = cv2.GaussianBlur(l1, (15, 15), 2.0)
            lmin, lmax = l2.min(), l2.max()
            hill = (l2 - lmin) / (lmax - lmin + 1e-8)

            from backend.adversarial import compute_adversarial_gradient_map
            grad_map = compute_adversarial_gradient_map(image_np)
            if grad_map.ndim == 3:
                sens = np.mean(np.abs(grad_map), axis=2)
            else:
                sens = np.abs(grad_map)
            smin, smax = sens.min(), sens.max()
            sens_n = (sens - smin) / (smax - smin + 1e-8)

            # Reduce safety where detector is most sensitive
            hill_adv = hill * (1.0 - 0.4 * sens_n)
            hamin, hamax = hill_adv.min(), hill_adv.max()
            hill_adv = (hill_adv - hamin) / (hamax - hamin + 1e-8)
            H_edge = 0.5 * H_edge + 0.5 * hill_adv
        except Exception as e:
            print("Advanced cost-map post-process fallback:", e)

    final_map = gamma * rho + (1.0 - gamma) * H_edge
    final_map = np.clip(final_map, 0.0, 1.0)

    # Quantize to 0.05 buckets so ±1 modifications cannot reorder zones/ranking
    final_map = np.round(final_map * 20.0) / 20.0
    return final_map.astype(np.float32)
