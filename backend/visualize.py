"""
Visualization Module for SecureStegVault
Generates diff heatmap overlay, binary modification mask, and zone maps for steganography inspection.

Visual Embedding Map:
1. Difference: D(x, y) = |stego(x, y) - cover(x, y)|
2. Heatmap: Red/Magenta highlight overlay on cover image for modified pixels.
3. Binary Mask: Pure black/white image (white = modified pixel).
4. Zone Map: Color-coded overlay (Zone A = light purple, Zone B = pink, Zone C = vivid magenta).
"""

import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from typing import Dict, Tuple


def image_to_base64_png(img_np: np.ndarray) -> str:
    """Converts RGB uint8 numpy array to base64 PNG data URL string."""
    pil_img = Image.fromarray(img_np.astype(np.uint8))
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_b64}"


def generate_gradient_overlay(cover_np: np.ndarray, gradient_map: np.ndarray) -> str:
    """
    Generates a glowing cyan/blue heatmap overlay on cover_np sourced from gradient_map magnitude.
    """
    if len(gradient_map.shape) == 3:
        grad_mag = np.mean(np.abs(gradient_map), axis=2)
    else:
        grad_mag = np.abs(gradient_map)

    g_min, g_max = np.min(grad_mag), np.max(grad_mag)
    norm_grad = (grad_mag - g_min) / (g_max - g_min + 1e-8)

    overlay = cover_np.copy().astype(np.float32)
    cyan_highlight = np.array([6, 182, 212], dtype=np.float32)  # Cyan #06B6D4

    # Apply cyan highlight proportional to gradient magnitude
    alpha = np.clip(norm_grad * 0.75, 0.0, 0.85)
    for c in range(3 if len(cover_np.shape) == 3 else 1):
        overlay[:, :, c] = (1.0 - alpha) * overlay[:, :, c] + alpha * cyan_highlight[c]

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return image_to_base64_png(overlay)


def generate_visualizations(
    cover_np: np.ndarray,
    stego_np: np.ndarray,
    cost_map: np.ndarray,
    zones_map: np.ndarray,
    gradient_map: np.ndarray = None,
) -> Dict[str, str]:
    """
    Generates base64 PNG data URLs for:
    - stego_image
    - diff_heatmap
    - binary_mask
    - zone_map
    - gradient_overlay
    - highlight_overlay (bright gold pixels that received data bits)
    - rgb_bits (RGB mode: pure R/G/B show which channel holds the bits)
    """
    # 1. Absolute difference (fully vectorized)
    diff = np.abs(cover_np.astype(np.int16) - stego_np.astype(np.int16))
    if len(cover_np.shape) == 3:
        diff_mag = np.max(diff, axis=2).astype(np.float32)
        modified_mask = np.any(diff > 0, axis=2)
        dR = diff[:, :, 0]
        dG = diff[:, :, 1]
        dB = diff[:, :, 2]
    else:
        diff_mag = diff.astype(np.float32)
        modified_mask = diff > 0
        dR = dG = dB = diff

    # 2. Binary Mask (white = modified)
    binary_mask = np.zeros((cover_np.shape[0], cover_np.shape[1], 3), dtype=np.uint8)
    binary_mask[modified_mask] = [255, 255, 255]

    # 3. Diff Heatmap Overlay (vectorized)
    heatmap = cover_np.copy().astype(np.float32)
    highlight_color = np.array([236, 72, 153], dtype=np.float32)  # Pink/Magenta
    alpha = np.clip(0.4 + 0.2 * diff_mag, 0.0, 0.8)
    alpha3 = alpha[:, :, np.newaxis]
    heatmap = np.where(
        modified_mask[:, :, np.newaxis],
        (1.0 - alpha3) * heatmap + alpha3 * highlight_color,
        heatmap,
    )
    heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)

    # 4. Zone Map visualization
    zone_visual = np.zeros((cover_np.shape[0], cover_np.shape[1], 3), dtype=np.uint8)
    z_2d = zones_map[:, :, 0] if len(zones_map.shape) == 3 else zones_map
    zone_visual[z_2d == 0] = [216, 180, 254]  # Light Lilac
    zone_visual[z_2d == 1] = [249, 168, 212]  # Pink
    zone_visual[z_2d == 2] = [232, 121, 249]  # Vivid Fuchsia

    if len(cover_np.shape) == 3:
        gray_bg = cv2.cvtColor(cover_np, cv2.COLOR_RGB2GRAY)
        gray_3d = cv2.cvtColor(gray_bg, cv2.COLOR_GRAY2RGB)
    else:
        gray_3d = cv2.cvtColor(cover_np, cv2.COLOR_GRAY2RGB)

    zone_map_composite = (0.55 * zone_visual.astype(np.float32) + 0.45 * gray_3d.astype(np.float32)).astype(np.uint8)

    # 5. Adversarial Gradient Overlay
    if gradient_map is None:
        try:
            from backend.adversarial import compute_adversarial_gradient_map
            gradient_map = compute_adversarial_gradient_map(cover_np)
        except Exception:
            gradient_map = np.zeros_like(cover_np, dtype=np.float32)

    grad_overlay_b64 = generate_gradient_overlay(cover_np, gradient_map)

    # 6. Bright gold highlight overlay
    highlight = stego_np.copy().astype(np.float32) * 0.35
    gold = np.array([255, 230, 40], dtype=np.float32)
    highlight = np.where(modified_mask[:, :, np.newaxis], gold, highlight)
    highlight = np.clip(highlight, 0, 255).astype(np.uint8)

    # 7. RGB Bits Mode – pure channel colours show exactly where data bits live
    rgb_bits = np.full((cover_np.shape[0], cover_np.shape[1], 3), [8, 8, 12], dtype=np.uint8)
    if len(cover_np.shape) == 3:
        rgb_bits[dR > 0, 0] = 255  # bright red
        rgb_bits[dG > 0, 1] = 255  # bright green
        rgb_bits[dB > 0, 2] = 255  # bright blue
    else:
        rgb_bits[modified_mask] = [255, 255, 255]

    return {
        "stego_b64": image_to_base64_png(stego_np),
        "heatmap_b64": image_to_base64_png(heatmap),
        "mask_b64": image_to_base64_png(binary_mask),
        "zone_map_b64": image_to_base64_png(zone_map_composite),
        "gradient_overlay_b64": grad_overlay_b64,
        "highlight_overlay_b64": image_to_base64_png(highlight),
        "rgb_bits_b64": image_to_base64_png(rgb_bits),
    }

