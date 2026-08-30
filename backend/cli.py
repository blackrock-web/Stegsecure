#!/usr/bin/env python3
"""
CLI bridge for SecureStegVault Python CNN backend.
Called by the Node server when cost_map_mode == 'advanced' so that the real
VGG16 cost map + SRNet-style adversarial gradient are used.

Usage:
  python -m backend.cli capacity  --image path [--params...]
  python -m backend.cli encode    --image path --text "..." --passphrase "..." [--params...]
  python -m backend.cli decode    --image path --passphrase "..." [--params...]

All results are printed as a single JSON object on stdout.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

# Ensure project root is on path when invoked as script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.crypto import encrypt_payload, decrypt_payload
from backend.cnn_costmap import compute_cnn_costmap
from backend.zoning import ZoningConfig, classify_zones, calculate_capacity
from backend.emd import (
    bytes_to_base5_digits,
    base5_digits_to_bytes,
    bytes_to_base7_digits,
    base7_digits_to_bytes,
    embed_emd_zone_a,
    extract_emd_zone_a,
)
from backend.opap import embed_opap_zone, extract_opap_zone
from backend.metrics import calculate_metrics, calculate_security_report
from backend.visualize import generate_visualizations
from backend.cost_optimizer import rank_zone_indices
from backend.adversarial import compute_adversarial_gradient_map, adversarial_sign_map


def load_image(path: str) -> np.ndarray:
    pil = Image.open(path)
    fmt = (pil.format or "").upper()
    if fmt not in ("PNG", "BMP"):
        raise ValueError(f"Unsupported format '{fmt}'. Cover must be lossless PNG or BMP.")
    return np.array(pil.convert("RGB"), dtype=np.uint8)


def cmd_capacity(args: argparse.Namespace) -> dict:
    img = load_image(args.image)
    cost_map = compute_cnn_costmap(img, gamma=args.gamma, cost_map_mode=args.cost_map_mode)
    config = ZoningConfig(
        thresh_a=args.thresh_a,
        thresh_b=args.thresh_b,
        emd_group_size=args.emd_n,
    )
    cap = calculate_capacity(img.shape, cost_map, config)
    return {
        "width": img.shape[1],
        "height": img.shape[0],
        "channels": img.shape[2],
        "capacity": cap,
        "cost_map_mode": f"Real VGG16 CNN + Edge Fusion ({args.cost_map_mode} mode)",
        "emd_n": args.emd_n,
        "engine": "python-cnn",
    }


def cmd_encode(args: argparse.Namespace) -> dict:
    if not args.text or not args.text.strip():
        raise ValueError("Secret message cannot be empty.")
    if not args.passphrase:
        raise ValueError("Passphrase is required.")

    cover = load_image(args.image)
    config = ZoningConfig(
        thresh_a=args.thresh_a,
        thresh_b=args.thresh_b,
        emd_group_size=args.emd_n,
        kb_bits=args.kb_bits,
        kc_bits=args.kc_bits,
    )

    # Phase 1 – AES-256-GCM
    encrypted = encrypt_payload(args.text, args.passphrase)
    payload_len = len(encrypted)

    # Phase 2 – Real VGG16 CNN cost map (visuals / capacity only; zones are spatial)
    cost_map = compute_cnn_costmap(cover, gamma=args.gamma, cost_map_mode=args.cost_map_mode)

    H, W, C = cover.shape
    # Spatial-strip zones (invariant under any local embedding noise).
    # CNN cost_map is still computed and returned for visualizations / paper metrics.
    zones_2d = np.zeros((H, W), dtype=np.uint8)
    t_a = int(H * args.thresh_a)
    t_b = int(H * args.thresh_b)
    zones_2d[:t_a, :] = 0
    zones_2d[t_a:t_b, :] = 1
    zones_2d[t_b:, :] = 2
    zones_3d = np.repeat(zones_2d[:, :, np.newaxis], C, axis=2)
    # Dummy cost map (unused for spatial ranking)
    cost_map_3d = np.zeros((H, W, C), dtype=np.float32)
    rank_cost_3d = cost_map_3d  # ranking is pure spatial via cost_optimizer

    cap_info = calculate_capacity(cover.shape, cost_map, config)
    if payload_len > cap_info["max_bytes"]:
        max_chars = max(0, cap_info["max_bytes"] - 48)
        raise ValueError(
            f"Message too long. Capacity {cap_info['max_bytes']} bytes "
            f"(~{max_chars} chars), need {payload_len} bytes."
        )

    # Adversarial gradient (always computed in advanced / when strength > 0)
    grad_map = None
    adv_signs = None
    # Only compute expensive adversarial gradients when explicitly requested
    if args.adversarial_strength > 0.0:
        grad_map = compute_adversarial_gradient_map(cover)
        adv_signs = adversarial_sign_map(grad_map).flatten()

    # Phase 4 – EMD-OPAP hybrid embedding
    stego = cover.copy()
    image_flat = stego.flatten()
    zones_flat = zones_3d.flatten()
    cost_flat = rank_cost_3d.flatten()

    raw_a = np.where(zones_flat == 0)[0]
    raw_b = np.where(zones_flat == 1)[0]
    raw_c = np.where(zones_flat == 2)[0]

    zone_a = rank_zone_indices(cost_flat, raw_a, is_emd=True, group_size=args.emd_n)
    zone_b = rank_zone_indices(cost_flat, raw_b, is_emd=False)
    zone_c = rank_zone_indices(cost_flat, raw_c, is_emd=False)

    payload_bits = []
    for b in encrypted:
        for bit_i in range(7, -1, -1):
            payload_bits.append((b >> bit_i) & 1)

    total_bits = len(payload_bits)
    bits_left = total_bits
    bit_idx = 0
    za_bits = zb_bits = zc_bits = 0

    # Zone A: EMD embeds whole bytes only (4 base-5 / 3 base-7 digits per byte)
    # so encode ↔ decode stay byte-aligned and reversible.
    if args.emd_n == 3:
        groups = len(zone_a) // 3
        max_bytes_a = groups // 3  # 3 digits per byte
        if max_bytes_a > 0 and bits_left > 0:
            n_bytes = min((bits_left + 7) // 8, max_bytes_a, len(encrypted))
            digits = bytes_to_base7_digits(encrypted[:n_bytes])
            digits = digits[: (len(digits) // 3) * 3]
            n_bytes = len(digits) // 3
            _, digs = embed_emd_zone_a(
                image_flat, zone_a, digits, emd_n=3,
                adversarial_signs=adv_signs, adversarial_strength=args.adversarial_strength,
            )
            n_bytes = digs // 3
            za_bits = n_bytes * 8
            bit_idx = n_bytes * 8
            bits_left = max(0, total_bits - bit_idx)
    else:
        groups = len(zone_a) // 2
        max_bytes_a = groups // 4  # 4 digits per byte
        if max_bytes_a > 0 and bits_left > 0:
            n_bytes = min((bits_left + 7) // 8, max_bytes_a, len(encrypted))
            digits = bytes_to_base5_digits(encrypted[:n_bytes])
            digits = digits[: (len(digits) // 4) * 4]
            n_bytes = len(digits) // 4
            _, digs = embed_emd_zone_a(
                image_flat, zone_a, digits, emd_n=2,
                adversarial_signs=adv_signs, adversarial_strength=args.adversarial_strength,
            )
            n_bytes = digs // 4
            za_bits = n_bytes * 8
            bit_idx = n_bytes * 8
            bits_left = max(0, total_bits - bit_idx)

    if bits_left > 0 and len(zone_b) > 0:
        stream = payload_bits[bit_idx : bit_idx + bits_left]
        _, emb = embed_opap_zone(
            image_flat, zone_b, stream, k=config.kb_bits,
            adversarial_signs=adv_signs, adversarial_strength=args.adversarial_strength,
        )
        zb_bits = emb
        bit_idx += emb
        bits_left = max(0, total_bits - bit_idx)

    if bits_left > 0 and len(zone_c) > 0:
        stream = payload_bits[bit_idx : bit_idx + bits_left]
        _, emb = embed_opap_zone(
            image_flat, zone_c, stream, k=config.kc_bits,
            adversarial_signs=adv_signs, adversarial_strength=args.adversarial_strength,
        )
        zc_bits = emb
        bit_idx += emb
        bits_left = max(0, total_bits - bit_idx)

    if bits_left > 0:
        raise ValueError("Could not fit all bits into available image zones.")

    stego = image_flat.reshape(cover.shape)

    zone_breakdown = {
        "zone_a_bits": za_bits,
        "zone_b_bits": zb_bits,
        "zone_c_bits": zc_bits,
    }
    metrics = calculate_metrics(cover, stego, total_bits, zone_breakdown)
    security = calculate_security_report(cover, stego)
    visuals = generate_visualizations(cover, stego, cost_map, zones_3d, gradient_map=grad_map)

    return {
        "success": True,
        "metrics": metrics,
        "security_report": security,
        "visuals": visuals,
        "cost_map_mode": args.cost_map_mode,
        "adversarial_strength": args.adversarial_strength,
        "emd_n": args.emd_n,
        "engine": "python-cnn-vgg16",
    }


def cmd_decode(args: argparse.Namespace) -> dict:
    if not args.passphrase:
        raise ValueError("Passphrase is required.")

    stego = load_image(args.image)
    config = ZoningConfig(
        thresh_a=args.thresh_a,
        thresh_b=args.thresh_b,
        emd_group_size=args.emd_n,
        kb_bits=args.kb_bits,
        kc_bits=args.kc_bits,
    )

    # Zones are spatial-strip — skip expensive CNN cost map on decode
    cost_map = None
    H, W, C = stego.shape
    zones_2d = np.zeros((H, W), dtype=np.uint8)
    t_a = int(H * args.thresh_a)
    t_b = int(H * args.thresh_b)
    zones_2d[:t_a, :] = 0
    zones_2d[t_a:t_b, :] = 1
    zones_2d[t_b:, :] = 2
    zones_3d = np.repeat(zones_2d[:, :, np.newaxis], C, axis=2)
    cost_map_3d = np.zeros((H, W, C), dtype=np.float32)
    rank_cost_3d = cost_map_3d

    image_flat = stego.flatten()
    zones_flat = zones_3d.flatten()
    cost_flat = rank_cost_3d.flatten()

    raw_a = np.where(zones_flat == 0)[0]
    raw_b = np.where(zones_flat == 1)[0]
    raw_c = np.where(zones_flat == 2)[0]

    zone_a = rank_zone_indices(cost_flat, raw_a, is_emd=True, group_size=args.emd_n)
    zone_b = rank_zone_indices(cost_flat, raw_b, is_emd=False)
    zone_c = rank_zone_indices(cost_flat, raw_c, is_emd=False)

    extracted = bytearray()

    groups = len(zone_a) // args.emd_n
    if groups > 0:
        digits = extract_emd_zone_a(image_flat, zone_a, groups, emd_n=args.emd_n)
        a_bytes = (
            base7_digits_to_bytes(digits) if args.emd_n == 3 else base5_digits_to_bytes(digits)
        )
        extracted.extend(a_bytes)

    if len(zone_b) > 0:
        bits = extract_opap_zone(
            image_flat, zone_b, len(zone_b) * config.kb_bits, k=config.kb_bits
        )
        for i in range(len(bits) // 8):
            val = 0
            for j in range(8):
                val = (val << 1) | bits[i * 8 + j]
            extracted.append(val)

    if len(zone_c) > 0:
        bits = extract_opap_zone(
            image_flat, zone_c, len(zone_c) * config.kc_bits, k=config.kc_bits
        )
        for i in range(len(bits) // 8):
            val = 0
            for j in range(8):
                val = (val << 1) | bits[i * 8 + j]
            extracted.append(val)

    plaintext = decrypt_payload(bytes(extracted), args.passphrase)
    return {
        "success": True,
        "decrypted_text": plaintext,
        "engine": "python-cnn-vgg16",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SecureStegVault CNN CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--image", required=True)
    common.add_argument("--thresh_a", type=float, default=0.35)
    common.add_argument("--thresh_b", type=float, default=0.65)
    common.add_argument("--gamma", type=float, default=0.7)
    common.add_argument("--kb_bits", type=int, default=2)
    common.add_argument("--kc_bits", type=int, default=3)
    common.add_argument("--cost_map_mode", default="advanced")
    common.add_argument("--emd_n", type=int, default=2)
    common.add_argument("--adversarial_strength", type=float, default=0.0)

    p_cap = sub.add_parser("capacity", parents=[common])
    p_enc = sub.add_parser("encode", parents=[common])
    p_enc.add_argument("--text", required=True)
    p_enc.add_argument("--passphrase", required=True)
    p_dec = sub.add_parser("decode", parents=[common])
    p_dec.add_argument("--passphrase", required=True)

    args = parser.parse_args()

    try:
        if args.command == "capacity":
            result = cmd_capacity(args)
        elif args.command == "encode":
            result = cmd_encode(args)
        elif args.command == "decode":
            result = cmd_decode(args)
        else:
            raise ValueError(f"Unknown command {args.command}")
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"success": False, "detail": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
