"""CNN + EMD + OPAP (+ optional adversarial / experimental STC) strategies."""

from __future__ import annotations
from typing import Optional, Tuple
import time
import numpy as np

from .base import EmbeddingStrategy, StrategyResult
from .registry import register


def _run_pipeline(
    cover: np.ndarray,
    message: str,
    passphrase: str,
    *,
    cost_map_mode: str = "cnn",
    emd_n: int = 2,
    adversarial_strength: float = 0.0,
    use_stc_approx: bool = False,
    kb_bits: int = 2,
    kc_bits: int = 3,
    thresh_a: float = 0.35,
    thresh_b: float = 0.65,
    gamma: float = 0.7,
):
    from backend.crypto import encrypt_payload
    from backend.cnn_costmap import compute_cnn_costmap
    from backend.zoning import ZoningConfig, classify_zones
    from backend.emd import (
        bytes_to_base5_digits,
        bytes_to_base7_digits,
        embed_emd_zone_a,
    )
    from backend.opap import embed_opap_zone
    from backend.cost_optimizer import rank_zone_indices
    from backend.metrics import calculate_metrics
    from backend.security.evaluator import evaluate_security
    from backend.adversarial import compute_adversarial_gradient_map, adversarial_sign_map
    import math

    t0 = time.perf_counter()
    config = ZoningConfig(
        thresh_a=thresh_a, thresh_b=thresh_b,
        emd_group_size=emd_n, kb_bits=kb_bits, kc_bits=kc_bits,
    )
    encrypted = encrypt_payload(message, passphrase)
    cost_map = compute_cnn_costmap(cover, gamma=gamma, cost_map_mode=cost_map_mode)
    H, W, C = cover.shape
    cost_3d = np.repeat(cost_map[:, :, np.newaxis], C, axis=2)
    zones = classify_zones(cost_3d, config)

    grad = None
    signs = None
    if adversarial_strength > 0:
        try:
            grad = compute_adversarial_gradient_map(cover)
            signs = adversarial_sign_map(grad).flatten()
        except Exception:
            pass

    stego = cover.copy()
    flat = stego.flatten()
    zf = zones.flatten()
    cf = cost_3d.flatten()
    za = rank_zone_indices(cf, np.where(zf == 0)[0], is_emd=True, group_size=emd_n)
    zb = rank_zone_indices(cf, np.where(zf == 1)[0], is_emd=False)
    zc = rank_zone_indices(cf, np.where(zf == 2)[0], is_emd=False)

    bits = [(b >> i) & 1 for b in encrypted for i in range(7, -1, -1)]
    total = len(bits)
    rem = total
    idx = 0
    za_bits = zb_bits = zc_bits = 0

    # Optional experimental STC on a portion of medium zone
    if use_stc_approx and len(zb) > 0 and rem > 0:
        from backend.stc import stc_embed_bits
        n_stc = min(rem, len(zb) // 2)
        flat, n_emb = stc_embed_bits(
            flat, zb[: len(zb) // 2], bits[idx : idx + n_stc], cf,
            adversarial_signs=signs, adversarial_weight=adversarial_strength,
        )
        idx += n_emb
        rem = max(0, total - idx)
        zb_bits += n_emb
        zb = zb[len(zb) // 2 :]  # remaining for OPAP

    if emd_n == 2:
        groups = len(za) // 2
        max_b = int(groups * math.log2(5))
        if max_b > 0 and rem > 0:
            nbits = min(rem, max_b)
            nbytes = (nbits + 7) // 8
            digs = bytes_to_base5_digits(encrypted[:nbytes])
            _, used = embed_emd_zone_a(
                flat, za, digs, emd_n=2,
                adversarial_signs=signs, adversarial_strength=adversarial_strength,
            )
            za_bits = int(used * math.log2(5))
            idx = (used // 4) * 8
            rem = max(0, total - idx)
    else:
        groups = len(za) // 3
        max_b = int(groups * math.log2(7))
        if max_b > 0 and rem > 0:
            nbits = min(rem, max_b)
            nbytes = (nbits + 7) // 8
            digs = bytes_to_base7_digits(encrypted[:nbytes])
            _, used = embed_emd_zone_a(
                flat, za, digs, emd_n=3,
                adversarial_signs=signs, adversarial_strength=adversarial_strength,
            )
            za_bits = int(used * math.log2(7))
            idx = (used // 3) * 8
            rem = max(0, total - idx)

    if rem > 0 and len(zb):
        stream = bits[idx:idx + rem]
        _, n = embed_opap_zone(
            flat, zb, stream, k=kb_bits,
            adversarial_signs=signs, adversarial_strength=adversarial_strength,
        )
        zb_bits += n
        idx += n
        rem = max(0, total - idx)
    if rem > 0 and len(zc):
        stream = bits[idx:idx + rem]
        _, n = embed_opap_zone(
            flat, zc, stream, k=kc_bits,
            adversarial_signs=signs, adversarial_strength=adversarial_strength,
        )
        zc_bits += n
        idx += n
        rem = max(0, total - idx)

    stego = flat.reshape(cover.shape)
    t_embed = time.perf_counter() - t0

    metrics = calculate_metrics(
        cover, stego, idx,
        {"zone_a_bits": za_bits, "zone_b_bits": zb_bits, "zone_c_bits": zc_bits},
    )
    metrics["embed_time_s"] = t_embed
    metrics["bits_remaining"] = rem
    sec = evaluate_security(cover, stego).to_dict()

    return stego, metrics, sec, encrypted, rem


def _run_extract_pipeline(
    stego: np.ndarray,
    passphrase: str,
    *,
    cost_map_mode: str = "cnn",
    emd_n: int = 2,
    kb_bits: int = 2,
    kc_bits: int = 3,
    thresh_a: float = 0.35,
    thresh_b: float = 0.65,
    gamma: float = 0.7,
    use_stc_approx: bool = False,
) -> str:
    """
    Shared extract helper mirroring main.py::decode_steganography and the
    zone ranking used by _run_pipeline.  Extracts full zone capacity then
    decrypts (AES-GCM).  STC-approx path is not inverted here; strategies
    that used STC at embed may return FAILED decryption and must be handled
    by the caller as N/A for BER when that happens.
    """
    from backend.crypto import decrypt_payload
    from backend.cnn_costmap import compute_cnn_costmap
    from backend.zoning import ZoningConfig, classify_zones
    from backend.emd import (
        base5_digits_to_bytes,
        base7_digits_to_bytes,
        extract_emd_zone_a,
    )
    from backend.opap import extract_opap_zone
    from backend.cost_optimizer import rank_zone_indices

    config = ZoningConfig(
        thresh_a=thresh_a, thresh_b=thresh_b,
        emd_group_size=emd_n, kb_bits=kb_bits, kc_bits=kc_bits,
    )
    cost_map = compute_cnn_costmap(stego, gamma=gamma, cost_map_mode=cost_map_mode)
    H, W, C = stego.shape
    cost_3d = np.repeat(cost_map[:, :, np.newaxis], C, axis=2)
    zones = classify_zones(cost_3d, config)

    image_flat = stego.flatten().astype(np.uint8, copy=False)
    zones_flat = zones.flatten()
    cost_flat = cost_3d.flatten()

    za = rank_zone_indices(
        cost_flat, np.where(zones_flat == 0)[0], is_emd=True, group_size=emd_n,
    )
    zb = rank_zone_indices(cost_flat, np.where(zones_flat == 1)[0], is_emd=False)
    zc = rank_zone_indices(cost_flat, np.where(zones_flat == 2)[0], is_emd=False)

    # STC approx (if used at embed) occupies the first half of zone-B indices;
    # without a matching STC decoder we skip that half so remaining OPAP bits
    # stay aligned.  Bits placed via experimental STC are not recovered.
    if use_stc_approx and len(zb) > 0:
        zb = zb[len(zb) // 2 :]

    extracted = bytearray()

    zone_a_groups = len(za) // emd_n
    if zone_a_groups > 0:
        digits = extract_emd_zone_a(image_flat, za, zone_a_groups, emd_n=emd_n)
        a_bytes = (
            base7_digits_to_bytes(digits) if emd_n == 3 else base5_digits_to_bytes(digits)
        )
        extracted.extend(a_bytes)

    if len(zb) > 0:
        b_bits_avail = len(zb) * kb_bits
        b_bits = extract_opap_zone(image_flat, zb, b_bits_avail, k=kb_bits)
        for byte_i in range(len(b_bits) // 8):
            b_val = 0
            for bit_i in range(8):
                b_val = (b_val << 1) | b_bits[byte_i * 8 + bit_i]
            extracted.append(b_val)

    if len(zc) > 0:
        c_bits_avail = len(zc) * kc_bits
        c_bits = extract_opap_zone(image_flat, zc, c_bits_avail, k=kc_bits)
        for byte_i in range(len(c_bits) // 8):
            c_val = 0
            for bit_i in range(8):
                c_val = (c_val << 1) | c_bits[byte_i * 8 + bit_i]
            extracted.append(c_val)

    # Try progressive truncation so trailing capacity garbage does not break
    # AES-GCM (v3 has no ciphertext length field).  Prefer longest success.
    raw = bytes(extracted)
    last_err: Optional[Exception] = None
    # Minimum plausible v3 payload: 16 header + 16 salt + 12 nonce + 16 tag
    min_len = 16 + 16 + 12 + 16
    for end in range(len(raw), min_len - 1, -1):
        try:
            return decrypt_payload(raw[:end], passphrase)
        except Exception as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    raise ValueError("Message could not be decrypted — wrong passphrase or corrupted image.")


@register
class EMD_OPAP(EmbeddingStrategy):
    name = "emd_opap"

    def embed(self, cover, message, passphrase, **kw):
        stego, metrics, sec, _, rem = _run_pipeline(
            cover, message, passphrase, cost_map_mode="fast", adversarial_strength=0.0, **kw
        )
        return StrategyResult(
            stego=stego, metrics=metrics, security=sec,
            meta={"strategy": self.name, "bits_remaining": rem},
        )

    def extract(self, stego, passphrase, **kw):
        return _run_extract_pipeline(
            stego, passphrase, cost_map_mode="fast", use_stc_approx=False, **kw
        )


@register
class CNN_EMD_OPAP(EmbeddingStrategy):
    name = "cnn_emd_opap"

    def embed(self, cover, message, passphrase, **kw):
        stego, metrics, sec, _, rem = _run_pipeline(
            cover, message, passphrase, cost_map_mode="cnn", adversarial_strength=0.0, **kw
        )
        return StrategyResult(
            stego=stego, metrics=metrics, security=sec, meta={"strategy": self.name}
        )

    def extract(self, stego, passphrase, **kw):
        return _run_extract_pipeline(
            stego, passphrase, cost_map_mode="cnn", use_stc_approx=False, **kw
        )


@register
class CNN_EMD_OPAP_ADV(EmbeddingStrategy):
    name = "cnn_emd_opap_adv"

    def embed(self, cover, message, passphrase, **kw):
        stego, metrics, sec, _, rem = _run_pipeline(
            cover, message, passphrase, cost_map_mode="cnn", adversarial_strength=0.7, **kw
        )
        return StrategyResult(
            stego=stego, metrics=metrics, security=sec, meta={"strategy": self.name}
        )

    def extract(self, stego, passphrase, **kw):
        return _run_extract_pipeline(
            stego, passphrase, cost_map_mode="cnn", use_stc_approx=False, **kw
        )


@register
class CNN_STC_EMD_OPAP(EmbeddingStrategy):
    name = "cnn_stc_emd_opap"

    def embed(self, cover, message, passphrase, **kw):
        stego, metrics, sec, _, rem = _run_pipeline(
            cover, message, passphrase, cost_map_mode="cnn",
            adversarial_strength=0.0, use_stc_approx=True, **kw
        )
        return StrategyResult(
            stego=stego, metrics=metrics, security=sec,
            meta={"strategy": self.name, "stc": "experimental_approx"},
        )

    def extract(self, stego, passphrase, **kw):
        return _run_extract_pipeline(
            stego, passphrase, cost_map_mode="cnn", use_stc_approx=True, **kw
        )


@register
class CNN_STC_EMD_OPAP_ADV(EmbeddingStrategy):
    name = "cnn_stc_emd_opap_adv"

    def embed(self, cover, message, passphrase, **kw):
        stego, metrics, sec, _, rem = _run_pipeline(
            cover, message, passphrase, cost_map_mode="advanced",
            adversarial_strength=0.7, use_stc_approx=True, **kw
        )
        return StrategyResult(
            stego=stego, metrics=metrics, security=sec,
            meta={"strategy": self.name, "stc": "experimental_approx"},
        )

    def extract(self, stego, passphrase, **kw):
        return _run_extract_pipeline(
            stego, passphrase, cost_map_mode="advanced", use_stc_approx=True, **kw
        )
