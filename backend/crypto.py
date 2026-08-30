"""
SecureStegVault v3 Cryptography Module
AES-256-GCM with versioned payload + configurable KDF (PBKDF2 / Argon2id when available).

Payload format (v3):
  MAGIC (4) | VERSION (1) | KDF_ID (1) | FLAGS (1) | reserved (1)
  | SALT_LEN (1) | NONCE_LEN (1) |  ...padded header to 16 bytes...
  | SALT | NONCE | CIPHERTEXT || TAG (16)

Legacy v1 (unversioned) is still readable for backward compatibility.
"""

from __future__ import annotations

import struct
import os
from typing import Tuple, Optional

from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag

MAGIC = b"SSV3"
VERSION = 3
KDF_PBKDF2 = 1
KDF_ARGON2ID = 2

PBKDF2_ITERATIONS_DEFAULT = 200_000
KEY_LENGTH = 32
SALT_LENGTH = 16
NONCE_LENGTH = 12
HEADER_V1_LENGTH = 4  # legacy

# Try Argon2
HAS_ARGON2 = False
try:
    from argon2.low_level import hash_secret_raw, Type
    HAS_ARGON2 = True
except ImportError:
    pass


def derive_key_pbkdf2(passphrase: str, salt: bytes, iterations: int = PBKDF2_ITERATIONS_DEFAULT) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=KEY_LENGTH,
        salt=salt,
        iterations=iterations,
    )
    return kdf.derive(passphrase.encode("utf-8"))


def derive_key_argon2id(passphrase: str, salt: bytes, time_cost: int = 3, memory_cost: int = 65536, parallelism: int = 2) -> bytes:
    if not HAS_ARGON2:
        raise RuntimeError("argon2-cffi not installed")
    return hash_secret_raw(
        secret=passphrase.encode("utf-8"),
        salt=salt,
        time_cost=time_cost,
        memory_cost=memory_cost,
        parallelism=parallelism,
        hash_len=KEY_LENGTH,
        type=Type.ID,
    )


def encrypt_payload(
    plaintext: str,
    passphrase: str,
    *,
    kdf: str = "pbkdf2",
    pbkdf2_iterations: int = PBKDF2_ITERATIONS_DEFAULT,
) -> bytes:
    """
    Encrypt with AES-256-GCM. Returns versioned binary payload ready for embedding.
    kdf: "pbkdf2" (default) or "argon2id" (if argon2-cffi available).
    """
    salt = os.urandom(SALT_LENGTH)
    nonce = os.urandom(NONCE_LENGTH)

    if kdf == "argon2id" and HAS_ARGON2:
        key = derive_key_argon2id(passphrase, salt)
        kdf_id = KDF_ARGON2ID
    else:
        key = derive_key_pbkdf2(passphrase, salt, iterations=pbkdf2_iterations)
        kdf_id = KDF_PBKDF2

    aesgcm = AESGCM(key)
    ct_and_tag = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)

    # Header: MAGIC(4) VER(1) KDF(1) FLAGS(1) reserved(1) SALT_LEN(1) NONCE_LEN(1) pad to 16
    header = bytearray(16)
    header[0:4] = MAGIC
    header[4] = VERSION
    header[5] = kdf_id
    header[6] = 0  # flags
    header[7] = 0  # reserved
    header[8] = SALT_LENGTH
    header[9] = NONCE_LENGTH
    # bytes 10-15: optional params (pbkdf2 iters as 3-byte BE if pbkdf2)
    if kdf_id == KDF_PBKDF2:
        header[10:13] = struct.pack(">I", pbkdf2_iterations)[1:]  # 24-bit

    return bytes(header) + salt + nonce + ct_and_tag


def _decrypt_v3(payload: bytes, passphrase: str) -> str:
    header = payload[:16]
    if header[0:4] != MAGIC:
        raise ValueError("Invalid payload magic")
    ver = header[4]
    kdf_id = header[5]
    salt_len = header[8]
    nonce_len = header[9]
    if salt_len != SALT_LENGTH or nonce_len != NONCE_LENGTH:
        raise ValueError("Unsupported salt/nonce length")

    off = 16
    salt = payload[off : off + salt_len]
    off += salt_len
    nonce = payload[off : off + nonce_len]
    off += nonce_len
    ct_and_tag = payload[off:]

    if kdf_id == KDF_ARGON2ID and HAS_ARGON2:
        key = derive_key_argon2id(passphrase, salt)
    else:
        iters = PBKDF2_ITERATIONS_DEFAULT
        if kdf_id == KDF_PBKDF2:
            try:
                iters = struct.unpack(">I", b"\x00" + header[10:13])[0]
                if iters < 10_000:
                    iters = PBKDF2_ITERATIONS_DEFAULT
            except Exception:
                pass
        key = derive_key_pbkdf2(passphrase, salt, iterations=iters)

    aesgcm = AESGCM(key)
    pt = aesgcm.decrypt(nonce, ct_and_tag, None)
    return pt.decode("utf-8")


def _decrypt_legacy_v1(payload: bytes, passphrase: str) -> str:
    """Backward-compatible decrypt for v1/v2 unversioned payloads."""
    min_len = 4 + SALT_LENGTH + NONCE_LENGTH + 16
    if len(payload) < min_len:
        raise ValueError("Message could not be decrypted — wrong passphrase or corrupted image.")
    N = struct.unpack(">I", payload[:4])[0]
    expected = 4 + SALT_LENGTH + NONCE_LENGTH + N
    if len(payload) < expected:
        raise ValueError("Message could not be decrypted — wrong passphrase or corrupted image.")
    salt = payload[4:20]
    nonce = payload[20:32]
    ct = payload[32:32 + N]
    # Try both iteration counts used historically
    for iters in (200_000, 100_000):
        try:
            key = derive_key_pbkdf2(passphrase, salt, iterations=iters)
            pt = AESGCM(key).decrypt(nonce, ct, None)
            return pt.decode("utf-8")
        except (InvalidTag, Exception):
            continue
    raise ValueError("Message could not be decrypted — wrong passphrase or corrupted image.")


def decrypt_payload(payload_bytes: bytes, passphrase: str) -> str:
    if len(payload_bytes) < 16:
        raise ValueError("Message could not be decrypted — wrong passphrase or corrupted image.")
    try:
        if payload_bytes[:4] == MAGIC:
            return _decrypt_v3(payload_bytes, passphrase)
        return _decrypt_legacy_v1(payload_bytes, passphrase)
    except ValueError:
        raise
    except Exception:
        raise ValueError("Message could not be decrypted — wrong passphrase or corrupted image.")


def payload_info(payload_bytes: bytes) -> dict:
    """Inspect payload without decrypting."""
    if len(payload_bytes) >= 4 and payload_bytes[:4] == MAGIC:
        return {
            "format": "v3",
            "version": payload_bytes[4],
            "kdf": "argon2id" if payload_bytes[5] == KDF_ARGON2ID else "pbkdf2",
            "length": len(payload_bytes),
        }
    return {"format": "legacy_v1", "length": len(payload_bytes)}
