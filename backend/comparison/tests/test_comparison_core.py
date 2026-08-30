"""Focused unit tests for comparison package."""

from __future__ import annotations
import numpy as np
import pytest


def test_paper4_roundtrip():
    from backend.strategies.registry import get_strategy
    cover = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    msg, pw = "hello research", "testpass123"
    s = get_strategy("paper4_lsb_magicmatrix")
    r = s.embed(cover, msg, pw)
    assert r.stego.shape == cover.shape
    out = s.extract(r.stego, pw)
    assert out == msg


def test_bitstream_adapter():
    from backend.comparison.adapters import BitstreamAdapter
    a = BitstreamAdapter()
    cover = np.zeros((8, 8, 3), dtype=np.uint8)
    res = a.adapt_for_embed(b"abc", cover)
    assert res.adapter_used == "bitstream"
    assert res.payload == b"abc"
    assert a.recover_from_extract(b"abc") == b"abc"


def test_image_tile_capacity_limit():
    from backend.comparison.adapters import ImageTileAdapter
    a = ImageTileAdapter(tile_h=4, tile_w=4, channels=3)
    cover = np.zeros((16, 16, 3), dtype=np.uint8)
    big = b"x" * 200
    res = a.adapt_for_embed(big, cover)
    assert res.capacity_limited is True


def test_normalize_na_safe():
    from backend.comparison.scoring.normalize import normalize_value
    vals = [10.0, None, 20.0, 10.0]
    out = normalize_value(vals, higher_is_better=True)
    assert out[1] is None
    assert out[0] == 0.0
    assert out[2] == 1.0


def test_pareto_domination():
    from backend.comparison.scoring.pareto import pareto_analysis
    methods = [
        {"method_id": "a", "psnr": 40.0, "mse": 1.0},
        {"method_id": "b", "psnr": 30.0, "mse": 2.0},
        {"method_id": "c", "psnr": 40.0, "mse": 0.5},
    ]
    directions = {"psnr": True, "mse": False}
    result = pareto_analysis(methods, directions)
    assert "c" in result["pareto_optimal_methods"]
    assert "b" in result["dominated_methods"]


def test_quality_metrics():
    from backend.comparison.metrics.quality import compute_quality
    a = np.zeros((16, 16, 3), dtype=np.uint8)
    b = a.copy()
    b[0, 0, 0] = 10
    q = compute_quality(a, b)
    assert q["psnr"] is not None
    assert q["ssim"] is not None


def test_ber():
    from backend.comparison.metrics.reliability import hamming_ber, compute_reliability
    assert hamming_ber(b"aa", b"aa") == 0.0
    rel = compute_reliability("hello", "hello")
    assert rel["exact_match"] is True
    assert rel["ber"] == 0.0


def test_robustness_attacks_run():
    from backend.comparison.metrics.robustness import attack_jpeg, attack_gaussian_noise
    img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    j = attack_jpeg(img, 70)
    assert j.shape == img.shape
    n = attack_gaussian_noise(img, 5)
    assert n.shape == img.shape


def test_weights_sum():
    from backend.comparison.scoring.weights import DEFAULT_WEIGHTS
    DEFAULT_WEIGHTS.validate()


def test_checkpoint_discovery_paper1_no_weights():
    from backend.comparison.checkpoint.manager import discover_checkpoints, OFFICIAL_STATUS
    info = discover_checkpoints("paper1_joint_cnn")
    assert info.exists is False or info.verification_status in (
        "not_found", "directory_missing", "file_present_unverified"
    )
    assert OFFICIAL_STATUS["paper1_joint_cnn"]["repo_contains_weights"] is False


def test_checkpoint_discovery_paper2_none():
    from backend.comparison.checkpoint.manager import discover_checkpoints, OFFICIAL_STATUS
    info = discover_checkpoints("paper2_cyclegan_steg")
    assert info.exists is False
    assert "NO OFFICIAL" in OFFICIAL_STATUS["paper2_cyclegan_steg"]["notes"]


def test_checkpoint_discovery_paper3_none():
    from backend.comparison.checkpoint.manager import discover_checkpoints, OFFICIAL_STATUS
    info = discover_checkpoints("paper3_block_prep_net")
    assert info.exists is False
    assert "NO OFFICIAL" in OFFICIAL_STATUS["paper3_block_prep_net"]["notes"]


def test_paper1_official_arch_imports():
    from backend.comparison.external_models.paper1_joint_cnn.official_arch import (
        HAS_TORCH, OfficialPaper1Model,
    )
    if HAS_TORCH:
        m = OfficialPaper1Model()
        assert m is not None
        n = sum(p.numel() for p in m.parameters())
        assert n > 1000


def test_checkpoint_manager_report():
    from backend.comparison.checkpoint.manager import CheckpointManager
    report = CheckpointManager().status_report()
    assert "paper1_joint_cnn" in report
    assert "paper2_cyclegan_steg" in report
    assert "paper3_block_prep_net" in report
