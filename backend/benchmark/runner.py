"""
Benchmark and ablation engines for SecureStegVault v3.

All results are generated experimentally — never fabricated.
"""

from __future__ import annotations

import csv
import json
import time
import hashlib
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

import numpy as np
from PIL import Image


@dataclass
class BenchmarkConfig:
    strategies: List[str] = field(default_factory=lambda: [
        "emd_opap", "cnn_emd_opap", "cnn_emd_opap_adv",
        "cnn_stc_emd_opap", "cnn_stc_emd_opap_adv",
    ])
    bpp_list: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.20, 0.30])
    seed: int = 42
    message_template: str = "SecureStegVault benchmark payload at {bpp:.2f} bpp. Seed={seed}."
    passphrase: str = "benchmark-passphrase-v3"
    output_dir: str = "experiments"
    max_images: int = 5
    image_paths: List[str] = field(default_factory=list)


def _image_id(path: str) -> str:
    return hashlib.sha1(path.encode()).hexdigest()[:12]


def _load(path: str, max_side: int = 256) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def _synthetic_cover(size: int = 128, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    base = rng.randint(40, 200, (size, size, 3), dtype=np.uint8)
    # add texture
    noise = rng.randn(size, size, 3) * 12
    return np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def run_benchmark(cfg: BenchmarkConfig) -> Dict[str, Any]:
    from backend.strategies import get_strategy, list_strategies

    available = list_strategies()
    strategies = [s for s in cfg.strategies if s in available]
    if not strategies:
        strategies = available[:3]

    images: List[tuple] = []
    if cfg.image_paths:
        for p in cfg.image_paths[: cfg.max_images]:
            try:
                images.append((p, _load(p)))
            except Exception:
                pass
    if not images:
        # fallback synthetic covers so the engine always runs
        for i in range(min(cfg.max_images, 3)):
            images.append((f"synthetic_{i}.png", _synthetic_cover(128, seed=cfg.seed + i)))

    results = []
    exp_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(cfg.output_dir) / f"benchmark_{exp_id}"
    out.mkdir(parents=True, exist_ok=True)

    for path, cover in images:
        H, W, C = cover.shape
        pixels = H * W * C
        for bpp in cfg.bpp_list:
            target_bits = int(bpp * pixels)
            target_chars = max(8, (target_bits // 8) - 64)
            msg = (cfg.message_template.format(bpp=bpp, seed=cfg.seed) + " X" * 200)[:target_chars]
            for strat_name in strategies:
                try:
                    strat = get_strategy(strat_name)
                    t0 = time.perf_counter()
                    res = strat.embed(cover, msg, cfg.passphrase)
                    elapsed = time.perf_counter() - t0
                    row = {
                        "experiment_id": exp_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "image": path,
                        "image_id": _image_id(path),
                        "strategy": strat_name,
                        "bpp_target": bpp,
                        "H": H, "W": W, "C": C,
                        "seed": cfg.seed,
                        "embed_time_s": elapsed,
                        **{f"metric_{k}": v for k, v in (res.metrics or {}).items() if not isinstance(v, (dict, list))},
                        "cnn_stego_prob": (res.security or {}).get("cnn_stego_probability"),
                        "composite_suspicion": (res.security or {}).get("composite_suspicion"),
                        "stc_note": (res.meta or {}).get("stc", ""),
                    }
                    results.append(row)
                except Exception as e:
                    results.append({
                        "experiment_id": exp_id,
                        "image": path,
                        "strategy": strat_name,
                        "bpp_target": bpp,
                        "error": str(e),
                    })

    # write CSV + JSON
    csv_path = out / "results.csv"
    json_path = out / "results.json"
    if results:
        keys = sorted({k for r in results for k in r.keys()})
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(results)
    with open(json_path, "w") as f:
        json.dump({"config": asdict(cfg), "results": results}, f, indent=2, default=str)

    config_path = out / "config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    return {
        "experiment_id": exp_id,
        "n_results": len(results),
        "output_dir": str(out),
        "csv": str(csv_path),
        "json": str(json_path),
        "results": results,
        "results_preview": results[:5],
    }


def run_ablation(cfg: Optional[BenchmarkConfig] = None) -> Dict[str, Any]:
    """
    Fixed ablation suite:
      A  emd_opap
      B  cnn_emd_opap
      C  cnn_emd_opap_adv
      D  cnn_stc_emd_opap
      E  cnn_stc_emd_opap_adv
    """
    cfg = cfg or BenchmarkConfig()
    cfg.strategies = [
        "emd_opap",
        "cnn_emd_opap",
        "cnn_emd_opap_adv",
        "cnn_stc_emd_opap",
        "cnn_stc_emd_opap_adv",
    ]
    cfg.bpp_list = [0.10, 0.20]
    return run_benchmark(cfg)
