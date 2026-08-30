"""
FastAPI Main Application for SecureStegVault Backend
Endpoints:
- GET  /api/health          -> service + model status
- POST /api/capacity        -> Image capacity check
- POST /api/encode          -> Full Phase 1-5 encoding pipeline
- POST /api/decode          -> Decoding + AES-256-GCM recovery
- POST /api/train_models    -> (dev) retrain CostMapCNN + SteganalyzerNet
"""

from __future__ import annotations

import io
import math
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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

app = FastAPI(title="SecureStegVault API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_image_bytes(file_bytes: bytes) -> np.ndarray:
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        fmt = (pil_img.format or "").upper()
        if fmt not in ["PNG", "BMP"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format '{fmt}'. Cover image must be a lossless PNG or BMP file.",
            )
        return np.array(pil_img.convert("RGB"), dtype=np.uint8)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or unreadable image file.")


def _build_zones_and_cost(
    image_np: np.ndarray,
    gamma: float,
    cost_map_mode: str,
    config: ZoningConfig,
):
    """Shared helper: cost map + zone labels (cost-based, quantized & median-filtered)."""
    cost_map = compute_cnn_costmap(image_np, gamma=gamma, cost_map_mode=cost_map_mode)
    H, W, C = image_np.shape
    cost_map_3d = np.repeat(cost_map[:, :, np.newaxis], C, axis=2)
    zones_3d = classify_zones(cost_map_3d, config)
    return cost_map, cost_map_3d, zones_3d


@app.get("/api/health")
def health_check():
    models_dir = Path(__file__).resolve().parents[1] / "models"
    costmap_ok = (models_dir / "costmap_cnn.pth").is_file()
    stega_ok = (models_dir / "steganalyzer_net.pth").is_file()
    torch_ok = False
    try:
        import torch
        torch_ok = True
    except ImportError:
        pass
    return {
        "status": "ok",
        "service": "SecureStegVault Backend",
        "version": "3.0.0",
        "torch_available": torch_ok,
        "models": {
            "costmap_cnn": costmap_ok,
            "steganalyzer_net": stega_ok,
        },
    }


@app.post("/api/capacity")
async def check_capacity(
    file: UploadFile = File(...),
    thresh_a: float = Form(0.35),
    thresh_b: float = Form(0.65),
    gamma: float = Form(0.7),
    cost_map_mode: str = Form("cnn"),
    emd_n: int = Form(2),
):
    contents = await file.read()
    img_np = load_image_bytes(contents)
    config = ZoningConfig(thresh_a=thresh_a, thresh_b=thresh_b, emd_group_size=emd_n)
    cost_map = compute_cnn_costmap(img_np, gamma=gamma, cost_map_mode=cost_map_mode)
    cap = calculate_capacity(img_np.shape, cost_map, config)
    return {
        "width": img_np.shape[1],
        "height": img_np.shape[0],
        "channels": img_np.shape[2],
        "capacity": cap,
        "cost_map_mode": cost_map_mode,
        "emd_n": emd_n,
    }


@app.post("/api/encode")
async def encode_steganography(
    file: UploadFile = File(...),
    secret_text: str = Form(...),
    passphrase: str = Form(...),
    thresh_a: float = Form(0.35),
    thresh_b: float = Form(0.65),
    gamma: float = Form(0.7),
    kb_bits: int = Form(2),
    kc_bits: int = Form(3),
    cost_map_mode: str = Form("cnn"),
    adversarial_strength: float = Form(0.0),
    emd_n: int = Form(2),
):
    if not secret_text.strip():
        raise HTTPException(status_code=400, detail="Secret message cannot be empty.")
    if not passphrase:
        raise HTTPException(status_code=400, detail="Passphrase is required.")

    contents = await file.read()
    cover_np = load_image_bytes(contents)

    config = ZoningConfig(
        thresh_a=thresh_a,
        thresh_b=thresh_b,
        emd_group_size=emd_n,
        kb_bits=kb_bits,
        kc_bits=kc_bits,
    )

    encrypted_payload = encrypt_payload(secret_text, passphrase)
    payload_len_bytes = len(encrypted_payload)

    cost_map, cost_map_3d, zones_3d = _build_zones_and_cost(
        cover_np, gamma, cost_map_mode, config
    )

    cap_info = calculate_capacity(cover_np.shape, cost_map, config)
    if payload_len_bytes > cap_info["max_bytes"]:
        max_chars = max(0, cap_info["max_bytes"] - 48)
        raise HTTPException(
            status_code=400,
            detail=(
                f"Message is too long for this image! Capacity is {cap_info['max_bytes']} bytes "
                f"(~{max_chars} chars), encrypted payload needs {payload_len_bytes} bytes."
            ),
        )

    grad_map = None
    adv_signs = None
    if adversarial_strength > 0.0 or cost_map_mode == "advanced":
        try:
            grad_map = compute_adversarial_gradient_map(cover_np)
            adv_signs = adversarial_sign_map(grad_map).flatten()
        except Exception as e:
            print("Adversarial map skipped:", e)

    stego_np = cover_np.copy()
    image_flat = stego_np.flatten()
    zones_flat = zones_3d.flatten()
    cost_flat = cost_map_3d.flatten()

    raw_zone_a = np.where(zones_flat == 0)[0]
    raw_zone_b = np.where(zones_flat == 1)[0]
    raw_zone_c = np.where(zones_flat == 2)[0]

    zone_a_indices = rank_zone_indices(cost_flat, raw_zone_a, is_emd=True, group_size=emd_n)
    zone_b_indices = rank_zone_indices(cost_flat, raw_zone_b, is_emd=False)
    zone_c_indices = rank_zone_indices(cost_flat, raw_zone_c, is_emd=False)

    payload_bits = []
    for b in encrypted_payload:
        for bit_i in range(7, -1, -1):
            payload_bits.append((b >> bit_i) & 1)
    total_bits_to_embed = len(payload_bits)
    bits_remaining = total_bits_to_embed
    current_bit_idx = 0
    zone_a_bits_used = zone_b_bits_used = zone_c_bits_used = 0

    if emd_n == 3:
        zone_a_groups = len(zone_a_indices) // 3
        zone_a_max_bits = int(zone_a_groups * math.log2(7))
        if zone_a_max_bits > 0 and bits_remaining > 0:
            a_bits_count = min(bits_remaining, zone_a_max_bits)
            a_bytes_count = (a_bits_count + 7) // 8
            a_payload_bytes = encrypted_payload[:a_bytes_count]
            a_digits = bytes_to_base7_digits(a_payload_bytes)
            _, digits_embedded = embed_emd_zone_a(
                image_flat, zone_a_indices, a_digits, emd_n=3,
                adversarial_signs=adv_signs, adversarial_strength=adversarial_strength,
            )
            zone_a_bits_used = int(digits_embedded * math.log2(7))
            a_bytes_embedded = digits_embedded // 3
            current_bit_idx = a_bytes_embedded * 8
            bits_remaining = max(0, total_bits_to_embed - current_bit_idx)
    else:
        zone_a_groups = len(zone_a_indices) // 2
        zone_a_max_bits = int(zone_a_groups * math.log2(5))
        if zone_a_max_bits > 0 and bits_remaining > 0:
            a_bits_count = min(bits_remaining, zone_a_max_bits)
            a_bytes_count = (a_bits_count + 7) // 8
            a_payload_bytes = encrypted_payload[:a_bytes_count]
            a_digits = bytes_to_base5_digits(a_payload_bytes)
            _, digits_embedded = embed_emd_zone_a(
                image_flat, zone_a_indices, a_digits, emd_n=2,
                adversarial_signs=adv_signs, adversarial_strength=adversarial_strength,
            )
            zone_a_bits_used = int(digits_embedded * math.log2(5))
            a_bytes_embedded = digits_embedded // 4
            current_bit_idx = a_bytes_embedded * 8
            bits_remaining = max(0, total_bits_to_embed - current_bit_idx)

    if bits_remaining > 0 and len(zone_b_indices) > 0:
        b_bits_stream = payload_bits[current_bit_idx : current_bit_idx + bits_remaining]
        _, b_embedded_count = embed_opap_zone(
            image_flat, zone_b_indices, b_bits_stream, k=config.kb_bits,
            adversarial_signs=adv_signs, adversarial_strength=adversarial_strength,
        )
        zone_b_bits_used = b_embedded_count
        current_bit_idx += b_embedded_count
        bits_remaining = max(0, total_bits_to_embed - current_bit_idx)

    if bits_remaining > 0 and len(zone_c_indices) > 0:
        c_bits_stream = payload_bits[current_bit_idx : current_bit_idx + bits_remaining]
        _, c_embedded_count = embed_opap_zone(
            image_flat, zone_c_indices, c_bits_stream, k=config.kc_bits,
            adversarial_signs=adv_signs, adversarial_strength=adversarial_strength,
        )
        zone_c_bits_used = c_embedded_count
        current_bit_idx += c_embedded_count
        bits_remaining = max(0, total_bits_to_embed - current_bit_idx)

    if bits_remaining > 0:
        raise HTTPException(status_code=400, detail="Could not fit all bits into available image zones.")

    stego_np = image_flat.reshape(cover_np.shape)

    zone_breakdown = {
        "zone_a_bits": zone_a_bits_used,
        "zone_b_bits": zone_b_bits_used,
        "zone_c_bits": zone_c_bits_used,
    }
    metrics = calculate_metrics(cover_np, stego_np, total_bits_to_embed, zone_breakdown)
    security_report = calculate_security_report(cover_np, stego_np)
    visuals = generate_visualizations(cover_np, stego_np, cost_map, zones_3d, gradient_map=grad_map)

    return {
        "success": True,
        "metrics": metrics,
        "security_report": security_report,
        "visuals": visuals,
        "cost_map_mode": cost_map_mode,
        "adversarial_strength": adversarial_strength,
        "emd_n": emd_n,
        "models_used": {
            "costmap": "CostMapCNN (trained)" if cost_map_mode in ("cnn", "advanced") else "classical",
            "steganalyzer": "SteganalyzerNet (trained)" if adversarial_strength > 0 or cost_map_mode == "advanced" else "none",
        },
    }


@app.post("/api/decode")
async def decode_steganography(
    file: UploadFile = File(...),
    passphrase: str = Form(...),
    thresh_a: float = Form(0.35),
    thresh_b: float = Form(0.65),
    gamma: float = Form(0.7),
    kb_bits: int = Form(2),
    kc_bits: int = Form(3),
    cost_map_mode: str = Form("cnn"),
    emd_n: int = Form(2),
):
    if not passphrase:
        raise HTTPException(status_code=400, detail="Passphrase is required.")

    contents = await file.read()
    stego_np = load_image_bytes(contents)

    config = ZoningConfig(
        thresh_a=thresh_a, thresh_b=thresh_b, emd_group_size=emd_n,
        kb_bits=kb_bits, kc_bits=kc_bits,
    )

    cost_map, cost_map_3d, zones_3d = _build_zones_and_cost(
        stego_np, gamma, cost_map_mode, config
    )

    image_flat = stego_np.flatten()
    zones_flat = zones_3d.flatten()
    cost_flat = cost_map_3d.flatten()

    raw_zone_a = np.where(zones_flat == 0)[0]
    raw_zone_b = np.where(zones_flat == 1)[0]
    raw_zone_c = np.where(zones_flat == 2)[0]

    zone_a_indices = rank_zone_indices(cost_flat, raw_zone_a, is_emd=True, group_size=emd_n)
    zone_b_indices = rank_zone_indices(cost_flat, raw_zone_b, is_emd=False)
    zone_c_indices = rank_zone_indices(cost_flat, raw_zone_c, is_emd=False)

    try:
        extracted_payload_bytes = bytearray()

        zone_a_groups = len(zone_a_indices) // emd_n
        if zone_a_groups > 0:
            extracted_digits = extract_emd_zone_a(image_flat, zone_a_indices, zone_a_groups, emd_n=emd_n)
            a_bytes = base7_digits_to_bytes(extracted_digits) if emd_n == 3 else base5_digits_to_bytes(extracted_digits)
            extracted_payload_bytes.extend(a_bytes)

        if len(zone_b_indices) > 0:
            b_bits_avail = len(zone_b_indices) * config.kb_bits
            b_bits = extract_opap_zone(image_flat, zone_b_indices, b_bits_avail, k=config.kb_bits)
            for byte_i in range(len(b_bits) // 8):
                b_val = 0
                for bit_i in range(8):
                    b_val = (b_val << 1) | b_bits[byte_i * 8 + bit_i]
                extracted_payload_bytes.append(b_val)

        if len(zone_c_indices) > 0:
            c_bits_avail = len(zone_c_indices) * config.kc_bits
            c_bits = extract_opap_zone(image_flat, zone_c_indices, c_bits_avail, k=config.kc_bits)
            for byte_i in range(len(c_bits) // 8):
                c_val = 0
                for bit_i in range(8):
                    c_val = (c_val << 1) | c_bits[byte_i * 8 + bit_i]
                extracted_payload_bytes.append(c_val)

        plaintext = decrypt_payload(bytes(extracted_payload_bytes), passphrase)
        return {"success": True, "decrypted_text": plaintext}

    except ValueError:
        raise HTTPException(status_code=400, detail="Message could not be decrypted — wrong passphrase or corrupted image.")
    except Exception:
        raise HTTPException(status_code=400, detail="Message could not be decrypted — wrong passphrase or corrupted image.")


@app.post("/api/train_models")
async def train_models_endpoint(
    costmap_samples: int = Form(48),
    costmap_epochs: int = Form(6),
    stega_samples: int = Form(96),
    stega_epochs: int = Form(10),
):
    try:
        from backend.models.costmap_net import train_costmap_model
        from backend.models.steganalyzer_net import train_steganalyzer_model

        p1 = train_costmap_model(num_samples=costmap_samples, epochs=costmap_epochs, verbose=True)
        p2 = train_steganalyzer_model(num_samples=stega_samples, epochs=stega_epochs, verbose=True)
        import backend.models.costmap_net as cm
        import backend.models.steganalyzer_net as sn
        cm._COSTMAP_MODEL = None
        sn._STEGA_MODEL = None
        return {"success": True, "costmap_weights": str(p1), "steganalyzer_weights": str(p2)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# v3 Research endpoints
# ---------------------------------------------------------------------------

@app.get("/api/strategies")
def api_strategies():
    from backend.strategies import list_strategies
    return {"strategies": list_strategies()}


@app.post("/api/benchmark")
async def api_benchmark(
    max_images: int = Form(3),
    seed: int = Form(42),
):
    """Run experimental benchmark (synthetic covers if no datasets/covers)."""
    from backend.benchmark import run_benchmark, BenchmarkConfig
    cfg = BenchmarkConfig(max_images=max_images, seed=seed)
    return run_benchmark(cfg)


@app.post("/api/ablation")
async def api_ablation(seed: int = Form(42)):
    from backend.benchmark import run_ablation, BenchmarkConfig
    return run_ablation(BenchmarkConfig(seed=seed, max_images=2))


@app.get("/api/dataset/stats")
def api_dataset_stats():
    from backend.dataset.loader import DatasetConfig, dataset_stats
    return dataset_stats(DatasetConfig())


@app.get("/api/system")
def api_system():
    import platform, sys
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        info["torch"] = None
        info["cuda_available"] = False
        info["device"] = "cpu"
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
    except Exception:
        pass
    return info


@app.post("/api/security/analyze")
async def api_security_analyze(file: UploadFile = File(...)):
    contents = await file.read()
    img = load_image_bytes(contents)
    from backend.security.evaluator import evaluate_security
    report = evaluate_security(None, img)
    return report.to_dict()



# ---------------------------------------------------------------------------
# Batch Lab API (SecureStegVault v3.2)
# ---------------------------------------------------------------------------

from fastapi import Request
from fastapi.responses import PlainTextResponse, FileResponse


@app.post("/api/batch/jobs")
async def batch_create_job(request: Request):
    """
    Create a batch encode/decode/experiment job.
    Multipart form:
      files: one or more images
      type: encode | decode | experiment
      secret_text, passphrase, strategy, workers, message_mode, ...
      optional message__<filename> for per-image messages
    """
    form = await request.form()
    job_type = str(form.get("type") or "encode").lower()
    files = []
    per_msgs = {}
    for key, val in form.multi_items():
        if hasattr(val, "filename") and hasattr(val, "read"):
            data = await val.read()
            files.append((val.filename or "image.png", data))
        elif key.startswith("message__"):
            per_msgs[key[len("message__"):]] = str(val)

    if not files:
        raise HTTPException(400, detail="No image files uploaded.")

    config = {
        "secret_text": str(form.get("secret_text") or ""),
        "passphrase": str(form.get("passphrase") or ""),
        "strategy": str(form.get("strategy") or "cnn_emd_opap"),
        "cost_map_mode": str(form.get("cost_map_mode") or "cnn"),
        "thresh_a": float(form.get("thresh_a") or 0.35),
        "thresh_b": float(form.get("thresh_b") or 0.65),
        "gamma": float(form.get("gamma") or 0.7),
        "kb_bits": int(form.get("kb_bits") or 2),
        "kc_bits": int(form.get("kc_bits") or 3),
        "emd_n": int(form.get("emd_n") or 2),
        "adversarial_strength": float(form.get("adversarial_strength") or 0.0),
        "workers": int(form.get("workers") or 2),
        "message_mode": str(form.get("message_mode") or "same"),
        "engine": str(form.get("engine") or "python"),
        "seed": int(form.get("seed") or 42),
    }
    if form.get("strategies"):
        try:
            import json as _json
            config["strategies"] = _json.loads(str(form.get("strategies")))
        except Exception:
            config["strategies"] = [s.strip() for s in str(form.get("strategies")).split(",") if s.strip()]
    if form.get("bpp_list"):
        try:
            import json as _json
            config["bpp_list"] = _json.loads(str(form.get("bpp_list")))
        except Exception:
            config["bpp_list"] = [float(x) for x in str(form.get("bpp_list")).split(",") if x.strip()]

    from backend.batch import get_batch_manager
    mgr = get_batch_manager(workers=config["workers"])
    try:
        if job_type == "decode":
            job = mgr.create_decode_job(files, config)
        elif job_type == "experiment":
            job = mgr.create_experiment_job(files, config)
        else:
            job = mgr.create_encode_job(files, config, per_image_messages=per_msgs or None)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, detail=f"Failed to create batch job: {e}")

    return {"job_id": job.job_id, "status": job.status, "total": job.total}


@app.get("/api/batch/jobs")
def batch_list_jobs(limit: int = 50):
    from backend.batch import get_batch_manager
    return {"jobs": get_batch_manager().list_jobs(limit=limit)}


@app.get("/api/batch/jobs/{job_id}")
def batch_get_job(job_id: str):
    from backend.batch import get_batch_manager
    data = get_batch_manager().get_job(job_id)
    if not data:
        raise HTTPException(404, detail="Job not found")
    return data


@app.post("/api/batch/jobs/{job_id}/cancel")
def batch_cancel_job(job_id: str):
    from backend.batch import get_batch_manager
    try:
        return get_batch_manager().cancel_job(job_id)
    except KeyError:
        raise HTTPException(404, detail="Job not found")


@app.post("/api/batch/jobs/{job_id}/retry")
def batch_retry_job(job_id: str):
    from backend.batch import get_batch_manager
    try:
        return get_batch_manager().retry_failed(job_id)
    except KeyError:
        raise HTTPException(404, detail="Job not found")


@app.get("/api/batch/jobs/{job_id}/summary")
def batch_job_summary(job_id: str):
    from backend.batch import get_batch_manager
    try:
        return get_batch_manager().summary(job_id)
    except KeyError:
        raise HTTPException(404, detail="Job not found")


@app.get("/api/batch/jobs/{job_id}/export")
def batch_export_job(job_id: str, format: str = "json"):
    from backend.batch import get_batch_manager
    mgr = get_batch_manager()
    try:
        if format == "csv":
            text = mgr.export(job_id, "csv")
            return PlainTextResponse(text, media_type="text/csv",
                                    headers={"Content-Disposition": f'attachment; filename="{job_id}.csv"'})
        if format == "zip":
            path = mgr.export(job_id, "zip")
            return FileResponse(path, media_type="application/zip",
                                filename=Path(path).name)
        text = mgr.export(job_id, "json")
        return PlainTextResponse(text, media_type="application/json",
                                headers={"Content-Disposition": f'attachment; filename="{job_id}.json"'})
    except KeyError:
        raise HTTPException(404, detail="Job not found")


# ---------------------------------------------------------------------------
# Comparison / research benchmarking endpoints (additive)
# ---------------------------------------------------------------------------

@app.get("/api/comparison/papers")
def api_comparison_papers():
    import json
    meta_path = Path(__file__).resolve().parent / "comparison" / "paper_metadata.json"
    if not meta_path.is_file():
        raise HTTPException(404, detail="paper_metadata.json not found")
    return json.loads(meta_path.read_text(encoding="utf-8"))


@app.get("/api/comparison/algorithms")
def api_comparison_algorithms():
    from backend.strategies.registry import list_strategies
    try:
        from backend.comparison.single_input_compare import DEFAULT_STRATEGIES
        defaults = DEFAULT_STRATEGIES
    except Exception:
        defaults = []
    return {
        "all_registered": list_strategies(),
        "comparison_defaults": defaults,
    }


@app.post("/api/comparison/compare-one")
async def api_comparison_compare_one(
    file: UploadFile = File(...),
    secret_text: str = Form(...),
    passphrase: str = Form(...),
    strategies: str = Form(""),
):
    """Run the same cover + secret + passphrase through all (or selected) methods."""
    if not secret_text.strip():
        raise HTTPException(400, detail="Secret message cannot be empty.")
    if not passphrase:
        raise HTTPException(400, detail="Passphrase is required.")
    contents = await file.read()
    cover = load_image_bytes(contents)
    strat_list = None
    if strategies.strip():
        strat_list = [s.strip() for s in strategies.split(",") if s.strip()]
    try:
        from backend.comparison.single_input_compare import compare_one
        return compare_one(cover, secret_text, passphrase, strategies=strat_list)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, detail=str(e))


@app.post("/api/comparison/run")
async def api_comparison_run(
    file: UploadFile = File(...),
    secret_text: str = Form(...),
    passphrase: str = Form(...),
    strategies: str = Form(""),
    run_robustness: bool = Form(False),
    run_security: bool = Form(True),
):
    """Synchronous orchestrated comparison (scoring + optional robustness)."""
    if not secret_text.strip():
        raise HTTPException(400, detail="Secret message cannot be empty.")
    if not passphrase:
        raise HTTPException(400, detail="Passphrase is required.")
    contents = await file.read()
    cover = load_image_bytes(contents)
    strat_list = [s.strip() for s in strategies.split(",") if s.strip()] or None
    try:
        from backend.comparison.orchestrator import run_single_orchestrated, persist_experiment
        result = run_single_orchestrated(
            cover, secret_text, passphrase,
            strategies=strat_list,
            run_robustness=run_robustness,
            run_security=run_security,
        )
        folder = persist_experiment(result)
        result["experiment_path"] = str(folder)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, detail=str(e))


@app.post("/api/comparison/ablation")
async def api_comparison_ablation(
    file: UploadFile = File(...),
    secret_text: str = Form(...),
    passphrase: str = Form(...),
):
    """SecureStegVault internal ablation ladder (not paper competition)."""
    contents = await file.read()
    cover = load_image_bytes(contents)
    try:
        from backend.comparison.ablation import run_ablation_ladder
        return run_ablation_ladder(cover, secret_text, passphrase)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, detail=str(e))


@app.get("/api/comparison/report/{run_id}")
def api_comparison_report(run_id: str, format: str = "json"):
    from fastapi.responses import PlainTextResponse
    exp_root = Path(__file__).resolve().parents[1] / "experiments"
    folder = exp_root / run_id
    if not folder.is_dir():
        # try prefix match
        matches = list(exp_root.glob(f"{run_id}*"))
        if not matches:
            raise HTTPException(404, detail="Experiment not found")
        folder = matches[0]
    exp_json = folder / "experiment.json"
    if not exp_json.is_file():
        raise HTTPException(404, detail="experiment.json missing")
    data = exp_json.read_text(encoding="utf-8")
    if format == "csv":
        from backend.comparison.export import export_csv
        import json as _json
        return PlainTextResponse(
            export_csv(_json.loads(data)),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{folder.name}.csv"'},
        )
    return PlainTextResponse(
        data,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{folder.name}.json"'},
    )


@app.post("/api/comparison/dataset/import")
async def api_comparison_dataset_import(file: UploadFile = File(...)):
    """ZIP upload of cover images — user-triggered only."""
    contents = await file.read()
    dest = Path(__file__).resolve().parents[1] / "datasets" / "covers"
    try:
        from backend.comparison.dataset_support import import_from_zip
        manifest = import_from_zip(contents, dest)
        return {"success": True, "manifest": manifest}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(400, detail=str(e))


@app.post("/api/comparison/dataset/fetch")
async def api_comparison_dataset_fetch(
    dataset: str = Form(...),
    confirm: bool = Form(False),
):
    """Named public-set fetch — requires confirm=true; never automatic."""
    if not confirm:
        raise HTTPException(400, detail="confirm=true is required for public dataset fetch.")
    # Explicit refusal of silent downloads: only acknowledge the request contract.
    raise HTTPException(
        501,
        detail=(
            f"Public dataset fetch for '{dataset}' is opt-in only and not auto-wired. "
            "Place images under datasets/covers/ or use /api/comparison/dataset/import."
        ),
    )


@app.get("/api/comparison/checkpoints")
def api_comparison_checkpoints():
    """Report official checkpoint discovery status for each paper method."""
    from backend.comparison.checkpoint.manager import CheckpointManager
    return CheckpointManager().status_report()
