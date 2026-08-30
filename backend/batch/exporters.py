"""Export batch results as CSV, JSON, and ZIP — no secrets in exports."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .aggregation import aggregate_job
from .storage import EXPERIMENTS_ROOT


def _flat_rows(job: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for it in job.get("items") or []:
        r = it.get("result") or {}
        m = r.get("metrics") or {}
        s = r.get("security") or {}
        rows.append({
            "job_id": job.get("job_id"),
            "item_id": it.get("id"),
            "filename": it.get("filename"),
            "status": it.get("status"),
            "error": it.get("error") or "",
            "processing_time_s": it.get("processing_time_s"),
            "strategy": it.get("strategy") or r.get("strategy") or "",
            "bpp_target": it.get("bpp_target"),
            "psnr_db": m.get("psnr_db"),
            "ssim": m.get("ssim"),
            "mse": m.get("mse"),
            "achieved_bpp": m.get("achieved_bpp"),
            "modified_pixel_percentage": m.get("modified_pixel_percentage"),
            "composite_suspicion": s.get("composite_suspicion"),
            "cnn_stego_prob": s.get("cnn_stego_probability"),
            "decode_success": r.get("success") if job.get("type") == "decode" else None,
            # never include decrypted_text or passphrase
        })
    return rows


def export_csv(job: Dict[str, Any]) -> str:
    rows = _flat_rows(job)
    if not rows:
        return "job_id,filename,status\n"
    keys = list(rows[0].keys())
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def export_json(job: Dict[str, Any]) -> str:
    summary = aggregate_job(job)
    payload = {
        "job": {
            "job_id": job.get("job_id"),
            "type": job.get("type"),
            "status": job.get("status"),
            "created_at": job.get("created_at"),
            "completed_at": job.get("completed_at"),
            "configuration": job.get("configuration"),
            "experiment_id": job.get("experiment_id"),
        },
        "summary": summary,
        "items": [
            {
                **{k: v for k, v in it.items() if k not in ("result", "secret_text")},
                "metrics": (it.get("result") or {}).get("metrics"),
                "security": (it.get("result") or {}).get("security"),
                "decode_ok": (it.get("result") or {}).get("success")
                if job.get("type") == "decode"
                else None,
            }
            for it in (job.get("items") or [])
        ],
    }
    return json.dumps(payload, indent=2, default=str)


def export_zip(job: Dict[str, Any], output_dir: Optional[Path] = None) -> Path:
    """
    Build SecureStegVault_Batch_<job_id>.zip with:
      stego/ | metrics.csv | results.json | experiment.json | README.txt
    """
    job_id = job.get("job_id") or "batch"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(output_dir) if output_dir else EXPERIMENTS_ROOT
    out_root.mkdir(parents=True, exist_ok=True)
    zip_path = out_root / f"SecureStegVault_Batch_{job_id}_{ts}.zip"

    csv_text = export_csv(job)
    json_text = export_json(job)
    summary = aggregate_job(job)
    readme = (
        f"SecureStegVault Batch Export\n"
        f"Job ID: {job_id}\n"
        f"Type: {job.get('type')}\n"
        f"Status: {job.get('status')}\n"
        f"Successful: {summary.get('successful')} / {summary.get('total')}\n"
        f"Failed: {summary.get('failed')}\n"
        f"Generated: {ts}\n"
        f"\nSecrets (passphrases, plaintext) are intentionally omitted.\n"
    )
    experiment = {
        "experiment_id": job.get("experiment_id") or job_id,
        "configuration": job.get("configuration"),
        "summary": summary,
    }

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metrics.csv", csv_text)
        zf.writestr("results.json", json_text)
        zf.writestr("experiment.json", json.dumps(experiment, indent=2, default=str))
        zf.writestr("README.txt", readme)
        for it in job.get("items") or []:
            if it.get("status") != "completed":
                continue
            outp = it.get("output_path")
            if outp and Path(outp).is_file():
                arc = f"stego/{it.get('safe_filename') or Path(outp).name}"
                zf.write(outp, arcname=arc)
            # recovered payloads for decode (optional, no passphrase)
            if job.get("type") == "decode":
                text = (it.get("result") or {}).get("decrypted_text")
                if text is not None:
                    name = (it.get("safe_filename") or it.get("filename") or "item").rsplit(".", 1)[0]
                    zf.writestr(f"payloads/{name}.txt", text)

    return zip_path
