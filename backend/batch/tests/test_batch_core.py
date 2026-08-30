"""Lightweight tests for batch models / aggregation / validation (no GPU)."""
import io
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from PIL import Image
from backend.batch.models import BatchJob, BatchItem, JobStatus, ItemStatus
from backend.batch.aggregation import aggregate_job
from backend.batch.validation import validate_image_bytes, validate_batch_config, detect_duplicates
from backend.batch.exporters import export_csv, export_json


def _png(seed=0, size=32):
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 255, (size, size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_validate_image_ok():
    ok, err, meta = validate_image_bytes(_png(), "a.png")
    assert ok and meta["width"] == 32


def test_validate_image_empty():
    ok, err, _ = validate_image_bytes(b"", "x.png")
    assert not ok


def test_duplicates():
    assert detect_duplicates(["a.png", "b.png", "a.png"]) == ["a.png"]


def test_config_requires_passphrase():
    errs = validate_batch_config({"type": "encode", "secret_text": "hi"}, 1)
    assert any("Passphrase" in e for e in errs)


def test_aggregation_partial_failure():
    job = {
        "job_id": "t1",
        "type": "encode",
        "status": "completed_with_errors",
        "total": 3,
        "items": [
            {"status": "completed", "result": {"metrics": {"psnr_db": 40.0, "ssim": 0.99}, "security": {}}, "processing_time_s": 1.0},
            {"status": "completed", "result": {"metrics": {"psnr_db": 50.0, "ssim": 0.98}, "security": {}}, "processing_time_s": 2.0},
            {"status": "failed", "error": "capacity", "result": None},
        ],
    }
    s = aggregate_job(job)
    assert s["successful"] == 2 and s["failed"] == 1
    assert abs(s["metrics"]["psnr_db"]["mean"] - 45.0) < 1e-6


def test_export_no_secrets():
    job = {
        "job_id": "t2",
        "type": "encode",
        "status": "completed",
        "configuration": {"strategy": "emd_opap"},
        "items": [{"id": "i1", "filename": "a.png", "status": "completed", "result": {"metrics": {"psnr_db": 42}}}],
    }
    j = export_json(job)
    assert "passphrase" not in j
    c = export_csv(job)
    assert "a.png" in c


def test_job_finalize():
    job = BatchJob()
    job.items = [
        BatchItem(status=ItemStatus.COMPLETED.value),
        BatchItem(status=ItemStatus.FAILED.value),
    ]
    job.finalize_status()
    assert job.status == JobStatus.COMPLETED_WITH_ERRORS.value


if __name__ == "__main__":
    for fn in [test_validate_image_ok, test_validate_image_empty, test_duplicates,
               test_config_requires_passphrase, test_aggregation_partial_failure,
               test_export_no_secrets, test_job_finalize]:
        fn()
        print("OK", fn.__name__)
    print("all passed")
