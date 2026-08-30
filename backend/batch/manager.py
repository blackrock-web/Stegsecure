"""
BatchManager — orchestrates job creation, queue, workers, cancel/retry, export.

Single-image pipeline remains the processing unit; this layer only schedules it.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    BatchJob,
    BatchItem,
    BatchConfig,
    JobStatus,
    ItemStatus,
    JobType,
    new_job_id,
)
from .queue import WorkQueue
from .worker import WorkerPool
from .storage import (
    job_tmp_dir,
    save_job_meta,
    load_job_meta,
    list_job_ids,
    sanitize_filename,
    write_bytes,
    cleanup_job_tmp,
    experiment_dir,
)
from .validation import validate_image_bytes, validate_batch_config, detect_duplicates, MAX_BATCH_IMAGES
from .aggregation import aggregate_job
from .exporters import export_csv, export_json, export_zip


class BatchManager:
    def __init__(self, default_workers: int = 2):
        self._jobs: Dict[str, BatchJob] = {}
        self._lock = threading.RLock()
        self._queue = WorkQueue()
        self._workers = default_workers
        self._pool = WorkerPool(
            queue=self._queue,
            get_job=self._get_job,
            on_item_done=self._on_item_done,
            n_workers=default_workers,
        )
        self._pool.start()

    def _get_job(self, job_id: str) -> Optional[BatchJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def _on_item_done(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.recompute_counts()
            pending = sum(
                1
                for i in job.items
                if i.status in (ItemStatus.QUEUED.value, ItemStatus.PROCESSING.value)
            )
            if pending == 0 and job.status == JobStatus.RUNNING.value:
                job.finalize_status()
            save_job_meta(job.to_dict(include_item_results=True))

    def create_encode_job(
        self,
        files: List[Tuple[str, bytes]],
        config: Dict[str, Any],
        per_image_messages: Optional[Dict[str, str]] = None,
    ) -> BatchJob:
        """
        files: list of (filename, raw_bytes)
        config: includes secret_text, passphrase, strategy, workers, etc.
        """
        cfg_errors = validate_batch_config({**config, "type": "encode"}, len(files))
        if cfg_errors:
            raise ValueError("; ".join(cfg_errors))

        dups = detect_duplicates([f[0] for f in files])
        if dups:
            raise ValueError(f"Duplicate filenames: {', '.join(dups[:5])}")

        workers = int(config.get("workers") or self._workers)
        job = BatchJob(
            type=JobType.ENCODE.value,
            status=JobStatus.QUEUED.value,
            workers=workers,
            configuration={
                k: v
                for k, v in config.items()
                if k not in ("passphrase", "secret_text")
            },
        )
        # keep secrets only in-memory on the job object
        job._full_config = dict(config)  # type: ignore

        tmp = job_tmp_dir(job.job_id)
        items: List[BatchItem] = []
        mode = config.get("message_mode") or "same"

        for fname, data in files[:MAX_BATCH_IMAGES]:
            ok, err, meta = validate_image_bytes(data, fname)
            safe = sanitize_filename(fname)
            item = BatchItem(filename=fname, safe_filename=safe)
            if not ok:
                item.status = ItemStatus.FAILED.value
                item.error = err
                items.append(item)
                continue
            in_path = tmp / "input" / safe
            out_path = tmp / "output" / safe
            write_bytes(in_path, data)
            item.input_path = str(in_path)
            item.output_path = str(out_path)
            if mode == "per_image" and per_image_messages:
                item.secret_text = per_image_messages.get(fname) or per_image_messages.get(safe)
                if not item.secret_text:
                    item.status = ItemStatus.FAILED.value
                    item.error = "Missing per-image message"
            items.append(item)

        job.items = items
        job.recompute_counts()
        job.experiment_id = f"experiment_{job.job_id}"

        with self._lock:
            self._jobs[job.job_id] = job
            save_job_meta(job.to_dict(include_item_results=False))

        # enqueue only non-failed
        self._queue.put_many(job.job_id, [i for i in items if i.status == ItemStatus.QUEUED.value])
        job.status = JobStatus.RUNNING.value
        from datetime import datetime, timezone

        job.started_at = datetime.now(timezone.utc).isoformat()
        save_job_meta(job.to_dict(include_item_results=False))
        return job

    def create_decode_job(
        self,
        files: List[Tuple[str, bytes]],
        config: Dict[str, Any],
    ) -> BatchJob:
        cfg_errors = validate_batch_config({**config, "type": "decode"}, len(files))
        if cfg_errors:
            raise ValueError("; ".join(cfg_errors))

        workers = int(config.get("workers") or self._workers)
        job = BatchJob(
            type=JobType.DECODE.value,
            status=JobStatus.QUEUED.value,
            workers=workers,
            configuration={k: v for k, v in config.items() if k not in ("passphrase", "secret_text")},
        )
        job._full_config = dict(config)  # type: ignore
        tmp = job_tmp_dir(job.job_id)
        items: List[BatchItem] = []

        for fname, data in files[:MAX_BATCH_IMAGES]:
            ok, err, meta = validate_image_bytes(data, fname)
            safe = sanitize_filename(fname)
            item = BatchItem(filename=fname, safe_filename=safe)
            if not ok:
                item.status = ItemStatus.FAILED.value
                item.error = err
                items.append(item)
                continue
            in_path = tmp / "input" / safe
            write_bytes(in_path, data)
            item.input_path = str(in_path)
            items.append(item)

        job.items = items
        job.recompute_counts()
        with self._lock:
            self._jobs[job.job_id] = job
        self._queue.put_many(job.job_id, [i for i in items if i.status == ItemStatus.QUEUED.value])
        job.status = JobStatus.RUNNING.value
        from datetime import datetime, timezone

        job.started_at = datetime.now(timezone.utc).isoformat()
        save_job_meta(job.to_dict(include_item_results=False))
        return job

    def create_experiment_job(
        self,
        files: List[Tuple[str, bytes]],
        config: Dict[str, Any],
    ) -> BatchJob:
        """
        Expand strategies × bpp_list × images into individual items.
        Still one queue; each item calls the existing pipeline once.
        """
        strategies = config.get("strategies") or [config.get("strategy") or "cnn_emd_opap"]
        bpp_list = config.get("bpp_list") or [0.1]
        # For experiment mode we still need a base message; length is adjusted per bpp in worker via strategy
        base_msg = config.get("secret_text") or "SecureStegVault experiment payload."
        expanded_files: List[Tuple[str, bytes]] = []
        tags: List[Dict[str, Any]] = []
        for fname, data in files:
            for strat in strategies:
                for bpp in bpp_list:
                    tag_name = f"{Path(fname).stem}__{strat}__bpp{bpp}{Path(fname).suffix}"
                    expanded_files.append((tag_name, data))
                    tags.append({"strategy": strat, "bpp_target": float(bpp), "orig": fname})

        # message length hint via config
        cfg = {**config, "secret_text": base_msg, "message_mode": "same"}
        job = self.create_encode_job(expanded_files, cfg)
        for item, tag in zip(job.items, tags):
            item.strategy = tag["strategy"]
            item.bpp_target = tag["bpp_target"]
        job.type = JobType.EXPERIMENT.value
        job.configuration["strategies"] = strategies
        job.configuration["bpp_list"] = bpp_list
        save_job_meta(job.to_dict(include_item_results=False))
        return job

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.recompute_counts()
                return job.to_dict(include_item_results=True)
        meta = load_job_meta(job_id)
        return meta

    def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            ids = list(self._jobs.keys())[-limit:]
            out = [self._jobs[i].to_dict(include_item_results=False) for i in reversed(ids)]
        # also surface persisted
        for jid in list_job_ids()[:limit]:
            if jid not in self._jobs:
                m = load_job_meta(jid)
                if m:
                    out.append(m)
        return out[:limit]

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            job._cancel_requested = True
            removed = self._queue.cancel_job(job_id)
            for it in job.items:
                if it.status == ItemStatus.QUEUED.value:
                    it.status = ItemStatus.CANCELLED.value
            job.recompute_counts()
            pending = sum(
                1 for i in job.items if i.status == ItemStatus.PROCESSING.value
            )
            if pending == 0:
                job.finalize_status()
            save_job_meta(job.to_dict(include_item_results=True))
            return job.to_dict(include_item_results=False)

    def retry_failed(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            retried = []
            for it in job.items:
                if it.status == ItemStatus.FAILED.value:
                    it.status = ItemStatus.QUEUED.value
                    it.error = None
                    it.result = None
                    it.progress = 0
                    retried.append(it)
            if not retried:
                return job.to_dict(include_item_results=False)
            job.status = JobStatus.RUNNING.value
            job._cancel_requested = False
            job.recompute_counts()
            self._queue.put_many(job_id, retried)
            save_job_meta(job.to_dict(include_item_results=False))
            return job.to_dict(include_item_results=False)

    def summary(self, job_id: str) -> Dict[str, Any]:
        data = self.get_job(job_id)
        if not data:
            raise KeyError(job_id)
        return aggregate_job(data)

    def export(self, job_id: str, fmt: str = "json") -> Any:
        data = self.get_job(job_id)
        if not data:
            raise KeyError(job_id)
        if fmt == "csv":
            return export_csv(data)
        if fmt == "zip":
            return str(export_zip(data))
        return export_json(data)

    def cleanup(self, job_id: str) -> None:
        cleanup_job_tmp(job_id)


_MANAGER: Optional[BatchManager] = None
_MANAGER_LOCK = threading.Lock()


def get_batch_manager(workers: int = 2) -> BatchManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            # auto workers: min(4, cpu)
            auto = min(4, max(1, (os.cpu_count() or 2)))
            _MANAGER = BatchManager(default_workers=workers or auto)
        return _MANAGER
