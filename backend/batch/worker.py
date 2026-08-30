"""Worker pool that executes single-image pipeline calls for batch items."""

from __future__ import annotations

import base64
import io
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
from PIL import Image

from .models import BatchJob, BatchItem, ItemStatus, JobStatus, JobType
from .queue import WorkQueue
from .storage import write_bytes


def _load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _save_png(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr.astype(np.uint8)).save(path, format="PNG")


def _encode_one(cover: np.ndarray, message: str, passphrase: str, cfg: Dict) -> Dict:
    """Call existing strategy pipeline — no algorithm duplication."""
    from backend.strategies import get_strategy

    strategy_name = cfg.get("strategy") or "cnn_emd_opap"
    strat = get_strategy(strategy_name)
    # Strategies set cost_map_mode / adversarial themselves; only pass non-conflicting knobs.
    kw = {}
    if "emd_n" in cfg:
        kw["emd_n"] = int(cfg["emd_n"])
    if "kb_bits" in cfg:
        kw["kb_bits"] = int(cfg["kb_bits"])
    if "kc_bits" in cfg:
        kw["kc_bits"] = int(cfg["kc_bits"])
    res = strat.embed(cover, message, passphrase, **kw)
    # Encode stego as PNG base64 for optional UI preview
    buf = io.BytesIO()
    Image.fromarray(res.stego.astype(np.uint8)).save(buf, format="PNG")
    stego_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {
        "success": True,
        "metrics": res.metrics or {},
        "security": res.security or {},
        "meta": res.meta or {},
        "stego_b64": stego_b64,
        "strategy": strategy_name,
    }


def _decode_one(stego: np.ndarray, passphrase: str, cfg: Dict) -> Dict:
    """
    Strategy.extract is not fully implemented for all hybrids.
    Fall back to the main FastAPI-style decode path via backend pipeline pieces.
    """
    from backend.crypto import decrypt_payload
    # Prefer full decode via main encode inverse when strategy.extract is stubbed
    try:
        from backend.strategies import get_strategy
        strategy_name = cfg.get("strategy") or "cnn_emd_opap"
        strat = get_strategy(strategy_name)
        text = strat.extract(stego, passphrase)
        return {"success": True, "decrypted_text": text, "strategy": strategy_name}
    except NotImplementedError:
        # Use main.py style: costmap + zones + extract (simplified path)
        from backend.cnn_costmap import compute_cnn_costmap
        from backend.zoning import ZoningConfig, classify_zones
        from backend.emd import extract_emd_zone_a, base5_digits_to_bytes, base7_digits_to_bytes
        from backend.opap import extract_opap_zone
        import numpy as np

        strategy_name = cfg.get("strategy") or "cnn_emd_opap"
        emd_n = int(cfg.get("emd_n", 2))
        kb = int(cfg.get("kb_bits", 2))
        kc = int(cfg.get("kc_bits", 3))
        cost_map_mode = "fast" if strategy_name == "emd_opap" else "cnn"
        cost = compute_cnn_costmap(stego, cost_map_mode=cost_map_mode)
        H, W, C = stego.shape
        cost3 = np.repeat(cost[:, :, np.newaxis], C, axis=2)
        config = ZoningConfig(emd_group_size=emd_n, kb_bits=kb, kc_bits=kc)
        zones = classify_zones(cost3, config)
        flat = stego.reshape(-1)
        # Capacity-driven extract length is unknown; strategies store length in payload header via crypto
        # Recover by extracting max and decrypting
        za = np.where(zones.reshape(-1) == 0)[0]
        zb = np.where(zones.reshape(-1) == 1)[0]
        zc = np.where(zones.reshape(-1) == 2)[0]
        digs = extract_emd_zone_a(flat, za, emd_n=emd_n)
        raw = base5_digits_to_bytes(digs) if emd_n == 2 else base7_digits_to_bytes(digs)
        # append OPAP bits as bytes-ish
        bits_b = extract_opap_zone(flat, zb, k=kb)
        bits_c = extract_opap_zone(flat, zc, k=kc)
        # crypto layer expects full encrypted blob; try progressive lengths
        for n in range(len(raw), 32, -1):
            try:
                text = decrypt_payload(raw[:n], passphrase)
                return {"success": True, "decrypted_text": text, "strategy": strategy_name}
            except Exception:
                continue
        raise RuntimeError("Decode failed: could not recover payload with given passphrase")


class WorkerPool:
    def __init__(
        self,
        queue: WorkQueue,
        get_job: Callable[[str], Optional[BatchJob]],
        on_item_done: Callable[[str], None],
        n_workers: int = 2,
    ):
        self.queue = queue
        self.get_job = get_job
        self.on_item_done = on_item_done
        self.n_workers = max(1, min(n_workers, 16))
        self._threads: list = []
        self._stop = threading.Event()
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._stop.clear()
        for i in range(self.n_workers):
            t = threading.Thread(target=self._loop, name=f"batch-worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()

    def set_workers(self, n: int) -> None:
        # Lightweight: only affects future pools; running pool size is fixed at start
        self.n_workers = max(1, min(n, 16))

    def _loop(self) -> None:
        while not self._stop.is_set():
            task = self.queue.get(timeout=0.4)
            if task is None:
                continue
            job_id, item_id = task
            job = self.get_job(job_id)
            if job is None:
                continue
            if job._cancel_requested:
                item = next((x for x in job.items if x.id == item_id), None)
                if item and item.status == ItemStatus.QUEUED.value:
                    item.status = ItemStatus.CANCELLED.value
                    self.on_item_done(job_id)
                continue
            if job._pause_requested:
                # re-queue later
                self.queue.put(job_id, item_id)
                time.sleep(0.3)
                continue
            self._process(job, item_id)

    def _process(self, job: BatchJob, item_id: str) -> None:
        item = next((x for x in job.items if x.id == item_id), None)
        if item is None or item.status not in (ItemStatus.QUEUED.value,):
            return
        item.status = ItemStatus.PROCESSING.value
        item.progress = 0.1
        item.started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        t0 = time.perf_counter()
        cfg = dict(job.configuration or {})
        # restore secrets held on job object if manager attached them
        full_cfg = getattr(job, "_full_config", cfg)

        try:
            if not item.input_path or not Path(item.input_path).is_file():
                raise RuntimeError("Input image missing")
            arr = _load_image(Path(item.input_path))
            item.progress = 0.3

            if job.type == JobType.DECODE.value:
                result = _decode_one(arr, full_cfg.get("passphrase", ""), full_cfg)
                item.result = result
            else:
                msg = item.secret_text if item.secret_text is not None else full_cfg.get("secret_text", "")
                if item.strategy:
                    full_cfg = {**full_cfg, "strategy": item.strategy}
                result = _encode_one(arr, msg, full_cfg.get("passphrase", ""), full_cfg)
                # write stego output
                stego_b64 = result.pop("stego_b64", None)
                if stego_b64 and item.output_path:
                    raw = base64.b64decode(stego_b64)
                    write_bytes(Path(item.output_path), raw)
                    result["output_file"] = item.output_path
                    result["stego_b64"] = stego_b64  # keep for single-item detail
                item.result = result

            item.progress = 1.0
            item.status = ItemStatus.COMPLETED.value
        except Exception as e:
            item.status = ItemStatus.FAILED.value
            item.error = str(e)[:500]
            item.progress = 0.0
            # avoid huge traces in stored state
            traceback.print_exc()
        finally:
            item.processing_time_s = round(time.perf_counter() - t0, 4)
            item.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            # free large arrays
            try:
                del arr  # type: ignore
            except Exception:
                pass
            self.on_item_done(job.job_id)
