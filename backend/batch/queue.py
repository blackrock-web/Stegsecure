"""In-memory bounded work queue for batch items."""

from __future__ import annotations

import threading
from collections import deque
from typing import Deque, Optional, Tuple

from .models import BatchItem, ItemStatus


class WorkQueue:
    """Thread-safe FIFO of (job_id, item_id) pairs."""

    def __init__(self):
        self._q: Deque[Tuple[str, str]] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    def put(self, job_id: str, item_id: str) -> None:
        with self._not_empty:
            self._q.append((job_id, item_id))
            self._not_empty.notify()

    def put_many(self, job_id: str, items: list) -> None:
        with self._not_empty:
            for it in items:
                if it.status == ItemStatus.QUEUED.value:
                    self._q.append((job_id, it.id))
            self._not_empty.notify_all()

    def get(self, timeout: float = 0.5) -> Optional[Tuple[str, str]]:
        with self._not_empty:
            if not self._q:
                self._not_empty.wait(timeout=timeout)
            if not self._q:
                return None
            return self._q.popleft()

    def cancel_job(self, job_id: str) -> int:
        """Remove queued items for a job. Returns count removed."""
        with self._lock:
            kept = deque()
            removed = 0
            while self._q:
                jid, iid = self._q.popleft()
                if jid == job_id:
                    removed += 1
                else:
                    kept.append((jid, iid))
            self._q = kept
            return removed

    def size(self) -> int:
        with self._lock:
            return len(self._q)

    def clear(self) -> None:
        with self._lock:
            self._q.clear()
