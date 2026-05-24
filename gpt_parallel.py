"""Run independent GPT batch jobs concurrently; merge results under a lock."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")
_LockType = type(threading.Lock())


def gpt_worker_count(default: int = 4) -> int:
    raw = (
        (os.environ.get("EDGE_GPT_WORKERS") or "")
        or (os.environ.get("GPT_WORKER_THREADS") or "")
    ).strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def run_parallel_batches(
    batches: list[T],
    *,
    worker: Callable[[int, T], list[R]],
    on_results: Callable[[list[R]], None],
    progress: Callable[[str], None],
    label: str = "GPT",
    workers: Optional[int] = None,
    lock: Optional[_LockType] = None,
) -> None:
    """Call ``worker`` for each batch in parallel; merge via ``on_results`` under ``lock``."""
    if not batches:
        return

    n_workers = workers if workers is not None else gpt_worker_count()
    n_workers = max(1, min(n_workers, len(batches)))
    merge_lock = lock or threading.Lock()
    total = len(batches)

    def run_one(batch_idx: int, batch: T) -> list[R]:
        return worker(batch_idx, batch)

    if n_workers == 1:
        for batch_idx, batch in enumerate(batches):
            results = run_one(batch_idx, batch)
            with merge_lock:
                on_results(results)
            progress(f"{label} batch {batch_idx + 1}/{total} complete.")
        return

    progress(f"{label}: {total} batch(es) across {n_workers} worker thread(s).")
    completed = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(run_one, i, batch) for i, batch in enumerate(batches)]
        for future in as_completed(futures):
            results = future.result()
            with merge_lock:
                on_results(results)
                completed += 1
                progress(f"{label} batch {completed}/{total} complete.")
