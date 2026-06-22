"""Dask cluster helpers."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _distributed_client():
    """Return the active dask.distributed Client, or None."""
    try:
        from dask.distributed import get_client

        return get_client()
    except Exception:
        return None


def _client_is_in_process(client) -> bool:
    """True if *client* runs its worker in this process (processes=False).

    An in-process worker shares the GIL. A long task that holds the GIL
    (e.g. a Cellpose/torch eval) starves the worker heartbeat, the scheduler
    declares it dead, and the P2P merge barrier drops its inputs →
    "FutureCancelledError: lost dependencies".
    """
    try:
        for addr in client.scheduler_info().get("workers", {}):
            if str(addr).startswith("inproc://"):
                return True
    except Exception:
        pass
    return False


def make_local_cluster(
    use_gpu: bool = False,
    n_workers: int | None = None,
    threads_per_worker: int = 1,
    memory_limit: str | None = None,
    **cluster_kwargs,
):
    """Create a process-based Dask cluster for tiled processing.

    Always uses worker subprocesses (``processes=True``). An in-process
    (threaded) worker breaks the label merge when ``segment_fn`` holds the
    GIL — see the patchworks docs for details.

    For GPU work defaults to a single worker (one CUDA context, no contention).
    For CPU scales to available cores.

    Parameters
    ----------
    use_gpu:
        Single-worker cluster for GPU. When False, use multiple CPU workers.
    n_workers:
        Override the worker count. Defaults to 1 for GPU, min(8, cpu_count).
    threads_per_worker:
        Keep at 1 so a GIL-holding tile function doesn't block heartbeats.
    memory_limit:
        Per-worker memory cap (e.g. ``"8GB"``).
    **cluster_kwargs:
        Extra arguments forwarded to ``dask.distributed.LocalCluster``.

    Returns
    -------
    (client, cluster)

    Examples
    --------
    >>> client, cluster = make_local_cluster(use_gpu=True)
    >>> print("dashboard:", client.dashboard_link)
    >>> result = tile_process("image.zarr", fn, write_to="labels.zarr")
    >>> client.close(); cluster.close()
    """
    from dask.distributed import Client, LocalCluster

    if n_workers is None:
        n_workers = 1 if use_gpu else min(8, os.cpu_count() or 1)

    cluster = LocalCluster(
        processes=True,
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        **cluster_kwargs,
    )
    client = Client(cluster)
    logger.info(
        "Started %d-worker process cluster (use_gpu=%s). Dashboard: %s",
        n_workers,
        use_gpu,
        client.dashboard_link,
    )
    return client, cluster
