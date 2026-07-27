"""Shared GPU helpers: device selection and surviving transient OOM.

Cluster GPUs are shared. A co-tenant job's footprint can grow mid-run and push
an otherwise-fine tile over the edge, so an out-of-memory error is often
transient rather than a sign the work does not fit. Retrying on the GPU beats
falling back to the CPU, where a single tile can take well over an hour.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

OOM_RETRIES = 4
OOM_BACKOFF_SECONDS = 30


def is_oom(exc: BaseException) -> bool:
    """Return whether *exc* is an out-of-memory error from any GPU stack.

    Matching only ``RuntimeError`` missed cupy entirely:
    ``cupy.cuda.memory.OutOfMemoryError`` derives from ``Exception``, not
    ``RuntimeError``, so the DoG plugin and ``dilate_labels(use_gpu=True)``
    got no retry at all. Class name is checked first because cupy's message
    reads "Out of memory allocating ..." while torch's reads "CUDA out of
    memory", and a future backend may word it differently again.

    Parameters
    ----------
    exc : BaseException
        The exception to classify.

    Returns
    -------
    bool
        True when the exception reports exhausted device memory.
    """
    if type(exc).__name__ in ("OutOfMemoryError", "CUDAOutOfMemoryError"):
        return True
    return "out of memory" in str(exc).lower()


def free_gpu_caches() -> None:
    """Release cached (but unused) device memory from torch and cupy.

    Both keep their own allocator pool and neither releases it on its own, so
    a process that has used either holds that memory for its lifetime. Freeing
    both also defragments, which is often what actually lets a retry succeed.
    Each is optional and skipped when not imported.
    """
    torch = __import__("sys").modules.get("torch")
    if torch is not None and getattr(torch, "cuda", None) is not None:
        try:
            torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - defensive
            logger.debug("torch.cuda.empty_cache() failed", exc_info=True)

    cupy = __import__("sys").modules.get("cupy")
    if cupy is not None:
        try:
            cupy.get_default_memory_pool().free_all_blocks()
            cupy.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:  # pragma: no cover - defensive
            logger.debug("cupy pool release failed", exc_info=True)


def retry_on_oom(
    call: Callable[[], Any],
    *,
    enabled: bool = True,
    on_release: Callable[[], None] | None = None,
    retries: int = OOM_RETRIES,
    backoff: int = OOM_BACKOFF_SECONDS,
) -> Any:
    """Run *call*, retrying with a backoff while the GPU is out of memory.

    Parameters
    ----------
    call : callable
        Zero-argument callable doing the GPU work.
    enabled : bool
        Set False on a CPU path so real errors surface immediately.
    on_release : callable, optional
        Called before each backoff, to drop anything big this process is
        holding on the device. Sleeping while still pinning a cached model is
        self-defeating -- that memory is exactly what the co-tenant needs.
    retries : int
        Retries after the first attempt.
    backoff : int
        Base seconds; the wait grows linearly with the attempt number.

    Returns
    -------
    Any
        Whatever *call* returns.
    """
    for attempt in range(retries + 1):
        try:
            return call()
        except Exception as exc:
            if not enabled or not is_oom(exc):
                raise
            if on_release is not None:
                on_release()
            free_gpu_caches()
            if attempt == retries:
                logger.error(
                    "GPU OOM persisted after %d retries; giving up.", retries
                )
                raise
            wait = backoff * (attempt + 1)
            logger.warning(
                "GPU OOM (likely contention on a shared device); released "
                "caches, retrying in %ds (attempt %d/%d).",
                wait,
                attempt + 1,
                retries,
            )
            time.sleep(wait)


def visible_device_index() -> int:
    """NVML index of the device this process is actually allowed to use.

    ``CUDA_VISIBLE_DEVICES`` remaps indices for the CUDA runtime but not for
    NVML, which always enumerates every GPU on the node. Querying NVML index 0
    unconditionally therefore reads a *different* GPU's free memory whenever
    SLURM granted anything other than the first one -- and tile sizing was
    built on that number.

    Returns
    -------
    int
        NVML device index, defaulting to 0 when nothing constrains us.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return 0
    first = visible.split(",")[0].strip()
    if not first or first.startswith("GPU-") or first.startswith("MIG-"):
        # A UUID form: NVML can resolve it directly, so leave index lookup
        # alone and let the caller fall back.
        return 0
    try:
        return max(0, int(first))
    except ValueError:
        return 0


def visible_device_uuid() -> "str | None":
    """UUID of the allowed device, when ``CUDA_VISIBLE_DEVICES`` uses one."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    first = visible.split(",")[0].strip() if visible else ""
    return first if first.startswith(("GPU-", "MIG-")) else None
