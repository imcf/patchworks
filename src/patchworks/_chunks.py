"""Auto tile-shape estimation."""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def auto_overlap(diameter: float, safety: float = 1.0) -> int:
    """Recommended overlap (halo) for a given cell diameter.

    Rule: overlap >= diameter so the segmentation function always sees at
    least one full cell's worth of context on every tile edge. Cells near
    tile boundaries are then segmented correctly and only genuinely split
    cells produce touching labels at the boundary → correct merge.

    Parameters
    ----------
    diameter:
        Expected cell diameter in pixels (same unit as your image).
    safety:
        Multiplier on top of diameter. Default 1.0 (= one cell width).
        Use 1.5–2.0 for elongated or irregularly-shaped cells.

    Returns
    -------
    int
        Overlap depth to pass to ``tile_process(..., overlap=...)``.

    Examples
    --------
    >>> from patchworks import auto_overlap, tile_process
    >>> from patchworks.plugins.cellpose import cellpose_fn
    >>>
    >>> fn = cellpose_fn("cyto3", gpu=True, diameter=30)
    >>> result = tile_process("image.zarr", fn,
    ...                       tile_shape=(1, 2048, 2048),
    ...                       overlap=auto_overlap(30))
    """
    return max(1, int(np.ceil(diameter * safety)))


_GPU_MEMORY_FALLBACK = 8 * 1024**3


def _get_available_memory() -> int:
    """Return available system RAM in bytes.

    Returns
    -------
    int
        Available memory via ``psutil``, or an 8 GiB fallback if it is not
        installed.
    """
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return 8 * 1024**3


def safe_worker_count(
    tile_nbytes: int,
    *,
    use_gpu: bool = False,
    fn_overhead: int = 4,
    ram_fraction: float = 0.8,
) -> int:
    """Concurrent tiles that fit the machine without OOM or a CPU freeze.

    Bounds the threaded scheduler by two limits and takes the smaller:

    * **CPU** — leaves at least one core free so the box stays responsive
      (never pins every core).
    * **RAM** — at most ``ram_fraction`` of available memory, assuming each
      in-flight tile needs ``fn_overhead`` copies (halo + output + temporaries).

    On GPU the answer is always 1: one evaluation at a time so concurrent
    tiles can never exhaust VRAM. Without ``psutil`` it returns a conservative
    default rather than guessing high.

    Parameters
    ----------
    tile_nbytes : int
        Size of one tile in bytes (``prod(tile_shape) * dtype.itemsize``).
    use_gpu : bool, optional
        Whether tiles are processed on the GPU.
    fn_overhead : int, optional
        Assumed peak number of tile-sized buffers alive per worker.
    ram_fraction : float, optional
        Fraction of available RAM the staging step may use.

    Returns
    -------
    int
        Worker-thread count (always >= 1).
    """
    cpu_cap = max(1, (os.cpu_count() or 1) - 1)
    if use_gpu:
        return 1
    avail = _get_available_memory()
    per_tile = max(1, int(tile_nbytes) * max(1, fn_overhead))
    mem_cap = max(1, int(avail * ram_fraction) // per_tile)
    return max(1, min(cpu_cap, mem_cap))


def _get_gpu_memory() -> int:
    """Return free GPU VRAM in bytes.

    Returns
    -------
    int
        Free VRAM of GPU 0 via ``nvidia-ml-py``, or an 8 GiB fallback if the
        query fails.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return int(info.free)
    except Exception:
        logger.warning(
            "GPU memory query failed (nvidia-ml-py not installed?); "
            "using %.0f GiB default.",
            _GPU_MEMORY_FALLBACK / 1024**3,
        )
        return _GPU_MEMORY_FALLBACK


def auto_tile_shape(
    shape: tuple[int, ...],
    dtype: Any,
    target_bytes: int = 64 * 1024**2,
    use_gpu: bool = False,
    gpu_memory: int | None = None,
    available_memory: int | None = None,
    n_workers: int | None = None,
    verbose: bool = False,
) -> tuple[int, ...]:
    """Balanced tile shape for general-purpose 3-D processing.

    Sizes the last three axes (spatial) to stay within the memory budget while
    keeping the shape as cubic as possible. Leading axes (t, c) are always 1.

    Parameters
    ----------
    shape:
        Full array shape, e.g. ``(z, y, x)`` or ``(t, c, z, y, x)``.
    dtype:
        Array dtype.
    target_bytes:
        Memory ceiling per tile. Default 64 MiB.
    use_gpu:
        Size tiles against GPU VRAM rather than host RAM.
    gpu_memory:
        Available GPU VRAM in bytes; auto-queried when None.
    available_memory:
        Available host RAM in bytes; auto-queried when None.
    n_workers:
        Number of parallel workers (divides the RAM budget).
    verbose:
        Log the chosen shape and estimated tile size.

    Returns
    -------
    tuple[int, ...]
        Tile shape with the same number of dimensions as *shape*.

    Examples
    --------
    >>> tile = auto_tile_shape((128, 2048, 2048), "uint16")
    >>> tile
    (128, 512, 512)
    """
    n_workers = n_workers or os.cpu_count() or 1
    itemsize = np.dtype(dtype).itemsize
    n_spatial = min(3, len(shape))

    if use_gpu:
        mem = gpu_memory if gpu_memory is not None else _get_gpu_memory()
        budget = min(target_bytes * 2, mem // 2)
    else:
        mem = available_memory or _get_available_memory()
        budget = min(target_bytes, mem // (n_workers * 4))

    budget = max(32 * 1024**2, budget)

    leading = [1] * (len(shape) - n_spatial)
    spatial = list(shape[-n_spatial:])
    target_voxels = budget / itemsize
    target_side = int(target_voxels ** (1.0 / n_spatial))
    chunk_spatial = [min(s, target_side) for s in spatial]

    capped = [
        i for i, (c, s) in enumerate(zip(chunk_spatial, spatial)) if c == s
    ]
    uncapped = [i for i in range(n_spatial) if i not in capped]
    if uncapped:
        used_by_capped = (
            np.prod([chunk_spatial[i] for i in capped]) if capped else 1
        )
        remaining = target_voxels / max(1, used_by_capped)
        new_side = int(remaining ** (1.0 / len(uncapped)))
        for i in uncapped:
            chunk_spatial[i] = min(spatial[i], new_side)

    result = tuple(leading + chunk_spatial)

    if verbose:
        mib = np.prod(result) * itemsize / 1024**2
        logger.info(
            "auto_tile_shape: shape=%s dtype=%s → tiles=%s (~%.0f MiB/tile)",
            shape,
            np.dtype(dtype).name,
            result,
            mib,
        )

    return result


def auto_tile_shape_cellpose(
    shape: tuple[int, ...],
    dtype: Any,
    diameter: float | None = None,
    do_3D: bool = False,
    use_gpu: bool = False,
    gpu_memory: int | None = None,
    available_memory: int | None = None,
    n_workers: int | None = None,
    model_memory_bytes: int = 2 * 1024**3,
    cellpose_memory_factor: int = 20,
    verbose: bool = False,
) -> tuple[int, ...]:
    """Cellpose-optimised tile shape.

    Cellpose is fundamentally 2-D: even in 3-D mode it runs 2-D segmentation
    on orthogonal planes and takes a consensus.

    **do_3D=False (default)**
        z is set to 1. Each tile is one 2-D ``(y, x)`` slice.

    **do_3D=True**
        z is kept at its full extent per tile. y and x are tiled based on the
        available memory, accounting for the 3× overhead of three plane orientations.

    Parameters
    ----------
    shape:
        Spatial shape, e.g. ``(z, y, x)``.
    dtype:
        Array dtype.
    diameter:
        Expected cell diameter in pixels. Tile will be at least ``4 × diameter``.
    do_3D:
        Whether Cellpose will run in 3-D mode.
    use_gpu:
        Size tiles for GPU VRAM.
    gpu_memory, available_memory, n_workers:
        Memory parameters (auto-queried when None).
    model_memory_bytes:
        Memory consumed by the Cellpose model weights (default 2 GiB).
    cellpose_memory_factor:
        Cellpose allocates roughly this multiple of raw input bytes (default 20×).
    verbose:
        Log the chosen shape and memory estimates.

    Returns
    -------
    tuple[int, ...]
        Tile shape with the same number of dimensions as *shape*.

    Examples
    --------
    >>> tile = auto_tile_shape_cellpose((128, 2048, 2048), "uint16", diameter=30)
    >>> tile
    (1, 2048, 2048)
    """
    n_workers = n_workers or os.cpu_count() or 1
    itemsize = np.dtype(dtype).itemsize

    if use_gpu:
        total_mem = gpu_memory if gpu_memory is not None else _get_gpu_memory()
    else:
        total_mem = (available_memory or _get_available_memory()) // n_workers

    usable = max(32 * 1024**2, total_mem - model_memory_bytes)
    max_raw_bytes = usable // cellpose_memory_factor

    n_spatial = min(3, len(shape))
    leading = [1] * (len(shape) - n_spatial)
    min_tile = int(4 * diameter) if diameter is not None else 1

    if n_spatial == 2 or not do_3D:
        max_pixels_2d = max(1, max_raw_bytes // itemsize)
        tile_side = max(min_tile, int(max_pixels_2d**0.5))
        if n_spatial == 2:
            y, x = shape[-2], shape[-1]
            chunk_spatial = [min(y, tile_side), min(x, tile_side)]
        else:
            z, y, x = shape[-3], shape[-2], shape[-1]
            chunk_spatial = [1, min(y, tile_side), min(x, tile_side)]
    else:
        z, y, x = shape[-3], shape[-2], shape[-1]
        max_pixels_per_slice = max(1, (max_raw_bytes // 3) // (z * itemsize))
        tile_side = max(min_tile, int(max_pixels_per_slice**0.5))
        chunk_spatial = [z, min(y, tile_side), min(x, tile_side)]

    result = tuple(leading + chunk_spatial)

    if verbose:
        raw_mib = np.prod(result) * itemsize / 1024**2
        logger.info(
            "auto_tile_shape_cellpose: shape=%s dtype=%s do_3D=%s "
            "→ tiles=%s (~%.0f MiB raw, ~%.0f MiB Cellpose estimate)",
            shape,
            np.dtype(dtype).name,
            do_3D,
            result,
            raw_mib,
            raw_mib * cellpose_memory_factor,
        )

    return result
