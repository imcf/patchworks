"""Auto tile-shape estimation."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)


def auto_overlap(
    diameter: float,
    safety: float = 1.0,
    voxel_size: Union[Sequence[float], None] = None,
) -> Union[int, tuple[int, ...]]:
    """Recommended overlap (halo) for a given cell diameter.

    Rule: overlap >= diameter so the segmentation function always sees at
    least one full cell's worth of context on every tile edge. Cells near
    tile boundaries are then segmented correctly and only genuinely split
    cells produce touching labels at the boundary → correct merge.

    With *voxel_size* the halo is returned per axis instead of as one number.
    That matters on anisotropic stacks: a halo big enough laterally is far
    more than one cell deep in z, and the extra planes are read and
    segmented only to be trimmed away again.

    Parameters
    ----------
    diameter:
        Expected cell diameter in **lateral** pixels (same unit as your
        image's x/y).
    safety:
        Multiplier on top of diameter. Default 1.0 (= one cell width).
        Use 1.5–2.0 for elongated or irregularly-shaped cells.
    voxel_size:
        Physical size per axis, in any single unit (e.g. ``(2.0, 0.1, 0.1)``
        for a 2 µm z-step and 100 nm pixels). ``None`` → one isotropic
        number, as before.

    Returns
    -------
    int or tuple of int
        Overlap depth to pass to ``tile_process(..., overlap=...)``. A tuple
        (one entry per axis) when *voxel_size* is given.

    Examples
    --------
    >>> from patchworks import auto_overlap, tile_process
    >>> from patchworks.plugins.cellpose import cellpose_fn
    >>>
    >>> fn = cellpose_fn("cyto3", gpu=True, diameter=30)
    >>> result = tile_process("image.zarr", fn,
    ...                       tile_shape=(1, 2048, 2048),
    ...                       overlap=auto_overlap(30))
    >>> auto_overlap(15, voxel_size=(2.0, 0.1, 0.1))
    (1, 15, 15)
    """
    lateral = max(1, int(np.ceil(diameter * safety)))
    if voxel_size is None:
        return lateral
    # Convert the lateral halo to a physical distance, then back into pixels
    # along each axis using that axis' own voxel size.
    physical = diameter * safety * float(voxel_size[-1])
    return tuple(max(1, int(np.ceil(physical / float(v)))) for v in voxel_size)


_GPU_MEMORY_FALLBACK = 8 * 1024**3


def cpu_allocation() -> int:
    """Return the number of CPUs this process may actually use.

    ``os.cpu_count()`` reports the machine's cores, which on a shared cluster
    node is wildly more than a job was granted -- a 4-core allocation on a
    128-core node would size itself for 128. Prefer what the scheduler says,
    then the process' CPU affinity mask, and only then the machine.

    Returns
    -------
    int
        Usable CPU count (always >= 1).
    """
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        try:
            value = int(os.environ[var])
        except (KeyError, ValueError):
            continue
        if value > 0:
            return value
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # not POSIX
        return max(1, os.cpu_count() or 1)


def _cgroup_memory_limit() -> "int | None":
    """Memory ceiling from the process' cgroup, if one applies.

    SLURM confines jobs with cgroups, so this is the limit that actually gets
    the process OOM-killed -- unlike the node-wide figure ``psutil`` reports.
    """
    for path in (
        "/sys/fs/cgroup/memory.max",  # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
    ):
        try:
            raw = Path(path).read_text().strip()
        except OSError:
            continue
        if raw == "max":
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        # v1 reports a sentinel near 2**63 when unlimited.
        if 0 < value < 2**62:
            return value
    return None


def _get_available_memory() -> int:
    """Return the memory this process may actually use, in bytes.

    Takes the **smallest** of everything that can constrain it: the SLURM
    allocation, the cgroup limit, and the node's free RAM. A node with 512 GB
    free must never convince a 128 GB job that it has room -- that mismatch is
    exactly how a job sizes itself into an OOM kill.

    Returns
    -------
    int
        Usable memory in bytes, or an 8 GiB fallback if nothing is knowable.
    """
    limits = []

    per_node = os.environ.get("SLURM_MEM_PER_NODE")
    if per_node:
        try:
            limits.append(int(per_node) * 1024**2)  # SLURM reports MB
        except ValueError:
            pass
    per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
    if per_cpu:
        try:
            limits.append(int(per_cpu) * 1024**2 * cpu_allocation())
        except ValueError:
            pass

    cgroup = _cgroup_memory_limit()
    if cgroup is not None:
        limits.append(cgroup)

    try:
        import psutil

        limits.append(int(psutil.virtual_memory().available))
    except Exception:
        pass

    if not limits:
        return 8 * 1024**3
    return max(1, min(limits))


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
    cpu_cap = max(1, cpu_allocation() - 1)
    if use_gpu:
        return 1
    avail = _get_available_memory()
    per_tile = max(1, int(tile_nbytes) * max(1, fn_overhead))
    mem_cap = max(1, int(avail * ram_fraction) // per_tile)
    return max(1, min(cpu_cap, mem_cap))


# Leave room for a co-tenant: info.free is a point-in-time reading of a device
# we usually do not own outright, and sizing a tile to fill all of it is what
# turns another job's growth into our OOM.
_GPU_HEADROOM = 0.8


def _get_gpu_memory() -> int:
    """Return usable GPU VRAM in bytes for the device we were granted.

    NVML enumerates every GPU on the node regardless of ``--gres=gpu:1``, so
    asking for index 0 unconditionally reads the *wrong* device's free memory
    whenever SLURM granted anything but the first one -- and tile sizing was
    built on that number. Resolve the device from ``CUDA_VISIBLE_DEVICES``,
    then keep a headroom fraction rather than claiming every free byte.

    Returns
    -------
    int
        Usable VRAM, or an 8 GiB fallback if the query fails.
    """
    try:
        import pynvml

        from ._gpu import visible_device_index, visible_device_uuid

        pynvml.nvmlInit()
        uuid = visible_device_uuid()
        if uuid is not None:
            handle = pynvml.nvmlDeviceGetHandleByUUID(uuid.encode())
        else:
            handle = pynvml.nvmlDeviceGetHandleByIndex(visible_device_index())
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return int(int(info.free) * _GPU_HEADROOM)
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
    n_channels: int = 1,
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
    n_channels:
        Channels each tile carries (default 1). Above 1 the per-voxel cost
        scales with it, so the tile shrinks accordingly -- e.g. the workflow's
        ``nuclei_channel`` hands Cellpose a cyto+nuclei pair.
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
    n_workers = n_workers or cpu_allocation()
    # A tile holds n_channels planes per voxel (e.g. Cellpose's
    # cyto+nuclei pair), so the per-voxel cost -- and every budget
    # derived from it below -- scales with them.
    if n_channels < 1:
        raise ValueError(f"n_channels must be >= 1; got {n_channels!r}")
    itemsize = np.dtype(dtype).itemsize * n_channels
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
    n_channels: int = 1,
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
    n_channels:
        Channels each tile carries (default 1). Above 1 the per-voxel cost
        scales with it, so the tile shrinks accordingly -- e.g. the workflow's
        ``nuclei_channel`` hands Cellpose a cyto+nuclei pair.
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
    n_workers = n_workers or cpu_allocation()
    # A tile holds n_channels planes per voxel (e.g. Cellpose's
    # cyto+nuclei pair), so the per-voxel cost -- and every budget
    # derived from it below -- scales with them.
    if n_channels < 1:
        raise ValueError(f"n_channels must be >= 1; got {n_channels!r}")
    itemsize = np.dtype(dtype).itemsize * n_channels

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
