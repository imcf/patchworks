"""Linear sequential relabelling (O(voxels), not O(n_chunks²))."""
from __future__ import annotations

import logging

import numpy as np
import zarr

logger = logging.getLogger(__name__)


def relabel_sequential_array(labels: np.ndarray) -> np.ndarray:
    """Remap *labels* to a contiguous ``0, 1, … N`` range.

    Background (0) stays 0. Runs in one ``np.unique`` + a lookup-table gather,
    i.e. O(voxels) — unlike dask's ``relabel_sequential`` which is O(n_chunks²).

    Examples
    --------
    >>> relabel_sequential_array(np.array([0, 500000, 500000, 7]))
    array([0, 2, 2, 1])
    """
    uniq = np.unique(labels)
    lut = np.zeros(int(uniq[-1]) + 1, dtype=np.int64)
    lut[uniq] = np.arange(uniq.size)
    out = lut[labels]
    n = uniq.size - 1 if uniq[0] == 0 else uniq.size
    dtype = np.uint16 if n < np.iinfo(np.uint16).max else np.uint32
    return out.astype(dtype)


def relabel_sequential_zarr(store_path: str, component: str = "labels") -> int:
    """Relabel a written label zarr to contiguous ids, in place. Returns N.

    Two-pass streaming algorithm — safe for arrays far larger than RAM.
    Pass 1 collects unique ids (bounded memory: a set). Pass 2 applies the
    lookup-table remap chunk by chunk.
    """
    root = zarr.open_group(store_path, mode="r+")
    z = root[component]
    uniq: set[int] = set()
    z_shape, z_chunks = z.shape, z.chunks
    step = z_chunks[0] if z_chunks else z_shape[0]
    for i0 in range(0, z_shape[0], step):
        uniq.update(np.unique(z[i0:i0 + step]).tolist())
    sorted_ids = np.array(sorted(uniq), dtype=np.int64)
    lut = np.zeros(int(sorted_ids[-1]) + 1, dtype=np.int64)
    lut[sorted_ids] = np.arange(sorted_ids.size)
    n = sorted_ids.size - 1 if sorted_ids[0] == 0 else sorted_ids.size
    for i0 in range(0, z_shape[0], step):
        block = z[i0:i0 + step]
        z[i0:i0 + step] = lut[block].astype(z.dtype)
    logger.info("relabel_sequential_zarr: %d objects renumbered to 1..%d", n, n)
    return int(n)
