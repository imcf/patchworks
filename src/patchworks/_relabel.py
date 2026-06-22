"""Linear sequential relabelling (O(voxels), not O(n_chunks²))."""

from __future__ import annotations

import logging
from itertools import product as _iproduct

import numpy as np
import zarr

logger = logging.getLogger(__name__)


_LUT_WARN_THRESHOLD = 100_000_000  # warn when max_label > 100 M (LUT > 800 MB)


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
    max_label = int(uniq[-1])
    if max_label > _LUT_WARN_THRESHOLD:
        logger.warning(
            "relabel_sequential_array: max_label=%d → LUT size ~%.0f MB. "
            "Consider using write_to= so labels never need to be in RAM.",
            max_label,
            max_label * 8 / 1024**2,
        )
    lut = np.zeros(max_label + 1, dtype=np.int64)
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
    z_shape, z_chunks = z.shape, z.chunks

    # Iterate over actual zarr chunks in ALL dimensions. The z-slab approach
    # (step = z_chunks[0], slice z[i0:i0+step]) reads the full y/x extent per
    # step — for chunks like (120, 731, 731) that means (120, 37888, 27392)
    # = 464 GiB in one allocation (MemoryError).
    n_per_dim = [(s + c - 1) // c for s, c in zip(z_shape, z_chunks)]
    chunk_slices = [
        tuple(
            slice(i * c, min((i + 1) * c, s))
            for i, c, s in zip(idx, z_chunks, z_shape)
        )
        for idx in _iproduct(*[range(n) for n in n_per_dim])
    ]

    uniq: set[int] = set()
    for sl in chunk_slices:
        uniq.update(np.unique(np.asarray(z[sl])).tolist())
    sorted_ids = np.array(sorted(uniq), dtype=np.int64)
    max_label = int(sorted_ids[-1])
    if max_label > _LUT_WARN_THRESHOLD:
        logger.warning(
            "relabel_sequential_zarr: max_label=%d → LUT size ~%.0f MB.",
            max_label,
            max_label * 8 / 1024**2,
        )
    lut = np.zeros(max_label + 1, dtype=np.int64)
    lut[sorted_ids] = np.arange(sorted_ids.size)
    n = sorted_ids.size - 1 if sorted_ids[0] == 0 else sorted_ids.size
    # Use same dtype logic as relabel_sequential_array so output never overflows.
    out_dtype = np.uint16 if n < np.iinfo(np.uint16).max else np.uint32
    for sl in chunk_slices:
        block = np.asarray(z[sl])
        z[sl] = lut[block].astype(out_dtype)
    logger.info("relabel_sequential_zarr: %d objects renumbered to 1..%d", n, n)
    return int(n)
