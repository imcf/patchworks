"""Linear sequential relabelling (O(voxels), not O(n_chunks²))."""

from __future__ import annotations

import logging
import numpy as np
import zarr

from ._chunks import chunk_slices

logger = logging.getLogger(__name__)


_LUT_WARN_THRESHOLD = 100_000_000  # warn when max_label > 100 M (LUT > 800 MB)


def _sequential_lut(ids: np.ndarray) -> tuple[np.ndarray, int]:
    """LUT mapping the sorted distinct *ids* onto ``1..N`` (0 stays 0).

    Numbering from 1 whether or not 0 is among *ids*: counting from 0
    whenever background happened to be absent turned the smallest object
    into background.

    Parameters
    ----------
    ids : np.ndarray
        Sorted distinct label ids.

    Returns
    -------
    tuple
        ``(lut, n)`` -- the lookup table and the object count ``N``.
    """
    if ids.size and ids[0] < 0:
        raise ValueError(f"labels must be non-negative, found {int(ids[0])}")
    objects = ids[ids > 0]
    max_label = int(ids[-1]) if ids.size else 0
    lut = np.zeros(max_label + 1, dtype=np.int64)
    lut[objects] = np.arange(1, objects.size + 1)
    return lut, int(objects.size)


def relabel_sequential_array(labels: np.ndarray) -> np.ndarray:
    """Remap *labels* to a contiguous ``0, 1, … N`` range.

    Background (0) stays 0. Runs in one ``np.unique`` + a lookup-table gather,
    i.e. O(voxels) — unlike dask's ``relabel_sequential`` which is O(n_chunks²).

    Parameters
    ----------
    labels : np.ndarray
        Integer label array (may have gappy ids).

    Returns
    -------
    np.ndarray
        Labels remapped to a contiguous ``0, 1, … N`` range.

    Examples
    --------
    >>> relabel_sequential_array(np.array([0, 500000, 500000, 7]))
    array([0, 2, 2, 1], dtype=uint16)
    """
    uniq = np.unique(labels)
    max_label = int(uniq[-1]) if uniq.size else 0
    if max_label > _LUT_WARN_THRESHOLD:
        logger.warning(
            "relabel_sequential_array: max_label=%d → LUT size ~%.0f MB. "
            "Consider using write_to= so labels never need to be in RAM.",
            max_label,
            max_label * 8 / 1024**2,
        )
    lut, n = _sequential_lut(uniq)
    out = lut[labels]
    dtype = np.uint16 if n < np.iinfo(np.uint16).max else np.uint32
    return out.astype(dtype)


def relabel_sequential_zarr(store_path: str, component: str = "labels") -> int:
    """Relabel a written label zarr to contiguous ids, in place.

    Two-pass streaming algorithm — safe for arrays far larger than RAM.
    Pass 1 collects each chunk's unique ids (memory bounded by the id count,
    not the voxels themselves). Pass 2 applies the lookup-table remap chunk by
    chunk, writing back into the same store.

    Parameters
    ----------
    store_path : str
        Path to the zarr store containing the label array.
    component : str, optional
        Array name inside the store to relabel in place (default
        ``"labels"``).

    Returns
    -------
    int
        Number of distinct objects (``N``); the array now holds ``1..N``
        (background ``0`` unchanged).

    Examples
    --------
    >>> import zarr
    >>> root = zarr.open_group("staged.zarr", mode="w")  # doctest: +SKIP
    >>> root.create_array(
    ...     "labels", shape=(4, 4), chunks=(4, 4), dtype="int32"
    ... )[:] = [
    ...     [0, 500000, 500000, 0],
    ...     [0, 0, 0, 7],
    ...     [0, 0, 0, 0],
    ...     [0, 0, 0, 0],
    ... ]  # doctest: +SKIP
    >>> relabel_sequential_zarr("staged.zarr")  # doctest: +SKIP
    2
    """
    root = zarr.open_group(store_path, mode="r+")
    z = root[component]
    z_shape, z_chunks = z.shape, z.chunks

    # Iterate over actual zarr chunks in ALL dimensions. The z-slab approach
    # (step = z_chunks[0], slice z[i0:i0+step]) reads the full y/x extent per
    # step — for chunks like (120, 731, 731) that means (120, 37888, 27392)
    # = 464 GiB in one allocation (MemoryError).
    slices = chunk_slices(z_shape, z_chunks)

    # Per-chunk unique arrays, merged by one np.unique -- no Python set of
    # every id.
    sorted_ids = np.unique(
        np.concatenate(
            [np.unique(np.asarray(z[sl])).astype(np.int64) for sl in slices]
            or [np.empty(0, dtype=np.int64)]
        )
    )
    max_label = int(sorted_ids[-1]) if sorted_ids.size else 0
    if max_label > _LUT_WARN_THRESHOLD:
        logger.warning(
            "relabel_sequential_zarr: max_label=%d → LUT size ~%.0f MB.",
            max_label,
            max_label * 8 / 1024**2,
        )
    lut, n = _sequential_lut(sorted_ids)
    # Use same dtype logic as relabel_sequential_array so output never overflows.
    out_dtype = np.uint16 if n < np.iinfo(np.uint16).max else np.uint32
    for sl in slices:
        block = np.asarray(z[sl])
        z[sl] = lut[block].astype(out_dtype)
    logger.info("relabel_sequential_zarr: %d objects renumbered to 1..%d", n, n)
    return int(n)
