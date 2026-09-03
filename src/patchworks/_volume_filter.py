"""Drop label objects below a volume threshold, in place, after merge.

Meant to run once, globally, on the fully merged label array -- not per
tile, where an object's true size isn't known yet (a tile only sees
whatever fragment of it landed inside that tile's bounds, so a per-tile
filter would clip or drop objects that are only small *within one tile*).

Two-pass streaming algorithm, mirroring :func:`patchworks.relabel_sequential_zarr`
-- safe for arrays far larger than RAM. Pass 1 does a chunk-wise
unique+count to get every label's voxel count (bounded memory: a Python
dict keyed by label id, not the voxels themselves). Pass 2 builds a LUT
that zeroes labels under the threshold -- optionally renumbering the
survivors to a contiguous range in the same pass -- and applies it chunk by
chunk, writing back into the same store.
"""

from __future__ import annotations

import logging
import math
from itertools import product as _iproduct

import numpy as np
import zarr

logger = logging.getLogger(__name__)

_LUT_WARN_THRESHOLD = 100_000_000  # warn when max_label > 100 M (LUT > 800 MB)


def voxel_volume(voxel_size: "dict[str, float]") -> float:
    """Physical volume of one voxel, from a per-axis calibration.

    Axes missing from *voxel_size* are treated as 1.0 -- e.g. a 2-D
    calibration with no ``z`` gives an area, not a bogus volume shrunk by a
    fake axis. Units follow whatever *voxel_size* is in (micrometers for
    :func:`patchworks.plugins.ome_zarr.read_pixel_size`).

    Parameters
    ----------
    voxel_size : dict
        Per-axis physical size, e.g. ``{"z": .., "y": .., "x": ..}``.

    Returns
    -------
    float
        Product of the given axis sizes.

    Examples
    --------
    >>> voxel_volume({"z": 0.24, "y": 0.10833, "x": 0.10833})
    0.0028164933359999997
    """
    vol = 1.0
    for size in voxel_size.values():
        vol *= size
    return vol


def min_voxels_for_volume(
    min_volume: float, voxel_size: "dict[str, float]"
) -> int:
    """Convert a physical volume threshold to a voxel count.

    Rounds up: an object must reach *min_volume* to survive, so a partial
    voxel's worth of extra volume should not tip it over the line.

    Parameters
    ----------
    min_volume : float
        Minimum object volume to keep, in the same physical units as
        *voxel_size* (micrometers³ for an NGFF calibration).
    voxel_size : dict
        Per-axis physical size -- see :func:`voxel_volume`.

    Returns
    -------
    int
        Minimum voxel count for an object to survive filtering.

    Examples
    --------
    >>> min_voxels_for_volume(5.0, {"z": 0.24, "y": 0.10833, "x": 0.10833})
    1776
    """
    return math.ceil(min_volume / voxel_volume(voxel_size))


def _chunk_slices(shape, chunks):
    """Every zarr chunk's index expression, in all dimensions.

    Iterating actual chunk boundaries (rather than z-slabs) keeps each read
    bounded to one chunk's worth of memory, whatever the array's shape.
    """
    n_per_dim = [(s + c - 1) // c for s, c in zip(shape, chunks)]
    return [
        tuple(
            slice(i * c, min((i + 1) * c, s))
            for i, c, s in zip(idx, chunks, shape)
        )
        for idx in _iproduct(*[range(n) for n in n_per_dim])
    ]


def filter_labels_by_size(
    store_path: str,
    component: str,
    min_voxels: int,
    *,
    relabel: bool = True,
) -> "tuple[int, int]":
    """Drop label objects smaller than *min_voxels*, in place.

    Two-pass streaming scan (see module docstring) -- the array never has
    to fit in RAM.

    Parameters
    ----------
    store_path : str
        Path to the zarr store containing the label array.
    component : str
        Array name inside the store to filter in place.
    min_voxels : int
        Objects with fewer voxels than this are zeroed (dropped). Use
        :func:`min_voxels_for_volume` to derive this from a physical
        volume and calibration.
    relabel : bool, optional
        Renumber the surviving objects to a contiguous ``1..N`` range in
        the same LUT that drops the small ones (default ``True``) --
        otherwise the removed ids leave permanent gaps and survivors keep
        their original ids.

    Returns
    -------
    tuple of int
        ``(n_kept, n_removed)``.

    Examples
    --------
    >>> import zarr
    >>> root = zarr.open_group("labels.zarr", mode="w")  # doctest: +SKIP
    >>> root.create_array(
    ...     "labels", shape=(4, 4), chunks=(4, 4), dtype="int32"
    ... )[:] = [
    ...     [0, 1, 1, 0],
    ...     [0, 1, 1, 0],
    ...     [0, 0, 0, 2],
    ...     [0, 0, 0, 0],
    ... ]  # doctest: +SKIP
    >>> filter_labels_by_size("labels.zarr", "labels", min_voxels=2)  # doctest: +SKIP
    (1, 1)
    """
    root = zarr.open_group(store_path, mode="r+")
    z = root[component]
    slices = _chunk_slices(z.shape, z.chunks)

    counts: "dict[int, int]" = {}
    for sl in slices:
        ids, n = np.unique(np.asarray(z[sl]), return_counts=True)
        for label_id, count in zip(ids.tolist(), n.tolist()):
            if label_id == 0:
                continue
            counts[label_id] = counts.get(label_id, 0) + count

    kept = sorted(i for i, c in counts.items() if c >= min_voxels)
    n_kept = len(kept)
    n_removed = len(counts) - n_kept

    # Sized to the largest id *seen*, not just the largest surviving one --
    # a removed object's id can still exceed every kept id and must stay
    # in bounds so the LUT gather below maps it to 0 rather than indexing
    # past the end.
    max_label = max(counts) if counts else 0
    if max_label > _LUT_WARN_THRESHOLD:
        logger.warning(
            "filter_labels_by_size: max_label=%d -> LUT size ~%.0f MB.",
            max_label,
            max_label * 8 / 1024**2,
        )
    lut = np.zeros(max_label + 1, dtype=np.int64)
    if kept:
        lut[kept] = np.arange(1, n_kept + 1) if relabel else np.asarray(kept)

    max_out = n_kept if relabel else max_label
    out_dtype = np.uint16 if max_out < np.iinfo(np.uint16).max else np.uint32
    for sl in slices:
        block = np.asarray(z[sl])
        z[sl] = lut[block].astype(out_dtype)

    logger.info(
        "filter_labels_by_size: dropped %d/%d object(s) under %d voxels, "
        "%d remain",
        n_removed,
        len(counts),
        min_voxels,
        n_kept,
    )
    return n_kept, n_removed
