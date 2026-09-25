"""Drop label objects outside a size range, in place, after merge.

Meant to run once, globally, on the fully merged label array -- not per
tile, where an object's true size isn't known yet (a tile only sees
whatever fragment of it landed inside that tile's bounds, so a per-tile
filter would clip or drop objects that are only small, or only large,
*within one tile*).

Two-pass streaming algorithm, mirroring :func:`patchworks.relabel_sequential_zarr`
-- safe for arrays far larger than RAM. Pass 1 does a chunk-wise
unique+count to get every label's voxel count (bounded memory: a Python
dict keyed by label id, not the voxels themselves). Pass 2 builds a LUT
that zeroes labels outside ``[min_voxels, max_voxels]`` -- optionally
renumbering the survivors to a contiguous range in the same pass -- and
applies it chunk by chunk, writing back into the same store.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import zarr

from ._chunks import chunk_slices

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


def max_voxels_for_volume(
    max_volume: float, voxel_size: "dict[str, float]"
) -> int:
    """Convert a physical volume threshold to a voxel count.

    Rounds down, the mirror image of :func:`min_voxels_for_volume`: an
    object must not *exceed* *max_volume*, so a voxel count whose volume
    would tip past it must not survive.

    Parameters
    ----------
    max_volume : float
        Maximum object volume to keep, in the same physical units as
        *voxel_size* (micrometers³ for an NGFF calibration).
    voxel_size : dict
        Per-axis physical size -- see :func:`voxel_volume`.

    Returns
    -------
    int
        Maximum voxel count for an object to survive filtering.

    Examples
    --------
    >>> max_voxels_for_volume(5.0, {"z": 0.24, "y": 0.10833, "x": 0.10833})
    1775
    """
    return math.floor(max_volume / voxel_volume(voxel_size))


def filter_labels_by_size(
    store_path: str,
    component: str,
    min_voxels: "int | None" = None,
    max_voxels: "int | None" = None,
    *,
    relabel: bool = True,
) -> "tuple[int, int]":
    """Drop label objects outside ``[min_voxels, max_voxels]``, in place.

    Two-pass streaming scan (see module docstring) -- the array never has
    to fit in RAM.

    Parameters
    ----------
    store_path : str
        Path to the zarr store containing the label array.
    component : str
        Array name inside the store to filter in place.
    min_voxels : int, optional
        Objects with fewer voxels than this are zeroed (dropped). ``None``
        (default) sets no lower bound. Use :func:`min_voxels_for_volume` to
        derive this from a physical volume and calibration.
    max_voxels : int, optional
        Objects with more voxels than this are zeroed (dropped) -- e.g. a
        segmentation artifact where several objects merged into one giant
        blob. ``None`` (default) sets no upper bound. Use
        :func:`max_voxels_for_volume` to derive this from a physical volume
        and calibration.
    relabel : bool, optional
        Renumber the surviving objects to a contiguous ``1..N`` range in
        the same LUT that drops the out-of-range ones (default ``True``)
        -- otherwise the removed ids leave permanent gaps and survivors
        keep their original ids.

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
    if min_voxels is None and max_voxels is None:
        raise ValueError(
            "filter_labels_by_size needs min_voxels, max_voxels, or both"
        )

    root = zarr.open_group(store_path, mode="r+")
    z = root[component]
    slices = chunk_slices(z.shape, z.chunks)

    # Per-chunk (id, count) arrays, summed once at the end: vectorised, where
    # a per-object Python loop crawls on millions of objects.
    chunk_ids, chunk_counts = [], []
    for sl in slices:
        ids, n = np.unique(np.asarray(z[sl]), return_counts=True)
        fg = ids != 0
        chunk_ids.append(ids[fg].astype(np.int64))
        chunk_counts.append(n[fg].astype(np.int64))
    if chunk_ids:
        all_ids = np.concatenate(chunk_ids)
        all_counts = np.concatenate(chunk_counts)
    else:
        all_ids = all_counts = np.empty(0, dtype=np.int64)
    ids, inverse = np.unique(all_ids, return_inverse=True)
    totals = np.zeros(ids.size, dtype=np.int64)
    np.add.at(totals, inverse, all_counts)

    keep = np.ones(ids.size, dtype=bool)
    if min_voxels is not None:
        keep &= totals >= min_voxels
    if max_voxels is not None:
        keep &= totals <= max_voxels
    kept = ids[keep]
    n_kept = int(kept.size)
    n_removed = int(ids.size) - n_kept

    # Sized to the largest id *seen*, not just the largest surviving one --
    # a removed object's id can still exceed every kept id and must stay
    # in bounds so the LUT gather below maps it to 0 rather than indexing
    # past the end.
    max_label = int(ids[-1]) if ids.size else 0
    if max_label > _LUT_WARN_THRESHOLD:
        logger.warning(
            "filter_labels_by_size: max_label=%d -> LUT size ~%.0f MB.",
            max_label,
            max_label * 8 / 1024**2,
        )
    lut = np.zeros(max_label + 1, dtype=np.int64)
    if n_kept:
        lut[kept] = np.arange(1, n_kept + 1) if relabel else kept

    max_out = n_kept if relabel else max_label
    out_dtype = np.uint16 if max_out < np.iinfo(np.uint16).max else np.uint32
    for sl in slices:
        block = np.asarray(z[sl])
        z[sl] = lut[block].astype(out_dtype)

    bounds = "-".join(
        str(v) if v is not None else "" for v in (min_voxels, max_voxels)
    )
    logger.info(
        "filter_labels_by_size: dropped %d/%d object(s) outside [%s] voxels, "
        "%d remain",
        n_removed,
        ids.size,
        bounds,
        n_kept,
    )
    return n_kept, n_removed
