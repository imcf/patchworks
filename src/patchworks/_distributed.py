"""Per-tile building blocks for distributed processing.

``tile_process`` runs every tile and merges in one process. To spread tiles
across separate jobs (e.g. one SLURM GPU job per tile) you need to process a
*single* tile independently and merge later. These helpers expose exactly that:
:func:`spatial_tiles` enumerates the tiles, :func:`create_stage` makes the
shared output store, and :func:`stage_tile` runs ``fn`` on one tile and writes
it into that store. Stitch the result with
:func:`patchworks.merge_tile_labels` (or ``zarr_native_merge``).
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Callable, Union

import numpy as np
import zarr


def spatial_tiles(
    shape: tuple[int, ...], tile_shape: tuple[int, ...]
) -> list[tuple[slice, ...]]:
    """Enumerate the tiles covering *shape*, in row-major order.

    Parameters
    ----------
    shape : tuple of int
        Spatial array shape.
    tile_shape : tuple of int
        Tile shape.

    Returns
    -------
    list of tuple of slice
        One slice tuple per tile (the same order ``estimate_empty_tiles``'s
        ``occupancy`` grid uses when ravelled).
    """
    grids = [range(0, s, t) for s, t in zip(shape, tile_shape)]
    return [
        tuple(
            slice(o, min(o + t, s))
            for o, t, s in zip(starts, tile_shape, shape)
        )
        for starts in itertools.product(*grids)
    ]


def create_stage(
    stage_path: Union[str, Path],
    shape: tuple[int, ...],
    tile_shape: tuple[int, ...],
    *,
    component: str = "staged",
    dtype=np.int32,
) -> str:
    """Create the empty (zero-filled) shared stage store for tiled writes.

    Parameters
    ----------
    stage_path : str or Path
        Destination ``.zarr`` store.
    shape : tuple of int
        Full (spatial) array shape.
    tile_shape : tuple of int
        Chunk = tile shape (one chunk per tile, so jobs write disjoint files).
    component : str, optional
        Array name inside the store (default ``"staged"``).
    dtype : data-type, optional
        Label dtype (default ``int32``). Tiles write local labels; the merge's
        first pass renumbers them to a compact global range that fits int32.

    Returns
    -------
    str
        The stage store path.
    """
    root = zarr.open_group(str(stage_path), mode="w")
    root.create_array(
        name=component, shape=shape, chunks=tile_shape, dtype=dtype
    )
    return str(stage_path)


def stage_tile(
    image,
    fn: Callable[[np.ndarray], np.ndarray],
    stage_path: Union[str, Path],
    index: int,
    *,
    tile_shape: tuple[int, ...],
    overlap: int = 0,
    component: str = "staged",
) -> int:
    """Run *fn* on a single tile and write it into the shared stage store.

    Reads the tile (expanded by *overlap* on every side for boundary context),
    runs *fn*, trims the halo back off, and writes the result to the tile's
    disjoint chunk of ``stage_path/component`` — so many of these can run
    concurrently (one per job) without conflicts.

    Parameters
    ----------
    image : array-like
        The full image (dask/zarr/NumPy), indexable by slices.
    fn : callable
        ``(ndarray) -> ndarray`` returning integer labels of the same shape.
    stage_path : str or Path
        Stage store created by :func:`create_stage`.
    index : int
        Tile index into :func:`spatial_tiles`.
    tile_shape : tuple of int
        Tile shape (must match the stage store's chunks).
    overlap : int, optional
        Halo added on every side before calling *fn*.
    component : str, optional
        Array name inside the stage store.

    Returns
    -------
    int
        The processed tile *index*.
    """
    shape = image.shape
    sl = spatial_tiles(shape, tile_shape)[index]
    expanded, trims = [], []
    for s, dim in zip(sl, shape):
        lo = max(0, s.start - overlap)
        hi = min(dim, s.stop + overlap)
        expanded.append(slice(lo, hi))
        trims.append((s.start - lo, hi - s.stop))
    block = np.asarray(image[tuple(expanded)])
    out = np.asarray(fn(block))
    sel = tuple(
        slice(left, out.shape[i] - right)
        for i, (left, right) in enumerate(trims)
    )
    # Local labels (1..N) are fine here — they collide across tiles, but the
    # merge's first pass makes them globally unique before stitching.
    dst = zarr.open_group(str(stage_path), mode="r+")[component]
    dst[sl] = out[sel].astype(dst.dtype)
    return index
