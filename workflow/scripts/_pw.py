"""Shared helpers for the patchworks Snakemake workflow.

These wrap patchworks' public API so each Snakemake rule can act on a single
tile (for SLURM scatter) or the whole store.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import zarr

from patchworks import load_ome_zarr


def spatial_tile_slices(
    shape: tuple[int, ...], tile_shape: tuple[int, ...]
) -> list[tuple[slice, ...]]:
    """Row-major list of per-tile slice tuples covering *shape*.

    Parameters
    ----------
    shape : tuple of int
        Spatial array shape.
    tile_shape : tuple of int
        Tile shape.

    Returns
    -------
    list of tuple of slice
        One slice tuple per tile, in row-major order.
    """
    grids = [range(0, s, t) for s, t in zip(shape, tile_shape)]
    tiles = []
    for starts in itertools.product(*grids):
        tiles.append(
            tuple(
                slice(o, min(o + t, s))
                for o, t, s in zip(starts, tile_shape, shape)
            )
        )
    return tiles


def process_one_tile(
    image, sl: tuple[slice, ...], overlap: int, fn
) -> np.ndarray:
    """Read one tile (with halo), run *fn*, and trim the halo back off.

    Parameters
    ----------
    image : array-like
        The full image (dask/zarr/numpy), indexable by slices.
    sl : tuple of slice
        The tile's slice (without halo).
    overlap : int
        Halo size added on every side before calling *fn*.
    fn : callable
        ``(ndarray) -> ndarray`` returning integer labels of the same shape.

    Returns
    -------
    np.ndarray
        Labels for exactly the region *sl* (halo trimmed).
    """
    shape = image.shape
    expanded, trims = [], []
    for s, dim in zip(sl, shape):
        lo = max(0, s.start - overlap)
        hi = min(dim, s.stop + overlap)
        expanded.append(slice(lo, hi))
        trims.append((s.start - lo, hi - s.stop))  # halo added left / right
    block = np.asarray(image[tuple(expanded)])
    out = np.asarray(fn(block))
    sel = tuple(
        slice(left, out.shape[i] - right)
        for i, (left, right) in enumerate(trims)
    )
    return out[sel]


def load_tiles_json(path: str | Path) -> dict:
    """Load the tile manifest written by ``prepare_tiles.py``.

    Parameters
    ----------
    path : str or Path
        Path to ``tiles.json``.

    Returns
    -------
    dict
        The manifest (``tile_shape``, ``overlap``, ``occupied`` indices, …).
    """
    return json.loads(Path(path).read_text())


def open_image(work_dir: str | Path, channel, level):
    """Open the converted image for segmentation.

    Parameters
    ----------
    work_dir : str or Path
        Workflow output directory containing ``image.zarr``.
    channel : int or None
        Channel to select.
    level : int
        Pyramid level to read.

    Returns
    -------
    da.Array
        The (lazy) image array.
    """
    store = str(Path(work_dir) / "image.zarr")
    return load_ome_zarr(store, channel=channel, level=level)


def stage_path(work_dir: str | Path) -> str:
    """Path of the staged-labels zarr store.

    Parameters
    ----------
    work_dir : str or Path
        Workflow output directory.

    Returns
    -------
    str
        ``<work_dir>/stage.zarr``.
    """
    return str(Path(work_dir) / "stage.zarr")


def open_stage(work_dir: str | Path, mode: str = "r+"):
    """Open the staged-labels array.

    Parameters
    ----------
    work_dir : str or Path
        Workflow output directory.
    mode : str
        Zarr open mode.

    Returns
    -------
    zarr.Array
        The ``staged`` array inside ``stage.zarr``.
    """
    return zarr.open_group(stage_path(work_dir), mode=mode)["staged"]
