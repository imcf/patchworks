"""Exact per-tile occupancy from a max-pooled summary of the image.

``estimate_empty_tiles`` decides whether a tile is background by reading a
small *centred* window of it. On a ``(16, 1024, 1024)`` tile with the default
``(24, 256, 256)`` window that is 6.25% of the tile's area, so an object
sitting in the tile's outer ring is never seen -- and when the result is used
as a skip list (as the Snakemake workflow does) that tile is never segmented.

This module trades that sampling for a one-off reduction. Each
``block``-sized brick of the image is reduced to its **maximum** and stored in
a small array beside the image. Two properties make the result exact rather
than approximate:

* ``max`` never loses a bright voxel, so no signal can hide between samples.
* ``block_max > threshold`` is true exactly when *some* voxel in the block
  exceeds ``threshold``. Comparing pooled maxima against a threshold derived
  from raw voxels therefore gives the same answer as scanning every voxel.

The map is ~1/16384 of the image at the default block size, is written once
next to ``image.zarr``, and is shared by every segmentation config that reads
that image -- replacing one sampling pass *per config* with one pooling pass
in total.
"""

from __future__ import annotations

import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from itertools import product as _iproduct
from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np
import zarr

from ._chunks import cpu_allocation

logger = logging.getLogger(__name__)

DEFAULT_BLOCK = (1, 128, 128)
# Blocks per tile edge. A block is the finest thing the map can resolve, so it
# has to be well under the tile or every tile over-covers the same block and
# comes back "occupied" -- correct, but useless as a skip list.
_BLOCKS_PER_TILE = 4


def block_for_tile(
    tile_shape: Sequence[int], cap: Sequence[int] = DEFAULT_BLOCK
) -> tuple[int, ...]:
    """Occupancy block sized so a tile spans several blocks.

    The map can only resolve whole blocks, so a block as large as the tile
    makes every tile hit the same block and test occupied. Sizing it to a
    fraction of the tile keeps the answer discriminating, while the *cap*
    keeps the map small on big tiles.

    Parameters
    ----------
    tile_shape : sequence of int
        The tile shape the map will be queried with.
    cap : sequence of int, optional
        Largest block per axis.

    Returns
    -------
    tuple of int
        Block shape, at least 1 per axis.

    Examples
    --------
    >>> block_for_tile((16, 1024, 1024))
    (1, 128, 128)
    >>> block_for_tile((8, 32, 32))
    (1, 8, 8)
    """
    return tuple(
        max(1, min(int(c), int(t) // _BLOCKS_PER_TILE or 1))
        for t, c in zip(tile_shape, cap)
    )


# Bytes to pull per read while pooling; keeps peak memory flat regardless of
# how large the image is.
_READ_TARGET_BYTES = 128 * 1024**2


def _block_max(arr: np.ndarray, block: tuple[int, ...]) -> np.ndarray:
    """Reduce *arr* to the maximum of each *block*-sized brick.

    Parameters
    ----------
    arr : np.ndarray
        Region to reduce.
    block : tuple of int
        Brick shape, one entry per axis.

    Returns
    -------
    np.ndarray
        Array of per-brick maxima, ``ceil(arr.shape / block)`` in shape.
    """
    pad = [(0, (-s) % b) for s, b in zip(arr.shape, block)]
    if any(after for _, after in pad):
        # Pad with the region's own minimum so the padding can never win a
        # max() and invent signal at the image edge.
        arr = np.pad(arr, pad, mode="constant", constant_values=arr.min())
    folded: list[int] = []
    for s, b in zip(arr.shape, block):
        folded.extend((s // b, b))
    return arr.reshape(folded).max(axis=tuple(range(1, len(folded), 2)))


def _leading_index(ndim: int, n_spatial: int, channel: int) -> tuple[int, ...]:
    """Index prefix selecting *channel* from the non-spatial leading axes."""
    n_leading = ndim - n_spatial
    if n_leading <= 0:
        return ()
    return (0,) * (n_leading - 1) + (channel,)


def _level_array(root: zarr.Group, level: int, store_path: str) -> Any:
    """Return the zarr array for one pyramid *level* of an OME-ZARR store."""
    attrs = dict(root.attrs)
    multiscales = attrs.get("multiscales") or attrs.get("ome", {}).get(
        "multiscales"
    )
    try:
        path = multiscales[0]["datasets"][level]["path"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            f"Cannot read OME-ZARR multiscales metadata at level {level} "
            f"in {store_path!r}"
        ) from exc
    return root[path]


def occupancy_path(image_store: Union[str, Path], level: int = 0) -> str:
    """Path of the occupancy map for one pyramid level of *image_store*.

    Deliberately a **sibling** of the image store, not a node inside it. The
    map is not an NGFF array, and zarr refuses to walk a hierarchy containing
    one ("Object at ... is not recognized as a component of a Zarr
    hierarchy"), which would make every ``members()``/``arrays()`` call on the
    user's image warn.
    """
    store = str(image_store).rstrip("/")
    base = store[:-5] if store.endswith(".zarr") else store
    return f"{base}.occupancy.zarr/{level}"


def build_occupancy_map(
    image_store: Union[str, Path],
    *,
    level: int = 0,
    block: tuple[int, ...] = DEFAULT_BLOCK,
    overwrite: bool = False,
) -> str:
    """Max-pool every channel of an OME-ZARR level into a small summary array.

    Reads the image once and writes ``<image_store>/occupancy/<level>``, an
    array of shape ``(n_channels, *ceil(spatial_shape / block))`` holding the
    maximum of each brick. Cheap to keep (~1/16384 of the image by default)
    and reusable by every config that segments this image.

    Idempotent: an existing map is reused unless *overwrite* is set, so
    concurrent segmentation runs against one ``work_dir`` build it at most
    once. The map is written to a temporary sibling and moved into place, so a
    crash mid-build never leaves a partial map behind.

    Parameters
    ----------
    image_store : str or Path
        OME-ZARR store to summarise.
    level : int, optional
        Pyramid level to read (default 0, full resolution).
    block : tuple of int, optional
        Brick shape over the spatial axes (default ``(1, 128, 128)``: no z
        reduction, so any z-tiling works).
    overwrite : bool, optional
        Rebuild even if a map already exists.

    Returns
    -------
    str
        Path of the occupancy array.

    Examples
    --------
    >>> build_occupancy_map("image.zarr")  # doctest: +SKIP
    'image.zarr/occupancy/0'
    """
    store = str(image_store)
    out_path = occupancy_path(store, level)
    if not overwrite and Path(out_path).exists():
        # Reuse only if it was built at the block we want. A map left over
        # from a run with a different tile_shape is coarser (or finer) than
        # this run needs, and silently reusing it would degrade every
        # occupancy answer that follows.
        try:
            existing = tuple(zarr.open_array(out_path, mode="r").attrs["block"])
        except Exception:
            existing = None
        if existing == tuple(block):
            logger.info("occupancy map already present at %s", out_path)
            return out_path
        logger.info(
            "rebuilding occupancy map at %s: it was built with block %s, "
            "this run needs %s",
            out_path,
            existing,
            tuple(block),
        )
        overwrite = True

    root = zarr.open_group(store, mode="r")
    src = _level_array(root, level, store)
    n_spatial = len(block)
    sp_shape = tuple(src.shape[-n_spatial:])
    n_leading = src.ndim - n_spatial
    n_channels = src.shape[n_leading - 1] if n_leading > 0 else 1

    grid = tuple(-(-s // b) for s, b in zip(sp_shape, block))
    itemsize = np.dtype(src.dtype).itemsize
    # Read this many output cells per axis at a time -- cube-ish, sized so one
    # read stays near _READ_TARGET_BYTES.
    voxels_per_block = int(np.prod(block))
    cells = max(1, _READ_TARGET_BYTES // (voxels_per_block * itemsize))
    step = max(1, int(round(cells ** (1.0 / n_spatial))))
    steps = tuple(min(step, g) for g in grid)

    logger.info(
        "building occupancy map: %d channel(s), block=%s, grid=%s (%.1f MB)",
        n_channels,
        block,
        grid,
        n_channels * float(np.prod(grid)) * itemsize / 1024**2,
    )

    tmp_path = f"{out_path}.building.{os.getpid()}"
    shutil.rmtree(tmp_path, ignore_errors=True)
    Path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    dst = zarr.open_array(
        tmp_path,
        mode="w",
        shape=(n_channels, *grid),
        chunks=(1, *steps),
        dtype=src.dtype,
    )

    ranges = [range(0, g, s) for g, s in zip(grid, steps)]
    regions = list(_iproduct(*ranges))

    def _one(starts: tuple[int, ...]) -> None:
        out_sl = tuple(
            slice(o, min(o + s, g)) for o, s, g in zip(starts, steps, grid)
        )
        src_sl = tuple(
            slice(o.start * b, min(o.stop * b, s))
            for o, b, s in zip(out_sl, block, sp_shape)
        )
        # Read every channel of this region in ONE go. Looping channels on the
        # outside would traverse the whole image once per channel -- three
        # full reads for a three-channel stack, where one does.
        for channel in range(n_channels):
            prefix = _leading_index(src.ndim, n_spatial, channel)
            region = np.asarray(src[prefix + src_sl])
            dst[(channel, *out_sl)] = _block_max(region, block)

    try:
        n_workers = max(1, min(cpu_allocation(), len(regions)))
        if n_workers <= 1:
            for starts in regions:
                _one(starts)
        else:
            # Reads and decompression release the GIL, so threads are enough
            # and there is no worker payload to pickle.
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                for _ in pool.map(_one, regions):
                    pass
        dst.attrs["block"] = list(block)
        dst.attrs["level"] = int(level)
        dst.attrs["source_shape"] = list(sp_shape)
    except BaseException:
        shutil.rmtree(tmp_path, ignore_errors=True)
        raise

    if Path(out_path).exists():
        if not overwrite:
            # A concurrent run finished first; theirs is as good as ours.
            shutil.rmtree(tmp_path, ignore_errors=True)
            return out_path
        # Explicit rebuild: clear the old one, or os.replace would fail on a
        # non-empty directory and we would keep serving the stale map.
        shutil.rmtree(out_path, ignore_errors=True)
    try:
        os.replace(tmp_path, out_path)
    except OSError:
        shutil.rmtree(tmp_path, ignore_errors=True)
        if not Path(out_path).exists():
            raise
    logger.info("occupancy map written to %s", out_path)
    return out_path


def tile_occupancy(
    image_store: Union[str, Path],
    tile_shape: tuple[int, ...],
    *,
    channel: int = 0,
    threshold: float,
    level: int = 0,
) -> dict[str, Any]:
    """Decide which tiles hold signal, using the whole tile, not a sample.

    Reduces the occupancy map over each tile's full footprint and marks the
    tile occupied when any brick maximum exceeds *threshold* -- equivalent to
    testing every voxel, because a brick maximum exceeds the threshold exactly
    when some voxel in that brick does.

    Blocks are only ever over-covered, never under-covered: when a tile edge
    falls inside a brick, that brick counts for both neighbours. A tile can
    therefore be occupied because of a neighbour's signal in a shared brick,
    which costs one extra segmentation job but can never drop a tile. Choosing
    a *block* that divides *tile_shape* (the default 128 divides 1024) avoids
    even that.

    Parameters
    ----------
    image_store : str or Path
        OME-ZARR store holding the occupancy map (see
        :func:`build_occupancy_map`).
    tile_shape : tuple of int
        Tile shape, in full-resolution voxels.
    channel : int, optional
        Channel to test.
    threshold : float
        Empty cutoff, derived from raw voxel values (signal <= threshold →
        empty).
    level : int, optional
        Pyramid level the map was built from.

    Returns
    -------
    dict
        ``threshold``, ``n_tiles``, ``n_occupied``, ``empty_fraction`` and
        ``occupancy`` (bool array over the tile grid), matching
        :func:`patchworks.estimate_empty_tiles`.
    """
    arr = zarr.open_array(occupancy_path(image_store, level), mode="r")
    block = tuple(arr.attrs["block"])
    sp_shape = tuple(arr.attrs["source_shape"])
    if len(tile_shape) != len(block):
        raise ValueError(
            f"tile_shape is {len(tile_shape)}-D but the occupancy map is "
            f"{len(block)}-D"
        )
    coarse = [
        (ax, b, t) for ax, (b, t) in enumerate(zip(block, tile_shape)) if b >= t
    ]
    if coarse:
        logger.warning(
            "occupancy blocks are as large as the tile on axes %s; every tile "
            "will over-cover the same block and test occupied. Rebuild the "
            "map with block=block_for_tile(tile_shape).",
            [ax for ax, _, _ in coarse],
        )

    tile_grid = tuple(-(-s // t) for s, t in zip(sp_shape, tile_shape))
    occupancy = np.zeros(tile_grid, dtype=bool)
    pooled = np.asarray(arr[channel])
    for idx in np.ndindex(*tile_grid):
        sl = tuple(
            slice(
                (i * t) // b,
                min(-(-min((i + 1) * t, s) // b), g),
            )
            for i, t, b, s, g in zip(
                idx, tile_shape, block, sp_shape, pooled.shape
            )
        )
        window = pooled[sl]
        occupancy[idx] = bool(window.size) and bool(window.max() > threshold)

    n_tiles = int(occupancy.size)
    n_occ = int(occupancy.sum())
    logger.info(
        "tile_occupancy: threshold=%.4g  occupied %d/%d tiles",
        threshold,
        n_occ,
        n_tiles,
    )
    return {
        "threshold": float(threshold),
        "n_tiles": n_tiles,
        "n_occupied": n_occ,
        "empty_fraction": 1.0 - n_occ / n_tiles if n_tiles else 0.0,
        "occupancy": occupancy,
    }
