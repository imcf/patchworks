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
import logging
import os
from pathlib import Path
from typing import Callable, Sequence, Union

import numpy as np
import zarr

from ._io import zarr_compressor_kwargs
from ._relabel import relabel_sequential_array

logger = logging.getLogger(__name__)

Overlap = Union[int, Sequence[int]]


def normalize_overlap(
    overlap: Overlap,
    ndim: int,
    tile_shape: "Sequence[int] | None" = None,
) -> tuple[int, ...]:
    """Expand an overlap spec to one halo width per axis.

    A scalar applies the same halo to every axis (the historical behaviour).
    A sequence gives the halo per axis, which matters for anisotropic tiles:
    a ``(16, 1024, 1024)`` tile with a scalar overlap of 30 reads
    ``76 x 1084 x 1084`` to keep ``16 x 1024 x 1024`` -- 5.3x more voxels than
    it uses, nearly all of it in z.

    With *tile_shape*, an axis only one voxel thick gets **no** halo. There is
    no context to gather along an axis the tile does not span, and a 2-D
    method handed the extra planes would read them as channels. This is the
    ``tile_shape: "auto"`` + ``do_3D: false`` case, where tiles come out one
    plane thick: a z-overlap of 4 would otherwise read 9 planes per tile to
    keep 1.

    Parameters
    ----------
    overlap : int or sequence of int
        Halo width, shared or per-axis.
    ndim : int
        Number of axes the halo is applied to.
    tile_shape : sequence of int, optional
        Tile extent per axis. Used to drop halos an axis has no room for.

    Returns
    -------
    tuple of int
        One non-negative halo width per axis.
    """
    if isinstance(overlap, (int, np.integer)):
        values = (int(overlap),) * ndim
    else:
        values = tuple(int(o) for o in overlap)
        if len(values) != ndim:
            raise ValueError(
                f"overlap has {len(values)} entries but the tile is {ndim}-D"
            )
    if any(o < 0 for o in values):
        raise ValueError(f"overlap must be non-negative, got {values}")

    if tile_shape is not None:
        clipped = tuple(
            0 if int(t) <= 1 else o for o, t in zip(values, tile_shape)
        )
        if clipped != values:
            dropped = [
                i for i, (a, b) in enumerate(zip(values, clipped)) if a != b
            ]
            logger.info(
                "dropping the halo on axis %s: the tile is 1 voxel thick "
                "there, so there is no context to read.",
                dropped,
            )
        values = clipped
    return values


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
    zarr_format: int | None = None,
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
    zarr_format : int, optional
        2 or 3. Give the image store's own format when staging straight into
        its ``labels/<name>`` group: a v3 group inside a v2 (OME-Zarr 0.4)
        store is something no reader can open. Default: zarr's.

    Returns
    -------
    str
        The stage store path.
    """
    root = zarr.open_group(
        str(stage_path),
        mode="w",
        **({"zarr_format": zarr_format} if zarr_format else {}),
    )
    root.create_array(
        name=component,
        shape=shape,
        chunks=tile_shape,
        dtype=dtype,
        **zarr_compressor_kwargs(root.metadata.zarr_format),
    )
    return str(stage_path)


def stage_tile(
    image,
    fn: Callable[[np.ndarray], np.ndarray],
    stage_path: Union[str, Path],
    index: int,
    *,
    tile_shape: tuple[int, ...],
    overlap: Overlap = 0,
    component: str = "staged",
    channel_axis: int | None = None,
    halo_dir: Union[str, Path, None] = None,
    parts_dir: Union[str, Path, None] = None,
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
    overlap : int or sequence of int, optional
        Halo added on every side before calling *fn*. A scalar applies to
        every axis; a sequence gives one width per axis (see
        :func:`normalize_overlap`).
    component : str, optional
        Array name inside the stage store.
    channel_axis : int or None, optional
        Axis of *image* holding channels, which is **not** tiled: it is read
        whole and handed to *fn* alongside the tile's voxels. ``tile_shape``,
        ``overlap`` and the stage store stay purely spatial, so *fn* still
        returns one label per voxel with no channel axis (e.g. Cellpose fed a
        cytoplasm + nuclei pair returns a single label volume). ``None`` (the
        default) means *image* is already single-channel.
    halo_dir : str or Path, optional
        Also keep what *fn* predicted in the halo, for IoU stitching
        (``merge_tile_labels(..., halo_dir=...)``): one ``<index>.npz`` per
        tile holding, for each face with a halo, the strip beyond the tile
        (keys ``"<axis>+"``/``"<axis>-"``), labelled with the same ``1..n``
        ids as the staged core. An object seen only in the halo is 0 there:
        it has no id in this tile. ``None`` (default) keeps nothing.
    parts_dir : str or Path, optional
        Also measure the tile's objects as written (size, centroid, spread,
        bounding box; :func:`patchworks._tables.tile_partial`) into
        ``<parts_dir>/<index>.npz``, keyed by the staged ``1..n`` ids. The
        merge combines them into the object table without reading the
        labels again (``merge_tile_labels(..., parts_dir=...)``). A tile
        without labels writes none.

    Returns
    -------
    int
        Number of labels this tile wrote, i.e. its ids are exactly ``1..n``.
        Record it (the workflow puts it in the tile's ``.done`` marker): with
        one count per tile the merge can derive every tile's global id range
        by a cumulative sum, instead of rewriting the whole store to make the
        ids unique.
    """
    shape = tuple(image.shape)
    # The channel axis is carried, not tiled: geometry (tiles, halo, the stage
    # store) stays spatial, so nothing downstream of fn learns about channels.
    if channel_axis is None:
        spatial_shape = shape
    else:
        channel_axis %= len(shape)
        spatial_shape = shape[:channel_axis] + shape[channel_axis + 1 :]
    sl = spatial_tiles(spatial_shape, tile_shape)[index]
    halo = normalize_overlap(overlap, len(sl), tile_shape=tile_shape)
    expanded, trims = [], []
    for s, dim, ov in zip(sl, spatial_shape, halo):
        lo = max(0, s.start - ov)
        hi = min(dim, s.stop + ov)
        expanded.append(slice(lo, hi))
        trims.append((s.start - lo, hi - s.stop))
    read = list(expanded)
    if channel_axis is not None:
        read.insert(channel_axis, slice(None))
    block = np.asarray(image[tuple(read)])
    # What fn owes us back: one label per voxel, channel axis consumed.
    block_spatial = tuple(e.stop - e.start for e in expanded)
    out = np.asarray(fn(block))
    if out.shape != block_spatial:
        # Named here, not as a broadcast error deep in zarr.
        name = getattr(fn, "__name__", type(fn).__name__)
        raise ValueError(
            f"segmentation function {name!r} returned shape {out.shape} for "
            f"a tile of shape {block_spatial} (tile {index}). It must return "
            "one label per input voxel. Some deconvolution backends crop "
            "their output -- pad or centre it back to the input shape before "
            "returning."
        )
    sel = tuple(
        slice(left, out.shape[i] - right)
        for i, (left, right) in enumerate(trims)
    )
    trimmed = out[sel]
    # Ids 1..n with no gaps, so offset[tile] + local is unique and compact.
    if halo_dir is not None:
        _save_halo(out, sel, trims, halo_dir, index)
    trimmed = relabel_sequential_array(trimmed)
    n_labels = int(trimmed.max())
    dst = zarr.open_group(str(stage_path), mode="r+")[component]
    dst[sl] = trimmed.astype(dst.dtype)
    if parts_dir is not None and n_labels:
        _save_partial(trimmed, sl, parts_dir, index)
    return n_labels


def _save_partial(trimmed, sl, parts_dir, index) -> None:
    """Write one tile's per-object sums to ``<parts_dir>/<index>.npz``.

    Through a temporary name and a rename, so a job killed mid-write leaves
    no truncated file for the merge to trust.
    """
    from ._tables import tile_partial

    part = tile_partial(trimmed, [s.start for s in sl])
    if part is None:
        return
    parts_dir = Path(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    tmp = parts_dir / f".{int(index)}.tmp.npz"
    np.savez(tmp, **part)
    os.replace(tmp, parts_dir / f"{int(index)}.npz")


def _core_lut(core: np.ndarray, max_id: int) -> np.ndarray:
    """LUT renumbering *core*'s ids to ``1..n`` (as ``relabel_sequential``);
    ids absent from the core map to 0."""
    ids = np.unique(core)
    ids = ids[ids > 0]
    lut = np.zeros(max(int(max_id), 0) + 1, dtype=np.int64)
    lut[ids] = np.arange(1, ids.size + 1)
    return lut


def _save_halo(
    out: np.ndarray,
    sel: tuple[slice, ...],
    trims: list[tuple[int, int]],
    halo_dir: Union[str, Path],
    index: int,
) -> None:
    """Write the halo strips of one tile's prediction to ``<index>.npz``.

    Each strip spans the halo along its axis and the tile's *core* along the
    others (corners are left out: a corner neighbour is reached through an
    edge neighbour anyway). Ids follow the core's ``1..n`` numbering, so the
    merge can offset them exactly like the staged tile.
    """
    out = np.asarray(out)
    if out.size and out.min() < 0:
        raise ValueError("labels must be non-negative")
    lut = _core_lut(out[sel], int(out.max()) if out.size else 0)
    strips: dict[str, np.ndarray] = {}
    for ax, (left, right) in enumerate(trims):
        for side, width in (("-", left), ("+", right)):
            if not width:
                continue
            region = list(sel)
            n = out.shape[ax]
            region[ax] = slice(0, left) if side == "-" else slice(n - right, n)
            strip = lut[out[tuple(region)]]
            strips[f"{ax}{side}"] = strip.astype(np.int32, copy=False)
    # The tile's label count rides along, so a merge can place its ids
    # globally without renumbering the staged store.
    strips["n"] = np.asarray(int(lut.max()))
    halo_dir = Path(halo_dir)
    halo_dir.mkdir(parents=True, exist_ok=True)
    # Written then renamed, so a killed job never leaves a torn file.
    tmp = halo_dir / f".{int(index)}.npz.tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **strips)
    tmp.replace(halo_dir / f"{int(index)}.npz")
