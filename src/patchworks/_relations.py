"""Relate two label images by voxel overlap (e.g. nucleus -> containing cell)."""

from __future__ import annotations

import logging
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Mapping, Union

import dask.array as da
import numpy as np
import zarr

from ._chunks import cpu_allocation
from ._progress import PROGRESS_INTERVAL_S as _PROGRESS_INTERVAL_S
from ._progress import log_progress

logger = logging.getLogger(__name__)


def _chunk_pairs(
    a_block: np.ndarray, b_block: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Overlap rows and *a*-label sizes for one chunk pair.

    Returns
    -------
    tuple of np.ndarray
        ``(pairs, sizes)``: ``(a_id, b_id, voxels)`` rows where both are
        non-background, and ``(a_id, voxels)`` rows counting *every* voxel of
        each *a* label -- including those over *b*'s background, which the
        overlap fraction has to be taken against.
    """
    a_fg = a_block > 0
    if not a_fg.any():
        empty = np.empty((0, 3), dtype=np.int64)
        return empty, np.empty((0, 2), dtype=np.int64)
    a_vals = a_block[a_fg].astype(np.int64)
    a_ids, a_counts = np.unique(a_vals, return_counts=True)
    sizes = np.stack([a_ids, a_counts], axis=1)
    b_vals = b_block[a_fg].astype(np.int64)
    both = b_vals > 0
    if not both.any():
        return np.empty((0, 3), dtype=np.int64), sizes
    a_vals, b_vals = a_vals[both], b_vals[both]
    # One int64 key per pair, so the count is a 1-D unique: np.unique over
    # rows (axis=0) is a lexsort of structured rows, ~20x slower on a full
    # chunk, and it was the bulk of a relation's hours.
    base = int(b_vals.max()) + 1
    keys, counts = np.unique(a_vals * base + b_vals, return_counts=True)
    return np.stack([keys // base, keys % base, counts], axis=1), sizes


def _stored_chunks(arr) -> "set[tuple[int, ...]] | None":
    """Chunk coordinates *arr* has actually written, from one listing of its
    store; ``None`` when that cannot be told (a dask array, a sharded one).

    A chunk never written holds only background, so a child's unwritten
    chunks need neither its read nor its parent's.
    """
    if not isinstance(arr, zarr.Array) or getattr(arr, "shards", None):
        return None
    try:
        from zarr.core.sync import sync

        prefix = f"{arr.path}/" if arr.path else ""

        async def _keys():
            return {k async for k in arr.store.list_prefix(prefix)}

        keys = sync(_keys())
        grid = [-(-s // c) for s, c in zip(arr.shape, arr.chunks)]
        return {
            idx
            for idx in np.ndindex(*grid)
            if prefix + arr.metadata.encode_chunk_key(idx) in keys
        }
    except Exception:  # an exotic store: read everything, as before
        return None


def _as_source(source, component: str):
    """A zarr array when *source* is one (or a path to one), else dask."""
    if isinstance(source, (str, Path)):
        try:
            return zarr.open_array(str(source), path=component, mode="r")
        except Exception:
            return da.from_zarr(str(source), component=component)
    return source


def _chunk_grid(arr) -> tuple:
    if isinstance(arr, zarr.Array):
        return tuple(int(c) for c in arr.chunks)
    return tuple(tuple(int(x) for x in c) for c in arr.chunks)


def _reader(arr, grid):
    """``read(idx)`` returning chunk *idx* of *arr* as numpy."""
    if isinstance(arr, zarr.Array):
        shape = arr.shape

        def read(idx):
            return np.asarray(
                arr[
                    tuple(
                        slice(i * c, min((i + 1) * c, s))
                        for i, c, s in zip(idx, grid, shape)
                    )
                ]
            )

        return read

    def read_block(idx):
        return np.asarray(arr.blocks[idx])

    return read_block


def label_relations_many(
    children: "Mapping[str, Union[da.Array, zarr.Array, str, Path]]",
    parent: "Union[da.Array, zarr.Array, str, Path]",
    *,
    component: str = "",
    n_workers: int | None = None,
) -> "dict[str, dict[int, dict[str, float]]]":
    """:func:`label_relations` of several children against one parent,
    reading the parent once.

    Each parent chunk is read only where some child has labels, and a
    child's chunks it never wrote are not read at all (one listing of its
    store tells them). With the parent read once for all its children, a
    cell segmentation that three label images are related to is read once,
    not three times -- on a networked filesystem the reads are the cost.

    Parameters
    ----------
    children : mapping of str to array or path
        Child label images by name, each of the parent's shape and chunking.
    parent : array or path
        The parent label image.
    component : str, optional
        Array path inside the stores given as paths (default: the root).
    n_workers : int or None, optional
        Chunks in flight. Default twice the CPUs this process may use: the
        work waits on reads more than it computes. Each holds a few chunks.

    Returns
    -------
    dict
        ``{child_name: label_relations(child, parent)}``.
    """
    b = _as_source(parent, component)
    kids = {name: _as_source(a, component) for name, a in children.items()}
    grid = _chunk_grid(b)
    for name, a in kids.items():
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch: {name}={a.shape} b={b.shape}")
        if _chunk_grid(a) != grid:
            raise ValueError(
                f"{name} and the parent must share the same chunk layout "
                f"({name}={a.chunks} b={b.chunks}); rechunk one to match the "
                "other, e.g. b = b.rechunk(a.chunks)"
            )
    if isinstance(b, zarr.Array):
        n_blocks = tuple(-(-s // c) for s, c in zip(b.shape, grid))
    else:
        n_blocks = b.numblocks
    read_b = _reader(b, grid)
    readers = {name: _reader(a, grid) for name, a in kids.items()}
    stored = {name: _stored_chunks(a) for name, a in kids.items()}
    chunks = list(np.ndindex(*n_blocks))
    todo = [
        idx
        for idx in chunks
        if any(s is None or idx in s for s in stored.values())
    ]
    nw = n_workers if n_workers is not None else min(64, 2 * cpu_allocation())
    logger.info(
        "label_relations: %s against one parent read: %d of %d chunk(s) hold "
        "child labels, %d worker(s)",
        ", ".join(kids),
        len(todo),
        len(chunks),
        nw,
    )

    def _one(idx):
        out = {}
        b_block = None
        for name, read in readers.items():
            if stored[name] is not None and idx not in stored[name]:
                continue
            a_block = read(idx)
            if not a_block.any():
                continue
            if b_block is None:
                b_block = read_b(idx)
            out[name] = _chunk_pairs(a_block, b_block)
        return out

    started = _time.monotonic()
    last = started
    parts: dict[str, list] = {name: [] for name in kids}
    with ThreadPoolExecutor(max_workers=max(1, nw)) as ex:
        futures = [ex.submit(_one, idx) for idx in todo]
        for done, future in enumerate(as_completed(futures), start=1):
            for name, part in future.result().items():
                parts[name].append(part)
            now = _time.monotonic()
            if now - last >= _PROGRESS_INTERVAL_S or done == len(todo):
                log_progress("label_relations", done, len(todo), started)
                last = now
    out = {}
    for name in kids:
        out[name] = _summarise(parts[name])
        logger.info(
            "label_relations: %s: %d labels matched", name, len(out[name])
        )
    return out


def _summarise(parts) -> dict[int, dict[str, float]]:
    """One best match per child label from the chunks' pair counts."""
    rows = [p for p, _ in parts if p.size]
    if not rows:
        return {}
    all_pairs = np.concatenate(rows, axis=0)

    # Merge duplicate (a_id, b_id) rows across chunks (a label can span
    # several chunks) with one sort on both columns and summing runs.
    order = np.lexsort((all_pairs[:, 1], all_pairs[:, 0]))
    all_pairs = all_pairs[order]
    new_run = np.any(np.diff(all_pairs[:, :2], axis=0) != 0, axis=1)
    starts = np.concatenate([[0], np.flatnonzero(new_run) + 1])
    a_ids = all_pairs[starts, 0]
    b_ids = all_pairs[starts, 1]
    counts = np.add.reduceat(all_pairs[:, 2], starts)

    # Best match per a: sort by a, then count descending, then b ascending
    # (ties go to the lowest b id), and keep each a's first row.
    order = np.lexsort((b_ids, -counts, a_ids))
    a_ids, b_ids, counts = a_ids[order], b_ids[order], counts[order]
    first = np.concatenate([[True], a_ids[1:] != a_ids[:-1]])
    a_ids, b_ids, counts = a_ids[first], b_ids[first], counts[first]

    # Every voxel of each a label, summed over the chunks it spans.
    sizes = np.concatenate([s for _, s in parts if s.size], axis=0)
    size_ids, inverse = np.unique(sizes[:, 0], return_inverse=True)
    totals = np.zeros(size_ids.size, dtype=np.int64)
    np.add.at(totals, inverse, sizes[:, 1])
    a_totals = totals[np.searchsorted(size_ids, a_ids)]
    return {
        int(a_id): {
            "match": int(b_id),
            "overlap_voxels": int(count),
            "overlap_fraction": float(count) / float(a_total),
        }
        for a_id, b_id, count, a_total in zip(
            a_ids.tolist(), b_ids.tolist(), counts.tolist(), a_totals.tolist()
        )
    }


def label_relations(
    a: Union["da.Array", str, Path],
    b: Union["da.Array", str, Path],
    *,
    a_component: str = "labels",
    b_component: str = "labels",
    n_workers: int | None = None,
) -> dict[int, dict[str, float]]:
    """Map each label in *a* to its best-overlapping label in *b*.

    For every non-background label in *a* (e.g. a nucleus segmentation),
    finds the label in *b* (e.g. a cell/cytoplasm segmentation of the same
    image) it shares the most voxels with. Streams both arrays chunk by
    chunk — memory is bounded by the number of distinct label pairs (one row
    per touching (a, b) pair per chunk), not by volume size.

    *a* and *b* must be two segmentations of the **same image** (identical
    shape and chunk layout) — e.g. two runs of the Snakemake workflow with
    different ``label_name``/``cellpose:`` config but the same ``tile_shape``.

    Parameters
    ----------
    a, b : da.Array, str or Path
        Two label arrays of identical shape and chunking. A path is read via
        ``dask.array.from_zarr(path, component=...)``.
    a_component, b_component : str, optional
        Zarr array name inside *a*/*b* when they're store paths (default
        ``"labels"``).
    n_workers : int or None, optional
        Parallel chunk workers. Default: the CPUs this process may use (the
        SLURM allocation on a cluster). Each holds two chunks and their
        pairs at a time.

    Returns
    -------
    dict
        ``{a_label: {"match": b_label, "overlap_voxels": int,
        "overlap_fraction": float}}`` — one entry per *a* label that touches
        at least one non-background *b* voxel. ``overlap_fraction`` is the
        matched voxel count over *a* label's total voxel count, background
        included (1.0 = fully contained). Labels in *a* with zero overlap are
        omitted. On a tie the lowest *b* id wins.

    Examples
    --------
    >>> from patchworks import label_relations
    >>> table = label_relations(
    ...     "scan.zarr/labels/nuclei", "scan.zarr/labels/cells"
    ... )  # doctest: +SKIP
    >>> table[2]  # nucleus 2 sits inside cell 3  # doctest: +SKIP
    {'match': 3, 'overlap_voxels': 4821, 'overlap_fraction': 0.94}
    """
    a = _as_source(a, a_component)
    b = _as_source(b, b_component)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: a={a.shape} b={b.shape}")
    if _chunk_grid(a) != _chunk_grid(b):
        raise ValueError(
            "a and b must share the same chunk layout "
            f"(a={a.chunks} b={b.chunks}); rechunk one to match the other, "
            "e.g. b = b.rechunk(a.chunks)"
        )
    return label_relations_many({"a": a}, b, n_workers=n_workers)["a"]
