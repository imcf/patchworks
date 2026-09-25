"""Relate two label images by voxel overlap (e.g. nucleus -> containing cell)."""

from __future__ import annotations

import logging
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Union

import dask.array as da
import numpy as np

from ._chunks import cpu_allocation
from ._progress import PROGRESS_INTERVAL_S as _PROGRESS_INTERVAL_S
from ._progress import log_progress

logger = logging.getLogger(__name__)


def _as_dask(
    source: Union["da.Array", str, Path], component: str
) -> "da.Array":
    if isinstance(source, (str, Path)):
        return da.from_zarr(str(source), component=component)
    return source


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
    a_ids, a_counts = np.unique(a_block[a_fg], return_counts=True)
    sizes = np.stack([a_ids, a_counts], axis=1).astype(np.int64)
    mask = a_fg & (b_block > 0)
    if not mask.any():
        return np.empty((0, 3), dtype=np.int64), sizes
    pairs = np.stack([a_block[mask], b_block[mask]], axis=1).astype(np.int64)
    uniq, counts = np.unique(pairs, axis=0, return_counts=True)
    return np.concatenate([uniq, counts[:, None]], axis=1), sizes


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
        Parallel chunk workers. Default ``min(4, cpu_count)``.

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
    a = _as_dask(a, a_component)
    b = _as_dask(b, b_component)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: a={a.shape} b={b.shape}")
    if a.chunks != b.chunks:
        raise ValueError(
            "a and b must share the same chunk layout "
            f"(a={a.chunks} b={b.chunks}); rechunk one to match the other, "
            "e.g. b = b.rechunk(a.chunks)"
        )

    n_blocks = a.numblocks
    total = int(np.prod(n_blocks))
    nw = n_workers if n_workers is not None else min(4, cpu_allocation())

    logger.info(
        "label_relations: scanning %d chunk(s) of %s with %d worker(s)",
        total,
        "x".join(str(n) for n in a.shape),
        nw,
    )

    def _one(flat_idx: int) -> tuple[np.ndarray, np.ndarray]:
        idx = np.unravel_index(flat_idx, n_blocks)
        return _chunk_pairs(
            np.asarray(a.blocks[idx]), np.asarray(b.blocks[idx])
        )

    # as_completed, not ex.map: map returns an iterator that yields in
    # submission order, so one slow early chunk withholds every later result
    # and the log stays silent however many have actually finished. This step
    # runs for hours in a batch job where the only question the log has to
    # answer is "working, or hung?".
    started = _time.monotonic()
    last = started
    parts = []
    with ThreadPoolExecutor(max_workers=nw) as ex:
        futures = {ex.submit(_one, i): i for i in range(total)}
        for done, future in enumerate(as_completed(futures), start=1):
            parts.append(future.result())
            now = _time.monotonic()
            if now - last >= _PROGRESS_INTERVAL_S or done == total:
                log_progress("label_relations", done, total, started)
                last = now

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

    logger.info(
        "label_relations: %d a-labels matched across %d chunks",
        a_ids.size,
        total,
    )
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
