"""Relate two label images by voxel overlap (e.g. nucleus -> containing cell)."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Union

import dask.array as da
import numpy as np

logger = logging.getLogger(__name__)


def _as_dask(source: Union["da.Array", str, Path], component: str) -> "da.Array":
    if isinstance(source, (str, Path)):
        return da.from_zarr(str(source), component=component)
    return source


def _chunk_pairs(a_block: np.ndarray, b_block: np.ndarray) -> np.ndarray:
    """Non-background ``(a_id, b_id) -> voxel count`` rows for one chunk pair."""
    mask = (a_block > 0) & (b_block > 0)
    if not mask.any():
        return np.empty((0, 3), dtype=np.int64)
    pairs = np.stack([a_block[mask], b_block[mask]], axis=1).astype(np.int64)
    uniq, counts = np.unique(pairs, axis=0, return_counts=True)
    return np.concatenate([uniq, counts[:, None]], axis=1)


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
        matched voxel count over *a* label's total voxel count (1.0 = fully
        contained). Labels in *a* with zero overlap are omitted.

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
    nw = n_workers if n_workers is not None else min(4, os.cpu_count() or 1)

    def _one(flat_idx: int) -> np.ndarray:
        idx = np.unravel_index(flat_idx, n_blocks)
        return _chunk_pairs(np.asarray(a.blocks[idx]), np.asarray(b.blocks[idx]))

    with ThreadPoolExecutor(max_workers=nw) as ex:
        parts = list(ex.map(_one, range(total)))

    rows = [p for p in parts if p.size]
    if not rows:
        return {}
    all_pairs = np.concatenate(rows, axis=0)

    # Merge duplicate (a_id, b_id) rows across chunks (a label can span
    # several chunks) by sorting on a combined key and summing runs.
    key = all_pairs[:, 0] * (int(all_pairs[:, 1].max()) + 1) + all_pairs[:, 1]
    order = np.argsort(key, kind="stable")
    all_pairs, key = all_pairs[order], key[order]
    starts = np.concatenate([[0], np.flatnonzero(np.diff(key)) + 1])
    merged_counts = np.add.reduceat(all_pairs[:, 2], starts)
    merged = np.stack(
        [all_pairs[starts, 0], all_pairs[starts, 1], merged_counts], axis=1
    )

    a_totals: dict[int, int] = {}
    for a_id, _, count in merged:
        a_id = int(a_id)
        a_totals[a_id] = a_totals.get(a_id, 0) + int(count)

    best: dict[int, tuple[int, int]] = {}
    for a_id, b_id, count in merged:
        a_id, b_id, count = int(a_id), int(b_id), int(count)
        cur = best.get(a_id)
        if cur is None or count > cur[1]:
            best[a_id] = (b_id, count)

    logger.info(
        "label_relations: %d a-labels matched across %d chunks", len(best), total
    )
    return {
        a_id: {
            "match": b_id,
            "overlap_voxels": count,
            "overlap_fraction": count / a_totals[a_id],
        }
        for a_id, (b_id, count) in best.items()
    }
