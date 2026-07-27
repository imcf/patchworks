"""Zarr-native label merge: boundary scan → scipy CC → parallel relabel.

Three steps, all zarr-native with no dask task graph:
  1. Scan thin boundary slabs → touching label pairs (O(n_faces × face_area))
  2. scipy sparse connected_components on pairs → relabeling LUT
  3. Apply LUT to each chunk in parallel via multiprocessing.Pool

Trade-off: touching-label merge only (overlap_depth=0 semantics for merge).
IoU-overlap merge is not supported here. Keep overlap > 0 during segmentation
for boundary-cell context; trim the halo before staging so chunk boundaries
in the staged zarr are clean for this merge.

Public API
----------
``merge_tile_labels(labeled, write_to, ...)`` — standalone merge for labeled
dask arrays or pre-staged zarr stores. Use this directly if you already have
per-tile labels and just need the boundary-stitching step.
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import nullcontext as _nullcontext
from itertools import product as _iproduct
from multiprocessing import Pool as _Pool
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import dask.array as da
import numpy as np
import zarr

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None

logger = logging.getLogger(__name__)

_ZARR_V3 = int(zarr.__version__.split(".")[0]) >= 3
_LUT_WARN_THRESHOLD = 100_000_000  # warn when max_label > 100 M (LUT > 800 MB)

# Per-worker globals set by _init_worker.
# LUT is memory-mapped from disk so it is shared read-only across all workers
# (OS page cache, no per-process copy). Passing the LUT directly via pickle
# would deserialize N separate copies — e.g. 4 workers × 800 MB = 3.2 GB wasted.
_merge_lut: "np.ndarray | None" = None
_merge_lut_path: "str | None" = None
_merge_staged_path: "str | None" = None
_merge_staged_comp: "str | None" = None
_merge_out_path: "str | None" = None
_merge_out_comp: "str | None" = None


def _init_worker(lut_path, staged_path, staged_comp, out_path, out_comp):
    """Initialise a merge worker process with the shared paths and LUT.

    Parameters
    ----------
    lut_path : str
        Path to the relabel lookup table (loaded memory-mapped, read-only).
    staged_path : str
        Path to the staged-labels zarr store.
    staged_comp : str
        Component name within the staged store.
    out_path : str
        Path to the output zarr store.
    out_comp : str
        Component name within the output store.

    Returns
    -------
    None
    """
    global _merge_lut, _merge_lut_path, _merge_staged_path, _merge_staged_comp
    global _merge_out_path, _merge_out_comp
    _merge_lut = np.load(
        lut_path, mmap_mode="r"
    )  # shared read-only via OS page cache
    _merge_lut_path = lut_path
    _merge_staged_path = staged_path
    _merge_staged_comp = staged_comp
    _merge_out_path = out_path
    _merge_out_comp = out_comp


def _relabel_chunk_worker(task: tuple) -> None:
    """Apply the relabel LUT to one chunk and write it to the output store.

    Parameters
    ----------
    task : tuple
        ``(chunk_slice, offset)``. *chunk_slice* selects this chunk in both
        stores. *offset* is added to the chunk's non-zero (tile-local) ids
        before the LUT lookup, which is what makes them globally unique; pass
        0 when the store already holds global ids.

    Returns
    -------
    None
    """
    chunk_slice, offset = task
    src = zarr.open_group(_merge_staged_path, mode="r")[_merge_staged_comp]
    dst = zarr.open_group(_merge_out_path, mode="r+")[_merge_out_comp]
    block = np.asarray(src[chunk_slice])
    nz = block > 0
    if not nz.any():
        dst[chunk_slice] = np.zeros(block.shape, dtype=dst.dtype)
        return
    lut = _merge_lut
    # Gather only the non-zero ids: labels are sparse, so this stays far
    # smaller than widening the whole block to int64.
    ids = block[nz].astype(np.int64) + int(offset)
    max_b = int(ids.max())
    if max_b >= len(lut):
        ext = np.arange(len(lut), max_b + 1, dtype=np.int64)
        lut = np.concatenate([lut, ext])
    out = np.zeros(block.shape, dtype=np.int64)
    out[nz] = lut[ids]
    dst[chunk_slice] = out.astype(dst.dtype)


def _boundary_face_specs(
    shape: tuple[int, ...], chunk_shape: tuple[int, ...]
) -> list[tuple[int, int]]:
    """Enumerate interior chunk boundaries to scan for touching labels.

    Parameters
    ----------
    shape : tuple of int
        Array shape.
    chunk_shape : tuple of int
        Chunk shape.

    Returns
    -------
    list of tuple of int
        ``(axis, position)`` pairs, one per interior chunk boundary.
    """
    specs = []
    for ax, (s, cs) in enumerate(zip(shape, chunk_shape)):
        pos = cs
        while pos < s:
            specs.append((ax, pos))
            pos += cs
    return specs


def _scan_touching_pairs(
    zarr_path: str,
    component: str,
    chunk_shape: tuple[int, ...],
    label_offsets: "np.ndarray | None" = None,
) -> np.ndarray:
    """Scan chunk-boundary slabs; return (N, 2) int64 array of touching pairs.

    Reads the boundary face one zarr-chunk column at a time so memory per read
    is bounded to one chunk (~200 MB). Reading the full face at once
    (slice(None) on face axes) would allocate face_area × 8 bytes in one shot —
    e.g. 37888 × 27392 × 8 = 8 GiB for a single z-face (OOM on real datasets).

    Parameters
    ----------
    zarr_path : str
        Path to the staged-labels zarr store.
    component : str
        Component name within the store.
    chunk_shape : tuple of int
        Chunk shape (sets the per-read column size).
    label_offsets : np.ndarray, optional
        Per-chunk id offset (row-major, see :func:`_offsets_from_counts`).
        When given, the store holds tile-local ids and the offsets are applied
        here, on the two chunks either side of each boundary -- so the pairs
        come out global without the store ever being rewritten. Each read is
        confined to one chunk column, so both sides are a single chunk and the
        offsets are plain scalars.

    Returns
    -------
    np.ndarray
        ``(N, 2)`` int64 array of unique label pairs touching across a
        boundary.
    """
    root = zarr.open_group(zarr_path, mode="r")
    arr = root[component]
    shape = arr.shape
    n_per_dim = [(s + c - 1) // c for s, c in zip(shape, chunk_shape)]
    specs = _boundary_face_specs(shape, chunk_shape)
    all_pairs: list[np.ndarray] = []
    for ax, pos in specs:
        # tile the face dimensions using chunk_shape columns
        face_axes = [a for a in range(arr.ndim) if a != ax]
        face_ranges = [range(0, shape[a], chunk_shape[a]) for a in face_axes]
        for offsets in _iproduct(*face_ranges):
            sl: list = [slice(None)] * arr.ndim
            sl[ax] = slice(pos - 1, pos + 1)
            for a, off in zip(face_axes, offsets):
                sl[a] = slice(off, min(off + chunk_shape[a], shape[a]))
            slab = np.moveaxis(np.asarray(arr[tuple(sl)]), ax, 0)
            a_vals = slab[0].ravel().astype(np.int64)
            b_vals = slab[1].ravel().astype(np.int64)
            if label_offsets is not None:
                grid = [0] * arr.ndim
                for a, off in zip(face_axes, offsets):
                    grid[a] = off // chunk_shape[a]
                grid[ax] = pos // chunk_shape[ax]
                b_idx = int(np.ravel_multi_index(tuple(grid), n_per_dim))
                grid[ax] -= 1  # the chunk on the near side of the boundary
                a_idx = int(np.ravel_multi_index(tuple(grid), n_per_dim))
                a_vals[a_vals > 0] += label_offsets[a_idx]
                b_vals[b_vals > 0] += label_offsets[b_idx]
            mask = (a_vals > 0) & (b_vals > 0) & (a_vals != b_vals)
            if mask.any():
                pairs = np.sort(
                    np.stack([a_vals[mask], b_vals[mask]], axis=1), axis=1
                )
                all_pairs.append(np.unique(pairs, axis=0))
    if not all_pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.unique(np.vstack(all_pairs), axis=0)


def _build_relabel_lut(pairs: np.ndarray, max_label: int) -> np.ndarray:
    """Build a relabel LUT from touching pairs via connected components.

    Parameters
    ----------
    pairs : np.ndarray
        ``(N, 2)`` array of touching label pairs.
    max_label : int
        Largest label id present.

    Returns
    -------
    np.ndarray
        Lookup table mapping each old label to its merged (component) id.
    """
    if max_label > _LUT_WARN_THRESHOLD:
        logger.warning(
            "_build_relabel_lut: max_label=%d → LUT ~%.0f MB. "
            "Memory use is bounded but large LUTs slow the merge.",
            max_label,
            max_label * 8 / 1024**2,
        )
    lut = np.arange(max_label + 1, dtype=np.int64)
    if len(pairs) == 0 or max_label == 0:
        return lut
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n = max_label + 1
    valid = (pairs[:, 0] < n) & (pairs[:, 1] < n)
    pairs = pairs[valid]
    if len(pairs) == 0:
        return lut
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    graph = csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n)
    )
    n_cc, cc_of = connected_components(graph, directed=False)
    cc_min = np.full(n_cc, n, dtype=np.int64)
    np.minimum.at(cc_min, cc_of, np.arange(n, dtype=np.int64))
    return cc_min[cc_of]


def _create_zarr_label_array(
    group: zarr.Group, name: str, shape: tuple, chunks: tuple, dtype=np.int32
) -> zarr.Array:
    """Create (replacing any existing) a label array in *group*.

    Parameters
    ----------
    group : zarr.Group
        Parent group.
    name : str
        Array name (may be a nested path).
    shape : tuple
        Array shape.
    chunks : tuple
        Chunk shape.
    dtype : data-type, optional
        Label dtype (default ``int32``). Matched to the staged store so the
        merge keeps wide (block-encoded) ids intact before they are compacted.

    Returns
    -------
    zarr.Array
        The newly created array (works on zarr v2 and v3).
    """
    if name in group:
        del group[name]
    if _ZARR_V3:
        return group.create_array(name, shape=shape, chunks=chunks, dtype=dtype)
    return group.zeros(
        name, shape=shape, chunks=chunks, dtype=dtype, overwrite=True
    )


def _make_globally_unique(arr, shape: tuple, chunk_shape: tuple) -> int:
    """Renumber per-chunk local labels to a globally unique, compact range.

    Each tile writes local labels (``1..N``) that repeat across tiles, so the
    same value means different objects in different tiles. This streams the
    chunks in row-major order and remaps every chunk's non-zero labels to a
    fresh contiguous block ``[base+1, …]``, leaving the store globally unique
    with ``max_label == total objects`` (compact, so the relabel LUT stays
    ``O(n_objects)``). Background (0) stays 0. In place, one chunk in RAM.

    Parameters
    ----------
    arr : zarr.Array
        Writable staged label array.
    shape : tuple
        Array shape.
    chunk_shape : tuple
        Chunk (= tile) shape.

    Returns
    -------
    int
        New maximum label (number of objects across all tiles, before the
        cross-boundary merge fuses touching pairs).
    """
    n_per_dim = [(s + c - 1) // c for s, c in zip(shape, chunk_shape)]
    base = 0
    for idx in _iproduct(*[range(n) for n in n_per_dim]):
        sl = tuple(
            slice(i * c, min((i + 1) * c, s))
            for i, c, s in zip(idx, chunk_shape, shape)
        )
        block = np.asarray(arr[sl])
        uniq = np.unique(block)
        uniq = uniq[uniq > 0]
        if uniq.size == 0:
            continue
        lut = np.zeros(int(uniq[-1]) + 1, dtype=np.int64)
        lut[uniq] = np.arange(1, uniq.size + 1) + base
        arr[sl] = lut[block].astype(arr.dtype)
        base += int(uniq.size)
    return base


def capped_output_chunks(
    chunk_shape: Sequence[int], caps: Sequence[int]
) -> tuple[int, ...]:
    """Shrink each chunk to at most *cap*, staying an exact divisor.

    The merge's workers write one staged chunk at a time. As long as the output
    chunking divides the staged chunking, each write still covers whole output
    chunks, so concurrent workers never read-modify-write a chunk they share.
    A non-divisor cap would break that, so the largest divisor at or below the
    cap is used instead.

    Parameters
    ----------
    chunk_shape : sequence of int
        Staged chunk (= tile) shape.
    caps : sequence of int
        Maximum chunk size per axis.

    Returns
    -------
    tuple of int
        Chunking for the merged output.

    Examples
    --------
    >>> capped_output_chunks((16, 2048, 2048), (16, 1024, 1024))
    (16, 1024, 1024)
    >>> capped_output_chunks((16, 1024, 1024), (16, 1024, 1024))
    (16, 1024, 1024)
    """
    out = []
    for c, cap in zip(chunk_shape, caps):
        c, cap = int(c), int(cap)
        if c <= cap:
            out.append(c)
            continue
        out.append(next(d for d in range(cap, 0, -1) if c % d == 0))
    return tuple(out)


def _offsets_from_counts(
    counts: Mapping[int, int] | Sequence[int], n_chunks: int
) -> np.ndarray:
    """Exclusive cumulative sum turning per-tile counts into id offsets.

    Each tile writes a dense ``1..n_i``. Adding ``offset[i] = sum(n_0..n_{i-1})``
    makes ids globally unique *and* compact in one step, with no read of the
    volume at all -- the whole job is O(n_tiles).

    Parameters
    ----------
    counts : mapping of int to int, or sequence of int
        Labels written per tile, keyed by (or ordered as) the row-major tile
        index. Tiles that were skipped may be absent from a mapping.
    n_chunks : int
        Total number of tiles/chunks in the store.

    Returns
    -------
    np.ndarray
        ``offset`` of length *n_chunks*; the total object count before
        boundary merging is ``offset[-1] + counts[-1]``.
    """
    per_chunk = np.zeros(n_chunks, dtype=np.int64)
    if isinstance(counts, Mapping):
        for idx, n in counts.items():
            idx = int(idx)
            if not 0 <= idx < n_chunks:
                raise ValueError(
                    f"tile index {idx} is outside the {n_chunks}-chunk store"
                )
            per_chunk[idx] = int(n)
    else:
        if len(counts) != n_chunks:
            raise ValueError(
                f"got {len(counts)} counts for a {n_chunks}-chunk store"
            )
        per_chunk[:] = [int(n) for n in counts]
    if (per_chunk < 0).any():
        raise ValueError("label counts must be non-negative")
    offsets = np.zeros(n_chunks, dtype=np.int64)
    np.cumsum(per_chunk[:-1], out=offsets[1:])
    return offsets


def zarr_native_merge(
    staged_path: str,
    staged_component: str,
    out_path: str,
    out_component: str,
    n_workers: int = 4,
    show_progress: bool = False,
    label_counts: "Mapping[int, int] | Sequence[int] | None" = None,
    sequential: bool = False,
    output_chunks: "Sequence[int] | None" = None,
) -> "int | None":
    """Zarr-native label merge: boundary scan → scipy CC → parallel relabel.

    Scales to 2000+ chunks where the dask_image approach stalls (O(n_chunks²)
    graph). Reads *staged_path/staged_component*, merges touching cross-boundary
    labels, writes result to *out_path/out_component*. No dask task graph.

    Parameters
    ----------
    staged_path : str
        Path to the staged-labels zarr store.
    staged_component : str
        Component name within the staged store.
    out_path : str
        Path to the output zarr store.
    out_component : str
        Component name within the output store.
    n_workers : int
        Number of worker processes for the parallel relabel.
    show_progress : bool
        Show a progress bar over the relabel chunks.
    label_counts : mapping or sequence of int, optional
        Labels written per chunk, when the producer recorded them (each chunk
        holding a dense ``1..n_i``). Global ids are then ``offset + local``
        with ``offset`` an exclusive cumulative sum -- O(n_chunks) arithmetic
        instead of reading and rewriting the entire store to renumber it.
        ``None`` falls back to the streaming renumber pass.
    sequential : bool
        Renumber the merged labels to a contiguous ``1..N``. Free here: the id
        domain is dense by construction, so the compaction is a ``np.unique``
        over the LUT (length = object count) and folds into the same lookup,
        costing no extra pass over the volume.
    output_chunks : sequence of int, optional
        Chunking for the merged store. Must divide the staged chunk shape, so
        each worker's write still covers whole chunks. ``None`` mirrors the
        staged chunking. Use :func:`capped_output_chunks` to derive it.

    Returns
    -------
    int or None
        The object count when *sequential* is set, else ``None``.
    """
    root = zarr.open_group(staged_path, mode="r+")
    arr = root[staged_component]
    shape, chunk_shape = arr.shape, arr.chunks

    n_per_dim = [(s + c - 1) // c for s, c in zip(shape, chunk_shape)]
    n_chunks = int(np.prod(n_per_dim))

    # Tiles write labels 1..n that collide across tiles (every tile has a "1"),
    # so they must be made globally unique before the boundary merge can tell
    # unrelated objects apart. With per-tile counts that is a cumulative sum;
    # without them, fall back to streaming every chunk and renumbering it in
    # place -- correct, but a full read+write of the volume.
    if label_counts is not None:
        offsets = _offsets_from_counts(label_counts, n_chunks)
        counts_arr = (
            np.asarray(
                [label_counts.get(i, 0) for i in range(n_chunks)],
                dtype=np.int64,
            )
            if isinstance(label_counts, Mapping)
            else np.asarray(label_counts, dtype=np.int64)
        )
        max_label = int(offsets[-1] + counts_arr[-1]) if n_chunks else 0
    else:
        offsets = None
        max_label = _make_globally_unique(arr, shape, chunk_shape)
    logger.info(
        "zarr_native_merge: shape=%s chunks=%s max_label=%d (%s)",
        shape,
        chunk_shape,
        max_label,
        "offsets from tile counts" if offsets is not None else "renumber pass",
    )

    n_faces = len(_boundary_face_specs(shape, chunk_shape))
    logger.info("zarr_native_merge: scanning %d boundary faces…", n_faces)
    pairs = _scan_touching_pairs(
        staged_path, staged_component, chunk_shape, label_offsets=offsets
    )
    logger.info(
        "zarr_native_merge: %d touching pairs → building LUT", len(pairs)
    )

    lut = _build_relabel_lut(pairs, max_label)
    n_remapped = int((lut != np.arange(len(lut), dtype=np.int64)).sum())
    logger.info(
        "zarr_native_merge: %d labels remapped across boundaries", n_remapped
    )

    n_objects = None
    if sequential:
        # Every id in 1..max_label exists in the store (both paths above make
        # the domain dense), so the surviving ids are exactly the distinct LUT
        # values -- no scan of the volume is needed to find them.
        uniq, inverse = np.unique(lut, return_inverse=True)
        lut = inverse.astype(np.int64)
        n_objects = int(uniq.size - (1 if uniq.size and uniq[0] == 0 else 0))
        logger.info(
            "zarr_native_merge: renumbered to 1..%d in the same LUT", n_objects
        )

    out_root = zarr.open_group(out_path, mode="a")
    out_chunks = tuple(chunk_shape)
    if output_chunks is not None:
        out_chunks = tuple(int(c) for c in output_chunks)
        bad = [
            (i, c, o)
            for i, (c, o) in enumerate(zip(chunk_shape, out_chunks))
            if o <= 0 or c % o
        ]
        if bad:
            raise ValueError(
                "output_chunks must divide the staged chunk shape so workers "
                f"write whole chunks; axis/staged/output mismatches: {bad}"
            )
    # Match the staged dtype: ids are already compact (dense by construction,
    # and compacted again above when sequential), so nothing needs a wider one.
    _create_zarr_label_array(
        out_root, out_component, shape, out_chunks, dtype=arr.dtype
    )

    # Row-major, matching spatial_tiles' order -- so chunk i is tile i and the
    # offsets line up with the per-tile counts.
    chunk_slices = [
        tuple(
            slice(i * c, min((i + 1) * c, s))
            for i, c, s in zip(idx, chunk_shape, shape)
        )
        for idx in _iproduct(*[range(n) for n in n_per_dim])
    ]
    tasks = [
        (sl, int(offsets[i]) if offsets is not None else 0)
        for i, sl in enumerate(chunk_slices)
    ]
    n_w = max(1, min(n_workers, n_chunks))
    logger.info(
        "zarr_native_merge: relabeling %d chunks with %d worker(s)…",
        n_chunks,
        n_w,
    )

    # Save LUT to a temp .npy file so workers memory-map it (shared OS page cache).
    # Pickling the LUT array directly via multiprocessing initargs would
    # deserialize a full copy per worker — e.g. 4 workers × 800 MB = 3.2 GB.
    _lut_dir = tempfile.mkdtemp(prefix="bb_lut_")
    lut_path = os.path.join(_lut_dir, "lut.npy")
    np.save(lut_path, lut)
    del lut  # parent no longer needs it; workers load via mmap

    try:
        if n_w <= 1:
            _init_worker(
                lut_path, staged_path, staged_component, out_path, out_component
            )
            it: Any = tasks
            if show_progress and _tqdm is not None:
                it = _tqdm(it, total=n_chunks, desc="relabel chunks")
            for task in it:
                _relabel_chunk_worker(task)
        else:
            with _Pool(
                processes=n_w,
                initializer=_init_worker,
                initargs=(
                    lut_path,
                    staged_path,
                    staged_component,
                    out_path,
                    out_component,
                ),
            ) as pool:
                it = pool.imap_unordered(_relabel_chunk_worker, tasks)
                if show_progress and _tqdm is not None:
                    it = _tqdm(it, total=n_chunks, desc="relabel chunks")
                for _ in it:
                    pass
    finally:
        import shutil

        shutil.rmtree(_lut_dir, ignore_errors=True)

    return n_objects


# ---------------------------------------------------------------------------
# Public standalone merge API
# ---------------------------------------------------------------------------


def merge_tile_labels(
    labeled: Union["da.Array", str, Path],
    write_to: Union[str, Path, None] = None,
    *,
    input_component: str = "labels",
    output_component: str = "labels",
    overlap: int = 0,
    sequential_labels: bool = False,
    n_workers: int | None = None,
    stage_dir: Union[str, Path, None] = None,
    keep_stage: bool = False,
    progress: bool = False,
    return_count: bool = False,
    label_counts: "Mapping[int, int] | Sequence[int] | None" = None,
    output_chunks: "Sequence[int] | None" = None,
) -> Union["da.Array", tuple["da.Array", Union[int, None]]]:
    """Merge per-tile labels into a globally consistent label array.

    Standalone merge step — use this when you already have per-tile labels
    (from your own segmentation pipeline) and just need the boundary stitching.

    Accepts either:

    - A **dask array** of per-tile integer labels (e.g. output of
      ``dask.array.map_blocks`` on your own segmentation function).
    - A **zarr store path** whose ``input_component`` array already contains
      per-tile labels written by your own pipeline.

    Labels that **touch** across tile boundaries are merged into a single ID.
    The merge is zarr-native (boundary scan → scipy connected components →
    parallel relabel) — no dask task graph, scales to thousands of tiles.

    Parameters
    ----------
    labeled:
        Per-tile label array. Either a dask array or a path to a zarr store
        that contains per-tile labels in ``input_component``.
    write_to:
        Output zarr store path. When None, an auto-temp store is used.
    input_component:
        Array name inside a zarr *input* store (ignored for dask arrays).
    output_component:
        Array name inside ``write_to``. Default ``"labels"``.
    overlap:
        If ``labeled`` is a dask array that was computed with ``da.overlap``,
        pass the same depth here to trim the halos before merging.
        Set 0 (default) if the array has no overlap halos.
    sequential_labels:
        Renumber the merged labels to a contiguous ``1..N`` range. Folded into
        the merge's own lookup table, so it costs no extra pass over the
        volume. Default False.
    label_counts:
        Labels written per tile, when the producer recorded them (see
        :func:`patchworks.stage_tile`, which returns the count). Lets the
        merge derive every tile's global id range arithmetically instead of
        streaming the whole store to renumber it. ``None`` keeps the
        renumber pass.
    output_chunks:
        Chunking for the merged store; must divide the staged chunk shape.
        Lets the merge write straight into a store you will keep (e.g. an
        OME-ZARR label group's level 0) instead of a scratch store that then
        has to be copied. See :func:`capped_output_chunks`.
    n_workers:
        Parallel workers for the relabel step. Default ``min(4, cpu_count)``.
    stage_dir:
        Directory for the temp stage zarr when *labeled* is a dask array.
        Default: a system temp directory.
    keep_stage:
        Keep the temp stage zarr after merging. Default False.
    progress:
        Show a progress bar during the relabel step.
    return_count:
        Also return the exact object count. Only meaningful (non-``None``)
        when ``sequential_labels=True``, which already computes it for free
        while renumbering to ``1..N`` — otherwise no step here knows the
        final count without an extra full scan, so the second element is
        ``None``. Useful to persist alongside the labels (e.g.
        ``write_labels(..., n_objects=...)``) so a downstream consumer with
        the count can skip re-deriving the id set from the array itself.

    Returns
    -------
    da.Array
        Merged label array (int32) backed by ``write_to``. Or, when
        ``return_count=True``, a ``(labels, n_objects)`` tuple —
        ``n_objects`` is ``None`` unless ``sequential_labels=True``.

    Examples
    --------
    **From a dask array of per-tile labels:**

    >>> import dask.array as da
    >>> from patchworks import merge_tile_labels
    >>>
    >>> # your own tiling + segmentation
    >>> image = da.from_zarr("image.zarr").rechunk((1, 1024, 1024))
    >>> labeled = image.map_blocks(my_segment_fn, dtype="int32",
    ...                            meta=np.empty((0,) * image.ndim, dtype="int32"))
    >>>
    >>> merged = merge_tile_labels(labeled, write_to="labels.zarr", progress=True)

    **From a pre-staged zarr store (your pipeline already wrote labels):**

    >>> merged = merge_tile_labels(
    ...     "my_staged_labels.zarr",
    ...     input_component="raw_labels",
    ...     write_to="merged_labels.zarr",
    ...     sequential_labels=True,
    ... )

    **Trim overlap halos before merging:**

    >>> # if labeled was computed with da.overlap.overlap(depth=20)
    >>> merged = merge_tile_labels(labeled, write_to="labels.zarr", overlap=20)
    """
    import dask.array as da

    nw = n_workers if n_workers is not None else min(4, os.cpu_count() or 1)

    # -- Stage dask array to zarr if needed --
    stage_path: str | None = None
    staged_component = "staged"

    if isinstance(labeled, (str, Path)):
        stage_path = str(labeled)
        staged_component = input_component
    else:
        # labeled is a dask array
        if overlap > 0:
            labeled = da.overlap.trim_overlap(
                labeled, depth=overlap, boundary="none"
            )

        _base = (
            str(stage_dir)
            if stage_dir is not None
            else tempfile.mkdtemp(prefix="pws_stage_")
        )
        stage_path = os.path.join(_base, "_pws_stage.zarr")

        import dask
        from dask.diagnostics import ProgressBar

        ctx = ProgressBar() if progress else _nullcontext()
        logger.info("Staging per-tile labels to %s …", stage_path)
        with ctx:
            dask.compute(
                labeled.to_zarr(
                    stage_path,
                    component=staged_component,
                    overwrite=True,
                    compute=False,
                )
            )

    # -- Resolve output path --
    if write_to is not None:
        effective_out = str(write_to)
    else:
        effective_out = os.path.join(
            tempfile.mkdtemp(prefix="bb_merge_"), "merged.zarr"
        )
        logger.info(
            "write_to not set — merged labels in auto-temp %s", effective_out
        )

    # -- Merge (the sequential renumber rides along inside the same LUT) --
    n_objects = zarr_native_merge(
        stage_path,
        staged_component,
        effective_out,
        output_component,
        n_workers=nw,
        show_progress=progress,
        label_counts=label_counts,
        sequential=sequential_labels,
        output_chunks=output_chunks,
    )

    # -- Cleanup temp stage (only when we created it) --
    if not isinstance(labeled, (str, Path)) and not keep_stage:
        import shutil

        shutil.rmtree(stage_path, ignore_errors=True)
        logger.info("Removed stage store %s", stage_path)

    result = da.from_zarr(effective_out, component=output_component)
    return (result, n_objects) if return_count else result
