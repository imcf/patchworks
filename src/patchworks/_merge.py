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
from typing import Any, Union

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
    global _merge_lut, _merge_lut_path, _merge_staged_path, _merge_staged_comp
    global _merge_out_path, _merge_out_comp
    _merge_lut = np.load(lut_path, mmap_mode="r")  # shared read-only via OS page cache
    _merge_lut_path = lut_path
    _merge_staged_path = staged_path
    _merge_staged_comp = staged_comp
    _merge_out_path = out_path
    _merge_out_comp = out_comp


def _relabel_chunk_worker(chunk_slice: tuple) -> None:
    src = zarr.open_group(_merge_staged_path, mode="r")[_merge_staged_comp]
    dst = zarr.open_group(_merge_out_path, mode="r+")[_merge_out_comp]
    block = np.asarray(src[chunk_slice], dtype=np.int64)
    max_b = int(block.max())
    if max_b == 0:
        dst[chunk_slice] = block.astype(np.int32)
        return
    lut = _merge_lut
    if max_b < len(lut):
        out = lut[block]
    else:
        ext = np.arange(len(lut), max_b + 1, dtype=np.int64)
        out = np.concatenate([lut, ext])[block]
    dst[chunk_slice] = out.astype(np.int32)


def _boundary_face_specs(
    shape: tuple[int, ...], chunk_shape: tuple[int, ...]
) -> list[tuple[int, int]]:
    specs = []
    for ax, (s, cs) in enumerate(zip(shape, chunk_shape)):
        pos = cs
        while pos < s:
            specs.append((ax, pos))
            pos += cs
    return specs


def _scan_touching_pairs(
    zarr_path: str, component: str, chunk_shape: tuple[int, ...]
) -> np.ndarray:
    """Scan chunk-boundary slabs; return (N, 2) int64 array of touching pairs.

    Reads the boundary face one zarr-chunk column at a time so memory per read
    is bounded to one chunk (~200 MB). Reading the full face at once
    (slice(None) on face axes) would allocate face_area × 8 bytes in one shot —
    e.g. 37888 × 27392 × 8 = 8 GiB for a single z-face (OOM on real datasets).
    """
    root = zarr.open_group(zarr_path, mode="r")
    arr = root[component]
    shape = arr.shape
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
            mask = (a_vals > 0) & (b_vals > 0) & (a_vals != b_vals)
            if mask.any():
                pairs = np.sort(np.stack([a_vals[mask], b_vals[mask]], axis=1), axis=1)
                all_pairs.append(np.unique(pairs, axis=0))
    if not all_pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.unique(np.vstack(all_pairs), axis=0)


def _build_relabel_lut(pairs: np.ndarray, max_label: int) -> np.ndarray:
    """Touching-pairs → scipy connected components → relabeling LUT."""
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
    group: zarr.Group, name: str, shape: tuple, chunks: tuple
) -> zarr.Array:
    if name in group:
        del group[name]
    if _ZARR_V3:
        return group.create_array(name, shape=shape, chunks=chunks, dtype=np.int32)
    return group.zeros(name, shape=shape, chunks=chunks, dtype=np.int32, overwrite=True)


def zarr_native_merge(
    staged_path: str,
    staged_component: str,
    out_path: str,
    out_component: str,
    n_workers: int = 4,
    show_progress: bool = False,
) -> None:
    """Zarr-native label merge: boundary scan → scipy CC → parallel relabel.

    Scales to 2000+ chunks where the dask_image approach stalls (O(n_chunks²)
    graph). Reads *staged_path/staged_component*, merges touching cross-boundary
    labels, writes result to *out_path/out_component*. No dask task graph.
    """
    root = zarr.open_group(staged_path, mode="r")
    arr = root[staged_component]
    shape, chunk_shape = arr.shape, arr.chunks

    max_label = int(
        da.from_zarr(staged_path, component=staged_component).max().compute()
    )
    logger.info(
        "zarr_native_merge: shape=%s chunks=%s max_label=%d",
        shape,
        chunk_shape,
        max_label,
    )

    n_faces = len(_boundary_face_specs(shape, chunk_shape))
    logger.info("zarr_native_merge: scanning %d boundary faces…", n_faces)
    pairs = _scan_touching_pairs(staged_path, staged_component, chunk_shape)
    logger.info("zarr_native_merge: %d touching pairs → building LUT", len(pairs))

    lut = _build_relabel_lut(pairs, max_label)
    n_remapped = int((lut != np.arange(len(lut), dtype=np.int64)).sum())
    logger.info("zarr_native_merge: %d labels remapped across boundaries", n_remapped)

    out_root = zarr.open_group(out_path, mode="a")
    _create_zarr_label_array(out_root, out_component, shape, chunk_shape)

    n_per_dim = [(s + c - 1) // c for s, c in zip(shape, chunk_shape)]
    chunk_slices = [
        tuple(
            slice(i * c, min((i + 1) * c, s))
            for i, c, s in zip(idx, chunk_shape, shape)
        )
        for idx in _iproduct(*[range(n) for n in n_per_dim])
    ]
    n_chunks = len(chunk_slices)
    n_w = max(1, min(n_workers, n_chunks))
    logger.info(
        "zarr_native_merge: relabeling %d chunks with %d worker(s)…", n_chunks, n_w
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
            it: Any = chunk_slices
            if show_progress and _tqdm is not None:
                it = _tqdm(it, total=n_chunks, desc="relabel chunks")
            for sl in it:
                _relabel_chunk_worker(sl)
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
                it = pool.imap_unordered(_relabel_chunk_worker, chunk_slices)
                if show_progress and _tqdm is not None:
                    it = _tqdm(it, total=n_chunks, desc="relabel chunks")
                for _ in it:
                    pass
    finally:
        import shutil

        shutil.rmtree(_lut_dir, ignore_errors=True)


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
) -> "da.Array":
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
        Renumber the merged labels to a contiguous ``1..N`` range via a cheap
        linear post-pass (O(voxels)). Default False.
    n_workers:
        Parallel workers for the relabel step. Default ``min(4, cpu_count)``.
    stage_dir:
        Directory for the temp stage zarr when *labeled* is a dask array.
        Default: a system temp directory.
    keep_stage:
        Keep the temp stage zarr after merging. Default False.
    progress:
        Show a progress bar during the relabel step.

    Returns
    -------
    da.Array
        Merged label array (int32) backed by ``write_to``.

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

    from ._relabel import relabel_sequential_zarr

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
            labeled = da.overlap.trim_overlap(labeled, depth=overlap, boundary="none")

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
        logger.info("write_to not set — merged labels in auto-temp %s", effective_out)

    # -- Merge --
    zarr_native_merge(
        stage_path,
        staged_component,
        effective_out,
        output_component,
        n_workers=nw,
        show_progress=progress,
    )

    if sequential_labels:
        logger.info("Relabelling to contiguous ids…")
        relabel_sequential_zarr(effective_out, output_component)

    # -- Cleanup temp stage (only when we created it) --
    if not isinstance(labeled, (str, Path)) and not keep_stage:
        import shutil

        shutil.rmtree(stage_path, ignore_errors=True)
        logger.info("Removed stage store %s", stage_path)

    return da.from_zarr(effective_out, component=output_component)
