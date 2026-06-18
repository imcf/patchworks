"""Core tile_process function."""
from __future__ import annotations

import logging
import os
from contextlib import nullcontext as _nullcontext
from pathlib import Path
from typing import Any, Callable, Union

import dask.array as da
import numpy as np


from ._chunks import auto_tile_shape
from ._cluster import _client_is_in_process, _distributed_client
from ._io import _auto_empty_threshold, load_ome_zarr
from ._merge import zarr_native_merge
from ._relabel import relabel_sequential_array, relabel_sequential_zarr

logger = logging.getLogger(__name__)


def _finalise(arr: da.Array, show_progress: bool,
              write_to: str | None, component: str) -> np.ndarray | None:
    """Compute or write *arr*, showing progress via the active scheduler."""
    import dask

    client = _distributed_client()
    if write_to is not None:
        lazy_write = arr.to_zarr(
            str(write_to), component=component, overwrite=True, compute=False
        )
        if client is not None:
            future = client.compute(lazy_write)
            if show_progress:
                from dask.distributed import progress as _dist_progress
                _dist_progress(future)
            future.result()
        else:
            from dask.diagnostics import ProgressBar
            ctx = ProgressBar() if show_progress else _nullcontext()
            with ctx:
                dask.compute(lazy_write)
        return None

    if client is not None:
        future = client.compute(arr)
        if show_progress:
            from dask.distributed import progress as _dist_progress
            _dist_progress(future)
        return future.result()

    from dask.diagnostics import ProgressBar
    ctx = ProgressBar() if show_progress else _nullcontext()
    with ctx:
        return arr.compute()


def tile_process(
    image: Union[da.Array, str, Path],
    fn: Callable[[np.ndarray], np.ndarray],
    *,
    tile_shape: Union[tuple[int, ...], Callable[[tuple, Any], tuple], str, None] = None,
    overlap: int = 0,
    iou_threshold: float = 0.8,
    channel: int | None = 0,
    level: int = 0,
    use_gpu: bool = False,
    compute: bool = False,
    progress: bool = False,
    write_to: Union[str, Path, None] = None,
    output_component: str = "labels",
    sequential_labels: bool = False,
    skip_empty: bool = False,
    empty_threshold: float | None = None,
    stage: bool = True,
    stage_dir: Union[str, Path, None] = None,
    keep_stage: bool = False,
    verbose: bool = False,
) -> Union[da.Array, np.ndarray]:
    """Apply *fn* to every tile of *image* and merge labels globally.

    The core workhorse of blockbuster. ``fn`` can be any callable that takes a
    NumPy array and returns an integer label array of the same shape — Cellpose,
    StarDist, Otsu threshold, your own model, anything.

    Parameters
    ----------
    image:
        Dask array *or* path to an OME-ZARR store.
    fn:
        ``(ndarray) -> ndarray`` returning integer labels of the same shape.
        Must be picklable when using distributed schedulers.
    tile_shape:
        Controls tiling before calling *fn*. Accepted values:

        - ``None`` : keep existing dask chunks.
        - ``tuple`` : use this exact tile shape.
        - ``"auto"`` : call ``auto_tile_shape`` based on shape and dtype.
        - ``Callable[[shape, dtype], tuple]`` : called with the image's shape
          and dtype; the return value is used. Use this with
          ``auto_tile_shape_cellpose``:

          .. code-block:: python

              from functools import partial
              from blockbuster import auto_tile_shape_cellpose, tile_process
              tile_fn = partial(auto_tile_shape_cellpose, diameter=30, use_gpu=True)
              result = tile_process("image.zarr", fn, tile_shape=tile_fn)

    overlap:
        Voxels of overlap between adjacent tiles passed to *fn*.
        - ``0`` : labels that *touch* across tile boundaries are merged.
        - ``>0`` : labels in the overlap region are merged by IoU ≥
          ``iou_threshold``. Prefer this for methods that need spatial context
          near boundaries (Cellpose, StarDist, …).
    iou_threshold:
        Minimum IoU for two overlapping labels to be merged (overlap > 0 only).
    channel:
        Channel index when *image* is a path. Ignored for arrays.
    level:
        Pyramid level when *image* is a path (0 = full resolution).
    use_gpu:
        When ``tile_shape="auto"``, size tiles against GPU VRAM instead of RAM.
    compute:
        Compute and return the result immediately as a NumPy array.
    progress:
        Show a progress bar while computing. Requires ``compute=True`` or
        ``write_to`` to be set.
    write_to:
        Zarr store path to stream-write labels while computing (avoids loading
        the full result into RAM). Implies ``compute=True``.
    output_component:
        Array name inside ``write_to``. Default ``"labels"``.
    sequential_labels:
        Renumber merged labels to a contiguous ``1..N`` range. Default False —
        labels stay globally unique but gappy (block-encoded), which is fine for
        counting/measurement. Uses a cheap linear post-pass (O(voxels)), not the
        O(n_chunks²) dask built-in.
    skip_empty:
        Skip *fn* on background tiles. A tile whose max signal is <=
        ``empty_threshold`` returns all-zeros immediately. Biggest speed-up for
        sparse/mostly-background volumes. Use ``estimate_empty_tiles()`` first
        to pick a threshold.
    empty_threshold:
        Intensity at or below which a tile is empty (``skip_empty=True`` only).
        None → auto-derive via Otsu on a bounded sample.
    stage:
        Segment each tile to a temporary zarr *once*, then merge by reading it
        back. Default True — without staging the merge re-evaluates *fn* ~3-4×
        per tile. Strongly recommended for slow models like Cellpose.
    stage_dir:
        Where to put the temp stage store. Default → next to ``write_to``.
    keep_stage:
        Keep the temp stage store after merging (default: delete it).
    verbose:
        Log each tile's location and shape as it is processed.

    Returns
    -------
    da.Array or np.ndarray
        Globally relabeled array (int32). Returns a lazy ``da.Array`` unless
        ``compute=True`` or ``write_to`` is set.

    Examples
    --------
    **Any threshold function:**

    >>> from skimage.filters import threshold_otsu
    >>> from skimage.measure import label
    >>>
    >>> def my_fn(tile):
    ...     return label(tile > threshold_otsu(tile)).astype("int32")
    >>>
    >>> result = tile_process("image.zarr", my_fn, compute=True)

    **Cellpose (via the plugin):**

    >>> from blockbuster.plugins.cellpose import cellpose_fn
    >>>
    >>> fn = cellpose_fn("cyto3", gpu=True, diameter=30)
    >>> result = tile_process(
    ...     "image.zarr", fn,
    ...     tile_shape=(1, 2048, 2048),
    ...     overlap=20,
    ...     write_to="labels.zarr",
    ...     progress=True,
    ... )

    **StarDist:**

    >>> from stardist.models import StarDist2D
    >>> model = StarDist2D.from_pretrained("2D_versatile_fluo")
    >>>
    >>> def stardist_fn(tile):
    ...     norm = tile.astype("float32") / tile.max()
    ...     labels, _ = model.predict_instances(norm)
    ...     return labels.astype("int32")
    >>>
    >>> result = tile_process("image.zarr", stardist_fn,
    ...                       tile_shape=(1, 1024, 1024), overlap=32)

    **Write directly to zarr (no RAM accumulation):**

    >>> tile_process("image.zarr", fn, write_to="labels.zarr", progress=True)
    """
    if progress and not compute and write_to is None:
        raise ValueError("progress=True requires compute=True or write_to")

    # In-process dask workers break the label merge. A GIL-holding fn starves
    # the worker heartbeat and the P2P barrier drops inputs →
    # "FutureCancelledError: lost dependencies".
    _active = _distributed_client()
    if _active is not None and _client_is_in_process(_active):
        raise RuntimeError(
            "Active Dask client uses an in-process worker (processes=False). "
            "This breaks the label merge when fn holds the GIL. Use a "
            "process-based cluster instead:\n"
            "    from blockbuster import make_local_cluster\n"
            "    client, cluster = make_local_cluster(use_gpu=True)\n"
            "or drop the client to use the threaded scheduler "
            "(client.close(); cluster.close())."
        )

    # Load + tile
    image_source_path = None if isinstance(image, da.Array) else str(image)
    _load_chunks: tuple[int, ...] | None = None

    if not isinstance(image, da.Array):
        _peek = load_ome_zarr(image, channel=channel, level=level)
        if callable(tile_shape):
            _load_chunks = tuple(tile_shape(_peek.shape, _peek.dtype))
        elif isinstance(tile_shape, str):
            if tile_shape != "auto":
                raise ValueError(f"Unknown tile_shape value: {tile_shape!r}. Use 'auto', a tuple, or a callable.")
            _load_chunks = auto_tile_shape(_peek.shape, _peek.dtype, use_gpu=use_gpu, verbose=True)
        elif tile_shape is not None:
            _load_chunks = tuple(tile_shape)
        tile_shape = None  # already handled at load time
        if _load_chunks is not None:
            logger.info("Loading zarr with target tiles %s", _load_chunks)
            image = load_ome_zarr(image, channel=channel, level=level, chunks=_load_chunks)
        else:
            image = _peek

    if callable(tile_shape):
        tile_shape = tile_shape(image.shape, image.dtype)
    elif isinstance(tile_shape, str):
        if tile_shape != "auto":
            raise ValueError(f"Unknown tile_shape value: {tile_shape!r}. Use 'auto', a tuple, or a callable.")
        tile_shape = auto_tile_shape(image.shape, image.dtype, use_gpu=use_gpu, verbose=True)

    if tile_shape is not None:
        image = image.rechunk(tile_shape)
        logger.info("Rechunked to %s", tile_shape)

    _eff_chunks: tuple[int, ...] | None = _load_chunks or (
        tile_shape if isinstance(tile_shape, tuple) else None
    )

    n_tiles = int(np.prod([len(c) for c in image.chunks]))
    logger.info(
        "Processing %d tiles (per-axis %s, tile shape %s)",
        n_tiles,
        tuple(len(c) for c in image.chunks),
        tuple(c[0] for c in image.chunks),
    )

    image_for_threshold = image

    # Overlap — build a per-axis depth dict (clips to fit each axis).
    # An integer depth raises if any axis is smaller than the depth, so we
    # cap per axis. In practice z-axis of size 1 (2-D Cellpose) gets depth=0.
    _depth: dict[int, int] = {
        ax: min(overlap, max(0, sum(c) - 1))
        for ax, c in enumerate(image.chunks)
    }

    if overlap > 0:
        # boundary="none" is required: only this boundary mode composes with
        # trim_overlap to recover the original shape. "reflect" keeps the halo.
        image = da.overlap.overlap(image, depth=_depth, boundary="none")

    # Wrap fn with optional empty-tile skipping
    _skip_thr = empty_threshold
    if skip_empty and _skip_thr is None:
        _skip_thr = _auto_empty_threshold(image_for_threshold, channel, level)

    def active_fn(block, block_info=None):
        loc = block_info[0].get("chunk-location") if block_info else "?"
        if skip_empty and block.size and block.max() <= _skip_thr:
            if verbose:
                logger.debug("skip empty tile %s (max<=%s)", loc, _skip_thr)
            return np.zeros(block.shape, dtype=np.int32)
        if verbose:
            logger.debug("process tile %s shape=%s", loc, block.shape)
        return fn(block)

    _use_zarr_merge = stage and (compute or write_to is not None)

    labeled = image.map_blocks(
        active_fn, dtype=np.int32, meta=np.empty((0,) * image.ndim, dtype=np.int32)
    )

    if overlap > 0 and _use_zarr_merge:
        labeled = da.overlap.trim_overlap(labeled, depth=_depth, boundary="none")

    import dask as _dask

    if _active is None and use_gpu:
        _sched_ctx: Any = _dask.config.set(scheduler="threads", num_workers=1)
    else:
        _sched_ctx = _nullcontext()

    # Stage: write each tile's labels to disk once, merge from disk
    stage_path = None
    if stage and (compute or write_to is not None):
        if stage_dir is not None:
            base = str(stage_dir)
        elif write_to is not None:
            base = os.path.dirname(os.path.abspath(str(write_to)))
        elif image_source_path is not None:
            base = os.path.dirname(os.path.abspath(image_source_path))
        else:
            import tempfile
            base = tempfile.mkdtemp(prefix="bb_stage_")
        stage_path = os.path.join(base, "_bb_stage.zarr")
        logger.info("Staging tiles to %s …", stage_path)
        with _sched_ctx:
            _finalise(labeled, progress, stage_path, "staged")
        labeled = da.from_zarr(stage_path, component="staged")

        if skip_empty and _skip_thr is not None:
            def _tile_max(block: np.ndarray) -> np.ndarray:
                return np.full((1,) * block.ndim, int(block.max()), dtype=np.int32)
            _tile_maxes = labeled.map_blocks(
                _tile_max, dtype=np.int32,
                chunks=tuple(tuple(1 for _ in c) for c in labeled.chunks),
            ).compute()
            _n_skip = int((_tile_maxes == 0).sum())
            logger.info(
                "skip_empty: %d/%d tiles ran fn, %d skipped (max<=%.4g)",
                int(_tile_maxes.size) - _n_skip, int(_tile_maxes.size), _n_skip, _skip_thr,
            )

    def _cleanup_stage():
        if stage_path is not None and not keep_stage:
            import shutil
            shutil.rmtree(stage_path, ignore_errors=True)
            logger.info("Removed stage store %s", stage_path)

    # Zarr-native merge (boundary scan → scipy CC → parallel relabel).
    # Works whether or not staging happened. If stage=False, we stage now.
    import tempfile

    if stage_path is None:
        # Wasn't staged yet (stage=False): write labeled tiles to a temp zarr.
        import dask
        from dask.diagnostics import ProgressBar

        _stage_base = str(stage_dir) if stage_dir is not None else tempfile.mkdtemp(prefix="bb_stage_")
        stage_path = os.path.join(_stage_base, "_bb_stage.zarr")
        ctx = ProgressBar() if progress else _nullcontext()
        logger.info("Staging per-tile labels to %s …", stage_path)
        with ctx:
            dask.compute(
                labeled.to_zarr(stage_path, component="staged", overwrite=True, compute=False)
            )

    _nw = min(4, os.cpu_count() or 1)

    if write_to is not None:
        _effective_out = str(write_to)
    else:
        _effective_out = os.path.join(
            tempfile.mkdtemp(prefix="bb_merge_"), "merged.zarr"
        )
        logger.info("write_to not set — merged labels in auto-temp %s", _effective_out)

    zarr_native_merge(
        stage_path, "staged", _effective_out, output_component,
        n_workers=_nw, show_progress=progress,
    )
    if sequential_labels:
        logger.info("Relabelling to contiguous ids…")
        relabel_sequential_zarr(_effective_out, output_component)
    _cleanup_stage()

    result_arr = da.from_zarr(_effective_out, component=output_component)
    if compute or write_to is None:
        return result_arr.compute()
    return result_arr
