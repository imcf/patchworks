"""Core tile_process function."""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from contextlib import nullcontext as _nullcontext
from pathlib import Path
from typing import Any, Callable, Union

import dask.array as da
import numpy as np

from ._chunks import auto_tile_shape, cpu_allocation, safe_worker_count
from ._cluster import _client_is_in_process, _distributed_client
from ._io import auto_empty_threshold, load_ome_zarr
from ._merge import _remove_scratch, _scratch_store, zarr_native_merge

logger = logging.getLogger(__name__)


def _attach_log_file(path: str) -> None:
    """Tee the ``patchworks`` INFO log to *path* (replacing a prior auto file).

    A previous auto-attached handler is removed first so repeated calls do not
    stack handlers. The package logger level is raised to INFO if needed so the
    messages are actually emitted.

    Parameters
    ----------
    path : str
        Destination log-file path.

    Returns
    -------
    None
    """
    pkg = logging.getLogger("patchworks")
    for h in list(pkg.handlers):
        if getattr(h, "_patchworks_auto", False):
            pkg.removeHandler(h)
            h.close()
    handler = logging.FileHandler(path)
    handler._patchworks_auto = True  # tag so we can find/replace it later
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    pkg.addHandler(handler)
    if pkg.level == logging.NOTSET or pkg.level > logging.INFO:
        pkg.setLevel(logging.INFO)


def _resolve_tile_shape(tile_shape, shape, dtype, use_gpu):
    """Turn a ``tile_shape`` argument into a tuple (or None to keep chunks).

    Parameters
    ----------
    tile_shape : tuple, callable, str or None
        As accepted by :func:`tile_process`.
    shape : tuple of int
        Image shape.
    dtype : data-type
        Image dtype.
    use_gpu : bool
        Size ``"auto"`` tiles against GPU VRAM instead of RAM.

    Returns
    -------
    tuple of int or None
        The tile shape, or None when the existing chunks are kept.
    """
    if tile_shape is None:
        return None
    if callable(tile_shape):
        return tuple(tile_shape(shape, dtype))
    if isinstance(tile_shape, str):
        if tile_shape != "auto":
            raise ValueError(
                f"Unknown tile_shape value: {tile_shape!r}. "
                "Use 'auto', a tuple, or a callable."
            )
        return tuple(
            auto_tile_shape(shape, dtype, use_gpu=use_gpu, verbose=True)
        )
    return tuple(tile_shape)


def _is_zarr_path(path: str) -> bool:
    """Whether *path* names a ``.zarr`` store (tolerating a trailing slash)."""
    return path.rstrip("/\\").endswith(".zarr")


def _start_dashboard_cluster() -> tuple[Any, Any]:
    """Start a 1-worker / 1-thread in-process cluster for a single-GPU run.

    It keeps GPU evals serial (no VRAM contention) while exposing a live Dask
    dashboard for progress.

    Returns
    -------
    tuple
        ``(cluster, client)``, or ``(None, None)`` when ``distributed`` (or
        its dashboard) is unavailable -- the threaded scheduler is used then.
    """
    try:
        from dask.distributed import Client, LocalCluster

        cluster = LocalCluster(
            n_workers=1, threads_per_worker=1, processes=False
        )
    except Exception as exc:  # no distributed/bokeh → threaded fallback
        logger.warning(
            "Could not start a dashboard cluster (%s); "
            "falling back to the threaded scheduler.",
            exc,
        )
        return None, None
    try:
        client = Client(cluster)
    except Exception as exc:
        cluster.close()
        logger.warning(
            "Could not connect to the dashboard cluster (%s); "
            "falling back to the threaded scheduler.",
            exc,
        )
        return None, None
    logger.info("Dask dashboard for this run: %s", client.dashboard_link)
    return cluster, client


def _fn_key(fn: Any) -> Any:
    """A stable description of *fn* for resume fingerprints.

    Module + qualified name, plus a partial's bound arguments (plugins are
    partials over a config dict, so a changed threshold changes the key).
    Object addresses are left out: they differ between runs.
    """
    if isinstance(fn, functools.partial):
        return [
            _fn_key(fn.func),
            repr(fn.args),
            repr(sorted((fn.keywords or {}).items())),
        ]
    module = getattr(fn, "__module__", type(fn).__module__)
    name = getattr(fn, "__qualname__", type(fn).__qualname__)
    return f"{module}.{name}"


def _run_fingerprint(**parts: Any) -> str:
    """Short hash naming a resumable run's stage store."""
    blob = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def _write_json_atomic(path: str, payload: Any) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh)
    os.replace(tmp, path)


def _stage_tiles(
    image: da.Array,
    fn: Callable[[np.ndarray], np.ndarray],
    stage_path: str,
    tile: tuple[int, ...],
    overlap: list[int],
    n_workers: int,
    halo_dir: str | None,
    checkpoint: str | None,
    progress: bool,
) -> dict[int, int]:
    """Run *fn* tile by tile into a stage store, via :func:`stage_tile`.

    The engine behind ``stitch="iou"`` (it keeps each tile's halo, which the
    fused dask pass trims away) and ``resume=True`` (it records each
    finished tile, so a rerun skips it). Returns every tile's label count,
    which also spares the merge its renumbering pass.
    """
    import dask
    from concurrent.futures import ThreadPoolExecutor

    from ._distributed import create_stage, spatial_tiles, stage_tile
    from ._progress import track

    done: dict[int, int] = {}
    if checkpoint is not None and os.path.exists(checkpoint):
        with open(checkpoint) as fh:
            done = {int(k): int(v) for k, v in json.load(fh).items()}
    if not done:
        create_stage(stage_path, image.shape, tile)
    else:
        logger.info(
            "resuming: %d tile(s) already staged in %s", len(done), stage_path
        )
    n_tiles = len(spatial_tiles(image.shape, tile))
    todo = [i for i in range(n_tiles) if i not in done]
    lock = threading.Lock()

    def one(index: int) -> None:
        n = stage_tile(
            image,
            fn,
            stage_path,
            index,
            tile_shape=tile,
            overlap=overlap,
            halo_dir=halo_dir,
        )
        with lock:
            done[index] = n
            if checkpoint is not None:
                _write_json_atomic(checkpoint, done)

    # Each thread reads its own tile synchronously: the parallelism is here,
    # not in a dask pool nested under every thread.
    with dask.config.set(scheduler="synchronous"):
        if n_workers <= 1:
            for _ in track(
                (one(i) for i in todo),
                "stage tiles",
                len(todo),
                enabled=progress,
            ):
                pass
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                for _ in track(
                    pool.map(one, todo),
                    "stage tiles",
                    len(todo),
                    enabled=progress,
                ):
                    pass
    return done


def _stage_fused(
    labeled: da.Array,
    stage_path: str,
    active_client: Any,
    use_gpu: bool,
    max_workers: int | None,
    tile_nbytes: int,
    progress: bool,
) -> None:
    """Stage *labeled* (the fused map_overlap graph) through dask.

    Bounds concurrency to the machine so staging can neither OOM nor pin
    every core -- GPU: one eval at a time; CPU: as many tiles as fit RAM,
    leaving a core free. A distributed client manages its own concurrency.
    """
    import dask as _dask

    temp_cluster = temp_client = None
    try:
        if active_client is None and use_gpu:
            temp_cluster, temp_client = _start_dashboard_cluster()

        if _distributed_client() is None:
            workers = (
                max_workers
                if max_workers is not None
                else safe_worker_count(tile_nbytes, use_gpu=use_gpu)
            )
            workers = max(1, min(workers, cpu_allocation()))
            logger.info("Staging with %d worker thread(s)", workers)
            sched_ctx: Any = _dask.config.set(
                scheduler="threads", num_workers=workers
            )
        else:
            sched_ctx = _nullcontext()

        logger.info("Staging tiles to %s …", stage_path)
        with sched_ctx:
            _stage_to_zarr(labeled, stage_path, "staged", progress)
    finally:
        if temp_client is not None:
            temp_client.close()
            temp_cluster.close()


def _read_amplification(
    native_chunks: tuple[int, ...], tile_shape: tuple[int, ...]
) -> float:
    """Estimate how much extra data each tile read pulls off disk.

    When the store's on-disk chunks are larger than the tile, every tile read
    decodes whole chunks and discards most of them. Returns the ratio of bytes
    read to bytes used (1.0 = no waste).

    Parameters
    ----------
    native_chunks : tuple of int
        The store's on-disk chunk shape.
    tile_shape : tuple of int
        The processing tile shape.

    Returns
    -------
    float
        Read-amplification factor (``bytes_read / bytes_used``).
    """
    import math

    read = used = 1.0
    for n, t in zip(native_chunks, tile_shape):
        if n <= 0 or t <= 0:
            continue
        touched = math.ceil(t / n)  # chunks a tile spans on this axis
        read *= touched * n
        used *= t
    return read / used if used else 1.0


def _stage_to_zarr(
    arr: da.Array, path: str, component: str, show_progress: bool
) -> None:
    """Write *arr* to zarr ``path/component``, never loading it into RAM.

    Parameters
    ----------
    arr : da.Array
        Array to materialise to disk.
    path : str
        Zarr store path.
    component : str
        Array name within the store.
    show_progress : bool
        Show a progress bar while computing.

    Returns
    -------
    None
    """
    import dask

    lazy_write = arr.to_zarr(
        str(path), component=component, overwrite=True, compute=False
    )
    client = _distributed_client()
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


def tile_process(
    image: Union[da.Array, str, Path],
    fn: Callable[[np.ndarray], np.ndarray],
    *,
    tile_shape: Union[
        tuple[int, ...], Callable[[tuple, Any], tuple], str, None
    ] = None,
    overlap: int = 16,
    channel: int | None = 0,
    level: int = 0,
    use_gpu: bool = False,
    max_workers: int | None = None,
    progress: bool = True,
    write_to: Union[str, Path, None] = None,
    output_component: str = "labels",
    pyramid_levels: int = 5,
    pyramid_downscale: int = 2,
    sequential_labels: bool = False,
    skip_empty: bool = False,
    empty_threshold: float | None = None,
    stage_dir: Union[str, Path, None] = None,
    keep_stage: bool = False,
    log_file: Union[str, Path, bool, None] = None,
    verbose: bool = False,
    stitch: str = "touch",
    iou_threshold: float = 0.5,
    resume: bool = False,
) -> da.Array:
    """Apply *fn* to every tile of *image* and merge labels globally.

    The core workhorse of patchworks. ``fn`` can be any callable that takes a
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
              from patchworks import auto_tile_shape_cellpose, tile_process
              tile_fn = partial(auto_tile_shape_cellpose, diameter=30, use_gpu=True)
              result = tile_process("image.zarr", fn, tile_shape=tile_fn)

    overlap:
        Voxels of overlap (halo) added to each tile before *fn* is called, so
        objects near tile boundaries have enough spatial context to be
        segmented correctly (Cellpose, StarDist, …). The halo is trimmed off
        before merging — the output has the original shape. Defaults to ``16``;
        set it to roughly one object diameter (see ``auto_overlap``) for best
        results, or ``0`` to disable.

        Merging is always **touching-label** based: after the halo is trimmed,
        labels that touch across a tile boundary are merged into one object.
    channel:
        Channel index when *image* is a path. Ignored for arrays.
    level:
        Pyramid level when *image* is a path (0 = full resolution).
    use_gpu:
        When ``tile_shape="auto"``, size tiles against GPU VRAM instead of RAM.
        Also forces staging to one tile at a time (no VRAM contention).
    max_workers:
        Cap the worker threads/processes used for staging and merging. ``None``
        (default) auto-sizes to the machine: bounded by available RAM (tile
        size) and CPU (leaves one core free) so a run can neither OOM nor pin
        every core. Ignored when a distributed client is active (it manages its
        own concurrency).
    progress:
        Show progress bars for staging, the label write and the pyramid
        (default ``True``). Set ``False`` to silence them.
    write_to:
        Explicit output zarr store path. Overrides the default behaviour: the
        merged labels are written here as a single-resolution array named
        ``output_component`` (no pyramid). When None (default) and *image* is a
        ``.zarr`` store, labels are written back into that store under the NGFF
        ``labels/<output_component>/`` group with an auto pyramid, so the image
        and its segmentation live in one file. When None and *image* is an
        array, an auto-temp store is used. Every array written uses the
        active :func:`patchworks.compression` codec (zstd level 1 unless
        changed); wrap the call in ``with compression("blosc"):`` to pick
        another.
    output_component:
        Label name. The array inside ``write_to``, or the NGFF label image name
        under ``labels/`` when writing into the input store. Default
        ``"labels"``.
    pyramid_levels:
        Number of resolution levels for the in-store label pyramid (only when
        writing into the input ``.zarr``). Default 5.
    pyramid_downscale:
        Per-level X/Y downsampling factor for that pyramid (Z is kept at full
        resolution). Default 2.
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
    stage_dir:
        Where to put the temporary stage store. ``fn`` is always run once per
        tile to this store, then the merge reads it back from disk (running
        ``fn`` again is never needed). Default → next to ``write_to``, else next
        to the input store, else a system temp directory. The store gets a
        unique ``_pws_stage_<id>.zarr`` name, so concurrent runs sharing a
        directory never overwrite each other's tiles.
    keep_stage:
        Keep the temp stage store after merging (default: delete it, also
        when the run fails). Its path is logged. Useful for debugging.
    log_file:
        Where to tee the ``patchworks`` INFO log (including a per-tile
        ``tile k done`` counter and ETA). ``None``/``False`` (default) writes
        no file -- configure :mod:`logging` as usual; ``True`` writes
        ``patchworks.log`` next to the output; a path writes there. Asking
        for a file also raises the ``patchworks`` logger to INFO.
    verbose:
        Log each tile's location and shape as it is processed.
    stitch:
        How labels are joined across tile boundaries. ``"touch"`` (default)
        joins any two labels that touch there. ``"iou"`` joins them only when
        both tiles' predictions of the overlap zone agree (IoU >=
        ``iou_threshold``), so two distinct cells pressed together at a seam
        stay two; it needs ``overlap > 0`` to have a zone to compare (an axis
        without one falls back to the IoU of the two boundary slices).
    iou_threshold:
        Minimum IoU for ``stitch="iou"`` (default 0.5).
    resume:
        Make an interrupted run resumable: tiles are staged into a store
        named after this run's inputs (image, tiling, overlap, ``fn`` and its
        bound arguments, output), with a record of finished tiles. Rerunning
        the same call skips those tiles; the store is removed once the run
        succeeds, and kept when it fails. ``stitch="iou"`` and ``resume``
        stage tile by tile in threads rather than through dask, so an
        active distributed client is not used for staging.

    Returns
    -------
    da.Array
        Globally relabeled array (int32) backed by the output zarr (the input
        store's ``labels/<name>/0`` by default, ``write_to`` when given, else an
        auto-temp zarr). Never loads the full volume into RAM. Call
        ``.compute()`` yourself only if the result fits in RAM.

    Examples
    --------
    **Any threshold function:**

    >>> from skimage.filters import threshold_otsu
    >>> from skimage.measure import label
    >>>
    >>> def my_fn(tile):
    ...     return label(tile > threshold_otsu(tile)).astype("int32")
    >>>
    >>> result = tile_process("image.zarr", my_fn, write_to="labels.zarr")

    **Cellpose (via the plugin):**

    >>> from patchworks.plugins.cellpose import cellpose_fn
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
    # In-process dask workers break the label merge. A GIL-holding fn starves
    # the worker heartbeat and the P2P barrier drops inputs →
    # "FutureCancelledError: lost dependencies".
    _active = _distributed_client()
    if _active is not None and _client_is_in_process(_active):
        raise RuntimeError(
            "Active Dask client uses an in-process worker (processes=False). "
            "This breaks the label merge when fn holds the GIL. Use a "
            "process-based cluster instead:\n"
            "    from patchworks import make_local_cluster\n"
            "    client, cluster = make_local_cluster(use_gpu=True)\n"
            "or drop the client to use the threaded scheduler "
            "(client.close(); cluster.close())."
        )

    if stitch not in ("touch", "iou"):
        raise ValueError(f"stitch must be 'touch' or 'iou', got {stitch!r}")
    per_tile = stitch == "iou" or resume

    # Load + tile
    image_source_path = None if isinstance(image, da.Array) else str(image)

    # Auto log file (default): tee patchworks' INFO logs to a file next to the
    # output, so a long run leaves a tailable record without notebook setup.
    if log_file:
        if log_file is True:
            if write_to is not None:
                _ldir = os.path.dirname(os.path.abspath(str(write_to)))
            elif image_source_path is not None:
                _ldir = os.path.dirname(os.path.abspath(image_source_path))
            else:
                _ldir = os.getcwd()
            log_file = os.path.join(_ldir, "patchworks.log")
        _attach_log_file(str(log_file))
        logger.info("patchworks log → %s", log_file)

    _load_chunks: tuple[int, ...] | None = None
    _native_chunks: tuple[int, ...] | None = None

    if not isinstance(image, da.Array):
        _peek = load_ome_zarr(image, channel=channel, level=level)
        _native_chunks = _peek.chunksize  # on-disk zarr chunk shape
        _load_chunks = _resolve_tile_shape(
            tile_shape, _peek.shape, _peek.dtype, use_gpu
        )
        tile_shape = None  # already handled at load time
        if _load_chunks is not None:
            logger.info("Loading zarr with target tiles %s", _load_chunks)
            image = load_ome_zarr(
                image, channel=channel, level=level, chunks=_load_chunks
            )
        else:
            image = _peek

    tile_shape = _resolve_tile_shape(
        tile_shape, image.shape, image.dtype, use_gpu
    )
    if tile_shape is not None:
        image = image.rechunk(tile_shape)
        logger.info("Rechunked to %s", tile_shape)

    n_tiles = int(np.prod([len(c) for c in image.chunks]))
    _tile = tuple(c[0] for c in image.chunks)
    logger.info(
        "Processing %d tiles (per-axis %s, tile shape %s)",
        n_tiles,
        tuple(len(c) for c in image.chunks),
        _tile,
    )

    # Warn when the store's on-disk chunks are much larger than the tile: every
    # tile read then decodes whole chunks and throws most away (slow I/O).
    if _native_chunks is not None:
        _amp = _read_amplification(_native_chunks, _tile)
        if _amp >= 4:
            logger.warning(
                "Input chunks %s are much larger than the tile %s → ~%.0fx "
                "read amplification (each tile decodes whole chunks and "
                "discards most). Re-chunk the store near the tile size "
                "(e.g. to_ome_zarr(..., chunks=...) without shard=) or read "
                "the source file directly to avoid wasted I/O.",
                _native_chunks,
                _tile,
                _amp,
            )

    image_for_threshold = image

    # Overlap — build a per-axis depth dict (clips to fit each axis).
    # An integer depth raises if any axis is smaller than the depth, so we
    # cap per axis. In practice z-axis of size 1 (2-D Cellpose) gets depth=0.
    _depth: dict[int, int] = {
        ax: min(overlap, max(0, sum(c) - 1))
        for ax, c in enumerate(image.chunks)
    }

    # Wrap fn with optional empty-tile skipping
    _skip_thr = empty_threshold
    if skip_empty and _skip_thr is None:
        _skip_thr = auto_empty_threshold(image_for_threshold, channel, level)

    # Up-front heads-up for a big 3-D GPU job (z-stack tiles, many of them):
    # an accurate ETA is logged after the first few tiles, but warn early that
    # this is the expensive path and point at the faster alternatives.
    if use_gpu and image.ndim >= 3 and _tile[0] > 4 and n_tiles >= 50:
        logger.warning(
            "Large 3-D GPU job: %d tiles of %s. Per-tile 3-D segmentation is "
            "slow on a single device (a live ETA is logged after the first "
            "tiles). If per-slice results are acceptable, 2-D (z=1 tiles) is "
            "typically ~10x faster, or segment a lower pyramid level.",
            n_tiles,
            _tile,
        )

    # Tile counters + timing so the log shows live "tile k/N + ETA". The
    # threaded scheduler runs tiles concurrently and ``+=`` is not atomic, so
    # the counters are updated under a lock.
    _progress = {"done": 0, "seen": 0, "time": 0.0}
    _progress_lock = threading.Lock()

    def active_fn(block, block_info=None):
        """Run *fn* on one tile, or return zeros for an empty tile.

        Parameters
        ----------
        block : np.ndarray
            One image tile.
        block_info : dict or None
            Dask block metadata (used for logging the tile location).

        Returns
        -------
        np.ndarray
            Integer labels, or an all-zero tile when skipped.
        """
        loc = block_info[0].get("chunk-location") if block_info else "?"
        with _progress_lock:
            _progress["seen"] += 1
        if skip_empty and block.size and block.max() <= _skip_thr:
            if verbose:
                logger.debug("skip empty tile %s (max<=%s)", loc, _skip_thr)
            return np.zeros(block.shape, dtype=np.int32)

        t0 = time.perf_counter()
        out = np.asarray(fn(block))
        dt = time.perf_counter() - t0
        if out.shape != block.shape:
            # Otherwise this surfaces deep in dask/zarr as a broadcast error
            # that names neither fn nor the tile.
            name = getattr(fn, "__name__", type(fn).__name__)
            raise ValueError(
                f"segmentation function {name!r} returned shape {out.shape} "
                f"for a tile of shape {block.shape} (tile {loc}). It must "
                "return one label per input voxel."
            )

        with _progress_lock:
            _progress["done"] += 1
            _progress["time"] += dt
            done, seen = _progress["done"], _progress["seen"]
            avg = _progress["time"] / done
        # Extrapolate remaining non-empty tiles from the empty fraction seen.
        remaining_nonempty = max(0, n_tiles - seen) * (done / seen)
        eta_h = avg * remaining_nonempty / 3600
        logger.info(
            "tile %d done in %.1fs (avg %.1fs); %d/%d tiles seen; ETA ~%.1fh",
            done,
            dt,
            avg,
            seen,
            n_tiles,
            eta_h,
        )
        return out

    _meta = np.empty((0,) * image.ndim, dtype=np.int32)
    if overlap > 0:
        # One fused pass: add the halo, run fn, trim it back off. map_overlap
        # materialises only the halos it needs (no separate overlapped array)
        # and keeps the task graph small. boundary="none" + trim recovers the
        # original shape, so the boundary-slab scan reads clean tiles.
        labeled = da.map_overlap(
            active_fn,
            image,
            depth=_depth,
            boundary="none",
            trim=True,
            dtype=np.int32,
            meta=_meta,
        )
    else:
        labeled = image.map_blocks(active_fn, dtype=np.int32, meta=_meta)

    _tile_nbytes = int(np.prod(labeled.chunksize)) * labeled.dtype.itemsize

    # Stage: run fn once per tile to a temp zarr, then the zarr-native merge
    # reads concrete data from disk (fn is never re-run). Required because the
    # merge scans the labels directly on disk.
    if stage_dir is not None:
        base: str | None = str(stage_dir)
    elif write_to is not None:
        base = os.path.dirname(os.path.abspath(str(write_to)))
    elif image_source_path is not None:
        base = os.path.dirname(os.path.abspath(image_source_path))
    else:
        base = None  # a fresh system temp dir
    halo_dir: str | None = None
    checkpoint: str | None = None
    if resume:
        # Named after the run, so the same call finds it again -- not
        # unique per call like a scratch store, which is the point.
        fingerprint = _run_fingerprint(
            source=image_source_path or image.name,
            shape=image.shape,
            tile=tuple(image.chunksize),
            overlap=_depth,
            fn=_fn_key(fn),
            skip_empty=skip_empty,
            threshold=_skip_thr,
            out=[str(write_to), output_component],
        )
        root = base if base is not None else tempfile.gettempdir()
        os.makedirs(root, exist_ok=True)
        stage_path = os.path.join(root, f"_pws_resume_{fingerprint}.zarr")
        stage_cleanup = stage_path
        checkpoint = os.path.join(stage_path, ".patchworks_done.json")
        logger.info("resumable stage store: %s", stage_path)
    else:
        stage_path, stage_cleanup = _scratch_store(base, "stage")
    if stitch == "iou":
        halo_dir = f"{stage_path}.halo"
    label_counts: dict[int, int] | None = None
    succeeded = False

    # Default: input is a .zarr store and no explicit write_to → labels go back
    # *into* the input store under the NGFF labels/<name>/ group with an auto
    # pyramid, so image + segmentation live in one OME-ZARR.
    _into_input = (
        write_to is None
        and image_source_path is not None
        and _is_zarr_path(image_source_path)
    )
    _merge_cleanup: str | None = None

    # Everything from here on can fail halfway (fn raising, disk full, a
    # killed worker). The dashboard cluster and the scratch stores are torn
    # down whatever happens, so a failed run leaves no process or a stage
    # the size of the whole image behind.
    try:
        if per_tile:
            _workers = (
                max_workers
                if max_workers is not None
                else safe_worker_count(_tile_nbytes, use_gpu=use_gpu)
            )
            _workers = max(1, min(_workers, cpu_allocation()))
            logger.info(
                "Staging tile by tile with %d thread(s) to %s …",
                _workers,
                stage_path,
            )
            label_counts = _stage_tiles(
                image,
                active_fn,
                stage_path,
                tuple(image.chunksize),
                [_depth[ax] for ax in range(image.ndim)],
                _workers,
                halo_dir,
                checkpoint,
                progress,
            )
        else:
            _stage_fused(
                labeled,
                stage_path,
                _active,
                use_gpu,
                max_workers,
                _tile_nbytes,
                progress,
            )

        # NB: no post-staging skip-count pass here — counting skipped tiles by
        # re-reading the whole staged store off disk would double the I/O of
        # the entire run just for a log line. Use estimate_empty_tiles() up
        # front for that figure instead.

        # Merge runs in worker processes (each holds one chunk + an mmap'd
        # LUT); size it to RAM/CPU like staging, capped so we don't spawn a
        # process storm.
        _nw = max_workers or max(1, min(safe_worker_count(_tile_nbytes), 8))

        # The merge always writes its result to a concrete store first.
        if write_to is not None:
            _merge_out = str(write_to)
        else:
            _merge_out, _merge_tmp = _scratch_store(None, "merge")
            if _into_input:
                _merge_cleanup = _merge_tmp

        # sequential=True folds the contiguous renumbering into the merge's
        # own LUT, so it costs a np.unique over the object count rather than
        # the extra full read+write (plus a Python set of every id) that a
        # separate relabel_sequential_zarr pass would.
        zarr_native_merge(
            stage_path,
            "staged",
            _merge_out,
            output_component,
            n_workers=_nw,
            show_progress=progress,
            sequential=sequential_labels,
            label_counts=label_counts,
            halo_dir=halo_dir,
            iou_threshold=iou_threshold,
        )
        succeeded = True
    finally:
        if keep_stage:
            logger.info("Keeping stage store %s", stage_path)
        elif resume and not succeeded:
            logger.info(
                "Keeping stage store %s to resume from: rerun the same call",
                stage_path,
            )
        else:
            _remove_scratch(stage_cleanup)
            _remove_scratch(halo_dir)

    merged = da.from_zarr(_merge_out, component=output_component)
    if not _into_input:
        # Lazy dask array backed by the merge store. Never loads the full
        # volume into RAM. Caller can .compute() if it fits.
        return merged

    # Stream the merged labels into the input store as an NGFF label pyramid,
    # then drop the temporary merge store. write_labels uses da.to_zarr, so
    # this is chunk-streamed and OOM-safe.
    from .plugins.ome_zarr import write_labels

    try:
        label_group = write_labels(
            image_source_path,
            merged,
            name=output_component,
            n_levels=pyramid_levels,
            downscale=pyramid_downscale,
            progress=progress,
            overwrite=True,
            # Segmented at `level`, so calibrated as that level, not level 0.
            level=level,
        )
    finally:
        _remove_scratch(_merge_cleanup)
    logger.info("labels stored in input OME-ZARR under %s", label_group)
    return da.from_zarr(label_group, component="0")
