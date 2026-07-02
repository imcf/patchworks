"""OME-ZARR conversion plugin for patchworks.

Write any array **or any image file** to a pyramidal OME-NGFF (OME-ZARR)
store, generating downsampled resolution levels for fast multi-scale viewing.

Input handling, in order:

* a dask or NumPy array → used directly (NumPy is wrapped lazily);
* a ``.zarr`` path → read through :func:`patchworks.load_ome_zarr`;
* a ``.ims`` (Imaris) path → read lazily with ``imaris-ims-file-reader``
  (HDF5, no JVM); install with ``pip install "patchworks[imaris]"``;
* any other path (CZI, LIF, ND2, OME-TIFF, …) → opened lazily with
  `bioio <https://github.com/bioio-devs/bioio>`_.

Pixel calibration (physical voxel size) is read from the input — bioio's
``physical_pixel_sizes``, the Imaris resolution metadata, or an existing
OME-ZARR's scale transform — and written into the NGFF ``coordinate
Transformations`` so the µm/pixel sizing is preserved. Pass ``pixel_size=`` to
override or to supply it for bare arrays.

Downsampling uses strided (nearest-neighbour) subsampling — the correct,
label-preserving choice — and only on **X and Y**; ``z`` (and channel/time)
stay at full resolution. Every level is built by reading the *previous level
back from disk* and streaming the downsampled result out through dask with
bounded chunks, so the pyramid never materialises a whole volume in RAM.

This module also exposes :func:`add_pyramid` (add levels to an existing store)
and :func:`write_labels` (store a label image under the NGFF ``labels/`` group).

Usage
-----
>>> from patchworks.plugins.ome_zarr import to_ome_zarr
>>> to_ome_zarr("scan.ims", "scan.zarr")
'scan.zarr'
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Union

import dask.array as da
import numpy as np
import zarr

from .._io import load_ome_zarr

logger = logging.getLogger(__name__)

PixelSize = dict[str, float]

_NGFF_VERSION = "0.4"
_SPATIAL_AXES = frozenset("zyx")
# Only X and Y are downsampled when building pyramids; Z is kept at full
# resolution (microscopy stacks are already coarse and anisotropic in Z).
_DOWNSAMPLE_AXES = frozenset("yx")
# axis names assigned to a bare N-D array, taken from the right:
_DEFAULT_ORDER = "tczyx"
# Per-axis chunk caps so each written chunk stays small (≈tens of MB) and the
# pyramid never needs a whole plane/volume in RAM.
_CHUNK_CAP = {"t": 1, "c": 1, "z": 16, "y": 1024, "x": 1024}


def _default_axes(ndim: int) -> str:
    """Assign trailing OME axis names to an unlabelled array.

    A 2-D array becomes ``"yx"``, 3-D ``"zyx"``, 4-D ``"czyx"``.

    Parameters
    ----------
    ndim : int
        Number of array dimensions.

    Returns
    -------
    str
        The axes string for those trailing dimensions.
    """
    if ndim > len(_DEFAULT_ORDER):
        raise ValueError(
            f"cannot infer axes for a {ndim}-D array; pass axes= explicitly"
        )
    return _DEFAULT_ORDER[len(_DEFAULT_ORDER) - ndim :]


def _axis_type(name: str) -> str:
    """Map an axis letter to its NGFF axis type.

    Parameters
    ----------
    name : str
        Axis letter (``z``/``y``/``x``/``c``/``t``).

    Returns
    -------
    str
        ``"space"``, ``"time"`` or ``"channel"``.
    """
    if name in _SPATIAL_AXES:
        return "space"
    return "time" if name == "t" else "channel"


def _axes_meta(axes: str, calibrated: bool) -> list[dict]:
    """Build NGFF ``axes`` metadata for an axes string.

    Parameters
    ----------
    axes : str
        One letter per axis.
    calibrated : bool
        When True, spatial axes carry a micrometer unit.

    Returns
    -------
    list of dict
        One ``{name, type[, unit]}`` entry per axis.
    """
    meta = []
    for a in axes:
        entry = {"name": a, "type": _axis_type(a)}
        if calibrated and a in _SPATIAL_AXES:
            entry["unit"] = "micrometer"
        meta.append(entry)
    return meta


def _strides(axes: str, downscale: int) -> tuple[int, ...]:
    """Per-axis downsampling stride for one pyramid step.

    Parameters
    ----------
    axes : str
        One letter per axis.
    downscale : int
        Downsampling factor for X/Y.

    Returns
    -------
    tuple of int
        ``downscale`` for X/Y axes, ``1`` for Z/C/T.
    """
    return tuple(downscale if a in _DOWNSAMPLE_AXES else 1 for a in axes)


def _default_chunks(shape: tuple[int, ...], axes: str) -> tuple[int, ...]:
    """Bounded chunk shape so writing a level never blows up RAM.

    Parameters
    ----------
    shape : tuple of int
        Array shape.
    axes : str
        One letter per axis (selects the per-axis cap).

    Returns
    -------
    tuple of int
        Per-axis chunk size, capped by ``_CHUNK_CAP``.
    """
    return tuple(min(s, _CHUNK_CAP.get(a, s)) for s, a in zip(shape, axes))


ShardSpec = Union[bool, tuple[int, ...]]
_SHARD_TARGET_BYTES = 512 * 1024**2  # aim for ~512 MB shards
_ZARR_V3 = int(zarr.__version__.split(".")[0]) >= 3


def _effective_shard(
    requested: tuple[int, ...],
    chunks: tuple[int, ...],
    shape: tuple[int, ...],
) -> tuple[int, ...]:
    """Clamp a requested shard shape to a valid one.

    A shard must be a whole multiple of the inner chunk and should not exceed
    the array's chunk-padded extent.

    Parameters
    ----------
    requested : tuple of int
        Desired shard shape.
    chunks : tuple of int
        Inner chunk shape.
    shape : tuple of int
        Array shape.

    Returns
    -------
    tuple of int
        A shard shape that is a chunk-multiple within the array.
    """
    out = []
    for r, c, s in zip(requested, chunks, shape):
        cap = math.ceil(s / c) * c  # array dim padded up to a whole chunk
        out.append(min(max(c, (r // c) * c), cap))
    return tuple(out)


def _auto_shard(
    chunks: tuple[int, ...], shape: tuple[int, ...], dtype
) -> tuple[int, ...]:
    """Pick a shard shape of roughly ``_SHARD_TARGET_BYTES``.

    Grows the two largest axes equally until the shard reaches the target size,
    then clamps to a valid chunk-multiple.

    Parameters
    ----------
    chunks : tuple of int
        Inner chunk shape.
    shape : tuple of int
        Array shape.
    dtype : data-type
        Array dtype, used to size the shard in bytes.

    Returns
    -------
    tuple of int
        The chosen shard shape.
    """
    itemsize = np.dtype(dtype).itemsize
    base = itemsize
    for c in chunks:
        base *= c
    big = sorted(range(len(chunks)), key=lambda i: shape[i], reverse=True)[:2]
    factor = max(1, int((_SHARD_TARGET_BYTES / max(1, base)) ** 0.5))
    shard = list(chunks)
    for i in big:
        shard[i] = chunks[i] * factor
    return _effective_shard(tuple(shard), chunks, shape)


def _shard_for(
    shard: ShardSpec,
    chunks: tuple[int, ...],
    shape: tuple[int, ...],
    dtype,
) -> Union[tuple[int, ...], None]:
    """Resolve the ``shard`` argument into a concrete shard shape.

    Parameters
    ----------
    shard : bool or tuple of int
        ``False`` → no sharding; ``True`` → auto;
        a tuple → an explicit shard shape.
    chunks : tuple of int
        Inner chunk shape.
    shape : tuple of int
        Array shape.
    dtype : data-type
        Array dtype.

    Returns
    -------
    tuple of int or None
        The shard shape, or ``None`` when not sharding
        (also when zarr is older than v3).
    """
    if not shard:
        return None
    if not _ZARR_V3:
        logger.warning("sharding requires zarr v3; writing unsharded.")
        return None
    if shard is True:
        return _auto_shard(chunks, shape, dtype)
    return _effective_shard(tuple(shard), chunks, shape)


def _progress_ctx(progress: bool, label: str):
    """Return a progress-bar context manager.

    Parameters
    ----------
    progress : bool
        Whether to show a dask progress bar.
    label : str
        Name logged just before the bar.

    Returns
    -------
    contextmanager
        A ``ProgressBar`` when *progress* is set, else a no-op
        context manager.
    """
    if not progress:
        from contextlib import nullcontext

        return nullcontext()
    from dask.diagnostics import ProgressBar

    logger.info("writing %s …", label)
    return ProgressBar()


def _to_zarr_level(
    arr: da.Array,
    group_path: str,
    component: str,
    shard: ShardSpec,
    progress: bool = True,
) -> None:
    """Write one array to ``group_path/component``, optionally sharded.

    Without sharding, ``da.to_zarr`` writes chunk by chunk. With sharding that
    is unsafe — many chunks share one shard file and per-chunk writes race —
    so we create the sharded array explicitly (inner *chunks* + *shards*) and
    store with the dask blocks rechunked to the **shard** size, so each task
    writes one whole shard atomically.

    Parameters
    ----------
    arr : da.Array
        Array to write (its chunk size becomes the inner chunk).
    group_path : str
        Path of the parent zarr group.
    component : str
        Array name within the group.
    shard : bool or tuple of int
        Sharding request; see :func:`_shard_for`.
    progress : bool
        Show a dask progress bar for the write.

    Returns
    -------
    None
    """
    inner = arr.chunksize
    sh = _shard_for(shard, inner, arr.shape, arr.dtype)
    ctx = _progress_ctx(progress, f"{Path(group_path).name}/{component}")
    if not sh:
        with ctx:
            da.to_zarr(arr, group_path, component=component, overwrite=True)
        return
    grp = zarr.open_group(group_path, mode="a")
    if component in grp:
        del grp[component]
    z = grp.create_array(
        name=component,
        shape=arr.shape,
        chunks=inner,
        shards=sh,
        dtype=arr.dtype,
    )
    with ctx:
        arr.rechunk(sh).store(z, lock=True, compute=True)


def _normalize_pixel_size(
    pixel_size: Union[PixelSize, tuple, None], axes: str
) -> PixelSize:
    """Coerce a pixel-size dict/tuple into ``{axis: size}``.

    Parameters
    ----------
    pixel_size : dict, tuple or None
        Voxel size as a per-axis dict or a tuple
        aligned to the spatial axes.
    axes : str
        One letter per axis.

    Returns
    -------
    dict
        ``{axis: size}`` for the spatial axes present (empty if none given).
    """
    if not pixel_size:
        return {}
    if isinstance(pixel_size, dict):
        return {a: float(v) for a, v in pixel_size.items() if a in axes}
    # tuple/list: aligned to the spatial axes present, in axes order
    spatial = [a for a in axes if a in _SPATIAL_AXES]
    return {a: float(v) for a, v in zip(spatial, pixel_size)}


def _base_scale(axes: str, pixel_size: PixelSize) -> list[float]:
    """Build the level-0 NGFF scale vector.

    Parameters
    ----------
    axes : str
        One letter per axis.
    pixel_size : dict
        ``{axis: size}`` for spatial axes.

    Returns
    -------
    list of float
        Physical size per spatial axis, ``1.0`` for C/T.
    """
    return [float(pixel_size.get(a, 1.0)) for a in axes]


def _dataset(name: str, scale: list[float]) -> dict:
    """Build one NGFF ``multiscales`` dataset entry.

    Parameters
    ----------
    name : str
        Component path of the level (e.g. ``"0"``).
    scale : list of float
        Per-axis scale (physical size × downsample factor).

    Returns
    -------
    dict
        A dataset dict with its ``path`` and ``coordinateTransformations``.
    """
    return {
        "path": name,
        "coordinateTransformations": [
            {"type": "scale", "scale": [float(s) for s in scale]}
        ],
    }


def _write_pyramid(
    arr: da.Array,
    axes: str,
    group_path: str,
    *,
    n_levels: int,
    downscale: int,
    chunks: Union[tuple[int, ...], None],
    base_scale: list[float],
    base_name: str = "0",
    write_base: bool = True,
    shard: ShardSpec = False,
    progress: bool = True,
) -> list[dict]:
    """Write pyramid levels into *group_path* and return NGFF datasets.

    Each deeper level is produced by reading the previous level **back from
    disk** and striding it, so the dask graph stays shallow and bounded — no
    whole-volume recomputation, no OOM. Level 0 is named *base_name*; when
    *write_base* is False it is assumed to already exist (used by
    :func:`add_pyramid`).

    Parameters
    ----------
    arr : da.Array
        Full-resolution array.
    axes : str
        One letter per axis.
    group_path : str
        Path of the zarr group to write into.
    n_levels : int
        Maximum number of levels including full resolution.
    downscale : int
        Per-level X/Y downsampling factor.
    chunks : tuple of int or None
        Chunk shape, or a bounded default.
    base_scale : list of float
        Level-0 physical scale per axis.
    base_name : str
        Component name of level 0.
    write_base : bool
        Write level 0, or assume it already exists.
    shard : bool or tuple of int
        Sharding request (see :func:`_shard_for`).
    progress : bool
        Show a per-level progress bar.

    Returns
    -------
    list of dict
        One NGFF dataset entry per written level.
    """
    strides = _strides(axes, downscale)

    if write_base:
        base_chunks = chunks or _default_chunks(arr.shape, axes)
        _to_zarr_level(
            arr.rechunk(base_chunks), group_path, base_name, shard, progress
        )
    datasets = [_dataset(base_name, base_scale)]

    prev_name = base_name
    prev_shape = arr.shape
    for i in range(1, n_levels):
        next_shape = tuple(s // st for s, st in zip(prev_shape, strides))
        if min(next_shape) < 1:
            logger.info("stopping pyramid at level %d (next too small)", i)
            break
        src = da.from_zarr(group_path, component=prev_name)
        nxt = src[tuple(slice(None, None, st) for st in strides)]
        nxt = nxt.rechunk(chunks or _default_chunks(nxt.shape, axes))
        _to_zarr_level(nxt, group_path, str(i), shard, progress)
        scale = [base_scale[k] * (strides[k] ** i) for k in range(len(axes))]
        datasets.append(_dataset(str(i), scale))
        logger.info("pyramid level %d: shape=%s", i, nxt.shape)
        prev_name = str(i)
        prev_shape = nxt.shape
    return datasets


def _write_multiscales(
    group_path: str,
    axes: str,
    datasets: list[dict],
    name: str,
    *,
    calibrated: bool,
) -> None:
    """Write NGFF ``multiscales`` metadata onto *group_path*.

    Parameters
    ----------
    group_path : str
        Path of the zarr group to annotate.
    axes : str
        One letter per axis.
    datasets : list of dict
        Per-level dataset entries (see :func:`_dataset`).
    name : str
        Multiscales name.
    calibrated : bool
        Whether spatial axes carry a micrometer unit.

    Returns
    -------
    None
    """
    group = zarr.open_group(group_path, mode="a")
    group.attrs["multiscales"] = [
        {
            "version": _NGFF_VERSION,
            "name": name,
            "axes": _axes_meta(axes, calibrated),
            "datasets": datasets,
        }
    ]


def _read_zarr_calibration(store: Union[str, Path], axes: str) -> PixelSize:
    """Read level-0 spatial scale from an existing OME-ZARR, if any.

    Parameters
    ----------
    store : str or Path
        Path of the OME-ZARR group.
    axes : str
        One letter per axis (unused for parsing, kept for symmetry).

    Returns
    -------
    dict
        ``{axis: size}`` for spatial axes with a non-unit scale (empty if
        the store has no multiscales metadata).
    """
    try:
        root = zarr.open_group(str(store), mode="r")
        ms = root.attrs["multiscales"][0]
        ax = [a["name"] for a in ms["axes"]]
        scale = ms["datasets"][0]["coordinateTransformations"][0]["scale"]
    except (KeyError, IndexError, TypeError):
        return {}
    return {
        a: float(s)
        for a, s in zip(ax, scale)
        if a in _SPATIAL_AXES and float(s) != 1.0
    }


def _open_bioio(path: str, scene: int) -> tuple[da.Array, str, PixelSize]:
    """Open *path* with bioio, lazily.

    Singleton non-spatial axes (T/C of size 1) are dropped.

    Parameters
    ----------
    path : str
        Image file path.
    scene : int
        Scene index for multi-scene files.

    Returns
    -------
    tuple
        ``(array, axes, pixel_size)`` — a lazy dask array, its axes string
        and a ``{axis: micrometers}`` calibration dict.
    """
    try:
        from bioio import BioImage
    except ImportError as exc:
        ext = Path(path).suffix or "this file type"
        raise ImportError(
            f"reading {ext} requires bioio. Install it with:\n"
            "    pip install 'patchworks[bioio]'\n"
            "plus the matching reader plugin, e.g. bioio-ome-tiff, "
            "bioio-czi, bioio-lif, bioio-nd2."
        ) from exc

    img = BioImage(path)
    img.set_scene(img.scenes[scene])
    order = img.dims.order  # e.g. "TCZYX"
    arr = img.get_image_dask_data(order)  # lazy dask array, no I/O yet

    # Drop leading singleton non-spatial axes for a tidy result.
    keep = [
        i
        for i, name in enumerate(order)
        if name.lower() in _SPATIAL_AXES or arr.shape[i] > 1
    ]
    index = tuple(slice(None) if i in keep else 0 for i in range(arr.ndim))
    arr = arr[index]
    axes = "".join(order[i].lower() for i in keep)

    pixel_size: PixelSize = {}
    pps = getattr(img, "physical_pixel_sizes", None)
    for axis in "zyx":
        val = getattr(pps, axis.upper(), None) if pps is not None else None
        if val:
            pixel_size[axis] = float(val)
    logger.info(
        "bioio opened %s as %s %s cal=%s", path, axes, arr.shape, pixel_size
    )
    return arr, axes, pixel_size


def _open_imaris(path: str, level: int = 0) -> tuple[da.Array, str, PixelSize]:
    """Open one Imaris ``.ims`` resolution level lazily.

    Reads the underlying HDF5 datasets directly (own handle, crops the Imaris
    chunk padding) and stacks the per-(timepoint, channel) 3-D arrays.

    Parameters
    ----------
    path : str
        Imaris ``.ims`` file path.
    level : int
        Resolution level to read (0 = full resolution).

    Returns
    -------
    tuple
        ``(array, axes, pixel_size)`` — a lazy dask array, its axes string
        and a ``{axis: micrometers}`` calibration dict.
    """
    try:
        from imaris_ims_file_reader.ims import ims
    except ImportError as exc:
        raise ImportError(
            "reading Imaris .ims files requires imaris-ims-file-reader. "
            "Install it with:\n    pip install 'patchworks[imaris]'"
        ) from exc

    # Read straight from the underlying HDF5 datasets. The reader's own
    # __getitem__ squeezes singletons and pads to chunk boundaries, which both
    # break da.from_array; the raw h5py datasets slice exactly and keep their
    # native chunking. Imaris stores one 3-D (Z, Y, X) Data array per
    # (timepoint, channel), padded to a chunk multiple — crop to the true
    # extent the reader reports.
    import h5py

    reader = ims(path, ResolutionLevelLock=level)
    n_t = int(getattr(reader, "TimePoints", 1) or 1)
    n_c = int(getattr(reader, "Channels", 1) or 1)
    z, y, x = (int(s) for s in reader.shape[-3:])

    # Open our own h5py handle (the reader closes its own on GC). The Dataset
    # objects keep this File alive for the lifetime of the dask graph, and
    # da.from_array's default read lock makes the (thread-unsafe) h5py reads
    # safe under the threaded scheduler.
    hf = h5py.File(path, "r")
    t_stack = []
    for t in range(n_t):
        c_stack = []
        for c in range(n_c):
            ds = hf[
                f"DataSet/ResolutionLevel {level}/TimePoint {t}/"
                f"Channel {c}/Data"
            ]
            c_stack.append(da.from_array(ds, chunks=ds.chunks)[:z, :y, :x])
        t_stack.append(da.stack(c_stack, axis=0))  # (c, z, y, x)
    arr = da.stack(t_stack, axis=0)  # (t, c, z, y, x)

    # Drop singleton non-spatial axes for a tidy result.
    order = "tczyx"
    keep = [
        i
        for i, name in enumerate(order)
        if name in _SPATIAL_AXES or arr.shape[i] > 1
    ]
    index = tuple(slice(None) if i in keep else 0 for i in range(arr.ndim))
    arr = arr[index]
    axes = "".join(order[i] for i in keep)

    pixel_size: PixelSize = {}
    res = getattr(reader, "resolution", None)  # (z, y, x) in micrometers
    if res is not None and len(res) >= 3:
        pixel_size = {
            "z": float(res[0]),
            "y": float(res[1]),
            "x": float(res[2]),
        }
    logger.info(
        "imaris opened %s as %s %s cal=%s", path, axes, arr.shape, pixel_size
    )
    return arr, axes, pixel_size


def _write_imaris_pyramid(
    path: str,
    out: str,
    *,
    chunks: Union[tuple[int, ...], None],
    overwrite: bool,
    shard: ShardSpec = False,
    progress: bool = True,
) -> str:
    """Copy an Imaris file's own resolution levels into an OME-ZARR.

    Each Imaris ``ResolutionLevel`` is written as a pyramid level with its own
    physical scale, so no downsampling is recomputed. Lazy (h5py-backed) reads
    stream straight to disk.

    Parameters
    ----------
    path : str
        Imaris ``.ims`` file path.
    out : str
        Destination ``.zarr`` store path.
    chunks : tuple of int or None
        Chunk shape, or a bounded default.
    overwrite : bool
        Overwrite an existing store.
    shard : bool or tuple of int
        Sharding request (see :func:`_shard_for`).
    progress : bool
        Show a per-level progress bar.

    Returns
    -------
    str
        The path to the written store.
    """
    from imaris_ims_file_reader.ims import ims

    base = ims(path, ResolutionLevelLock=0)
    n_levels = int(getattr(base, "ResolutionLevels", 1) or 1)

    zarr.open_group(out, mode="w" if overwrite else "w-")
    datasets: list[dict] = []
    axes = ""
    calibrated = False
    for level in range(n_levels):
        arr, axes, ps = _open_imaris(path, level=level)
        scale = _base_scale(axes, ps)
        calibrated = calibrated or bool(ps)
        _to_zarr_level(
            arr.rechunk(chunks or _default_chunks(arr.shape, axes)),
            out,
            str(level),
            shard,
            progress,
        )
        datasets.append(_dataset(str(level), scale))
        logger.info("imaris level %d copied: shape=%s", level, arr.shape)
    _write_multiscales(
        out, axes, datasets, Path(out).stem, calibrated=calibrated
    )
    return out


def _to_dask(
    source: Union[da.Array, np.ndarray, str, Path],
    axes: Union[str, None],
    scene: int,
) -> tuple[da.Array, str, PixelSize]:
    """Resolve *source* into a lazy ``(array, axes, pixel_size)`` triple.

    Dispatches by type: dask/NumPy arrays pass through; ``.zarr`` paths use the
    OME-ZARR loader; ``.ims`` paths use the Imaris reader; anything else uses
    bioio.

    Parameters
    ----------
    source : da.Array, np.ndarray, str or Path
        Array or path to resolve.
    axes : str or None
        Explicit axes, or ``None`` to infer them.
    scene : int
        Scene index for bioio inputs.

    Returns
    -------
    tuple
        ``(array, axes, pixel_size)``.
    """
    if isinstance(source, da.Array):
        return source, axes or _default_axes(source.ndim), {}
    if isinstance(source, np.ndarray):
        return da.asarray(source), axes or _default_axes(source.ndim), {}

    path = str(source)
    if path.endswith(".zarr"):
        arr = load_ome_zarr(source, channel=None)
        ax = axes or _default_axes(arr.ndim)
        return arr, ax, _read_zarr_calibration(source, ax)
    if path.lower().endswith(".ims"):
        arr, detected, ps = _open_imaris(path)
        return arr, axes or detected, ps

    arr, detected, ps = _open_bioio(path, scene)
    return arr, axes or detected, ps


def to_ome_zarr(
    source: Union[da.Array, np.ndarray, str, Path],
    out_path: Union[str, Path],
    *,
    axes: Union[str, None] = None,
    pixel_size: Union[PixelSize, tuple, None] = None,
    scene: int = 0,
    n_levels: int = 5,
    downscale: int = 2,
    chunks: Union[tuple[int, ...], None] = None,
    shard: ShardSpec = False,
    reuse_pyramid: bool = False,
    progress: bool = True,
    overwrite: bool = False,
) -> str:
    """Write *source* as a pyramidal, calibrated OME-ZARR store.

    *source* may be a dask/NumPy array, a ``.zarr`` store, an Imaris ``.ims``
    file, or any image format readable by bioio (CZI, LIF, ND2, OME-TIFF, …).
    File inputs are read lazily; the pyramid is built level-by-level from disk
    with bounded chunks, so the full volume never needs to fit in RAM. Only
    ``x``/``y`` are downsampled; ``z`` (and channel/time) stay full-resolution.

    Parameters
    ----------
    source : da.Array, np.ndarray, str or Path
        Array or path to convert.
    out_path : str or Path
        Destination ``.zarr`` store (a directory).
    axes : str, optional
        One character per array dimension, e.g. ``"zyx"`` or ``"cyx"``.
        ``None`` → inferred from the file metadata or the array dimensions.
    pixel_size : dict, tuple or None, optional
        Physical voxel size in micrometers, as ``{"z": .., "y": .., "x": ..}``
        or a tuple aligned to the spatial axes. ``None`` → read from the input
        (bioio/Imaris/OME-ZARR); falls back to 1.0 (uncalibrated) for bare
        arrays.
    scene : int, optional
        Scene index for multi-scene bioio files.
    n_levels : int, optional
        Maximum number of pyramid levels including full resolution.
    downscale : int, optional
        Per-level X/Y downsampling factor (default 2).
    chunks : tuple of int, optional
        Chunk shape for the written levels. ``None`` → a bounded default.
    shard : bool or tuple of int, optional
        Pack many chunks into one shard file (zarr v3), cutting the file count
        ~100× on huge arrays. ``False`` (default) → unsharded, maximum reader
        compatibility. ``True`` → auto-pick a ~512 MB shard. A tuple sets an
        explicit shard shape (clamped to a chunk multiple). Sharded writes hold
        ~one shard per worker in RAM. Requires zarr v3 (ignored otherwise).
    progress : bool, optional
        Show a per-level dask progress bar (default ``True``). Set ``False`` to
        silence it.
    reuse_pyramid : bool, optional
        *Imaris ``.ims`` only.* Copy the file's **own** resolution levels
        instead of rebuilding the pyramid (faster, no recompute), keeping each
        level's native scale. Ignored for other inputs; falls back to a
        rebuild if the Imaris levels can't be read. Default ``False`` (rebuild,
        for a consistent XY-only, nearest-neighbour NGFF pyramid).
    overwrite : bool, optional
        Overwrite an existing store at *out_path*.

    Returns
    -------
    str
        The path to the written store (``str(out_path)``).

    Examples
    --------
    >>> from patchworks.plugins.ome_zarr import to_ome_zarr
    >>> to_ome_zarr("scan.ims", "scan.zarr", n_levels=4)
    'scan.zarr'
    """
    if downscale < 2:
        raise ValueError("downscale must be >= 2")
    if n_levels < 1:
        raise ValueError("n_levels must be >= 1")

    # Reuse an Imaris file's own resolution pyramid instead of rebuilding it.
    if (
        reuse_pyramid
        and isinstance(source, (str, Path))
        and str(source).lower().endswith(".ims")
    ):
        try:
            return _write_imaris_pyramid(
                str(source),
                str(out_path),
                chunks=chunks,
                overwrite=overwrite,
                shard=shard,
                progress=progress,
            )
        except Exception as exc:
            logger.warning(
                "reuse_pyramid failed (%s); rebuilding the pyramid instead.",
                exc,
            )

    arr, axes, detected = _to_dask(source, axes, scene)
    if len(axes) != arr.ndim:
        raise ValueError(
            f"axes {axes!r} has {len(axes)} entries but array is {arr.ndim}-D"
        )

    ps = _normalize_pixel_size(pixel_size, axes) if pixel_size else detected
    base_scale = _base_scale(axes, ps)

    out = str(out_path)
    zarr.open_group(out, mode="w" if overwrite else "w-")
    datasets = _write_pyramid(
        arr,
        axes,
        out,
        n_levels=n_levels,
        downscale=downscale,
        chunks=chunks,
        base_scale=base_scale,
        shard=shard,
        progress=progress,
    )
    _write_multiscales(out, axes, datasets, Path(out).stem, calibrated=bool(ps))
    return out


def add_pyramid(
    group_path: Union[str, Path],
    *,
    base: str = "0",
    axes: Union[str, None] = None,
    pixel_size: Union[PixelSize, tuple, None] = None,
    n_levels: int = 5,
    downscale: int = 2,
    chunks: Union[tuple[int, ...], None] = None,
    shard: ShardSpec = False,
    progress: bool = True,
) -> str:
    """Add downsampled pyramid levels to an existing single-resolution zarr.

    Reads the full-resolution array already at ``group_path/base``, writes the
    missing levels next to it (lazily, from disk), and (re)writes the NGFF
    ``multiscales`` metadata. Existing calibration is preserved; pass
    *pixel_size* to set it.

    Parameters
    ----------
    group_path : str or Path
        Zarr group containing the full-resolution array at *base*.
    base : str, optional
        Component name of the existing full-resolution level (default
        ``"0"``). Auto-detected from existing ``multiscales`` metadata if
        present, overriding this.
    axes : str, optional
        One letter per axis, e.g. ``"zyx"``. ``None`` → inferred from
        existing metadata, or from the array's dimensionality.
    pixel_size : dict, tuple or None, optional
        Physical voxel size in micrometers. ``None`` → read from the store's
        existing calibration, if any.
    n_levels : int, optional
        Maximum number of levels including the existing full-resolution one
        (default 5).
    downscale : int, optional
        Per-level X/Y downsampling factor (default 2).
    chunks : tuple of int, optional
        Chunk shape for the written levels. ``None`` → a bounded default.
    shard : bool or tuple of int, optional
        Sharding request (see :func:`to_ome_zarr`'s *shard*).
    progress : bool, optional
        Show a per-level dask progress bar (default ``True``).

    Returns
    -------
    str
        The path to the updated group.

    Examples
    --------
    >>> add_pyramid("scan.zarr", n_levels=4)  # doctest: +SKIP
    'scan.zarr'
    """
    if downscale < 2:
        raise ValueError("downscale must be >= 2")
    if n_levels < 1:
        raise ValueError("n_levels must be >= 1")

    gp = str(group_path)
    root = zarr.open_group(gp, mode="r")
    multiscales = root.attrs.get("multiscales")
    if multiscales:
        base = multiscales[0]["datasets"][0]["path"]
        if axes is None:
            axes = "".join(a["name"] for a in multiscales[0]["axes"])

    base_arr = da.from_zarr(gp, component=base)
    if axes is None:
        axes = _default_axes(base_arr.ndim)
    if len(axes) != base_arr.ndim:
        raise ValueError(
            f"axes {axes!r} has {len(axes)} entries but array is "
            f"{base_arr.ndim}-D"
        )

    if pixel_size:
        ps = _normalize_pixel_size(pixel_size, axes)
    else:
        ps = _read_zarr_calibration(gp, axes)
    base_scale = _base_scale(axes, ps)

    datasets = _write_pyramid(
        base_arr,
        axes,
        gp,
        n_levels=n_levels,
        downscale=downscale,
        chunks=chunks,
        base_scale=base_scale,
        base_name=base,
        write_base=False,
        shard=shard,
        progress=progress,
    )
    _write_multiscales(gp, axes, datasets, Path(gp).stem, calibrated=bool(ps))
    return gp


def register_labels(
    image_store: Union[str, Path],
    name: str = "labels",
    *,
    axes: Union[str, None] = None,
    pixel_size: Union[PixelSize, tuple, None] = None,
    n_levels: int = 5,
    downscale: int = 2,
    chunks: Union[tuple[int, ...], None] = None,
    shard: ShardSpec = False,
    progress: bool = True,
) -> str:
    """Pyramidalise and register an existing ``labels/<name>/0`` base level.

    Assumes the full-resolution label array already exists at
    ``image_store/labels/<name>/0``. Adds the downsampled levels, tags the
    group with NGFF ``image-label`` metadata, lists *name* in
    ``labels/.zattrs``, and inherits the parent image's pixel calibration
    (unless *pixel_size* is given).

    Parameters
    ----------
    image_store : str or Path
        OME-ZARR store path containing the image this label belongs to.
    name : str, optional
        Label image name under ``labels/`` (default ``"labels"``).
    axes : str, optional
        One letter per axis. ``None`` → inferred from the label array.
    pixel_size : dict, tuple or None, optional
        Physical voxel size in micrometers. ``None`` → inherited from the
        parent image's own calibration.
    n_levels : int, optional
        Maximum number of pyramid levels including full resolution
        (default 5).
    downscale : int, optional
        Per-level X/Y downsampling factor (default 2).
    chunks : tuple of int, optional
        Chunk shape for the written levels. ``None`` → a bounded default.
    shard : bool or tuple of int, optional
        Sharding request (see :func:`to_ome_zarr`'s *shard*).
    progress : bool, optional
        Show a per-level dask progress bar (default ``True``).

    Returns
    -------
    str
        Path to the label group (``image_store/labels/<name>``).

    Examples
    --------
    >>> register_labels("scan.zarr", "cells")  # doctest: +SKIP
    'scan.zarr/labels/cells'
    """
    store = str(image_store)
    group = f"{store}/labels/{name}"
    if not pixel_size:
        arr0 = da.from_zarr(group, component="0")
        lab_axes = axes or _default_axes(arr0.ndim)
        pixel_size = _read_zarr_calibration(store, lab_axes)
    add_pyramid(
        group,
        base="0",
        axes=axes,
        pixel_size=pixel_size,
        n_levels=n_levels,
        downscale=downscale,
        chunks=chunks,
        shard=shard,
        progress=progress,
    )
    grp = zarr.open_group(group, mode="a")
    grp.attrs["image-label"] = {"version": _NGFF_VERSION}

    labels_grp = zarr.open_group(f"{store}/labels", mode="a")
    registered = list(labels_grp.attrs.get("labels", []))
    if name not in registered:
        registered.append(name)
    labels_grp.attrs["labels"] = registered
    return group


def write_labels(
    image_store: Union[str, Path],
    labels: Union[da.Array, np.ndarray],
    *,
    name: str = "labels",
    axes: Union[str, None] = None,
    pixel_size: Union[PixelSize, tuple, None] = None,
    n_levels: int = 5,
    downscale: int = 2,
    chunks: Union[tuple[int, ...], None] = None,
    shard: ShardSpec = False,
    progress: bool = True,
    overwrite: bool = False,
) -> str:
    """Store *labels* inside *image_store* under the NGFF ``labels/`` group.

    The labels are written as their own multi-scale pyramid at
    ``image_store/labels/<name>/`` and registered in
    ``image_store/labels/.zattrs``, so the image and its segmentation live in a
    single OME-ZARR store. Calibration is inherited from the parent image
    unless *pixel_size* is given.

    Parameters
    ----------
    image_store : str or Path
        OME-ZARR store this label image belongs to.
    labels : da.Array or np.ndarray
        Integer label array (0 = background), same spatial shape as the
        image.
    name : str, optional
        Label image name under ``labels/`` (default ``"labels"``).
    axes : str, optional
        One letter per axis. ``None`` → inferred from *labels*'
        dimensionality.
    pixel_size : dict, tuple or None, optional
        Physical voxel size in micrometers. ``None`` → inherited from the
        parent image's own calibration.
    n_levels : int, optional
        Maximum number of pyramid levels including full resolution
        (default 5).
    downscale : int, optional
        Per-level X/Y downsampling factor (default 2).
    chunks : tuple of int, optional
        Chunk shape for the written levels. ``None`` → a bounded default.
    shard : bool or tuple of int, optional
        Sharding request (see :func:`to_ome_zarr`'s *shard*).
    progress : bool, optional
        Show a per-level dask progress bar (default ``True``).
    overwrite : bool, optional
        Replace an existing label image of the same *name* (default
        ``False``).

    Returns
    -------
    str
        Path to the written label group (``image_store/labels/<name>``).

    Examples
    --------
    >>> from patchworks import merge_tile_labels
    >>> merged = merge_tile_labels(
    ...     "stage.zarr", input_component="staged", write_to="merged.zarr"
    ... )  # doctest: +SKIP
    >>> write_labels("scan.zarr", merged, name="cells")  # doctest: +SKIP
    'scan.zarr/labels/cells'
    """
    arr = labels if isinstance(labels, da.Array) else da.asarray(labels)
    if axes is None:
        axes = _default_axes(arr.ndim)
    if len(axes) != arr.ndim:
        raise ValueError(
            f"axes {axes!r} has {len(axes)} entries but array is {arr.ndim}-D"
        )

    store = str(image_store)
    root = zarr.open_group(store, mode="a")
    parent = root.require_group("labels")
    if overwrite and name in parent:
        del parent[name]
    parent.require_group(name)

    label_group = f"{store}/labels/{name}"
    base = arr.rechunk(chunks or _default_chunks(arr.shape, axes))
    _to_zarr_level(base, label_group, "0", shard, progress)
    return register_labels(
        store,
        name,
        axes=axes,
        pixel_size=pixel_size,
        n_levels=n_levels,
        downscale=downscale,
        chunks=chunks,
        shard=shard,
        progress=progress,
    )
