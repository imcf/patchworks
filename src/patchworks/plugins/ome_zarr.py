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
    """
    if ndim > len(_DEFAULT_ORDER):
        raise ValueError(
            f"cannot infer axes for a {ndim}-D array; pass axes= explicitly"
        )
    return _DEFAULT_ORDER[len(_DEFAULT_ORDER) - ndim :]


def _axis_type(name: str) -> str:
    if name in _SPATIAL_AXES:
        return "space"
    return "time" if name == "t" else "channel"


def _axes_meta(axes: str, calibrated: bool) -> list[dict]:
    """NGFF ``axes`` metadata; spatial axes get a µm unit when calibrated."""
    meta = []
    for a in axes:
        entry = {"name": a, "type": _axis_type(a)}
        if calibrated and a in _SPATIAL_AXES:
            entry["unit"] = "micrometer"
        meta.append(entry)
    return meta


def _strides(axes: str, downscale: int) -> tuple[int, ...]:
    """Per-axis stride: downsample X/Y only; Z, C and T stay at 1."""
    return tuple(downscale if a in _DOWNSAMPLE_AXES else 1 for a in axes)


def _default_chunks(shape: tuple[int, ...], axes: str) -> tuple[int, ...]:
    """Bounded chunk shape so writing a level never blows up RAM."""
    return tuple(min(s, _CHUNK_CAP.get(a, s)) for s, a in zip(shape, axes))


def _normalize_pixel_size(
    pixel_size: Union[PixelSize, tuple, None], axes: str
) -> PixelSize:
    """Coerce a pixel-size dict/tuple into ``{axis: size}`` for spatial axes."""
    if not pixel_size:
        return {}
    if isinstance(pixel_size, dict):
        return {a: float(v) for a, v in pixel_size.items() if a in axes}
    # tuple/list: aligned to the spatial axes present, in axes order
    spatial = [a for a in axes if a in _SPATIAL_AXES]
    return {a: float(v) for a, v in zip(spatial, pixel_size)}


def _base_scale(axes: str, pixel_size: PixelSize) -> list[float]:
    """Level-0 NGFF scale per axis: physical size for spatial, 1.0 else."""
    return [float(pixel_size.get(a, 1.0)) for a in axes]


def _dataset(name: str, scale: list[float]) -> dict:
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
) -> list[dict]:
    """Write pyramid levels into *group_path* and return NGFF datasets.

    Each deeper level is produced by reading the previous level **back from
    disk** and striding it, so the dask graph stays shallow and bounded — no
    whole-volume recomputation, no OOM. Level 0 is named *base_name*; when
    *write_base* is False it is assumed to already exist (used by
    :func:`add_pyramid`).
    """
    strides = _strides(axes, downscale)

    if write_base:
        base_chunks = chunks or _default_chunks(arr.shape, axes)
        da.to_zarr(
            arr.rechunk(base_chunks),
            group_path,
            component=base_name,
            overwrite=True,
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
        da.to_zarr(nxt, group_path, component=str(i), overwrite=True)
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
    """Write NGFF ``multiscales`` metadata onto *group_path*."""
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
    """Read level-0 spatial scale from an existing OME-ZARR, if any."""
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
    """Open *path* with bioio → ``(array, axes, pixel_size)``, all lazy."""
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


def _open_imaris(path: str) -> tuple[da.Array, str, PixelSize]:
    """Open an Imaris ``.ims`` file lazily → ``(array, axes, pixel_size)``."""
    try:
        from imaris_ims_file_reader.ims import ims
    except ImportError as exc:
        raise ImportError(
            "reading Imaris .ims files requires imaris-ims-file-reader. "
            "Install it with:\n    pip install 'patchworks[imaris]'"
        ) from exc

    # Full-resolution level; the object is array-like and h5py-backed (lazy).
    reader = ims(path, ResolutionLevelLock=0)
    order = _DEFAULT_ORDER[len(_DEFAULT_ORDER) - reader.ndim :]
    arr = da.from_array(reader, chunks=_default_chunks(reader.shape, order))

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


def _to_dask(
    source: Union[da.Array, np.ndarray, str, Path],
    axes: Union[str, None],
    scene: int,
) -> tuple[da.Array, str, PixelSize]:
    """Resolve *source* into a lazy ``(array, axes, pixel_size)`` triple."""
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
) -> str:
    """Add downsampled pyramid levels to an existing single-resolution zarr.

    Reads the full-resolution array already at ``group_path/base``, writes the
    missing levels next to it (lazily, from disk), and (re)writes the NGFF
    ``multiscales`` metadata. Existing calibration is preserved; pass
    *pixel_size* to set it.

    Returns
    -------
    str
        The path to the updated group.
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
) -> str:
    """Pyramidalise and register an existing ``labels/<name>/0`` base level.

    Assumes the full-resolution label array already exists at
    ``image_store/labels/<name>/0``. Adds the downsampled levels, tags the
    group with NGFF ``image-label`` metadata, lists *name* in
    ``labels/.zattrs``, and inherits the parent image's pixel calibration
    (unless *pixel_size* is given).

    Returns
    -------
    str
        Path to the label group (``image_store/labels/<name>``).
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
    overwrite: bool = False,
) -> str:
    """Store *labels* inside *image_store* under the NGFF ``labels/`` group.

    The labels are written as their own multi-scale pyramid at
    ``image_store/labels/<name>/`` and registered in
    ``image_store/labels/.zattrs``, so the image and its segmentation live in a
    single OME-ZARR store. Calibration is inherited from the parent image
    unless *pixel_size* is given.

    Returns
    -------
    str
        Path to the written label group (``image_store/labels/<name>``).
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
    da.to_zarr(base, label_group, component="0", overwrite=True)
    return register_labels(
        store,
        name,
        axes=axes,
        pixel_size=pixel_size,
        n_levels=n_levels,
        downscale=downscale,
        chunks=chunks,
    )
