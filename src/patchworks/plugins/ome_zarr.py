"""OME-ZARR conversion plugin for patchworks.

Write any array **or any image file** to a pyramidal OME-NGFF (OME-ZARR)
store, generating downsampled resolution levels for fast multi-scale viewing.

Input handling, in order:

* a dask or NumPy array → used directly (NumPy is wrapped lazily);
* a ``.zarr`` path → read through :func:`patchworks.load_ome_zarr`;
* any other path (CZI, LIF, ND2, OME-TIFF, …) → opened lazily with
  `bioio <https://github.com/bioio-devs/bioio>`_, so the pixels are streamed
  from disk and never fully loaded into RAM.

Downsampling uses strided (nearest-neighbour) subsampling, the correct,
label-preserving choice for segmentation results — interpolating label values
would invent objects that do not exist. The whole pipeline is lazy: each level
is written chunk by chunk via dask, so terabyte inputs convert in bounded RAM.

``bioio`` itself is an optional dependency, and each file format needs its own
reader plugin (``bioio-ome-tiff``, ``bioio-czi``, ``bioio-lif``, …). Install
the base support with ``pip install "patchworks[bioio]"`` plus the reader(s)
you need.

This module also exposes :func:`add_pyramid`, to add resolution levels to an
existing single-resolution store, and :func:`write_labels`, to store a label
image inside an OME-ZARR under the NGFF ``labels/`` group.

Usage
-----
>>> from patchworks.plugins.ome_zarr import to_ome_zarr
>>>
>>> # From any microscopy file (lazy, via bioio):
>>> to_ome_zarr("scan.czi", "scan.zarr")
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

_NGFF_VERSION = "0.4"
_SPATIAL_AXES = frozenset("zyx")
# axis names assigned to a bare N-D array, taken from the right:
_DEFAULT_ORDER = "tczyx"


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


def _axes_meta(axes: str) -> list[dict]:
    """NGFF ``axes`` metadata for an axes string."""
    return [{"name": a, "type": _axis_type(a)} for a in axes]


def _strides(axes: str, downscale: int) -> tuple[int, ...]:
    """Per-axis stride: downsample spatial axes only, others stay at 1."""
    return tuple(downscale if a in _SPATIAL_AXES else 1 for a in axes)


def _write_pyramid(
    arr: da.Array,
    axes: str,
    group_path: str,
    *,
    n_levels: int,
    downscale: int,
    chunks: Union[tuple[int, ...], None],
    base_name: str = "0",
    write_base: bool = True,
) -> list[dict]:
    """Write pyramid levels into *group_path* and return NGFF datasets.

    Level 0 is named *base_name*; deeper levels are ``"1"``, ``"2"``, …. When
    *write_base* is False the full-resolution array is assumed to already exist
    at ``group_path/base_name`` (used by :func:`add_pyramid`) and only the
    downsampled levels are written.
    """
    strides = _strides(axes, downscale)
    datasets: list[dict] = []
    level = arr.rechunk(chunks) if chunks is not None else arr
    for i in range(n_levels):
        comp = base_name if i == 0 else str(i)
        if i > 0 or write_base:
            da.to_zarr(level, group_path, component=comp, overwrite=True)
        scale = [float(s**i) for s in strides]
        datasets.append(
            {
                "path": comp,
                "coordinateTransformations": [
                    {"type": "scale", "scale": scale}
                ],
            }
        )
        logger.info(
            "pyramid level %s: shape=%s -> %s", comp, level.shape, group_path
        )
        next_shape = tuple(s // st for s, st in zip(level.shape, strides))
        if i + 1 < n_levels and min(next_shape) < 1:
            logger.info("stopping pyramid at level %d (next too small)", i)
            break
        level = level[tuple(slice(None, None, st) for st in strides)]
        if chunks is not None:
            level = level.rechunk(chunks)
    return datasets


def _write_multiscales(
    group_path: str, axes: str, datasets: list[dict], name: str
) -> None:
    """Write NGFF ``multiscales`` metadata onto *group_path*."""
    group = zarr.open_group(group_path, mode="a")
    group.attrs["multiscales"] = [
        {
            "version": _NGFF_VERSION,
            "name": name,
            "axes": _axes_meta(axes),
            "datasets": datasets,
        }
    ]


def _open_bioio(path: str, scene: int) -> tuple[da.Array, str]:
    """Open *path* with bioio and return a lazy ``(array, axes)`` pair.

    Singleton non-spatial axes (T/C of size 1) are dropped so the resulting
    array and axes string are as compact as possible while always keeping the
    spatial dimensions.
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
    logger.info("bioio opened %s as %s %s", path, axes, arr.shape)
    return arr, axes


def _to_dask(
    source: Union[da.Array, np.ndarray, str, Path],
    axes: Union[str, None],
    scene: int,
) -> tuple[da.Array, str]:
    """Resolve *source* into a lazy ``(dask_array, axes)`` pair."""
    if isinstance(source, da.Array):
        return source, axes or _default_axes(source.ndim)
    if isinstance(source, np.ndarray):
        return da.asarray(source), axes or _default_axes(source.ndim)

    path = str(source)
    if path.endswith(".zarr"):
        arr = load_ome_zarr(source, channel=None)
        return arr, axes or _default_axes(arr.ndim)

    arr, detected = _open_bioio(path, scene)
    return arr, axes or detected


def to_ome_zarr(
    source: Union[da.Array, np.ndarray, str, Path],
    out_path: Union[str, Path],
    *,
    axes: Union[str, None] = None,
    scene: int = 0,
    n_levels: int = 5,
    downscale: int = 2,
    chunks: Union[tuple[int, ...], None] = None,
    overwrite: bool = False,
) -> str:
    """Write *source* as a pyramidal OME-ZARR store.

    *source* may be a dask/NumPy array, a ``.zarr`` store, or any image file
    readable by bioio (CZI, LIF, ND2, OME-TIFF, …). File inputs are read
    lazily, and every pyramid level is streamed to disk through dask, so the
    full volume never needs to fit in RAM. Only the spatial axes
    (``z``/``y``/``x``) are downsampled; channel/time axes are kept intact.

    Parameters
    ----------
    source : da.Array, np.ndarray, str or Path
        Array or path to convert.
    out_path : str or Path
        Destination ``.zarr`` store (a directory).
    axes : str, optional
        One character per array dimension, e.g. ``"zyx"`` or ``"cyx"``.
        ``None`` → inferred from bioio metadata for files, or from the
        trailing dimensions for bare arrays.
    scene : int, optional
        Scene index to read from multi-scene files (bioio inputs only).
    n_levels : int, optional
        Maximum number of pyramid levels including full resolution.
    downscale : int, optional
        Per-level downsampling factor along each spatial axis (default 2).
    chunks : tuple of int, optional
        Chunk shape for the written levels. ``None`` keeps the chunks.
    overwrite : bool, optional
        Overwrite an existing store at *out_path*.

    Returns
    -------
    str
        The path to the written store (``str(out_path)``).

    Examples
    --------
    >>> from patchworks.plugins.ome_zarr import to_ome_zarr
    >>> to_ome_zarr("scan.czi", "scan.zarr", n_levels=4)
    'scan.zarr'
    """
    if downscale < 2:
        raise ValueError("downscale must be >= 2")
    if n_levels < 1:
        raise ValueError("n_levels must be >= 1")

    arr, axes = _to_dask(source, axes, scene)
    if len(axes) != arr.ndim:
        raise ValueError(
            f"axes {axes!r} has {len(axes)} entries but array is {arr.ndim}-D"
        )

    out = str(out_path)
    zarr.open_group(out, mode="w" if overwrite else "w-")
    datasets = _write_pyramid(
        arr,
        axes,
        out,
        n_levels=n_levels,
        downscale=downscale,
        chunks=chunks,
    )
    _write_multiscales(out, axes, datasets, Path(out).stem)
    return out


def add_pyramid(
    group_path: Union[str, Path],
    *,
    base: str = "0",
    axes: Union[str, None] = None,
    n_levels: int = 5,
    downscale: int = 2,
    chunks: Union[tuple[int, ...], None] = None,
) -> str:
    """Add downsampled pyramid levels to an existing single-resolution zarr.

    Reads the full-resolution array already stored at ``group_path/base``,
    writes the missing downsampled levels next to it, and (re)writes the NGFF
    ``multiscales`` metadata — turning a flat store into a multi-scale one
    in place, lazily.

    Parameters
    ----------
    group_path : str or Path
        Path to the zarr group holding the base array.
    base : str, optional
        Component name of the existing full-resolution array (default
        ``"0"``). Ignored if the group already has ``multiscales`` metadata,
        in which case its level-0 path and axes are reused.
    axes : str, optional
        Axes string. ``None`` → taken from existing metadata, else inferred
        from the array's dimensions.
    n_levels, downscale, chunks
        As in :func:`to_ome_zarr`.

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

    datasets = _write_pyramid(
        base_arr,
        axes,
        gp,
        n_levels=n_levels,
        downscale=downscale,
        chunks=chunks,
        base_name=base,
        write_base=False,
    )
    _write_multiscales(gp, axes, datasets, Path(gp).stem)
    return gp


def write_labels(
    image_store: Union[str, Path],
    labels: Union[da.Array, np.ndarray],
    *,
    name: str = "labels",
    axes: Union[str, None] = None,
    n_levels: int = 5,
    downscale: int = 2,
    chunks: Union[tuple[int, ...], None] = None,
    overwrite: bool = False,
) -> str:
    """Store *labels* inside *image_store* under the NGFF ``labels/`` group.

    The labels are written as their own multi-scale pyramid at
    ``image_store/labels/<name>/`` and registered in
    ``image_store/labels/.zattrs``, so the image and its segmentation live in a
    single OME-ZARR store (the NGFF *image-label* convention).

    Parameters
    ----------
    image_store : str or Path
        Existing OME-ZARR store to attach the labels to.
    labels : da.Array or np.ndarray
        Label array (integer).
    name : str, optional
        Label image name (default ``"labels"``).
    axes, n_levels, downscale, chunks
        As in :func:`to_ome_zarr`.
    overwrite : bool, optional
        Overwrite an existing label image of the same name.

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
    label_group = f"{store}/labels/{name}"
    zarr.open_group(label_group, mode="w" if overwrite else "a")
    datasets = _write_pyramid(
        arr,
        axes,
        label_group,
        n_levels=n_levels,
        downscale=downscale,
        chunks=chunks,
    )
    _write_multiscales(label_group, axes, datasets, name)
    group = zarr.open_group(label_group, mode="a")
    group.attrs["image-label"] = {"version": _NGFF_VERSION}

    # Register the label image in the parent `labels` group.
    labels_grp = zarr.open_group(f"{store}/labels", mode="a")
    registered = list(labels_grp.attrs.get("labels", []))
    if name not in registered:
        registered.append(name)
    labels_grp.attrs["labels"] = registered
    return label_group
