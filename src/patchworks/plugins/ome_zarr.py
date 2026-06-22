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

Usage
-----
>>> from patchworks.plugins.ome_zarr import to_ome_zarr
>>>
>>> # From any microscopy file (lazy, via bioio):
>>> to_ome_zarr("scan.czi", "scan.zarr")
'scan.zarr'
>>>
>>> # From the labels produced by tile_process:
>>> import dask.array as da
>>> to_ome_zarr(da.from_zarr("labels.zarr", component="labels"),
...             "labels_pyramid.zarr", axes="zyx")
'labels_pyramid.zarr'
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
_DEFAULT_ORDER = "tczyx"  # axis names assigned to a bare N-D array, from the right


def _default_axes(ndim: int) -> str:
    """Assign trailing OME axis names to an unlabelled array.

    A 3-D array becomes ``"zyx"``, 4-D ``"czyx"``, 5-D ``"tczyx"``.
    """
    if ndim > len(_DEFAULT_ORDER):
        raise ValueError(
            f"cannot infer axes for a {ndim}-D array; pass axes= explicitly"
        )
    return _DEFAULT_ORDER[len(_DEFAULT_ORDER) - ndim :]


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
            "plus the matching reader plugin, e.g. bioio-ome-tiff, bioio-czi, "
            "bioio-lif, bioio-nd2."
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
        One character per array dimension, e.g. ``"zyx"``, ``"cyx"`` or
        ``"tczyx"``. ``None`` → inferred from bioio metadata for files, or from
        the trailing dimensions for bare arrays. Length must equal the number
        of array dimensions.
    scene : int, optional
        Scene index to read from multi-scene files (bioio inputs only).
    n_levels : int, optional
        Maximum number of pyramid levels including full resolution. Fewer
        levels are written if a spatial dimension would shrink below 1 px.
    downscale : int, optional
        Per-level downsampling factor along each spatial axis (default 2).
    chunks : tuple of int, optional
        Chunk shape for the written levels. ``None`` keeps the array's chunks.
    overwrite : bool, optional
        Overwrite an existing store at *out_path*.

    Returns
    -------
    str
        The path to the written store (``str(out_path)``).

    Raises
    ------
    ValueError
        If ``len(axes)`` does not match the array, ``n_levels < 1`` or
        ``downscale < 2``.
    ImportError
        If a non-zarr file is given but bioio is not installed.

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

    # Per-axis stride: downsample spatial axes only, leave c/t at stride 1.
    strides = tuple(downscale if a in _SPATIAL_AXES else 1 for a in axes)

    out = str(out_path)
    # Create (or wipe) the group; component arrays are written by dask below.
    zarr.open_group(out, mode="w" if overwrite else "w-")

    datasets: list[dict] = []
    level = arr.rechunk(chunks) if chunks is not None else arr
    for i in range(n_levels):
        da.to_zarr(level, out, component=str(i), overwrite=overwrite)
        # NGFF scale transform: downscale**i on spatial axes, 1.0 elsewhere.
        scale = [float(s ** i) for s in strides]
        datasets.append(
            {
                "path": str(i),
                "coordinateTransformations": [{"type": "scale", "scale": scale}],
            }
        )
        logger.info("OME-ZARR level %d: shape=%s -> %s", i, level.shape, out)

        # Stop once another downsample would erase a spatial dimension.
        next_shape = tuple(s // st for s, st in zip(level.shape, strides))
        if i + 1 < n_levels and min(next_shape) < 1:
            logger.info("stopping pyramid at level %d (next level too small)", i)
            break
        level = level[tuple(slice(None, None, st) for st in strides)]
        if chunks is not None:
            level = level.rechunk(chunks)

    root = zarr.open_group(out, mode="a")
    root.attrs["multiscales"] = [
        {
            "version": _NGFF_VERSION,
            "name": Path(out).stem,
            "axes": [
                {
                    "name": a,
                    "type": "space"
                    if a in _SPATIAL_AXES
                    else "time"
                    if a == "t"
                    else "channel",
                }
                for a in axes
            ],
            "datasets": datasets,
        }
    ]
    return out
