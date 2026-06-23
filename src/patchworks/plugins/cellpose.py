"""Cellpose plugin for patchworks.

Requires cellpose >= 3.0 (compatible with v3 and v4).

Usage
-----
>>> from patchworks.plugins.cellpose import cellpose_fn
>>> from patchworks import tile_process
>>>
>>> fn = cellpose_fn("cyto3", gpu=True, diameter=30)
>>> result = tile_process("image.zarr", fn, tile_shape=(1, 2048, 2048),
...                       overlap=20, write_to="labels.zarr", progress=True)
"""

from __future__ import annotations

import importlib.metadata
import logging
from functools import partial
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

try:
    from cellpose import models as _cellpose_models

    _CELLPOSE_VERSION: tuple[int, ...] = tuple(
        int(x) for x in importlib.metadata.version("cellpose").split(".")[:2]
    )
    _CELLPOSE_V4 = _CELLPOSE_VERSION[0] >= 4
except ImportError as _e:
    _cellpose_models = None  # type: ignore[assignment]
    _CELLPOSE_VERSION = (0, 0)
    _CELLPOSE_V4 = False

# Per-process model cache keyed by (model_type, gpu)
_model_cache: dict[tuple, Any] = {}


def _require_cellpose():
    """Raise an actionable ImportError if cellpose is not installed.

    Returns
    -------
    None
    """
    if _cellpose_models is None:
        raise ImportError(
            "cellpose is not installed. Install it with:\n"
            "    pip install cellpose\n"
            "or:\n"
            "    pip install patchworks[cellpose]"
        )


def cellpose_fn(
    model: str = "cyto3",
    *,
    gpu: bool = False,
    diameter: float | None = None,
    do_3D: bool = False,
    channels: list[int] | None = None,
    channel_axis: int | None = None,
    **cellpose_kwargs: Any,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a ready-to-use Cellpose function for ``tile_process``.

    One-liner convenience wrapper: combines model configuration and function
    creation into a single call.

    Parameters
    ----------
    model:
        Cellpose model type: ``"cyto3"``, ``"cyto2"``, ``"nuclei"``, etc.
    gpu:
        Use GPU for inference.
    diameter:
        Expected cell diameter in pixels. ``None`` → Cellpose auto-estimates.
    do_3D:
        Run in 3-D mode. Each tile must contain the full z-stack — use
        ``auto_tile_shape_cellpose(do_3D=True)`` for appropriate tile shapes.
    channels:
        *Cellpose 3 only.* ``[cytoplasm_channel, nucleus_channel]`` (1-based,
        0 = greyscale). ``[0, 0]`` → greyscale. ``[1, 2]`` → cyto=ch1, nuc=ch2.
    channel_axis:
        *Cellpose 4 only.* Index of the channel axis in the input array.
        ``None`` → greyscale input.
    **cellpose_kwargs:
        Extra kwargs forwarded to ``model.eval()``
        (e.g. ``flow_threshold``, ``cellprob_threshold``, ``anisotropy``).

    Returns
    -------
    Callable[[ndarray], ndarray]
        Picklable function ready for ``tile_process``.

    Examples
    --------
    Greyscale 2-D:

    >>> fn = cellpose_fn("cyto3", gpu=True, diameter=30)
    >>> result = tile_process("image.zarr", fn, tile_shape=(1, 2048, 2048), overlap=20)

    Nuclear segmentation:

    >>> fn = cellpose_fn("nuclei", diameter=15)
    >>> result = tile_process("image.zarr", fn, channel=1)

    3-D with anisotropy:

    >>> fn = cellpose_fn("cyto3", gpu=True, do_3D=True, anisotropy=3.0, diameter=20)
    >>> from functools import partial
    >>> from patchworks import auto_tile_shape_cellpose, tile_process
    >>> tile_fn = partial(auto_tile_shape_cellpose, do_3D=True, use_gpu=True, diameter=20)
    >>> result = tile_process("image.zarr", fn, tile_shape=tile_fn, overlap=10)
    """
    _require_cellpose()
    cfg = _make_config(
        model, gpu, channels, channel_axis, diameter, do_3D, **cellpose_kwargs
    )
    return partial(_run, cellpose_dict=cfg)


def _make_config(
    model: str = "cyto3",
    gpu: bool = False,
    channels: list[int] | None = None,
    channel_axis: int | None = None,
    diameter: float | None = None,
    do_3D: bool = False,
    **cellpose_kwargs: Any,
) -> dict[str, Any]:
    """Build a picklable Cellpose configuration dict.

    Parameters
    ----------
    model : str
        Cellpose model type.
    gpu : bool
        Run on the GPU.
    channels : list of int or None
        Cellpose-3 ``[cyto, nucleus]`` channels; defaults to ``[0, 0]``.
    channel_axis : int or None
        Cellpose-4 channel axis.
    diameter : float or None
        Expected cell diameter in pixels.
    do_3D : bool
        Segment in 3-D.
    **cellpose_kwargs : Any
        Extra arguments forwarded to ``model.eval()``.

    Returns
    -------
    dict
        The configuration consumed by :func:`_get_model` and :func:`_run`.
    """
    return {
        "model": model,
        "gpu": gpu,
        "channels": channels if channels is not None else [0, 0],
        "channel_axis": channel_axis,
        "diameter": diameter,
        "do_3D": do_3D,
        "cellpose_kwargs": cellpose_kwargs,
    }


def _get_model(cellpose_dict: dict[str, Any]) -> Any:
    """Return a worker-local cached Cellpose model.

    Parameters
    ----------
    cellpose_dict : dict
        Configuration from :func:`_make_config`.

    Returns
    -------
    Any
        A Cellpose model instance (cached per ``(model, gpu)`` per process).
    """
    _require_cellpose()
    key = (cellpose_dict["model"], cellpose_dict.get("gpu", False))
    if key not in _model_cache:
        gpu = cellpose_dict.get("gpu", False)
        model_type = cellpose_dict["model"]
        if _CELLPOSE_V4:
            _model_cache[key] = _cellpose_models.CellposeModel(
                model_type=model_type, gpu=gpu
            )
        else:
            _model_cache[key] = _cellpose_models.Cellpose(
                model_type=model_type, gpu=gpu
            )
    return _model_cache[key]


def _run(block: np.ndarray, cellpose_dict: dict[str, Any]) -> np.ndarray:
    """Segment one tile with a cached Cellpose model.

    Parameters
    ----------
    block : np.ndarray
        One image tile.
    cellpose_dict : dict
        Configuration from :func:`_make_config`.

    Returns
    -------
    np.ndarray
        Integer (``int32``) label array of the same spatial shape.
    """
    model = _get_model(cellpose_dict)
    do_3D = cellpose_dict["do_3D"]

    if _CELLPOSE_V4:
        kwargs: dict[str, Any] = dict(
            channel_axis=cellpose_dict.get("channel_axis"),
            diameter=cellpose_dict["diameter"],
            do_3D=do_3D,
            **cellpose_dict.get("cellpose_kwargs", {}),
        )
    else:
        kwargs = dict(
            channels=cellpose_dict["channels"],
            diameter=cellpose_dict["diameter"],
            do_3D=do_3D,
            **cellpose_dict.get("cellpose_kwargs", {}),
        )

    if do_3D:
        kwargs["z_axis"] = 0
        return model.eval(block, **kwargs)[0].astype("int32")
    else:
        # Squeeze singleton z so Cellpose gets a clean 2-D image
        squeeze = block.ndim == 3 and block.shape[0] == 1
        img = block[0] if squeeze else block
        masks = model.eval(img, **kwargs)[0].astype("int32")
        return masks[np.newaxis] if squeeze else masks


# Keep the lower-level names available for advanced users
make_cellpose_config = _make_config
get_cellpose_model = _get_model
run_cellpose = _run
