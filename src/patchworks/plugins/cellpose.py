"""Cellpose plugin for patchworks.

Requires cellpose >= 3.0 (compatible with v3 and v4).

Usage
-----
>>> from patchworks.plugins.cellpose import cellpose_fn  # doctest: +SKIP
>>> from patchworks import tile_process  # doctest: +SKIP
>>>
>>> fn = cellpose_fn("cyto3", gpu=True, diameter=30)  # doctest: +SKIP
>>> result = tile_process("image.zarr", fn, tile_shape=(1, 2048, 2048),  # doctest: +SKIP
...                       overlap=20, write_to="labels.zarr", progress=True)
"""

from __future__ import annotations

import importlib.metadata
import logging
from functools import partial
from typing import Any, Callable

import numpy as np

from .._gpu import retry_on_oom

logger = logging.getLogger(__name__)


def _parse_version(raw: str) -> tuple[int, ...]:
    """Leading numeric components of a version string.

    ``int(x)`` over the raw segments blows up on any suffixed release
    (``"4.0rc1"``, ``"4.0.1.dev0"``) — and at *import* time, where only
    ImportError was being caught, so the package became unimportable rather
    than degrading.
    """
    parts = []
    for segment in raw.split(".")[:2]:
        digits = "".join(c for c in segment if c.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) or (0,)


try:
    from cellpose import models as _cellpose_models

    _CELLPOSE_VERSION: tuple[int, ...] = _parse_version(
        importlib.metadata.version("cellpose")
    )
    _CELLPOSE_V4 = _CELLPOSE_VERSION[0] >= 4
except ImportError:
    _cellpose_models = None
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


def cellpose_anisotropy(voxel_size: dict[str, float]) -> float | None:
    """Cellpose's ``anisotropy`` (z voxel size / lateral voxel size) from a calibration.

    ``do_3D`` assumes isotropic voxels unless told otherwise: without this,
    Cellpose builds its 3-D flow-field consensus across z-planes it thinks
    are spaced the same as the xy pixels, which for real data rarely holds
    and fragments or distorts objects across z. Getting this wrong does not
    fail loudly -- it just produces a subtly (or not so subtly) wrong
    segmentation, which is why deriving it beats retyping it.

    Parameters
    ----------
    voxel_size : dict
        Image calibration, e.g. from
        :func:`patchworks.plugins.ome_zarr.read_pixel_size`. Needs ``z`` and
        ``x`` (or ``y``) for the lateral size.

    Returns
    -------
    float or None
        ``z / lateral``, or None if the calibration lacks what is needed.

    Examples
    --------
    >>> cellpose_anisotropy({"z": 0.24, "y": 0.10833, "x": 0.10833})
    2.215452783162559
    """
    lateral = voxel_size.get("x") or voxel_size.get("y")
    z = voxel_size.get("z")
    if not lateral or not z:
        return None
    return z / lateral


def cellpose_fn(
    model: str = "cyto3",
    *,
    gpu: bool = False,
    diameter: float | None = None,
    do_3D: bool = False,
    channels: list[int] | None = None,
    channel_axis: int | None = None,
    voxel_size: dict[str, float] | None = None,
    intensity_range: "list | tuple | None" = None,
    fragments: "float | None | str" = "auto",
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
    voxel_size:
        Physical voxel size as ``{"z": .., "y": .., "x": ..}``. When
        ``do_3D`` is set and ``anisotropy`` is not already in
        ``cellpose_kwargs``, it is derived from this via
        :func:`cellpose_anisotropy` -- an explicit ``anisotropy=`` always
        wins. The Snakemake workflow passes the image's own calibration
        automatically; from the API,
        :func:`patchworks.plugins.ome_zarr.read_pixel_size` reads it from a
        store.
    intensity_range:
        ``(low, high)``, or one per channel, mapped to 0 and 1 in every
        tile before Cellpose sees it, with Cellpose's own per-input
        normalisation turned off. Without it, each tile is stretched from
        its own percentiles, so neighbouring tiles get different contrast
        and cells are cut at the seams. Measure it once with
        :func:`patchworks.intensity_range`; the workflow does by default.
    fragments:
        Merge objects smaller than this fraction of a typical cell into the
        neighbour they touch most, and drop those touching nothing
        (:func:`patchworks.absorb_fragments`), per tile before the halo is
        trimmed. 3-D Cellpose often splits a cell into a body and slivers,
        and leaves specks in dim regions. ``"auto"`` (default): 0.1 with
        ``do_3D``, off in 2-D; ``None`` turns it off. 3-D runs also smooth
        the flows (``flow3D_smooth=1``) unless told otherwise, Cellpose's own
        remedy for fragmentation along z.
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

    >>> fn = cellpose_fn("cyto3", gpu=True, diameter=30)  # doctest: +SKIP
    >>> result = tile_process("image.zarr", fn, tile_shape=(1, 2048, 2048), overlap=20)  # doctest: +SKIP

    Nuclear segmentation:

    >>> fn = cellpose_fn("nuclei", diameter=15)  # doctest: +SKIP
    >>> result = tile_process("image.zarr", fn, channel=1)  # doctest: +SKIP

    3-D with an explicit anisotropy:

    >>> fn = cellpose_fn("cyto3", gpu=True, do_3D=True, anisotropy=3.0, diameter=20)  # doctest: +SKIP
    >>> from functools import partial  # doctest: +SKIP
    >>> from patchworks import auto_tile_shape_cellpose, tile_process  # doctest: +SKIP
    >>> tile_fn = partial(auto_tile_shape_cellpose, do_3D=True, use_gpu=True, diameter=20)  # doctest: +SKIP
    >>> result = tile_process("image.zarr", fn, tile_shape=tile_fn, overlap=10)  # doctest: +SKIP

    3-D with anisotropy derived from the image's own calibration:

    >>> fn = cellpose_fn("cyto3", gpu=True, do_3D=True, diameter=20,  # doctest: +SKIP
    ...                  voxel_size={"z": 0.24, "y": 0.10833, "x": 0.10833})
    """
    _require_cellpose()
    if do_3D and voxel_size and "anisotropy" not in cellpose_kwargs:
        anisotropy = cellpose_anisotropy(voxel_size)
        if anisotropy is not None:
            logger.info(
                "anisotropy derived from image calibration: %.4g "
                "(z=%.4g, lateral=%.4g)",
                anisotropy,
                voxel_size.get("z"),
                voxel_size.get("x") or voxel_size.get("y"),
            )
            cellpose_kwargs = {**cellpose_kwargs, "anisotropy": anisotropy}
    defaults = applied_defaults(
        do_3D, {**cellpose_kwargs, "fragments": fragments}
    )
    fragments = defaults.pop("fragments", None)
    cellpose_kwargs = {**cellpose_kwargs, **defaults}
    if intensity_range is not None:
        if "normalize" in cellpose_kwargs:
            raise ValueError(
                "give either intensity_range or Cellpose's normalize, not both"
            )
        cellpose_kwargs = {**cellpose_kwargs, "normalize": False}
    cfg = _make_config(
        model, gpu, channels, channel_axis, diameter, do_3D, **cellpose_kwargs
    )
    if intensity_range is not None:
        ranges = np.asarray(intensity_range, dtype=float).reshape(-1, 2)
        cfg["intensity_range"] = ranges.tolist()
    cfg["fragments"] = fragments
    return partial(_run, cellpose_dict=cfg)


def applied_defaults(do_3D: bool, settings: dict[str, Any]) -> dict[str, Any]:
    """What :func:`cellpose_fn` fills in that *settings* left open.

    ``fragments`` (``"auto"`` or absent: 0.1 in 3-D, off in 2-D) and, in
    3-D where the installed Cellpose takes it, ``flow3D_smooth=1``. Used by
    ``cellpose_fn`` itself and by the workflow to record in the labels'
    provenance what actually ran.
    """
    out: dict[str, Any] = {}
    fragments = settings.get("fragments", "auto")
    out["fragments"] = (
        (0.1 if do_3D else None) if fragments == "auto" else fragments
    )
    if do_3D and "flow3D_smooth" not in settings and _accepts("flow3D_smooth"):
        out["flow3D_smooth"] = 1
    return out


def _accepts(name: str) -> bool:
    """Whether the installed Cellpose's ``eval`` takes *name*."""
    import inspect

    cls = getattr(_cellpose_models, "CellposeModel", None)
    try:
        return (
            cls is not None and name in inspect.signature(cls.eval).parameters
        )
    except (TypeError, ValueError):
        return False


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
        *Cellpose 3 only.* ``[cyto, nucleus]``, 1-based into the channel axis
        (0 = grayscale). ``None`` resolves per tile: ``[1, 2]`` when the tile
        carries two channels, else ``[0, 0]``. Cellpose 4 dropped this
        argument, so it is ignored there.
    channel_axis : int or None
        Axis of the tile holding channels, forwarded to ``eval`` for both
        Cellpose 3 and 4. ``None`` means single-channel tiles.
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
        # Left as None ("auto") rather than [0, 0]: _run only knows how many
        # channels a tile actually carries once it has one in hand.
        "channels": channels,
        "channel_axis": channel_axis,
        "diameter": diameter,
        "do_3D": do_3D,
        "cellpose_kwargs": cellpose_kwargs,
    }


def available_models() -> list[str]:
    """Pretrained model names the installed Cellpose actually accepts.

    The name set changed completely between major versions -- v3 ships
    ``cyto3``/``nuclei``/… , v4 replaced them with ``cpsam``-family names --
    and neither version raises on an unknown one. v4 falls back to its
    default model with only a log line, so a v3 name in a v4 environment
    segments everything with the wrong model and nothing fails. Checking the
    name against this list turns that into a config error instead.

    Returns
    -------
    list of str
        Model names, or an empty list when Cellpose isn't installed or
        doesn't publish them (in which case no name can be rejected).
    """
    names = getattr(_cellpose_models, "MODEL_NAMES", None)
    return list(names) if names else []


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
            # v4 ignores model_type=: the name goes to pretrained_model.
            _model_cache[key] = _cellpose_models.CellposeModel(
                pretrained_model=model_type, gpu=gpu
            )
        else:
            _model_cache[key] = _cellpose_models.Cellpose(
                model_type=model_type, gpu=gpu
            )
    return _model_cache[key]


def _drop_cached_models() -> None:
    """Evict the per-process model cache so its VRAM can be reclaimed.

    Waiting out an OOM while still holding a loaded model pinned on the device
    is self-defeating: that memory is exactly what the contending job needs.
    The model is reloaded from the (already downloaded) weights on the next
    call, which costs seconds against a backoff measured in minutes.
    """
    if _model_cache:
        logger.info(
            "releasing %d cached Cellpose model(s) before backing off",
            len(_model_cache),
        )
        _model_cache.clear()


def _eval_with_oom_fallback(
    img: np.ndarray,
    kwargs: dict[str, Any],
    cellpose_dict: dict[str, Any],
) -> np.ndarray:
    """Run ``model.eval``, surviving transient GPU contention.

    Cluster GPUs are often shared: another job's memory footprint can grow
    mid-run and push an otherwise-fine tile size over the edge. Staying on
    GPU matters — this tile can take well over an hour, so falling back to
    CPU would be far worse than waiting. See :func:`patchworks._gpu.retry_on_oom`.

    The model is fetched inside the retry, so an eviction during backoff is
    followed by a reload rather than a use-after-free of the cached handle.

    Parameters
    ----------
    img : np.ndarray
        Image to segment.
    kwargs : dict
        Keyword arguments for ``model.eval``.
    cellpose_dict : dict
        Configuration from :func:`_make_config`.

    Returns
    -------
    np.ndarray
        Label array from ``model.eval``.
    """
    return retry_on_oom(
        lambda: _get_model(cellpose_dict).eval(img, **kwargs)[0],
        enabled=bool(cellpose_dict.get("gpu", False)),
        on_release=_drop_cached_models,
    )


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
    do_3D = cellpose_dict["do_3D"]
    channel_axis = cellpose_dict.get("channel_axis")
    n_channels = block.shape[channel_axis] if channel_axis is not None else 1
    if cellpose_dict.get("intensity_range") is not None:
        from .._intensity import scale_intensity

        block = scale_intensity(
            block, cellpose_dict["intensity_range"], channel_axis
        )

    kwargs: dict[str, Any] = dict(
        channel_axis=channel_axis,
        diameter=cellpose_dict["diameter"],
        do_3D=do_3D,
        **cellpose_dict.get("cellpose_kwargs", {}),
    )
    if not _CELLPOSE_V4:
        # Cellpose 4 (cpsam) dropped `channels` and reads whatever channels
        # the array carries; Cellpose 3 needs the cyto/nucleus pairing named.
        channels = cellpose_dict.get("channels")
        if channels is None:
            channels = [1, 2] if n_channels >= 2 else [0, 0]
        kwargs["channels"] = channels

    # Where z sits once the channel axis is accounted for.
    z_axis = 1 if channel_axis == 0 else 0

    if do_3D:
        kwargs["z_axis"] = z_axis
        masks = _eval_with_oom_fallback(block, kwargs, cellpose_dict)
        return _clean(masks.astype("int32"), cellpose_dict)
    else:
        # Squeeze singleton z so Cellpose gets a clean 2-D image
        spatial = list(block.shape)
        if channel_axis is not None:
            spatial.pop(channel_axis)
        squeeze = len(spatial) == 3 and spatial[0] == 1
        if len(spatial) == 3 and not squeeze:
            raise ValueError(
                f"do_3D is False but this tile has {spatial[0]} z-planes. "
                "Cellpose would receive the stack with no z_axis and treat "
                "the leading axis as channels. Set do_3D: true, or tile with "
                "z=1 to segment plane by plane."
            )
        if squeeze:
            img = block[(slice(None),) * z_axis + (0,)]
            # Dropping z shifts any channel axis that sat behind it.
            if channel_axis is not None and channel_axis > z_axis:
                kwargs["channel_axis"] = channel_axis - 1
        else:
            img = block
        masks = _eval_with_oom_fallback(img, kwargs, cellpose_dict)
        masks = _clean(masks.astype("int32"), cellpose_dict)
        return masks[np.newaxis] if squeeze else masks


def _clean(masks: np.ndarray, cellpose_dict: dict[str, Any]) -> np.ndarray:
    """Fold fragments into the cells they belong to (``fragments=``)."""
    fraction = cellpose_dict.get("fragments")
    if not fraction:
        return masks
    from .._postprocess import absorb_fragments

    return absorb_fragments(masks, fraction=float(fraction))


# Keep the lower-level names available for advanced users
make_cellpose_config = _make_config
get_cellpose_model = _get_model
run_cellpose = _run
