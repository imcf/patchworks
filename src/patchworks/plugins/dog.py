"""Difference-of-Gaussians plugin for patchworks.

Optional deconvolution (pycudadecon) followed by a DoG blob/spot detector:
threshold the difference of two Gaussian blurs and label connected
components. CPU (scipy) or GPU (cupy) backed.

Usage
-----
>>> from patchworks.plugins.dog import dog_label_fn
>>> from patchworks import tile_process
>>>
>>> fn = dog_label_fn(low_sigma=1.0, high_sigma=3.0, threshold=0.02)
>>> result = tile_process("image.zarr", fn, tile_shape=(1, 2048, 2048),
...                       overlap=8, write_to="labels.zarr", progress=True)

With deconvolution first (widen ``overlap`` to cover the PSF support):

>>> fn = dog_label_fn(
...     low_sigma=1.0, high_sigma=3.0, threshold=0.02,
...     decon_kwargs=dict(psf=psf, dxpsf=xy_scale, dxdata=xy_scale,
...                        dzpsf=z_scale, dzdata=z_scale,
...                        wavelength=wavelength, na=numerical_aperture,
...                        nimm=refractive_index),
... )
>>> result = tile_process("image.zarr", fn, tile_shape=(1, 2048, 2048), overlap=32)

For the Snakemake workflow's ``method: "custom"`` (see
docs/guide/custom_segmentation.md), use the
:func:`segment` adapter instead of the factory directly:

>>> # custom: {module: "patchworks.plugins.dog", function: "segment",
>>> #          kwargs: {low_sigma: 1.0, high_sigma: 3.0, threshold: 0.02}}
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any, Callable

import numpy as np

from .._gpu import free_gpu_caches, retry_on_oom

logger = logging.getLogger(__name__)


def _require_cupy():
    """Raise an actionable ImportError if cupy is not installed.

    Returns
    -------
    None
    """
    try:
        import cupy  # noqa: F401
        import cupyx.scipy.ndimage  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "cupy is not installed. Install a build matching your CUDA "
            "version (e.g. pip install cupy-cuda12x), or use use_gpu=False "
            "for the CPU (scipy) path."
        ) from exc


def _require_pycudadecon():
    """Raise an actionable ImportError if pycudadecon is not installed.

    Returns
    -------
    None
    """
    try:
        import pycudadecon  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pycudadecon is not installed. Install it with:\n"
            "    pip install pycudadecon\n"
            "or drop decon_kwargs to skip deconvolution."
        ) from exc


def decon_voxel_kwargs(
    voxel_size: dict[str, float], psf_voxel_size: dict[str, float] | None = None
) -> dict[str, float]:
    """pycudadecon's voxel-size arguments, derived from a calibration.

    ``dxdata``/``dzdata`` describe the **image**, so they come straight from
    its calibration. ``dxpsf``/``dzpsf`` describe the **PSF file**, which is
    only the same when the PSF was acquired on the same system at the same
    sampling -- pass *psf_voxel_size* when it was not.

    Getting these wrong does not fail loudly: deconvolution just returns a
    subtly wrong result, which is why deriving them beats retyping them.

    Parameters
    ----------
    voxel_size : dict
        Image calibration, e.g. from
        :func:`patchworks.plugins.ome_zarr.read_pixel_size`. Needs ``x`` (or
        ``y``) for the lateral size and ``z`` for the axial one.
    psf_voxel_size : dict, optional
        The PSF's own calibration. Defaults to *voxel_size*.

    Returns
    -------
    dict
        ``dxdata``/``dzdata``/``dxpsf``/``dzpsf``, omitting any the
        calibration cannot supply.

    Examples
    --------
    >>> decon_voxel_kwargs({"z": 0.2, "y": 0.1, "x": 0.1})
    {'dxdata': 0.1, 'dzdata': 0.2, 'dxpsf': 0.1, 'dzpsf': 0.2}
    """

    def _lateral(cal):
        x, y = cal.get("x"), cal.get("y")
        if x and y and abs(x - y) > 1e-9 * max(x, y):
            logger.warning(
                "anisotropic lateral calibration (x=%s, y=%s); pycudadecon "
                "takes a single lateral size, using x.",
                x,
                y,
            )
        return x or y

    psf_cal = psf_voxel_size if psf_voxel_size is not None else voxel_size
    out: dict[str, float] = {}
    for key, value in (
        ("dxdata", _lateral(voxel_size)),
        ("dzdata", voxel_size.get("z")),
        ("dxpsf", _lateral(psf_cal)),
        ("dzpsf", psf_cal.get("z")),
    ):
        if value:
            out[key] = float(value)
    return out


def dog_label_fn(
    low_sigma: float | tuple[float, ...],
    high_sigma: float | tuple[float, ...],
    threshold: float,
    *,
    use_gpu: bool = False,
    decon_kwargs: dict[str, Any] | None = None,
    voxel_size: dict[str, float] | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a ready-to-use DoG labeler for ``tile_process``.

    Parameters
    ----------
    low_sigma, high_sigma:
        Gaussian sigmas (pixels) for the narrow/wide blur.
        ``dog = blur(low_sigma) - blur(high_sigma)``.
    threshold:
        Binary threshold applied to the DoG image (``dog > threshold``).
    use_gpu:
        Run gaussian_filter + label on GPU via cupy/cupyx instead of scipy.
        Independent of ``decon_kwargs`` — pycudadecon always needs a CUDA
        GPU regardless of this flag (it takes/returns plain NumPy), and this
        flag only picks the backend for the blur/label steps that follow.
    decon_kwargs:
        If given, each tile is first deconvolved via
        ``pycudadecon.decon(block, **decon_kwargs)`` before the DoG step.
        ``None`` (default) skips deconvolution. pycudadecon is CUDA-only, so
        a SLURM job running this needs a GPU allocated regardless of
        ``use_gpu`` above. Widen ``tile_process``'s ``overlap`` to cover the
        PSF support when deconvolving, so edge tiles don't get truncated
        context.

        ``dxdata``/``dzdata``/``dxpsf``/``dzpsf`` are filled in from
        *voxel_size* when you leave them out, so the voxel sizes cannot drift
        away from the image they describe. Anything you set explicitly wins.
    voxel_size:
        Physical voxel size as ``{"z": .., "y": .., "x": ..}``. The Snakemake
        workflow passes the image's own calibration automatically; from the
        API, :func:`patchworks.plugins.ome_zarr.read_pixel_size` reads it from
        a store.

    Returns
    -------
    Callable[[ndarray], ndarray]
        Picklable function ready for ``tile_process``.
    """
    if use_gpu:
        _require_cupy()
    if decon_kwargs is not None:
        _require_pycudadecon()
        if voxel_size:
            # Derived values fill gaps only -- an explicit dxpsf for a PSF
            # sampled differently from the data must not be overwritten.
            derived = decon_voxel_kwargs(voxel_size)
            missing = {
                k: v for k, v in derived.items() if k not in decon_kwargs
            }
            if missing:
                logger.info(
                    "decon voxel sizes taken from the image calibration: %s",
                    missing,
                )
                decon_kwargs = {**decon_kwargs, **missing}
    cfg = {
        "low_sigma": low_sigma,
        "high_sigma": high_sigma,
        "threshold": threshold,
        "use_gpu": use_gpu,
        "decon_kwargs": decon_kwargs,
    }
    return partial(_run, dog_dict=cfg)


def _run(block: np.ndarray, dog_dict: dict[str, Any]) -> np.ndarray:
    """Deconvolve (optional), then DoG-threshold-label one tile.

    Parameters
    ----------
    block : np.ndarray
        One image tile.
    dog_dict : dict
        Configuration from :func:`dog_label_fn`.

    Returns
    -------
    np.ndarray
        Integer (``int32``) label array of the same shape.
    """
    use_gpu = dog_dict["use_gpu"]
    # Deconvolution is CUDA regardless of use_gpu, so both paths can hit a
    # device OOM. Retrying is worth it on a shared GPU: the tile is fine, a
    # co-tenant just grew. cupy's OutOfMemoryError is not a RuntimeError, so
    # it went entirely uncaught before.
    gpu_involved = bool(use_gpu or dog_dict["decon_kwargs"] is not None)
    return retry_on_oom(
        lambda: _segment_once(block, dog_dict, use_gpu),
        enabled=gpu_involved,
    )


def _segment_once(
    block: np.ndarray, dog_dict: dict[str, Any], use_gpu: bool
) -> np.ndarray:
    """One decon + DoG + label pass; see :func:`_run` for the retry wrapper."""
    img = block.astype("float32")

    decon_kwargs = dog_dict["decon_kwargs"]
    if decon_kwargs is not None:
        # ponytail: re-inits the GPU/OTF context on every tile via the
        # one-shot decon() API. Ceiling: per-tile setup cost dominates on
        # many small tiles. Upgrade: cache a pycudadecon.RLContext per
        # worker process (see cellpose.py's _model_cache) if that shows up
        # in the per-tile timing that tile_process logs.
        from pycudadecon import decon

        img = decon(images=img, **decon_kwargs)

    if use_gpu:
        import cupy as cp
        from cupyx.scipy.ndimage import gaussian_filter, label

        img = cp.asarray(img)
    else:
        from scipy.ndimage import gaussian_filter, label

    low_blur = high_blur = dog_image = mask = labels = None
    try:
        low_blur = gaussian_filter(img, sigma=dog_dict["low_sigma"])
        high_blur = gaussian_filter(img, sigma=dog_dict["high_sigma"])
        dog_image = low_blur - high_blur
        low_blur = high_blur = None  # peak is here; drop what's already used
        mask = dog_image > dog_dict["threshold"]
        dog_image = None
        labels, _ = label(mask)
        mask = None
        labels = labels.astype("int32")
        out = cp.asnumpy(labels) if use_gpu else labels
    finally:
        if use_gpu:
            # Several tile-sized buffers are live at the peak above, and
            # cupy's pool never returns a block to the driver on its own — so
            # without this a long-lived worker's footprint only ever grows.
            # Drop our references first, or freeing the pool frees nothing.
            del low_blur, high_blur, dog_image, mask, labels, img
            free_gpu_caches()
    return out


def segment(tile: np.ndarray, **kwargs: Any) -> np.ndarray:
    """``dog_label_fn`` as a direct-call function for ``method: "custom"``.

    Snakemake's ``custom`` method (see docs/guide/snakemake.md) calls
    ``segment(tile, **kwargs)`` directly rather than building a factory
    first. This adapts :func:`dog_label_fn` to that contract, so this
    plugin needs no dedicated wiring in the workflow's ``build_fn`` —
    ``method: "custom"`` with ``module: "patchworks.plugins.dog"`` already
    covers it.

    Parameters
    ----------
    tile : np.ndarray
        One image tile.
    **kwargs : Any
        Forwarded to :func:`dog_label_fn` (``low_sigma``, ``high_sigma``,
        ``threshold``, ``use_gpu``, ``decon_kwargs``).

    Returns
    -------
    np.ndarray
        Integer (``int32``) label array of the same shape.
    """
    return dog_label_fn(**kwargs)(tile)


# ``segment`` takes **kwargs, so its own signature accepts anything. Point at
# the function that really validates them so the workflow can reject a typo'd
# key in `prepare` instead of on a GPU node hours later.
segment.patchworks_kwargs_target = dog_label_fn


# Keep the lower-level name available for advanced users
run_dog_label = _run
