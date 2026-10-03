"""PlantSeg plugin for patchworks: boundary U-Net + graph partitioning.

PlantSeg (https://github.com/kreshuklab/plant-seg) segments cells from a
membrane (or cell-wall) stain in two steps: a 3-D U-Net predicts where the
cell boundaries are, then the boundary map is partitioned into cells --
over-segmented into supervoxels by a distance-transform watershed, which an
agglomeration (GASP, mutex watershed, multicut) merges back across the weak
boundaries. A boundary U-Net trained on membranes sees walls a generalist
model misses, and the partitioning closes a wall that is faint in places.

With a nuclear channel, the nuclei constrain the result too:

``"lifted_multicut"``
    PlantSeg's lifted multicut: supervoxels in the same nucleus are pulled
    together, supervoxels in different nuclei pushed apart.
``"nuclei_watershed"``
    The boundary map flooded from the nuclei, one cell per nucleus (the
    seeded flooding of :mod:`patchworks.plugins.watershed`, on the U-Net's
    boundaries rather than the raw stain).

PlantSeg is installed from conda-forge (``plant-seg``), not PyPI -- see the
``plantseg`` environment of the workflow's pixi.toml. Its U-Nets download
from the PlantSeg model zoo on first use into ``$PLANTSEG_HOME`` (default
``~/.plantseg_models``): run :func:`fetch_model` once on a node with
internet access when the GPU nodes have none.

Usage
-----
>>> # method: "custom"
>>> # nuclei_channel: 1          # for lifted_multicut / nuclei_watershed
>>> # custom: {module: "patchworks.plugins.plantseg", function: "segment",
>>> #          kwargs: {model: "generic_confocal_3D_unet",
>>> #                   segmentation: "nuclei_watershed"}}

From the API:

>>> from patchworks.plugins.plantseg import plantseg_fn  # doctest: +SKIP
>>> fn = plantseg_fn("generic_confocal_3D_unet", segmentation="gasp",  # doctest: +SKIP
...                  voxel_size={"z": 0.5, "y": 0.2, "x": 0.2})
>>> tile_process("image.zarr", fn, tile_shape=(32, 512, 512), overlap=(4, 40, 40))  # doctest: +SKIP
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any, Callable

import numpy as np

from .._gpu import free_gpu_caches, retry_on_oom
from .watershed import (
    _sigma,
    foreground_mask,
    max_radius_px,
    nuclei_seeds,
    seeded_watershed,
    split_channels,
)

logger = logging.getLogger(__name__)

#: How the boundary map becomes cells.
SEGMENTATIONS = (
    "gasp",
    "mutex_ws",
    "multicut",
    "dt_watershed",
    "lifted_multicut",
    "nuclei_watershed",
)
#: The ones that need the nuclear channel (``nuclei_channel`` in the config).
NEEDS_NUCLEI = ("lifted_multicut", "nuclei_watershed")
#: Rescaling closer than this to 1 on every axis is skipped.
_RESCALE_TOLERANCE = 0.1


def _require_plantseg():
    """Raise an actionable ImportError if PlantSeg is not installed."""
    try:
        import plantseg.functionals.prediction  # noqa: F401
        import plantseg.functionals.segmentation  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "PlantSeg is not installed. It is on conda-forge, not PyPI:\n"
            "    pixi install -e plantseg      (workflow/pixi.toml)\n"
            "or\n"
            "    conda install -c conda-forge plant-seg"
        ) from exc


def available_models() -> list[str]:
    """Names in PlantSeg's model zoo (empty without PlantSeg)."""
    try:
        from plantseg.core.zoo import model_zoo
    except ImportError:
        return []
    return list(model_zoo.get_model_names())


def model_resolution(model: str) -> tuple[float, ...] | None:
    """The voxel size (z, y, x) in micrometres a zoo model was trained at."""
    try:
        from plantseg.core.zoo import model_zoo

        res = model_zoo.get_model_resolution(model)
    except Exception:
        return None
    return tuple(float(r) for r in res) if res else None


def fetch_model(model: str = "generic_confocal_3D_unet") -> None:
    """Download a zoo model into ``$PLANTSEG_HOME`` now, not on a GPU node.

    Compute nodes often have no internet access; the segment jobs then load
    the copy fetched here.
    """
    _require_plantseg()
    from plantseg.core.zoo import model_zoo

    model_zoo.get_model_by_name(model)


def rescale_factors(
    voxel_size: dict[str, float] | None,
    resolution: tuple[float, ...] | None,
    ndim: int,
) -> tuple[float, ...] | None:
    """Zoom per axis taking the image to the model's training resolution.

    ``None`` when either is unknown or every factor is within 10 % of 1.

    Examples
    --------
    >>> rescale_factors({"z": 0.47, "y": 0.3, "x": 0.3}, (0.235, 0.15, 0.15), 3)
    (2.0, 2.0, 2.0)
    >>> rescale_factors({"z": 0.24, "y": 0.15, "x": 0.15}, (0.235, 0.15, 0.15), 3)
    """
    if not voxel_size or not resolution:
        return None
    axes = "zyx"[-ndim:]
    res = tuple(resolution)[-ndim:]
    factors = []
    for axis, r in zip(axes, res):
        v = voxel_size.get(axis)
        factors.append(float(v) / float(r) if v and r else 1.0)
    if all(abs(f - 1.0) <= _RESCALE_TOLERANCE for f in factors):
        return None
    return tuple(round(f, 6) for f in factors)


def plantseg_fn(
    model: str | None = "generic_confocal_3D_unet",
    *,
    model_id: str | None = None,
    config_path: str | None = None,
    weights_path: str | None = None,
    segmentation: str = "gasp",
    beta: float = 0.6,
    post_minsize: int = 100,
    ws_threshold: float = 0.5,
    ws_sigma_seeds: float = 2.0,
    ws_min_size: int = 50,
    ws_stacked: bool = False,
    nuclei_sigma: Any = 1.0,
    nuclei_threshold: float | None = None,
    nuclei_min_size: int = 50,
    foreground: str | float | None = None,
    foreground_sigma: Any = 2.0,
    max_radius_um: float | None = None,
    boundary_channel: int = 0,
    rescale: bool = True,
    patch: tuple[int, ...] | None = None,
    device: str = "cuda",
    n_threads: int | None = None,
    voxel_size: dict[str, float] | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a PlantSeg segmentation for ``tile_process``.

    Parameters
    ----------
    model :
        PlantSeg zoo model (``available_models()``), e.g.
        ``"generic_confocal_3D_unet"`` or ``"generic_light_sheet_3D_unet"``.
    model_id :
        Or a BioImage.IO model zoo id instead of *model*.
    config_path, weights_path :
        Or your own trained U-Net (PlantSeg's training config and weights).
    segmentation :
        One of :data:`SEGMENTATIONS`. ``"gasp"`` (default), ``"mutex_ws"``
        and ``"multicut"`` agglomerate watershed supervoxels;
        ``"dt_watershed"`` stops at the supervoxels; ``"lifted_multicut"``
        and ``"nuclei_watershed"`` also use the nuclear channel.
    beta :
        Agglomeration bias: lower merges more (under-segments), higher
        splits more. PlantSeg's GUI default is 0.6.
    post_minsize :
        Cells smaller than this many voxels are merged into a neighbour.
    ws_threshold, ws_sigma_seeds, ws_min_size, ws_stacked :
        The supervoxel watershed (PlantSeg's ``dt_watershed``); *ws_stacked*
        runs it plane by plane, for a coarse z.
    nuclei_sigma, nuclei_threshold, nuclei_min_size :
        How nuclei are found in the nuclear channel, see
        :func:`patchworks.plugins.watershed.nuclei_seeds`.
    foreground, foreground_sigma, max_radius_um :
        Background masking, see
        :func:`patchworks.plugins.watershed.foreground_mask`: a boundary
        U-Net puts cells everywhere, empty space included.
    boundary_channel :
        Output channel holding the boundaries (0 for the boundary models).
    rescale :
        Resample each tile to the model's training voxel size before
        predicting (and the prediction back), from *voxel_size*. The single
        biggest factor in a pretrained U-Net's quality.
    patch :
        U-Net patch shape; ``None`` lets PlantSeg size it to the GPU.
    device :
        ``"cuda"`` or ``"cpu"``.
    n_threads :
        Threads for the watershed and agglomeration.
    voxel_size :
        ``{"z": .., "y": .., "x": ..}`` in micrometres; the workflow passes
        the image's own calibration.

    Returns
    -------
    Callable[[ndarray], ndarray]
        Picklable labeller: ``([2,] [z,] y, x) -> ([z,] y, x)``.
    """
    if segmentation not in SEGMENTATIONS:
        raise ValueError(
            f"segmentation must be one of {SEGMENTATIONS}, got {segmentation!r}"
        )
    if model is None and model_id is None and config_path is None:
        raise ValueError("give a zoo model, a model_id or a config_path")
    if max_radius_um is not None and not voxel_size:
        raise ValueError(
            "max_radius_um needs voxel_size (the image calibration)"
        )
    if max_radius_um is not None and segmentation not in NEEDS_NUCLEI:
        raise ValueError(
            "max_radius_um measures from the nuclei: use it with "
            f"segmentation {NEEDS_NUCLEI}"
        )
    _require_plantseg()
    factors = None
    if rescale and model and not (model_id or config_path):
        resolution = model_resolution(model)
        if voxel_size and resolution:
            factors = rescale_factors(voxel_size, resolution, len(resolution))
            if factors:
                logger.info(
                    "PlantSeg: resampling tiles by %s (z, y, x) to the "
                    "training voxel size of %s (%s um)",
                    factors,
                    model,
                    resolution,
                )
    cfg = dict(
        model=model,
        model_id=model_id,
        config_path=config_path,
        weights_path=weights_path,
        segmentation=segmentation,
        beta=beta,
        post_minsize=post_minsize,
        ws_threshold=ws_threshold,
        ws_sigma_seeds=ws_sigma_seeds,
        ws_min_size=ws_min_size,
        ws_stacked=ws_stacked,
        nuclei_sigma=nuclei_sigma,
        nuclei_threshold=nuclei_threshold,
        nuclei_min_size=nuclei_min_size,
        foreground=foreground,
        foreground_sigma=foreground_sigma,
        max_radius_um=max_radius_um,
        boundary_channel=boundary_channel,
        rescale=factors,
        patch=tuple(patch) if patch else None,
        device=device,
        n_threads=n_threads,
        voxel_size=voxel_size,
    )
    return partial(_run, cfg=cfg)


def _zoom_to(img: np.ndarray, shape: tuple[int, ...], order: int = 1):
    """Resample *img* to exactly *shape* (linear by default)."""
    from scipy import ndimage as ndi

    if img.shape == tuple(shape):
        return img
    factors = [s / d for s, d in zip(shape, img.shape)]
    out = ndi.zoom(img, factors, order=order, grid_mode=True, mode="nearest")
    # zoom rounds the output shape; crop/pad the odd voxel back.
    crop = tuple(slice(0, s) for s in shape)
    out = out[crop]
    pad = [(0, s - o) for s, o in zip(shape, out.shape)]
    if any(p for _, p in pad):
        out = np.pad(out, pad, mode="edge")
    return out


def predict_boundaries(membrane: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """The U-Net's boundary probability map for *membrane*, same shape."""
    from plantseg.functionals.prediction import unet_prediction

    shape = membrane.shape
    img = np.asarray(membrane, "float32")
    if cfg["rescale"]:
        target = tuple(
            max(1, int(round(s * f))) for s, f in zip(shape, cfg["rescale"])
        )
        img = _zoom_to(img, target)
    layout = "ZYX" if img.ndim == 3 else "YX"

    def _predict():
        return unet_prediction(
            img,
            input_layout=layout,
            model_name=cfg["model"],
            model_id=cfg["model_id"],
            patch=cfg["patch"],
            device=cfg["device"],
            disable_tqdm=True,
            config_path=cfg["config_path"],
            model_weights_path=cfg["weights_path"],
        )

    pmaps = retry_on_oom(
        _predict,
        enabled=str(cfg["device"]).startswith("cuda"),
        on_release=free_gpu_caches,
    )
    pmaps = np.asarray(pmaps)
    if pmaps.ndim > img.ndim:
        pmaps = pmaps[cfg["boundary_channel"]]
    pmaps = pmaps.reshape(img.shape).astype("float32")
    if pmaps.shape != shape:
        pmaps = _zoom_to(pmaps, shape)
    return np.clip(pmaps, 0.0, 1.0)


def _pixel_pitch(voxel_size, ndim: int):
    """Relative voxel spacing for the distance transform, or None."""
    if not voxel_size:
        return None
    axes = "zyx"[-ndim:]
    sizes = [float(voxel_size.get(a) or 0) for a in axes]
    if not all(sizes):
        return None
    smallest = min(sizes)
    pitch = tuple(s / smallest for s in sizes)
    return None if all(abs(p - 1) < 1e-6 for p in pitch) else pitch


def partition(
    pmaps: np.ndarray,
    cfg: dict[str, Any],
    seeds: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Cells from a boundary map, by the configured ``segmentation``.

    *mask* (where cells may be) bounds the flooding of
    ``"nuclei_watershed"``; the other modes cover the whole tile and are
    masked afterwards.
    """
    mode = cfg["segmentation"]
    if mode == "nuclei_watershed":
        return seeded_watershed(pmaps, seeds, mask=mask)

    from plantseg.functionals import segmentation as ps

    supervoxels = ps.dt_watershed(
        pmaps,
        threshold=cfg["ws_threshold"],
        sigma_seeds=cfg["ws_sigma_seeds"],
        stacked=cfg["ws_stacked"],
        min_size=cfg["ws_min_size"],
        # Distances in physical units on an anisotropic stack: the map is
        # back at the image's own sampling, whatever the U-Net saw.
        pixel_pitch=(
            None
            if cfg["ws_stacked"]
            else _pixel_pitch(cfg["voxel_size"], pmaps.ndim)
        ),
        n_threads=cfg["n_threads"],
    )
    if mode == "dt_watershed":
        return supervoxels
    threads = cfg["n_threads"] or 6
    if mode == "gasp":
        return ps.gasp(
            pmaps,
            supervoxels,
            beta=cfg["beta"],
            post_minsize=cfg["post_minsize"],
            n_threads=threads,
        )
    if mode == "mutex_ws":
        return ps.mutex_ws(
            pmaps,
            supervoxels,
            beta=cfg["beta"],
            post_minsize=cfg["post_minsize"],
            n_threads=threads,
        )
    if mode == "multicut":
        return ps.multicut(
            pmaps,
            supervoxels,
            beta=cfg["beta"],
            post_minsize=cfg["post_minsize"],
        )
    if mode == "lifted_multicut":
        return ps.lifted_multicut_from_nuclei_segmentation(
            pmaps,
            seeds,
            supervoxels,
            beta=cfg["beta"],
            post_minsize=cfg["post_minsize"],
        )
    raise ValueError(f"unknown segmentation {mode!r}")  # pragma: no cover


def _run(tile: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Predict boundaries, then partition them, for one tile."""
    membrane, nuclei = split_channels(tile)
    mode = cfg["segmentation"]
    if mode in NEEDS_NUCLEI and nuclei is None:
        raise ValueError(
            f'segmentation "{mode}" needs the nuclear channel: set '
            "nuclei_channel in the config (the tile must be [membrane, "
            f"nuclei] on its first axis), got a tile of shape {tile.shape}"
        )
    ndim = membrane.ndim
    cal = cfg["voxel_size"]
    seeds = None
    if mode in NEEDS_NUCLEI:
        seeds = nuclei_seeds(
            nuclei,
            sigma=_sigma(cfg["nuclei_sigma"], ndim, cal, "px"),
            threshold=cfg["nuclei_threshold"],
            min_size=cfg["nuclei_min_size"],
        )
        if not seeds.any():
            return np.zeros(membrane.shape, "int32")

    mask = None
    if cfg["foreground"] is not None or cfg["max_radius_um"] is not None:
        radius = (
            max_radius_px(cfg["max_radius_um"], ndim, cal)
            if cfg["max_radius_um"] is not None
            else None
        )
        mask = foreground_mask(
            membrane,
            seeds if seeds is not None else np.zeros(membrane.shape, "int32"),
            foreground=cfg["foreground"],
            sigma=_sigma(cfg["foreground_sigma"], ndim, cal, "px"),
            max_radius=radius,
        )

    pmaps = predict_boundaries(membrane, cfg)
    labels = np.asarray(partition(pmaps, cfg, seeds, mask)).astype("int32")
    if mask is not None:
        labels[~mask] = 0
    return labels


def segment(tile: np.ndarray, **kwargs: Any) -> np.ndarray:
    """:func:`plantseg_fn` as a direct call, for the workflow's
    ``method: "custom"`` (``function: "segment"``)."""
    return plantseg_fn(**kwargs)(tile)


setattr(segment, "patchworks_kwargs_target", plantseg_fn)
