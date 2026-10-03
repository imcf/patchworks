"""Nuclei-seeded watershed on a membrane channel: one cell per nucleus.

For epithelia stained for a membrane (or cortex) marker plus a nuclear dye,
the nuclei are the easy part -- bright, compact, well separated -- and they
say exactly how many cells there are and where. Flooding the membrane image
from them grows each nucleus out to its cell's walls, so a cell cannot be
split in two (one seed each) nor merged with its neighbour (two seeds never
join), which is where a generalist model such as Cellpose struggles on a
faint or broken membrane.

Needs only scipy and scikit-image. The tile must carry both channels: set
``nuclei_channel`` in the workflow config, which stacks ``[channel,
nuclei_channel]`` on the tile's first axis.

Usage
-----
>>> # method: "custom"
>>> # custom: {module: "patchworks.plugins.watershed", function: "segment",
>>> #          kwargs: {nuclei_min_size: 200, foreground: "otsu"}}

From the API:

>>> from patchworks.plugins.watershed import watershed_fn  # doctest: +SKIP
>>> fn = watershed_fn(max_radius_um=12, voxel_size={"z": 0.5, "y": 0.2, "x": 0.2})  # doctest: +SKIP
>>> tile_process(two_channel_image, fn, channel_axis=0, ...)  # doctest: +SKIP

The same seeded flooding runs on a U-Net boundary map in
:mod:`patchworks.plugins.plantseg` (``segmentation: "nuclei_watershed"``),
which is the better boundary image when the membrane stain is uneven.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

#: Accepted ``foreground`` modes besides a number.
FOREGROUND_MODES = (None, "otsu")
#: Where the seeds come from: found in a nuclear intensity channel, or
#: given as a label image (the workflow's ``seed_labels``).
SEED_MODES = ("channel", "labels")


def _sigma(sigma: Any, ndim: int, voxel_size: dict | None, units: str):
    """Per-axis pixel sigma for an ``ndim`` image from a scalar or tuple."""
    axes = "zyx"[-ndim:]
    values = (
        (float(sigma),) * ndim
        if np.isscalar(sigma)
        else tuple(float(v) for v in sigma)[-ndim:]
    )
    if units == "um":
        cal = voxel_size or {}
        values = tuple(
            v / float(cal.get(a) or 1.0) for v, a in zip(values, axes)
        )
    return values


def normalize(img: np.ndarray, low: float = 1.0, high: float = 99.8):
    """Rescale *img* to [0, 1] between two percentiles (robust to hot pixels).

    Examples
    --------
    >>> import numpy as np
    >>> out = normalize(np.arange(1000.0))
    >>> float(out.min()), float(out.max())
    (0.0, 1.0)
    """
    img = np.asarray(img, dtype="float32")
    lo, hi = np.percentile(img, [low, high])
    if hi <= lo:
        return np.zeros_like(img)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


def nuclei_seeds(
    nuclei: np.ndarray,
    *,
    sigma: Any = 1.0,
    threshold: float | None = None,
    min_size: int = 50,
) -> np.ndarray:
    """Label the nuclei of a nuclear-dye image, to seed the cells with.

    Smooth, threshold (Otsu unless *threshold* is given, in the image's own
    intensity units), fill holes, drop specks below *min_size* voxels and
    label what is left.

    Two touching nuclei come out as one seed, so the two cells around them
    as one cell: raise *threshold* if that happens, or give the seeds from a
    nuclei model (:func:`seeded_watershed` takes any label image).
    """
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu
    from skimage.measure import label

    smooth = ndi.gaussian_filter(np.asarray(nuclei, "float32"), sigma)
    if threshold is None:
        threshold = (
            float(threshold_otsu(smooth))
            if smooth.max() > smooth.min()
            else float("inf")
        )
    mask = smooth > threshold
    mask = ndi.binary_fill_holes(mask)
    return _drop_small(label(mask).astype("int32"), min_size)


def seeds_from_labels(labels: np.ndarray) -> np.ndarray:
    """A label image's objects as seeds, renumbered ``1..n`` (int32).

    The labels arrive with their global ids -- possibly millions, and
    promoted to the image's dtype when stacked with it -- so they are
    renumbered for the tile; which seed is which does not matter here, only
    that each object stays one seed.
    """
    labels = np.asarray(labels)
    if labels.dtype.kind == "f":
        labels = np.rint(labels)
    ids, inverse = np.unique(labels.astype("int64"), return_inverse=True)
    out = inverse.reshape(labels.shape).astype("int32")
    if ids[0] != 0:
        out += 1  # no background in this tile: every id shifts up one
    return out


def tile_seeds(
    second: np.ndarray, cfg: dict[str, Any], ndim: int, units: str = "px"
) -> np.ndarray:
    """The seeds for one tile from its second channel, per ``cfg["seeds"]``:
    the given labels, or nuclei found in the intensities."""
    if cfg.get("seeds", "channel") == "labels":
        return seeds_from_labels(second)
    cal = cfg.get("voxel_size")
    return nuclei_seeds(
        second,
        sigma=_sigma(cfg["nuclei_sigma"], ndim, cal, units),
        threshold=cfg["nuclei_threshold"],
        min_size=cfg["nuclei_min_size"],
    )


def missing_seeds_error(tile_shape) -> ValueError:
    """The error for a tile carrying no second channel to seed from."""
    return ValueError(
        "seeding needs a second channel: set nuclei_channel (a nuclear "
        "stain to find the nuclei in) or seed_labels (a label image, e.g. "
        "Cellpose's nuclei) in the config -- the tile must be [membrane, "
        f"nuclei] on its first axis; got a tile of shape {tuple(tile_shape)}"
    )


def foreground_mask(
    membrane: np.ndarray,
    seeds: np.ndarray,
    *,
    foreground: str | float | None = "otsu",
    sigma: Any = 2.0,
    max_radius: tuple[float, ...] | None = None,
) -> np.ndarray | None:
    """Where cells may grow: tissue and/or near a seed. ``None``: anywhere.

    *foreground* ``"otsu"`` (or an intensity) keeps what the smoothed
    membrane signal covers, holes filled -- the cell interiors are dark in a
    membrane stain but enclosed by it. *max_radius* (pixels per axis) keeps
    what lies within that distance of a nucleus, so a cell on the tissue's
    edge stops instead of flooding the empty space beyond it.
    """
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu

    mask = None
    if foreground is not None:
        smooth = ndi.gaussian_filter(np.asarray(membrane, "float32"), sigma)
        if foreground == "otsu":
            thr = (
                float(threshold_otsu(smooth))
                if smooth.max() > smooth.min()
                else float("inf")
            )
        else:
            thr = float(foreground)
        mask = smooth > thr
        # Interiors: in 3-D and plane by plane, since a cell cut by the tile
        # edge is enclosed in its planes but open to the border in 3-D.
        mask = ndi.binary_fill_holes(mask)
        if mask.ndim == 3:
            for z in range(mask.shape[0]):
                mask[z] = ndi.binary_fill_holes(mask[z])
    if max_radius is not None:
        # Distance in units of the radius per axis: <= 1 means within reach.
        near = (
            ndi.distance_transform_edt(
                seeds == 0, sampling=[1.0 / r for r in max_radius]
            )
            <= 1.0
        )
        mask = near if mask is None else (mask & near)
    if mask is not None:
        mask |= seeds > 0
    return mask


def seeded_watershed(
    boundaries: np.ndarray,
    seeds: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    compactness: float = 0.0,
    min_size: int = 0,
) -> np.ndarray:
    """Flood *boundaries* (high on cell walls) from the labelled *seeds*.

    Every seed becomes exactly one cell; with *mask*, cells stop at its
    edge. *compactness* > 0 favours rounder cells where a wall is missing.
    Cells smaller than *min_size* voxels are dropped (a seed in a speck of
    tissue).
    """
    from skimage.segmentation import watershed

    labels = watershed(
        np.asarray(boundaries, "float32"),
        markers=seeds,
        mask=mask,
        compactness=compactness,
    ).astype("int32")
    return _drop_small(labels, min_size)


def _drop_small(labels: np.ndarray, min_size: int) -> np.ndarray:
    """Zero the objects of *labels* smaller than *min_size* voxels."""
    if not min_size or not labels.any():
        return labels
    counts = np.bincount(labels.ravel())
    small = np.flatnonzero(counts < min_size)
    small = small[small > 0]
    if small.size:
        labels[np.isin(labels, small)] = 0
    return labels


def split_channels(tile: np.ndarray, spatial_ndim: int | None = None):
    """``(membrane, nuclei)`` from a ``(2, ...)`` tile; nuclei None if 1-ch.

    The workflow stacks ``[channel, nuclei_channel]`` on axis 0 when
    ``nuclei_channel`` is set; otherwise the tile is the bare image.
    """
    tile = np.asarray(tile)
    if spatial_ndim is not None and tile.ndim == spatial_ndim:
        return tile, None
    if (
        tile.ndim >= 3
        and tile.shape[0] == 2
        and (spatial_ndim is None or tile.ndim == spatial_ndim + 1)
    ):
        return tile[0], tile[1]
    return tile, None


def watershed_fn(
    *,
    boundary_sigma: Any = 1.0,
    nuclei_sigma: Any = 1.0,
    nuclei_threshold: float | None = None,
    nuclei_min_size: int = 50,
    foreground: str | float | None = "otsu",
    foreground_sigma: Any = 2.0,
    max_radius_um: float | None = None,
    compactness: float = 0.0,
    min_size: int = 0,
    sigma_units: str = "px",
    seeds: str = "channel",
    voxel_size: dict[str, float] | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a nuclei-seeded watershed for ``tile_process``.

    Parameters
    ----------
    boundary_sigma :
        Smoothing of the membrane channel before flooding it.
    nuclei_sigma, nuclei_threshold, nuclei_min_size :
        How the nuclei are found: see :func:`nuclei_seeds`. The threshold is
        an intensity of the nuclear channel; ``None`` picks it by Otsu, per
        tile.
    foreground, foreground_sigma :
        Tissue mask from the membrane channel: ``"otsu"`` (default), an
        intensity, or ``None`` to let cells fill the whole tile.
    max_radius_um :
        Furthest a cell reaches from its nucleus, in micrometres (needs
        *voxel_size*; the workflow passes it). ``None``: no limit.
    compactness :
        > 0 favours round cells where a wall is missing (skimage's
        ``watershed``). Small values (0.001--0.01) are typical.
    min_size :
        Drop cells smaller than this many voxels.
    sigma_units :
        ``"px"`` (default) or ``"um"`` for every sigma above.
    seeds :
        ``"channel"`` (default): the tile's second channel is a nuclear
        stain, and the nuclei are found in it (the ``nuclei_*`` options).
        ``"labels"``: it is a label image -- the workflow's ``seed_labels``,
        e.g. Cellpose's nuclei -- and each of its objects seeds one cell, as
        it is. The workflow sets this itself when ``seed_labels`` is used.
    voxel_size :
        ``{"z": .., "y": .., "x": ..}`` in micrometres.

    Returns
    -------
    Callable[[ndarray], ndarray]
        Picklable ``(2, [z,] y, x) -> ([z,] y, x)`` labeller.
    """
    if sigma_units not in ("px", "um"):
        raise ValueError(
            f'sigma_units must be "px" or "um", got {sigma_units!r}'
        )
    if sigma_units == "um" and not voxel_size:
        raise ValueError('sigma_units="um" needs voxel_size')
    if max_radius_um is not None and not voxel_size:
        raise ValueError(
            "max_radius_um needs voxel_size (the image calibration)"
        )
    if not (
        foreground in FOREGROUND_MODES or isinstance(foreground, (int, float))
    ):
        raise ValueError(
            f'foreground must be "otsu", a number or null, got {foreground!r}'
        )
    if seeds not in SEED_MODES:
        raise ValueError(f"seeds must be one of {SEED_MODES}, got {seeds!r}")
    cfg = dict(
        boundary_sigma=boundary_sigma,
        nuclei_sigma=nuclei_sigma,
        nuclei_threshold=nuclei_threshold,
        nuclei_min_size=nuclei_min_size,
        foreground=foreground,
        foreground_sigma=foreground_sigma,
        max_radius_um=max_radius_um,
        compactness=compactness,
        min_size=min_size,
        sigma_units=sigma_units,
        seeds=seeds,
        voxel_size=voxel_size,
    )
    return partial(_run, cfg=cfg)


def max_radius_px(radius_um: float, ndim: int, voxel_size: dict) -> tuple:
    """A physical radius as pixels per axis (``zyx`` order, last *ndim*)."""
    axes = "zyx"[-ndim:]
    return tuple(radius_um / float(voxel_size.get(a) or 1.0) for a in axes)


def _run(tile: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    from scipy import ndimage as ndi

    membrane, nuclei = split_channels(tile)
    if nuclei is None:
        raise missing_seeds_error(tile.shape)
    ndim = membrane.ndim
    units, cal = cfg["sigma_units"], cfg["voxel_size"]
    seeds = tile_seeds(nuclei, cfg, ndim, units)
    if not seeds.any():
        return np.zeros(membrane.shape, "int32")
    boundaries = ndi.gaussian_filter(
        normalize(membrane), _sigma(cfg["boundary_sigma"], ndim, cal, units)
    )
    radius = (
        max_radius_px(cfg["max_radius_um"], ndim, cal)
        if cfg["max_radius_um"] is not None
        else None
    )
    mask = foreground_mask(
        membrane,
        seeds,
        foreground=cfg["foreground"],
        sigma=_sigma(cfg["foreground_sigma"], ndim, cal, units),
        max_radius=radius,
    )
    return seeded_watershed(
        boundaries,
        seeds,
        mask=mask,
        compactness=cfg["compactness"],
        min_size=cfg["min_size"],
    )


def segment(tile: np.ndarray, **kwargs: Any) -> np.ndarray:
    """:func:`watershed_fn` as a direct call, for the workflow's
    ``method: "custom"`` (``function: "segment"``)."""
    return watershed_fn(**kwargs)(tile)


setattr(segment, "patchworks_kwargs_target", watershed_fn)
