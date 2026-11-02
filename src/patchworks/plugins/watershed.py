"""Nuclei-seeded watershed on a membrane channel: one cell per nucleus.

For epithelia stained for a membrane (or cortex) marker plus a nuclear dye,
the nuclei are the easy part -- bright, compact, well separated -- and they
say exactly how many cells there are and where. Flooding the membrane image
from them grows each nucleus out to its cell's walls, so a cell cannot be
split in two (one seed each) nor merged with its neighbour (two seeds never
join), which is where a generalist model such as Cellpose struggles on a
faint or broken membrane.

Needs only scipy and scikit-image. The tile must carry both channels: set
`nuclei_channel` in the workflow config, which stacks `[channel,
nuclei_channel]` on the tile's first axis.

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
:mod:`patchworks.plugins.plantseg` (`segmentation: "nuclei_watershed"`),
which is the better boundary image when the membrane stain is uneven.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

#: Accepted `foreground` modes besides a number.
FOREGROUND_MODES = (None, "otsu")
#: Where the seeds come from: found in a nuclear intensity channel, or
#: given as a label image (the workflow's `seed_labels`).
SEED_MODES = ("channel", "labels")


def _sigma(sigma: Any, ndim: int, voxel_size: dict | None, units: str):
    """Per-axis pixel sigma for an `ndim` image from a scalar or tuple.

    Parameters
    ----------
    sigma : float or sequence of float
        One sigma for every axis, or one per axis (`zyx` order, the last
        *ndim* used).
    ndim : int
        The image's number of axes.
    voxel_size : dict or None
        µm per voxel by axis, for *units* `"um"`.
    units : str
        `"px"`: *sigma* is in pixels; `"um"`: in µm, converted with
        *voxel_size* (1 µm per voxel where it lacks an axis).

    Returns
    -------
    tuple of float
    """
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

    Parameters
    ----------
    img : np.ndarray
        The image.
    low, high : float, optional
        The percentiles mapped to 0 and 1 (default 1 and 99.8); values
        beyond are clipped.

    Returns
    -------
    np.ndarray
        float32; zeros for a flat image.

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

    Smooth, threshold, fill holes, drop specks and label what is left.
    Two touching nuclei come out as one seed, so the two cells around them
    as one cell: raise *threshold* if that happens, or give the seeds from
    a nuclei model (:func:`seeded_watershed` takes any label image).

    Parameters
    ----------
    nuclei : np.ndarray
        The nuclear channel.
    sigma : float or sequence of float, optional
        Gaussian smoothing first, in pixels (default 1).
    threshold : float, optional
        The intensity above which is nucleus, in the image's own units;
        None picks it by Otsu.
    min_size : int, optional
        Drop nuclei smaller than this many voxels (default 50).

    Returns
    -------
    np.ndarray
        int32 seeds.
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
    """A label image's objects as seeds, renumbered `1..n` (int32).

    The labels arrive with their global ids -- possibly millions, and
    promoted to the image's dtype when stacked with it -- so they are
    renumbered for the tile; which seed is which does not matter here, only
    that each object stays one seed.

    Parameters
    ----------
    labels : np.ndarray
        The seed labels' tile, any dtype (floats are rounded).

    Returns
    -------
    np.ndarray
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
    """The seeds for one tile from its second channel, per `cfg["seeds"]`.

    Parameters
    ----------
    second : np.ndarray
        The tile's second channel: seed labels, or nuclear intensities.
    cfg : dict
        The settings: `seeds` (`"labels"` or `"channel"`), and for a
        channel `nuclei_sigma`, `nuclei_threshold`, `nuclei_min_size`,
        `voxel_size`.
    ndim : int
        The tile's spatial axes.
    units : str, optional
        The unit of `nuclei_sigma`: `"px"` (default) or `"um"`.

    Returns
    -------
    np.ndarray
        int32 seeds: the given labels, or nuclei found in the intensities.
    """
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
    """The error for a tile carrying no second channel to seed from.

    Parameters
    ----------
    tile_shape : tuple of int
        The tile's shape, named in the message.

    Returns
    -------
    ValueError
    """
    return ValueError(
        "seeding needs a second channel: set nuclei_channel (a nuclear "
        "stain to find the nuclei in) or seed_labels (a label image, e.g. "
        "Cellpose's nuclei) in the config -- the tile must be [membrane, "
        f"nuclei] on its first axis; got a tile of shape {tuple(tile_shape)}"
    )


#: How far (pixels) the mask is mirrored past the tile's edges before its
#: holes are filled: about the largest cell diameter.
EDGE_MIRROR = 128


def fill_holes_past_edges(
    mask: np.ndarray, pad: int = EDGE_MIRROR
) -> np.ndarray:
    """Fill holes as if the image continued past the tile's edges.

    A cell cut by the edge of the region a tile reads is open there, so a
    plain hole fill leaves its interior empty -- and the cell is cut at
    every tile seam. Mirroring the mask outward first closes such a cell
    with its own reflection, while background that runs off the edge stays
    connected to the (padded) border and is left alone.

    Parameters
    ----------
    mask : np.ndarray of bool
        The mask.
    pad : int, optional
        How far to mirror, in voxels (default `EDGE_MIRROR`).

    Returns
    -------
    np.ndarray of bool

    Examples
    --------
    >>> ring = np.zeros((5, 9), bool); ring[[0, 4], :] = True; ring[:, 0] = True
    >>> fill_holes_past_edges(ring)[2].tolist()  # open to the right edge
    [True, True, True, True, True, True, True, True, True]
    """
    from scipy import ndimage as ndi

    widths = [(min(n - 1, pad),) * 2 for n in mask.shape]
    padded = np.pad(mask, widths, mode="reflect")
    filled = ndi.binary_fill_holes(padded)
    inner = tuple(slice(w, w + n) for (w, _), n in zip(widths, mask.shape))
    return filled[inner] | mask


def foreground_mask(
    membrane: np.ndarray,
    seeds: np.ndarray,
    *,
    foreground: str | float | None = "otsu",
    sigma: Any = 2.0,
    max_radius: tuple[float, ...] | None = None,
) -> np.ndarray | None:
    """Where cells may grow: tissue and/or near a seed.

    Parameters
    ----------
    membrane : np.ndarray
        The membrane channel.
    seeds : np.ndarray
        The seeds, for *max_radius*.
    foreground : str or float, optional
        `"otsu"` (default) or an intensity: keeps what the smoothed membrane
        signal covers, holes filled -- the cell interiors are dark in a
        membrane stain but enclosed by it. None: no tissue mask.
    sigma : float or sequence of float, optional
        Smoothing before that threshold, in pixels (default 2).
    max_radius : tuple of float, optional
        Keep only what lies within this distance of a seed (pixels per
        axis), so a cell on the tissue's edge stops instead of flooding the
        empty space beyond it.

    Returns
    -------
    np.ndarray of bool or None
        The mask; None (anywhere) when neither applies.
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
        mask = fill_holes_past_edges(mask)
        if mask.ndim == 3:
            for z in range(mask.shape[0]):
                mask[z] = fill_holes_past_edges(mask[z])
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

    Every seed becomes exactly one cell.

    Parameters
    ----------
    boundaries : np.ndarray
        The membrane signal, high on cell walls.
    seeds : np.ndarray
        The seed labels.
    mask : np.ndarray of bool, optional
        Cells stop at its edge.
    compactness : float, optional
        > 0 favours rounder cells where a wall is missing (default 0).
    min_size : int, optional
        Drop cells smaller than this many voxels (a seed in a speck of
        tissue); 0 (default) keeps all.

    Returns
    -------
    np.ndarray
        int32 cells.
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
    """Zero the objects of *labels* smaller than *min_size* voxels.

    Parameters
    ----------
    labels : np.ndarray
        The labels, changed in place.
    min_size : int
        The smallest object kept; 0 keeps all.

    Returns
    -------
    np.ndarray
        *labels*.
    """
    if not min_size or not labels.any():
        return labels
    counts = np.bincount(labels.ravel())
    small = np.flatnonzero(counts < min_size)
    small = small[small > 0]
    if small.size:
        labels[np.isin(labels, small)] = 0
    return labels


def split_channels(tile: np.ndarray, spatial_ndim: int | None = None):
    """`(membrane, nuclei)` from a `(2, ...)` tile; nuclei None if 1-ch.

    The workflow stacks `[channel, nuclei_channel]` on axis 0 when
    `nuclei_channel` is set; otherwise the tile is the bare image.

    Parameters
    ----------
    tile : np.ndarray
        The tile.
    spatial_ndim : int, optional
        Its spatial axes, when known: a tile with only those is one
        channel, whatever its first axis's length.

    Returns
    -------
    tuple
        `(membrane, nuclei)`.
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
    """Return a nuclei-seeded watershed for `tile_process`.

    Parameters
    ----------
    boundary_sigma : float or sequence of float, optional
        Smoothing of the membrane channel before flooding it.
    nuclei_sigma : float or sequence of float, optional
        Smoothing of the nuclear channel before finding the nuclei (see
        :func:`nuclei_seeds`).
    nuclei_threshold : float, optional
        An intensity of the nuclear channel: above it is nucleus. `None`
        picks it by Otsu, per tile.
    nuclei_min_size : int, optional
        Smallest nucleus, in voxels.
    foreground : str or float, optional
        Tissue mask from the membrane channel: `"otsu"` (default), an
        intensity, or `None` to let cells fill the whole tile.
    foreground_sigma : float or sequence of float, optional
        Smoothing of the membrane channel before the foreground threshold.
    max_radius_um : float, optional
        Furthest a cell reaches from its nucleus, in micrometres (needs
        *voxel_size*; the workflow passes it). `None`: no limit.
    compactness : float, optional
        > 0 favours round cells where a wall is missing (skimage's
        `watershed`). Small values (0.001--0.01) are typical.
    min_size : int, optional
        Drop cells smaller than this many voxels.
    sigma_units : str, optional
        `"px"` (default) or `"um"` for every sigma above.
    seeds : str, optional
        `"channel"` (default): the tile's second channel is a nuclear
        stain, and the nuclei are found in it (the `nuclei_*` options).
        `"labels"`: it is a label image -- the workflow's `seed_labels`,
        e.g. Cellpose's nuclei -- and each of its objects seeds one cell, as
        it is. The workflow sets this itself when `seed_labels` is used.
    voxel_size : dict of str to float, optional
        `{"z": .., "y": .., "x": ..}` in micrometres.

    Returns
    -------
    Callable[[ndarray], ndarray]
        Picklable `(2, [z,] y, x) -> ([z,] y, x)` labeller.
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
    """A physical radius as pixels per axis.

    Parameters
    ----------
    radius_um : float
        The radius, µm.
    ndim : int
        The tile's spatial axes (the last *ndim* of `zyx`).
    voxel_size : dict
        µm per voxel by axis; 1 where it lacks one.

    Returns
    -------
    tuple of float
    """
    axes = "zyx"[-ndim:]
    return tuple(radius_um / float(voxel_size.get(a) or 1.0) for a in axes)


def _run(tile: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Seeded watershed of one `[membrane, nuclei]` tile.

    Parameters
    ----------
    tile : np.ndarray
        The tile, its second channel the seeds or nuclei.
    cfg : dict
        The settings :func:`watershed_fn` built.

    Returns
    -------
    np.ndarray
        int32 cells.

    Raises
    ------
    ValueError
        When the tile has no second channel.
    """
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
    labels = seeded_watershed(
        boundaries,
        seeds,
        mask=mask,
        compactness=cfg["compactness"],
        min_size=cfg["min_size"],
    )
    if mask is not None:
        # Holes the mask punched into cells go back to them.
        from .._postprocess import fill_enclosed_holes

        labels = fill_enclosed_holes(labels)
    return labels


def segment(tile: np.ndarray, **kwargs: Any) -> np.ndarray:
    """:func:`watershed_fn` as a direct call, for the workflow's
    `method: "custom"` (`function: "segment"`).

    Parameters
    ----------
    tile : np.ndarray
        The `[membrane, nuclei]` tile.
    **kwargs
        :func:`watershed_fn`'s keyword arguments.

    Returns
    -------
    np.ndarray
        int32 cells.
    """
    return watershed_fn(**kwargs)(tile)


setattr(segment, "patchworks_kwargs_target", watershed_fn)
# Every voxel goes to some cell, so neighbours touch at every tile seam:
# the workflow requires IoU stitching (stitch: "iou") with this function.
setattr(segment, "patchworks_stitch", "iou")
