"""One intensity range for the whole image, so every tile is scaled alike.

Models such as Cellpose rescale each input from its own percentiles. Tiled,
that gives every tile its own contrast: a dim tile is stretched, a bright one
squashed, and the same cell is segmented differently on either side of a
seam. Measuring the range once and scaling every tile with it removes that.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)


def _crop(region: Sequence[slice], shape: Sequence[int], size: Sequence[int]):
    """A crop of at most *size* centred in *region*."""
    out = []
    for sl, n, s in zip(region, shape, size):
        start, stop = sl.start or 0, n if sl.stop is None else min(sl.stop, n)
        s = min(s, stop - start)
        lo = start + (stop - start - s) // 2
        out.append(slice(lo, lo + s))
    return tuple(out)


def sample_crops(
    store: Union[str, Path],
    channel: Union[int, None] = 0,
    *,
    level: int = 0,
    regions: Sequence[Sequence[slice]] | None = None,
    n_samples: int = 48,
    sample_shape: Sequence[int] = (16, 256, 256),
    seed: int = 0,
) -> list[np.ndarray]:
    """Full-resolution crops of one channel, for image-wide statistics.

    *n_samples* crops of *sample_shape* at *level* (the level the tiles are
    read at: a coarser one averages thin bright structures away), centred in
    a random subset of *regions* -- e.g. the tiles that will be segmented,
    so background does not dominate -- or placed at random. The same *seed*
    gives the same crops, for every channel and every rerun.
    """
    from ._io import load_ome_zarr

    image = load_ome_zarr(str(store), channel=channel, level=level)
    shape = image.shape
    size = (
        tuple(shape[: len(shape) - len(sample_shape)]) + tuple(sample_shape)
    )[-len(shape) :]
    rng = np.random.default_rng(seed)
    if regions:
        pick = rng.choice(
            len(regions), min(n_samples, len(regions)), replace=False
        )
        crops = [_crop(regions[i], shape, size) for i in sorted(pick)]
    else:
        crops = []
        for _ in range(n_samples):
            starts = [
                int(rng.integers(0, max(1, n - s + 1)))
                for n, s in zip(shape, size)
            ]
            crops.append(tuple(slice(a, a + s) for a, s in zip(starts, size)))
    return [np.asarray(image[c]) for c in crops]


def _acquired(crops: Sequence[np.ndarray]) -> np.ndarray:
    """All voxels of *crops*, without exact zeros (unacquired padding)."""
    values = np.concatenate([np.ravel(c) for c in crops])
    return values[values != 0]


def intensity_range(
    store: Union[str, Path],
    channels: Union[int, Sequence[int], None] = 0,
    *,
    percentiles: tuple[float, float] = (1.0, 99.0),
    **sampling,
) -> list[tuple[float, float]]:
    """Image-wide intensity percentiles, one ``(low, high)`` per channel.

    Measured on full-resolution crops (:func:`sample_crops`, which takes
    *level*, *regions*, *n_samples*, *sample_shape* and *seed*), leaving
    out exact zeros: padding where nothing was acquired.

    Parameters
    ----------
    store : str or Path
        An OME-Zarr image.
    channels : int, sequence of int or None
        Channel(s) to measure; ``None`` for an image without a channel axis.
    percentiles : (float, float)
        Low and high percentiles (Cellpose's own default is 1 and 99).
    **sampling
        Forwarded to :func:`sample_crops`.

    Returns
    -------
    list of (float, float)
        One range per channel, in the order given.
    """
    chans = (
        [channels]
        if channels is None or np.isscalar(channels)
        else list(channels)
    )
    ranges = []
    for ch in chans:
        values = _acquired(sample_crops(store, ch, **sampling))
        if values.size == 0:
            lo, hi = 0.0, 1.0
        else:
            lo, hi = (float(v) for v in np.percentile(values, percentiles))
        if hi <= lo:
            hi = lo + 1.0
        logger.info("intensity range of channel %s: %.6g .. %.6g", ch, lo, hi)
        ranges.append((lo, hi))
    return ranges


def intensity_stats(
    store: Union[str, Path], channel: Union[int, None] = 0, **sampling
) -> tuple[float, float]:
    """Image-wide mean and standard deviation of one channel, from
    full-resolution crops (:func:`sample_crops`), exact zeros left out."""
    values = _acquired(sample_crops(store, channel, **sampling)).astype(
        "float64"
    )
    if values.size == 0:
        return 0.0, 1.0
    mean, std = float(values.mean()), float(values.std())
    logger.info("mean and std of channel %s: %.6g, %.6g", channel, mean, std)
    return mean, (std if std > 0 else 1.0)


def otsu_threshold(
    store: Union[str, Path],
    channel: Union[int, None] = 0,
    *,
    sigma: float | Sequence[float] = 0.0,
    **sampling,
) -> float:
    """One Otsu threshold for the whole image, from full-resolution crops.

    Per tile, Otsu moves with each tile's content -- a tile full of bright
    membrane gets a high cutoff, a dim one a low cutoff -- so a mask made
    with it changes at every seam. Each crop is smoothed by *sigma* (pixels)
    first, as the plugins smooth before thresholding.
    """
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu

    crops = sample_crops(store, channel, **sampling)
    if np.any(np.asarray(sigma) > 0):
        crops = [ndi.gaussian_filter(c.astype("float32"), sigma) for c in crops]
    values = _acquired(crops)
    if values.size == 0 or values.min() == values.max():
        return float("inf")
    thr = float(threshold_otsu(values))
    logger.info("Otsu threshold of channel %s: %.6g", channel, thr)
    return thr


def scale_intensity(
    block: np.ndarray,
    ranges: Sequence[Sequence[float]],
    channel_axis: Union[int, None] = None,
) -> np.ndarray:
    """Map each channel's ``(low, high)`` to ``(0, 1)``, as float32.

    Values outside the range are kept (not clipped), as Cellpose's own
    normalisation does.
    """
    out = np.asarray(block, dtype=np.float32)
    if channel_axis is None:
        lo, hi = ranges[0]
        return (out - lo) / (hi - lo)
    out = np.moveaxis(out.copy(), channel_axis, 0)
    if len(ranges) < out.shape[0]:
        raise ValueError(
            f"{len(ranges)} intensity range(s) for a tile with "
            f"{out.shape[0]} channels"
        )
    for c in range(out.shape[0]):
        lo, hi = ranges[c]
        out[c] = (out[c] - lo) / (hi - lo)
    return np.moveaxis(out, 0, channel_axis)
