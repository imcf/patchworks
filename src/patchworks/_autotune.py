"""Choose ``overlap`` from the data instead of guessing it.

The halo exists so that tiling does not change the result. That is directly
testable: segment a crop spanning a few tiles once without tiling (the
reference), then tiled at each candidate overlap, and see from which overlap
the tiled result stops differing. The smallest such overlap is the cheapest
one that is safe for this image and this method.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from typing import Any, Callable, Sequence, Union

import dask.array as da
import numpy as np

logger = logging.getLogger(__name__)


def object_f1(a: np.ndarray, b: np.ndarray, iou: float = 0.5) -> float:
    """Object-level F1 between two label images at an IoU threshold.

    Two objects match when their IoU exceeds *iou* (above 0.5 a match is
    necessarily one-to-one). 1.0 means every object in one has its
    counterpart in the other; background-only images agree perfectly.
    """
    from ._merge import _pair_stats

    pairs, inter, a_area, b_area = _pair_stats(a, b)
    na, nb = len(a_area), len(b_area)
    if na + nb == 0:
        return 1.0
    tp = sum(
        1
        for (ai, bi), c in zip(pairs.tolist(), inter.tolist())
        if c / (a_area[ai] + b_area[bi] - c) > iou
    )
    return 2 * tp / (na + nb)


def _centre_crop(
    shape: tuple[int, ...], tile: tuple[int, ...], tiles: int
) -> tuple[slice, ...]:
    """A crop *tiles* tiles wide per axis (where the image has room), centred
    on a tile corner so it contains seams on every tiled axis."""
    crop = []
    for n, t in zip(shape, tile):
        want = min(n, tiles * t)
        # Snap to the tile grid, so the crop's seams are the run's seams.
        start = max(0, min((n // 2 // t) * t - (tiles // 2) * t, n - want))
        start -= start % t
        crop.append(slice(start, start + want))
    return tuple(crop)


def suggest_overlap(
    image: Union[da.Array, np.ndarray],
    fn: Callable[[np.ndarray], np.ndarray],
    tile_shape: Sequence[int],
    *,
    candidates: Sequence[int] = (0, 4, 8, 16, 32, 64),
    region: "tuple[slice, ...] | None" = None,
    crop_tiles: int = 2,
    target: float = 0.99,
    stitch: str = "touch",
) -> dict[str, Any]:
    """Smallest overlap whose tiled result matches an untiled one.

    Segments a crop spanning ``crop_tiles`` tiles per axis once without
    tiling, then with ``tile_process`` at each candidate overlap, scoring
    each by :func:`object_f1` against the untiled reference. Candidates are
    tried in increasing order and the search stops at the first that reaches
    *target*.

    Parameters
    ----------
    image : dask or NumPy array
        The image as ``tile_process`` would see it (channel already chosen).
    fn : callable
        The segmentation function. It is run on the crop in one piece, so the
        crop must fit it (``crop_tiles=2`` means a 2x2(x2) block of tiles).
    tile_shape : sequence of int
        The tile shape the real run will use.
    candidates : sequence of int, optional
        Overlaps to try, in voxels (applied on every axis the tile allows).
    region : tuple of slice, optional
        The crop to use; default a centred block of tiles. Pick one with
        typical objects -- an empty crop agrees at any overlap.
    crop_tiles : int, optional
        Tiles per axis in the default crop (default 2).
    target : float, optional
        F1 counted as "tiling makes no difference" (default 0.99).
    stitch : str, optional
        Stitching mode of the real run (``"touch"`` or ``"iou"``).

    Returns
    -------
    dict
        ``{"overlap": chosen or None, "scores": {overlap: f1}, "region":
        crop, "reference_objects": n}``. ``overlap`` is None when no
        candidate reached *target* (raise the candidates, or the crop has
        objects larger than any of them).

    Examples
    --------
    >>> from patchworks import suggest_overlap  # doctest: +SKIP
    >>> suggest_overlap(img, fn, (16, 512, 512))["overlap"]  # doctest: +SKIP
    16
    """
    from ._core import tile_process

    tile = tuple(int(t) for t in tile_shape)
    arr = image if isinstance(image, da.Array) else da.from_array(image)
    region = region or _centre_crop(arr.shape, tile, crop_tiles)
    crop = np.asarray(arr[region])
    if crop.size == 0:
        raise ValueError(f"empty crop {region}")
    reference = np.asarray(fn(crop))
    n_ref = len(np.unique(reference)) - (1 if (reference == 0).any() else 0)
    if n_ref == 0:
        logger.warning(
            "suggest_overlap: the crop %s holds no objects, so every overlap "
            "agrees; pass region= with typical objects",
            region,
        )

    scores: dict[int, float] = {}
    chosen = None
    scratch = tempfile.mkdtemp(prefix="pws_overlap_")
    try:
        lazy = da.from_array(crop, chunks=tile)
        for ov in sorted(int(c) for c in candidates):
            tiled = tile_process(
                lazy,
                fn,
                overlap=ov,
                stitch=stitch,
                write_to=f"{scratch}/ov{ov}.zarr",
                progress=False,
                log_file=False,
            )
            scores[ov] = object_f1(reference, np.asarray(tiled))
            logger.info(
                "suggest_overlap: overlap %d -> F1 %.3f vs untiled",
                ov,
                scores[ov],
            )
            if scores[ov] >= target:
                chosen = ov
                break
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if chosen is None:
        logger.warning(
            "suggest_overlap: no candidate reached F1 %.2f (best %s); try "
            "larger overlaps",
            target,
            max(scores.items(), key=lambda kv: kv[1]) if scores else None,
        )
    return {
        "overlap": chosen,
        "scores": scores,
        "region": [(s.start, s.stop) for s in region],
        "reference_objects": int(n_ref),
    }
