"""Generic post-segmentation wrappers for patchworks.

These wrap any ``fn(tile) -> labels`` callable (a plugin, a custom function,
whatever ``method`` in the Snakemake workflow builds) so the same
post-processing applies regardless of which segmentation method produced the
labels.

Usage
-----
>>> from patchworks import tile_process, dilate_labels  # doctest: +SKIP
>>> from patchworks.plugins.dog import dog_label_fn  # doctest: +SKIP
>>>
>>> fn = dog_label_fn(low_sigma=1.0, high_sigma=3.0, threshold=0.02)  # doctest: +SKIP
>>> fn = dilate_labels(fn, iterations=2)  # doctest: +SKIP
>>> result = tile_process("image.zarr", fn, tile_shape=(1, 2048, 2048),  # doctest: +SKIP
...                       overlap=8, write_to="labels.zarr")
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import numpy as np


def dilate_labels(
    fn: Callable[[np.ndarray], np.ndarray],
    iterations: int = 1,
    *,
    use_gpu: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a segmentation callable to grow its labels after each tile.

    Applies a single-pass grey dilation to whatever ``fn`` returns, before
    ``tile_process``/``stage_tile`` trim the overlap halo and merge across
    tile boundaries — so dilated labels still stitch correctly at tile
    edges. Labels only grow into background: a voxel that already belongs
    to an object keeps it, so touching objects do not eat into each other
    (a bare max filter would let the higher id overwrite its neighbour).
    Where two labels grow into the same background voxel, the higher id
    takes it.

    Parameters
    ----------
    fn : Callable[[np.ndarray], np.ndarray]
        Any segmentation function with the ``tile_process``/``stage_tile``
        contract (one tile in, integer label array out).
    iterations : int, optional
        Pixels to grow each label by (grey-dilation footprint size
        ``2 * iterations + 1``, single pass). Default 1. Values ``<= 0``
        disable dilation — ``fn`` is returned unwrapped.
    use_gpu : bool, optional
        Dilate via cupyx instead of scipy. Independent of whatever backend
        ``fn`` itself uses internally.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        Picklable function ready for ``tile_process``/``stage_tile``. If
        ``iterations <= 0``, this is ``fn`` itself.
    """
    if iterations <= 0:
        return fn
    return partial(_run, fn=fn, iterations=iterations, use_gpu=use_gpu)


def _run(
    block: np.ndarray,
    fn: Callable[[np.ndarray], np.ndarray],
    iterations: int,
    use_gpu: bool,
) -> np.ndarray:
    """Run ``fn`` on ``block``, then grow the resulting labels.

    Parameters
    ----------
    block : np.ndarray
        One image tile.
    fn : Callable[[np.ndarray], np.ndarray]
        Segmentation function to run first.
    iterations : int
        Pixels to grow each label by.
    use_gpu : bool
        Dilate via cupyx instead of scipy.

    Returns
    -------
    np.ndarray
        Dilated integer label array, same shape as ``fn``'s output.
    """
    labels = fn(block)
    size = 2 * iterations + 1

    if use_gpu:
        import cupy as cp
        from cupyx.scipy.ndimage import grey_dilation

        gpu = cp.asarray(labels)
        grown = grey_dilation(gpu, size=size)
        return cp.asnumpy(cp.where(gpu == 0, grown, gpu))

    from scipy.ndimage import grey_dilation

    labels = np.asarray(labels)
    grown = grey_dilation(labels, size=size)
    return np.where(labels == 0, grown, labels)


def fill_holes(
    fn: Callable[[np.ndarray], np.ndarray], *, per_plane: bool = False
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a segmentation callable to fill holes inside its objects.

    A hole is background with no path to the tile's border; it takes the
    label of the nearest object voxel (so a hole inside one object joins
    it, and one shared by two objects is split between them). Runs on the
    tile *with* its halo, before the halo is trimmed and tiles are merged,
    so a hole near a tile edge is judged with context.

    Parameters
    ----------
    fn : Callable[[np.ndarray], np.ndarray]
        Any segmentation function with the ``tile_process`` contract.
    per_plane : bool, optional
        Fill each z-plane of a 3-D tile separately (the usual meaning of
        "holes" in a stack). A thin z-tile leaves little room for a 3-D
        hole to be enclosed at all, since its top and bottom planes are
        border. Default ``False`` (3-D holes).

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        Picklable function ready for ``tile_process``/``stage_tile``.
    """
    return partial(_run_fill_holes, fn=fn, per_plane=per_plane)


def _fill_label_holes(labels: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi

    background = labels == 0
    if not background.any() or background.all():
        return labels
    regions, n = ndi.label(background)
    # Background components touching the border are outside, not holes.
    border = np.zeros(labels.shape, bool)
    for ax in range(labels.ndim):
        index = [slice(None)] * labels.ndim
        index[ax] = slice(0, 1)
        border[tuple(index)] = True
        index[ax] = slice(-1, None)
        border[tuple(index)] = True
    outside = np.unique(regions[border & background])
    holes = background & ~np.isin(regions, outside)
    if not holes.any():
        return labels
    # Nearest object voxel for every background voxel.
    nearest = ndi.distance_transform_edt(
        background, return_distances=False, return_indices=True
    )
    filled = labels.copy()
    filled[holes] = labels[tuple(ix[holes] for ix in nearest)]
    return filled


def _run_fill_holes(
    block: np.ndarray,
    fn: Callable[[np.ndarray], np.ndarray],
    per_plane: bool,
) -> np.ndarray:
    labels = np.asarray(fn(block))
    if per_plane and labels.ndim == 3:
        return np.stack([_fill_label_holes(plane) for plane in labels])
    return _fill_label_holes(labels)


def open_labels(
    fn: Callable[[np.ndarray], np.ndarray], radius: int = 1
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a segmentation callable to remove thin spurs from its objects.

    A morphological opening applied to each object on its own, so touching
    objects never erode into or grow into each other: protrusions and
    bridges thinner than about ``2 * radius + 1`` voxels are cut, and an
    object thinner than that everywhere disappears. **Not for thin
    structures** -- on cilia or filaments this removes the objects
    themselves; use it for blob-like cells and nuclei.

    Parameters
    ----------
    fn : Callable[[np.ndarray], np.ndarray]
        Any segmentation function with the ``tile_process`` contract.
    radius : int, optional
        Structuring-element radius in voxels (a ``2r+1`` cube). ``<= 0``
        returns *fn* unwrapped.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        Picklable function ready for ``tile_process``/``stage_tile``.
    """
    if radius <= 0:
        return fn
    return partial(_run_open, fn=fn, radius=int(radius))


def _run_open(
    block: np.ndarray, fn: Callable[[np.ndarray], np.ndarray], radius: int
) -> np.ndarray:
    from scipy import ndimage as ndi

    labels = np.asarray(fn(block))
    # No extent along a one-voxel-thick axis: a 2-D tile stored as
    # (1, y, x) would otherwise be eroded away entirely through z.
    footprint = np.ones(
        tuple(1 if n == 1 else 2 * radius + 1 for n in labels.shape), bool
    )
    out = np.zeros_like(labels)
    for i, box in enumerate(ndi.find_objects(labels), start=1):
        if box is None:
            continue
        # Pad the box so the opening sees background around the object.
        padded = tuple(
            slice(max(0, s.start - radius), min(n, s.stop + radius))
            for s, n in zip(box, labels.shape)
        )
        mask = labels[padded] == i
        kept = ndi.binary_opening(mask, structure=footprint) & mask
        region = out[padded]
        region[kept] = i
    return out
