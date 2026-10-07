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


def typical_volume(labels: np.ndarray) -> float:
    """The volume of the object a typical labelled voxel belongs to.

    A voxel-weighted median: many small fragments barely move it, unlike
    the plain median object size, so it stays the size of a whole object.
    """
    sizes = np.bincount(np.asarray(labels).ravel())[1:]
    sizes = np.sort(sizes[sizes > 0])
    if sizes.size == 0:
        return 0.0
    weight = np.cumsum(sizes)
    return float(sizes[np.searchsorted(weight, weight[-1] / 2)])


def _contacts(labels: np.ndarray) -> np.ndarray:
    """``(a, b, n)`` rows: labels a != b, both non-zero, sharing n faces."""
    pairs = []
    for ax in range(labels.ndim):
        lo = [slice(None)] * labels.ndim
        hi = [slice(None)] * labels.ndim
        lo[ax], hi[ax] = slice(None, -1), slice(1, None)
        a, b = labels[tuple(lo)], labels[tuple(hi)]
        touch = (a != b) & (a > 0) & (b > 0)
        if touch.any():
            a, b = a[touch].astype(np.int64), b[touch].astype(np.int64)
            pairs.append(np.stack([a, b], 1))
            pairs.append(np.stack([b, a], 1))
    if not pairs:
        return np.empty((0, 3), np.int64)
    uniq, n = np.unique(np.vstack(pairs), axis=0, return_counts=True)
    return np.column_stack([uniq, n])


def absorb_fragments(
    labels: np.ndarray,
    min_voxels: float | None = None,
    *,
    fraction: float = 0.1,
    rounds: int = 3,
) -> np.ndarray:
    """Merge small fragments into the object they touch most; drop the rest.

    3-D segmentation often breaks a cell into one large piece and slivers
    beside it, or leaves specks in dim regions. An object smaller than
    *min_voxels* -- by default *fraction* of :func:`typical_volume` -- joins
    the neighbour it shares the most surface with (a larger one when it has
    any), so the cell keeps its full volume; one touching nothing is
    removed. Repeated up to *rounds* times, for fragments touching only
    other fragments; any small object left after that is removed. Objects
    touching the array's border are left alone: they may be cells cut by
    the tile edge, whose size is unknown here.

    A sliver (2) beside cell 1, a speck (4) between it and cell 3, one
    alone (5), and a small object on the border (6), which is kept:

    >>> lab = np.zeros((8, 14), int)
    >>> lab[1:7, 1:5] = 1; lab[1:7, 5] = 2; lab[1:7, 7:10] = 3
    >>> lab[1, 6] = 4; lab[4, 12] = 5; lab[7, 12] = 6
    >>> out = absorb_fragments(lab, 7)
    >>> out[1].tolist(), out[4].tolist(), out[7].tolist()[12]
    ([0, 1, 1, 1, 1, 1, 3, 3, 3, 3, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0, 3, 3, 3, 0, 0, 0, 0], 6)
    """
    labels = np.asarray(labels)
    if min_voxels is None:
        min_voxels = fraction * typical_volume(labels)
    if min_voxels <= 1:
        return labels
    out = labels.copy()
    edge = np.zeros(int(out.max()) + 1, bool)
    for ax in range(out.ndim):
        if out.shape[ax] > 1:  # a single plane is not an edge
            edge[np.take(out, [0, -1], axis=ax).ravel()] = True
    edge[0] = False
    for _ in range(rounds):
        sizes = np.bincount(out.ravel(), minlength=edge.size)
        small = (sizes > 0) & (sizes < min_voxels) & ~edge
        small[0] = False
        if not small.any():
            break
        contacts = _contacts(out)
        contacts = contacts[small[contacts[:, 0]]]
        if contacts.size == 0:
            break
        # Prefer a neighbour that is itself whole, then the longest contact.
        whole = ~small[contacts[:, 1]]
        order = np.lexsort((-contacts[:, 2], ~whole, contacts[:, 0]))
        contacts = contacts[order]
        first = np.unique(contacts[:, 0], return_index=True)[1]
        src, dst = contacts[first, 0], contacts[first, 1]
        # A fragment pair choosing each other merges one way only.
        keep = ~(small[dst] & (dst < src))
        lut = np.arange(sizes.size, dtype=out.dtype)
        lut[src[keep]] = dst[keep]
        # Follow chains a -> b -> c.
        for _ in range(rounds):
            lut = lut[lut]
        out = lut[out]
    sizes = np.bincount(out.ravel(), minlength=edge.size)
    small = (sizes > 0) & (sizes < min_voxels) & ~edge
    small[0] = False
    if small.any():
        out[small[out]] = 0
    return out
