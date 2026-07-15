"""Generic post-segmentation wrappers for patchworks.

These wrap any ``fn(tile) -> labels`` callable (a plugin, a custom function,
whatever ``method`` in the Snakemake workflow builds) so the same
post-processing applies regardless of which segmentation method produced the
labels.

Usage
-----
>>> from patchworks import tile_process, dilate_labels
>>> from patchworks.plugins.dog import dog_label_fn
>>>
>>> fn = dog_label_fn(low_sigma=1.0, high_sigma=3.0, threshold=0.02)
>>> fn = dilate_labels(fn, iterations=2)
>>> result = tile_process("image.zarr", fn, tile_shape=(1, 2048, 2048),
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
    edges.

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

        labels = cp.asnumpy(grey_dilation(cp.asarray(labels), size=size))
    else:
        from scipy.ndimage import grey_dilation

        labels = grey_dilation(labels, size=size)

    return labels
