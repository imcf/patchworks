"""OME-ZARR loading and empty-tile estimation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import dask.array as da
import numpy as np
import zarr

logger = logging.getLogger(__name__)

_ZARR_V3 = int(zarr.__version__.split(".")[0]) >= 3


def load_ome_zarr(
    store_path: Union[str, Path],
    channel: int | None = 0,
    level: int = 0,
    chunks: tuple[int, ...] | None = None,
) -> da.Array:
    """Load one spatial array from an OME-ZARR store.

    Parameters
    ----------
    store_path:
        Path to the OME-ZARR store (.zarr directory).
    channel:
        Channel index to select (axis is dropped). Pass ``None`` to keep it.
    level:
        Resolution pyramid level (0 = full resolution).
    chunks:
        Target chunk shape for the returned dask array.

    Returns
    -------
    da.Array
        Shape ``(z, y, x)`` when *channel* is an int, or ``(c, z, y, x)``
        when *channel* is None.

    Examples
    --------
    >>> arr = load_ome_zarr("image.zarr", channel=0)
    >>> arr.shape
    (128, 2048, 2048)
    """
    root = zarr.open_group(str(store_path), mode="r")
    try:
        path = root.attrs["multiscales"][0]["datasets"][level]["path"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            f"Cannot read OME-ZARR multiscales metadata at level {level} "
            f"in {store_path!r}"
        ) from exc

    zarr_chunks = chunks
    if chunks is not None and channel is not None:
        zarr_ndim = len(root[path].shape)
        if zarr_ndim > len(chunks):
            zarr_chunks = (1,) * (zarr_ndim - len(chunks)) + tuple(chunks)

    arr = da.from_zarr(str(store_path), component=path, chunks=zarr_chunks)
    if channel is not None:
        arr = arr[channel]
    return arr


def _otsu_threshold(sample: np.ndarray) -> float:
    """Otsu threshold of *sample*; falls back to 0 if degenerate.

    Operates on the full distribution including zeros — zeros are background
    pixels and must be included so Otsu can find the signal/background boundary.
    """
    try:
        from skimage.filters import threshold_otsu

        return float(threshold_otsu(sample))
    except Exception:
        # Degenerate (all same value) → no threshold needed; return 0 so
        # non-zero tiles are marked occupied.
        return 0.0


def _auto_empty_threshold(image: da.Array, channel: int | None, level: int) -> float:
    """Pick an empty-tile threshold from a cheap bounded sample (Otsu)."""
    n = image.ndim
    win = [min(64 if i >= n - 3 else s, s) for i, s in enumerate(image.shape)]
    win = [min(w, 256) if i >= n - 2 else w for i, w in enumerate(win)]
    samples = []
    for frac in (0.33, 0.5, 0.66):
        sl = tuple(
            slice(
                int(s * frac) - w // 2 if s > w else 0,
                (int(s * frac) - w // 2 if s > w else 0) + w,
            )
            for s, w in zip(image.shape, win)
        )
        samples.append(np.asarray(image[sl]).ravel())
    sample = np.concatenate(samples)
    thr = _otsu_threshold(sample)
    logger.info("Auto empty_threshold=%.3g (Otsu on %d samples)", thr, len(samples))
    return thr


def estimate_empty_tiles(
    image: Union[da.Array, str, Path],
    tile_shape: tuple[int, ...],
    threshold: float | None = None,
    channel: int | None = 0,
    level: int = 0,
    sample_window: tuple[int, ...] = (24, 256, 256),
) -> dict[str, Any]:
    """Fast preview of which tiles are background before processing.

    For each tile, reads a small centred window (``sample_window``) and tests
    whether its max exceeds *threshold*. Bounded I/O — runs in seconds to
    minutes on terabyte arrays.

    APPROXIMATE: only the tile centre is inspected. The actual ``tile_process``
    run always tests the whole tile inline. Use this only to pick a threshold
    and gauge the empty fraction before committing to a full run.

    Parameters
    ----------
    image:
        Dask array or OME-ZARR path.
    tile_shape:
        Tile shape you plan to use, e.g. ``(120, 697, 697)``.
    threshold:
        Empty cutoff (signal <= threshold → empty). None → Otsu on samples.
    channel, level:
        Used only when *image* is a path.
    sample_window:
        Size of the centred window read per tile.

    Returns
    -------
    dict with keys:
        ``threshold``, ``n_tiles``, ``n_occupied``, ``empty_fraction``,
        ``occupancy`` (bool ndarray, one entry per tile in the grid).

    Examples
    --------
    >>> info = estimate_empty_tiles("image.zarr", (120, 697, 697))
    >>> print(f"{info['empty_fraction']:.0%} of tiles are background")
    >>> labels = tile_process("image.zarr", fn, tile_shape=(120, 697, 697),
    ...                       skip_empty=True, empty_threshold=info["threshold"])
    """
    n_spatial = len(tile_shape)

    z_src: Any = None
    if isinstance(image, (str, Path)):
        _root = zarr.open_group(str(image), mode="r")
        try:
            _zpath = _root.attrs["multiscales"][0]["datasets"][level]["path"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                f"Cannot read OME-ZARR multiscales metadata at level {level} "
                f"in {image!r}"
            ) from exc
        z_src = _root[_zpath]
        sp_shape = tuple(z_src.shape[-n_spatial:])
    else:
        arr = image
        sp_shape = tuple(arr.shape[-n_spatial:])

    win = [min(w, t, s) for w, t, s in zip(sample_window, tile_shape, sp_shape)]
    grid = [int(np.ceil(s / t)) for s, t in zip(sp_shape, tile_shape)]

    _ch_prefix: tuple = ()
    if z_src is not None:
        n_leading = z_src.ndim - n_spatial
        if channel is not None and n_leading > 0:
            _ch_prefix = (0,) * (n_leading - 1) + (channel,)

    # Streaming single pass: store only per-tile max (a scalar) and a bounded
    # sample list for Otsu. The old approach stored every tile's full block in
    # `blocks` dict — for 2000 tiles × 24×256×256 × 2 bytes = ~6 GB in RAM.
    _MAX_OTSU_SAMPLES = 500
    samples: list[np.ndarray] = []
    tile_maxes: dict[tuple, float] = {}

    for idx in np.ndindex(*grid):
        sl: list[slice] = []
        for i, t, w, s in zip(idx, tile_shape, win, sp_shape):
            start = min(i * t + (t - w) // 2, s - w)
            sl.append(slice(start, start + w))

        if z_src is not None:
            block = np.asarray(z_src[_ch_prefix + tuple(sl)])
        else:
            sub = arr[(...,) + tuple(sl)] if arr.ndim > n_spatial else arr[tuple(sl)]
            block = np.asarray(sub)

        tile_maxes[idx] = float(block.max()) if block.size else 0.0
        if threshold is None and len(samples) < _MAX_OTSU_SAMPLES:
            samples.append(block.ravel())
        # block freed here — not stored

    if threshold is None:
        threshold = _otsu_threshold(np.concatenate(samples) if samples else np.zeros(1))

    occupancy = np.zeros(grid, dtype=bool)
    for idx, mx in tile_maxes.items():
        occupancy[idx] = mx > threshold

    n_tiles = int(occupancy.size)
    n_occ = int(occupancy.sum())
    empty_frac = 1.0 - n_occ / n_tiles if n_tiles else 0.0
    logger.info(
        "estimate_empty_tiles: threshold=%.4g  occupied %d/%d tiles  empty=%.0f%%",
        threshold,
        n_occ,
        n_tiles,
        empty_frac * 100,
    )
    return {
        "threshold": float(threshold),
        "n_tiles": n_tiles,
        "n_occupied": n_occ,
        "empty_fraction": empty_frac,
        "occupancy": occupancy,
    }
