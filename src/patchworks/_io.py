"""OME-ZARR loading and empty-tile estimation."""

from __future__ import annotations

import contextvars
import logging
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Union

import dask.array as da
import numpy as np
import zarr

logger = logging.getLogger(__name__)


# The codec every array patchworks creates is written with. A ContextVar, so
# `with compression(...)` scopes it to one call without leaking into
# concurrent writers; set_compression() changes the process default.
_COMPRESSION: "contextvars.ContextVar[str]" = contextvars.ContextVar(
    "patchworks_compression", default="zstd"
)
_BLOSC_CNAMES = ("zstd", "lz4", "lz4hc", "zlib", "blosclz")


def parse_compression(spec: str) -> tuple[str, str | None, int | None]:
    """Validate a compression spec and split it into its parts.

    Accepted: ``"zstd"`` / ``"zstd:<level>"`` (default level 1),
    ``"blosc"`` / ``"blosc:<cname>"`` / ``"blosc:<cname>:<level>"`` (default
    ``zstd`` at level 5, byte-shuffled), and ``"none"``.

    Returns
    -------
    tuple
        ``(kind, cname, level)``.

    Raises
    ------
    ValueError
        For anything else.
    """
    parts = str(spec).strip().lower().split(":")
    kind = parts[0]
    try:
        if kind == "none" and len(parts) == 1:
            return "none", None, None
        if kind == "zstd" and len(parts) <= 2:
            return "zstd", None, int(parts[1]) if len(parts) == 2 else 1
        if kind == "blosc" and len(parts) <= 3:
            cname = parts[1] if len(parts) >= 2 else "zstd"
            level = int(parts[2]) if len(parts) == 3 else 5
            if cname in _BLOSC_CNAMES:
                return "blosc", cname, level
    except ValueError:
        pass
    raise ValueError(
        f"unknown compression {spec!r}; use 'zstd', 'zstd:<level>', "
        "'blosc', 'blosc:<cname>', 'blosc:<cname>:<level>' "
        f"(cname one of {', '.join(_BLOSC_CNAMES)}), or 'none'"
    )


@contextmanager
def compression(spec: str):
    """Write every array created inside the block with *spec*.

    Covers everything patchworks writes -- stage stores, merged labels,
    pyramid levels -- which is how a whole ``tile_process`` or merge picks a
    codec. See :func:`parse_compression` for the accepted specs.

    Examples
    --------
    >>> from patchworks import compression, tile_process
    >>> with compression("blosc:lz4"):  # doctest: +SKIP
    ...     tile_process("image.zarr", fn)
    """
    parse_compression(spec)
    token = _COMPRESSION.set(spec)
    try:
        yield
    finally:
        _COMPRESSION.reset(token)


def set_compression(spec: str) -> None:
    """Set the codec for every array written from now on (default ``zstd``).

    Prefer :func:`compression` to scope it; this suits a script that writes
    everything with one codec.
    """
    parse_compression(spec)
    _COMPRESSION.set(spec)


def zarr_compressor_kwargs(zarr_format: int = 3) -> dict:
    """Keyword arguments pinning the compression codec for a new array.

    The codec is the active :func:`compression` (zstd level 1 unless
    changed). zstd is already zarr v3's default, but relying on a library
    default means the stores patchworks writes change silently if that
    default ever moves.

    The codec *object* depends on the format of the array being written, not
    on the installed zarr: zarr-python 3 can write a zarr-v2 array (which is
    what NGFF 0.4 needs), and a v2 array rejects ``zarr.codecs`` codecs --
    it wants the numcodecs ones.

    Parameters
    ----------
    zarr_format : int, optional
        Format of the array about to be created, 2 or 3 (default 3).

    Returns
    -------
    dict
        ``compressors=`` holding the codec that format expects (``None``
        for ``"none"``).
    """
    kind, cname, level = parse_compression(_COMPRESSION.get())
    if kind == "none":
        return {"compressors": None}
    if zarr_format != 2:
        if kind == "zstd":
            from zarr.codecs import ZstdCodec

            return {"compressors": (ZstdCodec(level=level),)}
        from zarr.codecs import BloscCodec

        return {
            "compressors": (
                BloscCodec(cname=cname, clevel=level, shuffle="shuffle"),
            )
        }
    import numcodecs

    if kind == "zstd":
        return {"compressors": (numcodecs.Zstd(level=level),)}
    return {
        "compressors": (
            numcodecs.Blosc(
                cname=cname, clevel=level, shuffle=numcodecs.Blosc.SHUFFLE
            ),
        )
    }


# A ".zip" path component: the archive name, then the end or a separator.
# A bare substring test also matched directories such as "my.zipfiles/".
_ZIP_PART = re.compile(r"^(.*?\.zip)(?=$|[/\\])(.*)$", re.IGNORECASE)


def split_zip_path(path: Union[str, Path]) -> "tuple[str, str] | None":
    """Split ``bundle.zip/inner/group`` into ``(archive, inner)``.

    Returns None when no path component ends in ``.zip``.
    """
    m = _ZIP_PART.match(str(path))
    if m is None:
        return None
    # Zarr keys always use "/", whatever the OS spelled the path with.
    return m.group(1), m.group(2).replace("\\", "/").strip("/")


def open_zarr_source(
    store_path: Union[str, Path],
) -> tuple[Union[str, "zarr.storage.StoreLike"], str]:
    """Resolve a store path, transparently opening a ``.zip`` bundle.

    A store packed by ``pixi run zip`` is one file holding
    ``<name>.zarr/...``. zarr reads it in place through a ``ZipStore``, so
    nothing has to be unpacked first -- but a plain path string does not,
    and every reader here takes a path. This returns what zarr and dask
    should actually be handed, plus the prefix to prepend to a component.

    Parameters
    ----------
    store_path : str or Path
        A ``.zarr`` directory, or a ``.zip`` bundle containing one.

    Returns
    -------
    tuple
        ``(source, prefix)``. For a directory, the path and ``""``. For a
        bundle, an open read-only ``ZipStore`` and the store's name inside
        it, so a component is addressed as ``f"{prefix}/{component}"``.

    Raises
    ------
    ValueError
        If a ``.zip`` does not hold exactly one top-level store.
    """
    split = split_zip_path(store_path)
    if split is None:
        return str(store_path), ""

    import zipfile

    # The bundle may be addressed with a group path after it, e.g.
    # "scan.zarr.zip/labels/cells" -- callers build those by string-joining.
    archive_path, inner = split
    with zipfile.ZipFile(archive_path) as archive:
        tops = {
            name.split("/", 1)[0] for name in archive.namelist() if "/" in name
        }
    if len(tops) != 1:
        raise ValueError(
            f"{archive_path} must contain exactly one top-level store; found "
            f"{sorted(tops) or 'nothing'}. Bundles written by "
            "`pixi run zip` always do."
        )
    prefix = tops.pop()
    if inner:
        prefix = f"{prefix}/{inner}"
    return zarr.storage.ZipStore(archive_path, mode="r"), prefix


def open_group_any(path: Union[str, Path], mode: str = "r"):
    """``zarr.open_group`` that also accepts a path *inside* a .zip bundle.

    Callers build group paths by string-joining (``f"{store}/labels"``),
    which a bundle breaks: the archive is a file, not a directory. Split on
    the ``.zip`` instead, so ``bundle.zip/labels/cells`` resolves to the
    right group inside it.
    """
    source, prefix = open_zarr_source(path)
    return zarr.open_group(source, path=prefix, mode=mode)


def from_zarr_any(path: Union[str, Path], component: str | None = None):
    """``dask.array.from_zarr`` that also accepts a .zip bundle."""
    import dask.array as _da

    source, prefix = open_zarr_source(path)
    inner = _component(prefix, component) if component else prefix
    return (
        _da.from_zarr(source, component=inner)
        if inner
        else _da.from_zarr(source)
    )


def _component(prefix: str, name: str) -> str:
    """Join a bundle prefix and a component, tolerating an empty prefix."""
    return f"{prefix}/{name}" if prefix else name


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
    source, prefix = open_zarr_source(store_path)
    root = zarr.open_group(source, path=prefix, mode="r")
    # OME-ZARR 0.5 nests under "ome" key; older stores use "multiscales" directly
    _attrs = dict(root.attrs)
    _ms = _attrs.get("multiscales") or _attrs.get("ome", {}).get("multiscales")
    try:
        path = _ms[0]["datasets"][level]["path"]
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

    arr = da.from_zarr(
        source, component=_component(prefix, path), chunks=zarr_chunks
    )
    if channel is not None:
        arr = _select_channel(arr, channel, _ms[0], store_path)
    return arr


def _select_channel(arr, channel: int, multiscale: dict, store_path):
    """Index *arr*'s channel axis, or leave it alone when there isn't one.

    ``arr[channel]`` used to be applied unconditionally, so on a
    single-channel store written as plain ``zyx`` the default ``channel: 0``
    silently sliced away **z** instead. The result stayed a valid array, just
    one dimension short, and surfaced much later as a tile-count mismatch
    against the occupancy grid rather than as anything about channels.

    Parameters
    ----------
    arr : da.Array
        The full array as stored.
    channel : int
        Requested channel index.
    multiscale : dict
        The store's multiscales entry, read for its ``axes``.
    store_path : str or Path
        Only used in messages.

    Returns
    -------
    da.Array
        *arr* with the channel axis indexed away, or unchanged when the store
        has no channel axis and channel 0 was requested.
    """
    axes = [
        (a.get("name") if isinstance(a, dict) else a) or ""
        for a in (multiscale.get("axes") or [])
    ]
    if "c" in axes:
        idx = axes.index("c")
        return arr[(slice(None),) * idx + (channel,)]

    # No axes metadata: fall back to shape. A 4-D array is c,z,y,x by the
    # convention this package writes; a 3-D one is z,y,x.
    if not axes and arr.ndim >= 4:
        return arr[channel]

    if channel:
        raise ValueError(
            f"channel={channel} was requested but {store_path!r} has no "
            f"channel axis (axes={axes or 'unknown'}, shape={arr.shape}). "
            "Set channel: null in the config for a single-channel image."
        )
    logger.info(
        "%s has no channel axis; ignoring channel=0 and using the whole "
        "array (shape %s).",
        store_path,
        arr.shape,
    )
    return arr


def _otsu_threshold(sample: np.ndarray) -> float:
    """Otsu threshold of *sample*; falls back to 0 if empty.

    Operates on the full distribution including zeros — zeros are background
    pixels and must be included so Otsu can find the signal/background boundary.

    A NumPy port of ``skimage.filters.threshold_otsu`` (same histogram: one
    bin per integer value for integer data, 256 bins otherwise, and the same
    between-class variance), so the empty-tile threshold does not quietly
    degrade to 0 when scikit-image, which is not a dependency, is missing.

    Parameters
    ----------
    sample : np.ndarray
        Intensity sample (any shape).

    Returns
    -------
    float
        The Otsu threshold; the single value for a constant sample, ``0.0``
        for an empty one.
    """
    sample = np.asarray(sample).ravel()
    if sample.size == 0:
        return 0.0
    if np.issubdtype(sample.dtype, np.floating):
        sample = sample[np.isfinite(sample)]
        if sample.size == 0:
            return 0.0
    lo, hi = sample.min(), sample.max()
    if lo == hi:
        return float(lo)

    if np.issubdtype(sample.dtype, np.integer):
        counts = np.bincount((sample.astype(np.int64) - int(lo)).ravel())
        centers = np.arange(int(lo), int(hi) + 1, dtype=np.float64)
    else:
        counts, edges = np.histogram(sample, bins=256, range=(lo, hi))
        centers = (edges[:-1] + edges[1:]) / 2

    counts = counts.astype(np.float64)
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean1 = np.cumsum(counts * centers) / weight1
        mean2 = (np.cumsum((counts * centers)[::-1]) / weight2[::-1])[::-1]
        variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    return float(centers[np.nanargmax(variance)])


def auto_empty_threshold(
    image: da.Array, channel: int | None, level: int
) -> float:
    """Pick an empty-tile threshold from a cheap bounded sample (Otsu).

    Parameters
    ----------
    image : da.Array
        Image to sample.
    channel : int or None
        Channel hint (kept for signature symmetry).
    level : int
        Pyramid level hint (kept for signature symmetry).

    Returns
    -------
    float
        Otsu threshold over a few small centred windows.
    """
    n = image.ndim
    win = [min(64 if i >= n - 3 else s, s) for i, s in enumerate(image.shape)]
    win = [min(w, 256) if i >= n - 2 else w for i, w in enumerate(win)]
    samples = []
    for frac in (0.33, 0.5, 0.66):
        # Clamp each window inside the array: an axis just longer than the
        # window gave a negative start, which Python reads from the end --
        # an empty or truncated sample instead of a full one.
        sl = tuple(
            slice(start, start + w)
            for start, w in (
                (min(max(0, int(s * frac) - w // 2), s - w), w)
                for s, w in zip(image.shape, win)
            )
        )
        samples.append(np.asarray(image[sl]).ravel())
    sample = np.concatenate(samples)
    thr = _otsu_threshold(sample)
    logger.info(
        "Auto empty_threshold=%.3g (Otsu on %d samples)", thr, len(samples)
    )
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
        _root = open_group_any(image)
        _rattr = dict(_root.attrs)
        _rms = _rattr.get("multiscales") or _rattr.get("ome", {}).get(
            "multiscales"
        )
        try:
            _zpath = _rms[0]["datasets"][level]["path"]
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
            # Centre the window in this tile, then clamp it to the tile's own
            # extent -- NOT to the array's. Clamping to ``s - w`` used to drag
            # the last (partial) tile's window backwards into its neighbour,
            # so an edge tile's verdict came partly from the tile before it.
            lo, hi = i * t, min((i + 1) * t, s)
            start = max(lo, min(lo + (hi - lo - w) // 2, hi - w))
            sl.append(slice(start, min(start + w, hi)))

        if z_src is not None:
            block = np.asarray(z_src[_ch_prefix + tuple(sl)])
        else:
            sub = (
                arr[(...,) + tuple(sl)]
                if arr.ndim > n_spatial
                else arr[tuple(sl)]
            )
            block = np.asarray(sub)

        tile_maxes[idx] = float(block.max()) if block.size else 0.0
        if threshold is None and len(samples) < _MAX_OTSU_SAMPLES:
            samples.append(block.ravel())
        # block freed here — not stored

    if threshold is None:
        threshold = _otsu_threshold(
            np.concatenate(samples) if samples else np.zeros(1)
        )

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
