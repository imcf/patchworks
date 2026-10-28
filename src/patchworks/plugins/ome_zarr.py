"""OME-ZARR conversion plugin for patchworks.

Write any array **or any image file** to a pyramidal OME-NGFF (OME-ZARR)
store, generating downsampled resolution levels for fast multi-scale viewing.

Input handling, in order:

* a dask or NumPy array → used directly (NumPy is wrapped lazily);
* a ``.zarr`` path → read through :func:`patchworks.load_ome_zarr`;
* a ``.ims`` (Imaris) path → read lazily with ``imaris-ims-file-reader``
  (HDF5, no JVM); install with ``pip install "patchworks[imaris]"``;
* with ``sequence_pattern=`` set, a glob over a folder of single-plane TIFFs
  → each file becomes one chunk of a lazy dask array via
  :func:`tifffile.TiffSequence`, no data duplicated;
* any other path (CZI, LIF, ND2, OME-TIFF, …) → opened lazily with
  `bioio <https://github.com/bioio-devs/bioio>`_.

Pixel calibration (physical voxel size) is read from the input — bioio's
``physical_pixel_sizes``, the Imaris resolution metadata, an existing
OME-ZARR's scale transform, or (for a TIFF sequence) the first file's own
ImageJ/resolution tags — and written into the NGFF ``coordinateTransformations``
so the µm/pixel sizing is preserved. Pass ``pixel_size=`` to override or to
supply it for bare arrays.

Downsampling is only on **X and Y**; ``z`` (and channel/time) stay at full
resolution. Images are **block-averaged** (each output pixel is the mean of
the ``downscale x downscale`` block it covers), which keeps overview levels
free of the aliasing and shot-noise speckle that plain decimation gives a
fluorescence image. Labels are **decimated** (nearest-neighbour), the
label-preserving choice: averaging ids would invent objects that never
existed. Every level is built by reading the *previous level
back from disk*, so the pyramid never materialises a whole volume in RAM.
Because neither method needs a halo, levels are written zarr-natively: each
level's chunks are chosen as ``ceil(src_chunk / stride)`` so one task reads
exactly one source chunk and writes exactly one output chunk, making peak
memory ``n_workers × one chunk`` by construction. Explicit ``chunks=`` or
sharding fall back to dask, which must rechunk and therefore reads several
source chunks per output chunk.

This module also exposes :func:`add_pyramid` (add levels to an existing store)
and :func:`write_labels` (store a label image under the NGFF ``labels/`` group).

Usage
-----
>>> from patchworks.plugins.ome_zarr import to_ome_zarr  # doctest: +SKIP
>>> to_ome_zarr("scan.ims", "scan.zarr")  # doctest: +SKIP
'scan.zarr'
"""

from __future__ import annotations

import contextvars
import glob
import logging
import math
import shutil
import time as _time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextlib import nullcontext as _nullcontext
from itertools import product as _iproduct
from pathlib import Path
from typing import Union

import dask
import dask.array as da
import numpy as np
import zarr

from .._chunks import cpu_allocation, safe_worker_count
from .._progress import (
    PROGRESS_INTERVAL_S as _PROGRESS_INTERVAL_S,
)
from .._progress import dask_progress, log_progress
from .._io import compression as _compression
from .._provenance import write_provenance
from .._io import (
    is_remote,
    load_ome_zarr,
    open_group_any,
    zarr_compressor_kwargs,
)

logger = logging.getLogger(__name__)

PixelSize = dict[str, float]

_NGFF_VERSION = "0.4"
_SPATIAL_AXES = frozenset("zyx")
# Only X and Y are downsampled when building pyramids; Z is kept at full
# resolution (microscopy stacks are already coarse and anisotropic in Z).
_DOWNSAMPLE_AXES = frozenset("yx")
# axis names assigned to a bare N-D array, taken from the right:
_DEFAULT_ORDER = "tczyx"
# Per-axis chunk caps so each written chunk stays small (≈tens of MB) and the
# pyramid never needs a whole plane/volume in RAM.
_CHUNK_CAP = {"t": 1, "c": 1, "z": 16, "y": 1024, "x": 1024}


def _default_axes(ndim: int) -> str:
    """Assign trailing OME axis names to an unlabelled array.

    A 2-D array becomes ``"yx"``, 3-D ``"zyx"``, 4-D ``"czyx"``.

    Parameters
    ----------
    ndim : int
        Number of array dimensions.

    Returns
    -------
    str
        The axes string for those trailing dimensions.
    """
    if ndim > len(_DEFAULT_ORDER):
        raise ValueError(
            f"cannot infer axes for a {ndim}-D array; pass axes= explicitly"
        )
    guess = _DEFAULT_ORDER[len(_DEFAULT_ORDER) - ndim :]
    if ndim >= 4:
        # From 4-D the leading axis could be channel or time.
        logger.warning(
            "no axes= given for a %d-D array; guessing %r from the number of "
            "dimensions. Pass axes= explicitly if the leading axis is not %r.",
            ndim,
            guess,
            guess[0],
        )
    return guess


def _axis_type(name: str) -> str:
    """Map an axis letter to its NGFF axis type.

    Parameters
    ----------
    name : str
        Axis letter (``z``/``y``/``x``/``c``/``t``).

    Returns
    -------
    str
        ``"space"``, ``"time"`` or ``"channel"``.
    """
    if name in _SPATIAL_AXES:
        return "space"
    return "time" if name == "t" else "channel"


def _axes_meta(axes: str, calibrated: bool) -> list[dict]:
    """Build NGFF ``axes`` metadata for an axes string.

    Parameters
    ----------
    axes : str
        One letter per axis.
    calibrated : bool
        When True, spatial axes carry a micrometer unit.

    Returns
    -------
    list of dict
        One ``{name, type[, unit]}`` entry per axis.
    """
    meta = []
    for a in axes:
        entry = {"name": a, "type": _axis_type(a)}
        if calibrated and a in _SPATIAL_AXES:
            entry["unit"] = "micrometer"
        meta.append(entry)
    return meta


def _strides(axes: str, downscale: int) -> tuple[int, ...]:
    """Per-axis downsampling stride for one pyramid step.

    Parameters
    ----------
    axes : str
        One letter per axis.
    downscale : int
        Downsampling factor for X/Y.

    Returns
    -------
    tuple of int
        ``downscale`` for X/Y axes, ``1`` for Z/C/T.
    """
    return tuple(downscale if a in _DOWNSAMPLE_AXES else 1 for a in axes)


def _default_chunks(
    shape: tuple[int, ...],
    axes: str,
    source_chunks: Union[tuple[int, ...], None] = None,
) -> tuple[int, ...]:
    """Bounded chunk shape so writing a level never blows up RAM.

    With *source_chunks*, the result never spans more than one source chunk on
    any axis. That matters when the source's own granularity is coarse: a
    folder of stitched TIFFs gives one chunk per **file**, so a plane can be
    gigabytes. Grouping 16 of those into one output chunk (the ``z`` cap) means
    holding 16 whole planes to write a 32 MB chunk -- tens of GB for the
    smallest unit of work, which is how conversion gets OOM-killed.

    Splitting a source chunk is fine (that is a slice of something already
    read); combining several is not.

    Parameters
    ----------
    shape : tuple of int
        Array shape.
    axes : str
        One letter per axis (selects the per-axis cap).
    source_chunks : tuple of int, optional
        The input array's own chunking.

    Returns
    -------
    tuple of int
        Per-axis chunk size, capped by ``_CHUNK_CAP`` and by *source_chunks*.
    """
    caps = [min(s, _CHUNK_CAP.get(a, s)) for s, a in zip(shape, axes)]
    if source_chunks is not None:
        caps = [min(c, max(1, int(sc))) for c, sc in zip(caps, source_chunks)]
    return tuple(caps)


ShardSpec = Union[bool, tuple[int, ...]]
_SHARD_TARGET_BYTES = 512 * 1024**2  # aim for ~512 MB shards
# NGFF 0.4 is defined over zarr v2 and puts its keys at the top level of
# .zattrs; 0.5 is the zarr-v3 revision and nests them under an "ome" key with
# the version there. Writing v3 data with 0.4's layout matches neither, so the
# layout follows the zarr version actually being written.
_NGFF_VERSION_V3 = "0.5"

# Which zarr format each NGFF version is defined over. The two are not
# independent choices: 0.4 is specified against zarr v2 and 0.5 against v3,
# so asking for a version is also asking for a store format.
_NGFF_ZARR_FORMAT = {_NGFF_VERSION: 2, _NGFF_VERSION_V3: 3}
# Released but deliberately not written; see _resolve_ngff_version.
_NGFF_UNSUPPORTED = ("0.6",)

# Set for the duration of one public write call. A ContextVar rather than a
# plain global so concurrent writers (the pyramid's own thread pool, or two
# stores written from one process) cannot see each other's target.
_ngff_target: "contextvars.ContextVar[Union[str, None]]" = (
    contextvars.ContextVar("patchworks_ngff_target", default=None)
)


def _resolve_ngff_version(requested: Union[str, None] = None) -> str:
    """Validate an NGFF version request and resolve ``"auto"``.

    ``"auto"`` (the default) picks 0.5, the version defined over the zarr v3
    that patchworks requires. An explicit version is honoured instead --
    including 0.4, which writes a zarr-v2 store.

    Parameters
    ----------
    requested : str or None, optional
        ``"auto"``/``None``, or an NGFF version string.

    Returns
    -------
    str
        The version that will be written.

    Raises
    ------
    ValueError
        For an unknown version, or for one patchworks cannot yet write.
    """
    if requested in (None, "auto"):
        return _NGFF_VERSION_V3
    requested = str(requested)
    if requested in _NGFF_UNSUPPORTED:
        raise ValueError(
            f"NGFF {requested} is released but patchworks does not write it "
            "yet: it replaces a multiscale's `axes` with `coordinateSystems` "
            "and requires `input`/`output` on every coordinate transformation "
            "(RFC-5), which is a different metadata document, not a version "
            "bump. No reader supports it yet either -- ome-zarr-py, and so "
            f'napari, still default to {_NGFF_VERSION_V3}. Use "auto".'
        )
    if requested not in _NGFF_ZARR_FORMAT:
        known = ", ".join(sorted(_NGFF_ZARR_FORMAT))
        raise ValueError(
            f'unknown ngff_version {requested!r}; expected "auto" or one '
            f"of {known}"
        )
    return requested


def ngff_version() -> str:
    """NGFF version currently being written.

    Inside a write call this reflects that call's ``ngff_version=``;
    outside one it is the version an ``"auto"`` write would produce.
    """
    return _resolve_ngff_version(_ngff_target.get())


def _zarr_format() -> int:
    """Zarr format matching the NGFF version currently being written."""
    return _NGFF_ZARR_FORMAT[ngff_version()]


def _group_zarr_format(group_or_path) -> int:
    """Zarr format to write into *group_or_path*.

    An existing store wins over the requested NGFF version: mixing v2 and v3
    arrays in one store leaves something no reader can open, and the label
    groups in particular are written by a later call (the merge's
    ``register_labels``) than the image was. ``ngff_version`` therefore
    decides the format only when there is no store yet.
    """
    try:
        if isinstance(group_or_path, (str, Path)):
            grp = zarr.open_group(str(group_or_path), mode="r")
        else:
            grp = group_or_path
        return int(grp.metadata.zarr_format)
    except Exception:
        return _zarr_format()


def _ngff_version_for(group_or_path) -> str:
    """NGFF version to record for a store, given the format it is in.

    The metadata layout is part of the version: 0.4 puts its keys at the top
    level of a zarr-v2 store, 0.5 nests them under ``ome`` in a v3 one.
    Writing one store's arrays in v2 and its metadata in the 0.5 layout would
    describe a store that does not exist, so both follow the same source of
    truth -- the store -- and the requested version only decides a new one.
    """
    fmt = _group_zarr_format(group_or_path)
    requested = ngff_version()
    if _NGFF_ZARR_FORMAT[requested] == fmt:
        return requested
    return _NGFF_VERSION if fmt == 2 else _NGFF_VERSION_V3


def _open_group(path, mode: str = "a") -> "zarr.Group":
    """Open (or create) a group in the zarr format being written.

    ``zarr_format=`` only matters when the group does not exist yet: an
    existing store keeps its own format, which is what re-opening one for
    ``labels/`` has to do. Passing it unconditionally would make zarr refuse
    to reopen a v2 store on a 0.5 write, so it is passed only on creation.
    """
    path = str(path)
    kwargs = {}
    if not _store_exists(path):
        kwargs["zarr_format"] = _zarr_format()
    return zarr.open_group(path, mode=mode, **kwargs)


def _store_exists(path: str) -> bool:
    """Whether a zarr group already exists at *path* (local or a URL).

    A local check on a URL always says no, which would re-open an existing
    remote NGFF 0.4 (zarr v2) store as v3.
    """
    if not is_remote(path):
        # The metadata file, not the directory, which Snakemake creates empty.
        root = Path(path)
        return (root / "zarr.json").is_file() or (root / ".zgroup").is_file()
    try:
        zarr.open_group(path, mode="r")
    except (
        FileNotFoundError,
        KeyError,
        ValueError,
        zarr.errors.GroupNotFoundError,
    ):
        return False
    return True


@contextmanager
def _writing_ngff(requested: Union[str, None]):
    """Pin the NGFF version for everything written inside the block.

    ``"auto"`` *inherits* an enclosing pin rather than re-deriving one from
    the installed zarr. The public writers call each other -- write_labels
    into register_labels into add_pyramid -- each with its own ``"auto"``
    default, and without inheritance the inner call would silently drag the
    store back to 0.5 halfway through a 0.4 write.
    """
    current = _ngff_target.get()
    if requested in (None, "auto") and current is not None:
        resolved = current
    else:
        resolved = _resolve_ngff_version(requested)
    token = _ngff_target.set(resolved)
    try:
        yield resolved
    finally:
        _ngff_target.reset(token)


def read_ngff_attr(attrs, key: str, default=None):
    """Read an NGFF key from either layout.

    Accepts both the 0.4 top-level placement and the 0.5 ``ome`` nesting, so
    stores written by any patchworks version (or another tool) still load.

    Parameters
    ----------
    attrs : Mapping
        A zarr group's attributes.
    key : str
        NGFF key, e.g. ``"multiscales"``, ``"labels"``, ``"image-label"``.
    default : Any, optional
        Returned when the key is absent from both layouts.
    """
    attrs = dict(attrs)
    if key in attrs:
        return attrs[key]
    nested = attrs.get("ome")
    if isinstance(nested, dict) and key in nested:
        return nested[key]
    return default


def write_ngff_attrs(group, **entries) -> None:
    """Write NGFF keys in the layout matching the store's zarr version.

    On zarr v3 the keys are merged into the ``ome`` attribute (0.5); on v2
    they go to the top level (0.4). Merging matters because several keys are
    written at different times onto the same group -- ``multiscales`` by the
    pyramid, then ``image-label`` when the labels are registered.
    """
    version = _ngff_version_for(group)
    if _NGFF_ZARR_FORMAT[version] == 2:
        for key, value in entries.items():
            group.attrs[key] = value
        return
    existing = dict(group.attrs).get("ome")
    merged = dict(existing) if isinstance(existing, dict) else {}
    merged["version"] = version
    merged.update(entries)
    group.attrs["ome"] = merged


# Channel colours, cycled, when the source gives none (as napari's own).
_CHANNEL_COLORS = ("00FF00", "FF00FF", "00FFFF", "FFFF00", "FFFFFF", "FF0000")


def channel_names(store: Union[str, Path]) -> "list[str] | None":
    """The channels' names, from the store's ``omero`` metadata; None when
    it has none."""
    from .._io import open_group_any

    try:
        omero = read_ngff_attr(open_group_any(store).attrs, "omero") or {}
    except Exception:  # noqa: BLE001 - unreadable: no names
        return None
    names = [str(c.get("label") or "") for c in omero.get("channels") or []]
    return names if names and all(names) else None


def write_channel_names(
    store: Union[str, Path], names: "list[str]", colors=None
) -> None:
    """Name the image's channels (NGFF ``omero`` metadata), in place.

    napari, Fiji/MoBIE and other OME-Zarr readers show these names instead
    of channel numbers. Raises ValueError when *names* does not match the
    channel axis.
    """
    group = _open_group(str(store), mode="r+")
    meta = read_ngff_attr(group.attrs, "multiscales")[0]
    axes = "".join(
        str(a.get("name", "") if isinstance(a, dict) else a)[:1]
        for a in meta["axes"]
    )
    n = (
        group[meta["datasets"][0]["path"]].shape[axes.index("c")]
        if ("c" in axes)
        else 1
    )
    if len(names) != n:
        raise ValueError(
            f"{len(names)} channel name(s) for {n} channel(s): {names}"
        )
    colors = list(colors or [])
    omero = dict(read_ngff_attr(group.attrs, "omero") or {})
    old = list(omero.get("channels") or [])
    omero["channels"] = [
        {
            **(old[i] if i < len(old) else {}),
            "label": str(name),
            "color": (
                colors[i]
                if i < len(colors) and colors[i]
                else (old[i].get("color") if i < len(old) else None)
                or _CHANNEL_COLORS[i % len(_CHANNEL_COLORS)]
            ),
            "active": True,
        }
        for i, name in enumerate(names)
    ]
    write_ngff_attrs(group, omero=omero)


def source_channel_names(
    source: Union[str, Path], scene: int = 0
) -> "list[str] | None":
    """The channels' names in an image file (CZI, LIF, ND2, OME-TIFF, ...
    through bioio; Imaris; OME-Zarr), or None when it gives none."""
    path = str(source)
    try:
        if path.endswith(".zarr") or ".zarr.zip" in path:
            return channel_names(path)
        if path.lower().endswith(".ims"):
            import h5py

            names = []
            with h5py.File(path, "r") as hf:
                info = hf["DataSetInfo"]
                c = 0
                while f"Channel {c}" in info:
                    raw = info[f"Channel {c}"].attrs.get("Name")
                    names.append(
                        b"".join(raw).decode(errors="replace")
                        if raw is not None
                        else ""
                    )
                    c += 1
        else:
            from bioio import BioImage

            img = BioImage(path)
            img.set_scene(img.scenes[scene])
            names = [str(n) for n in (img.channel_names or [])]
    except Exception:  # noqa: BLE001 - names are a nicety, never an error
        logger.debug("no channel names in %s", path, exc_info=True)
        return None
    return names if names and all(n.strip() for n in names) else None


def _ome_channel_names(xml: str) -> "list[str]":
    """The Channel names of an OME-XML document's first Image."""
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    image = next((e for e in root.iter() if e.tag.endswith("}Image")), root)
    return [
        e.attrib.get("Name", "")
        for e in image.iter()
        if e.tag.endswith("}Channel") or e.tag == "Channel"
    ]


def _tiff_channel_name(path: str) -> "str | None":
    """One single-channel TIFF's channel name: OME-XML, ImageJ's Labels,
    or Micro-Manager's metadata."""
    import tifffile

    with tifffile.TiffFile(path) as tif:
        names = _ome_channel_names(tif.ome_metadata or "")
        if len(names) == 1 and names[0]:
            return names[0]
        labels = (tif.imagej_metadata or {}).get("Labels")
        if isinstance(labels, (list, tuple)):
            labels = labels[0] if len(labels) == 1 else None
        if labels:
            return str(labels)
        mm = tif.micromanager_metadata or {}
        for key in ("Channel", "ChannelName"):
            value = (mm.get("FrameKey") or mm).get(key)
            if isinstance(value, str) and value:
                return value
    return None


def tiff_sequence_channel_names(
    pattern: str, sequence_pattern: str
) -> "list[str] | None":
    """The channels' names in a folder of single-plane TIFFs.

    From a companion OME-XML in the folder (``*.ome.xml``,
    ``*.companion.ome``) naming every channel, else from each channel's
    first file (its OME-XML, ImageJ ``Labels`` or Micro-Manager metadata).
    None when the files do not name every channel.
    """
    import glob
    import re

    files = sorted(glob.glob(pattern))
    if not files or "(?P<C>" not in sequence_pattern:
        return None
    first: dict[int, str] = {}
    for f in files:
        m = re.search(sequence_pattern, Path(f).name) or re.search(
            sequence_pattern, f
        )
        if m and m.group("C") is not None:
            first.setdefault(int(m.group("C")), f)
    n = len(first)
    if n < 2:
        return None
    folder = Path(files[0]).parent
    for companion in sorted(
        [*folder.glob("*.ome.xml"), *folder.glob("*.companion.ome")]
    ):
        try:
            names = _ome_channel_names(companion.read_text(errors="replace"))
        except OSError:
            continue
        if len(names) == n and all(names):
            return names
    try:
        names = [_tiff_channel_name(first[c]) for c in sorted(first)]
    except Exception:  # noqa: BLE001 - names are a nicety, never an error
        logger.debug("no channel names in %s", pattern, exc_info=True)
        return None
    if all(names) and len(set(names)) == n:
        return [str(x) for x in names]
    # The same OME-XML in every file, naming all the channels.
    try:
        import tifffile

        with tifffile.TiffFile(files[0]) as tif:
            names = _ome_channel_names(tif.ome_metadata or "")
    except Exception:  # noqa: BLE001
        return None
    return names if len(names) == n and all(names) else None


def _name_channels_from(
    source: object,
    out: str,
    scene: int = 0,
    sequence_pattern: "str | None" = None,
) -> None:
    """Carry the source file's channel names into the new store."""
    if not isinstance(source, (str, Path)):
        return
    names = (
        tiff_sequence_channel_names(str(source), sequence_pattern)
        if sequence_pattern
        else source_channel_names(source, scene)
    )
    if not names:
        return
    try:
        write_channel_names(out, names)
        logger.info("channel names from %s: %s", source, ", ".join(names))
    except ValueError as exc:  # e.g. a singleton channel axis dropped
        logger.info("channel names not kept: %s", exc)


def _effective_shard(
    requested: tuple[int, ...],
    chunks: tuple[int, ...],
    shape: tuple[int, ...],
) -> tuple[int, ...]:
    """Clamp a requested shard shape to a valid one.

    A shard must be a whole multiple of the inner chunk and should not exceed
    the array's chunk-padded extent.

    Parameters
    ----------
    requested : tuple of int
        Desired shard shape.
    chunks : tuple of int
        Inner chunk shape.
    shape : tuple of int
        Array shape.

    Returns
    -------
    tuple of int
        A shard shape that is a chunk-multiple within the array.
    """
    out = []
    for r, c, s in zip(requested, chunks, shape):
        cap = math.ceil(s / c) * c  # array dim padded up to a whole chunk
        out.append(min(max(c, (r // c) * c), cap))
    return tuple(out)


def _auto_shard(
    chunks: tuple[int, ...], shape: tuple[int, ...], dtype
) -> tuple[int, ...]:
    """Pick a shard shape of roughly ``_SHARD_TARGET_BYTES``.

    Grows the two largest axes equally until the shard reaches the target size,
    then clamps to a valid chunk-multiple.

    Parameters
    ----------
    chunks : tuple of int
        Inner chunk shape.
    shape : tuple of int
        Array shape.
    dtype : data-type
        Array dtype, used to size the shard in bytes.

    Returns
    -------
    tuple of int
        The chosen shard shape.
    """
    itemsize = np.dtype(dtype).itemsize
    base = itemsize
    for c in chunks:
        base *= c
    big = sorted(range(len(chunks)), key=lambda i: shape[i], reverse=True)[:2]
    factor = max(1, int((_SHARD_TARGET_BYTES / max(1, base)) ** 0.5))
    shard = list(chunks)
    for i in big:
        shard[i] = chunks[i] * factor
    return _effective_shard(tuple(shard), chunks, shape)


def _shard_for(
    shard: ShardSpec,
    chunks: tuple[int, ...],
    shape: tuple[int, ...],
    dtype,
    zarr_format: Union[int, None] = None,
) -> Union[tuple[int, ...], None]:
    """Resolve the ``shard`` argument into a concrete shard shape.

    Parameters
    ----------
    shard : bool or tuple of int
        ``False`` → no sharding; ``True`` → auto;
        a tuple → an explicit shard shape.
    chunks : tuple of int
        Inner chunk shape.
    shape : tuple of int
        Array shape.
    dtype : data-type
        Array dtype.
    zarr_format : int or None, optional
        Format of the store being written. ``None`` (default) falls back to
        the format the requested NGFF version implies. Zarr v2 has no
        sharding codec, so 2 always means "no shards".

    Returns
    -------
    tuple of int or None
        The shard shape, or ``None`` when not sharding
        (also when the target store is zarr v2 / NGFF 0.4).
    """
    if not shard:
        return None
    if (zarr_format if zarr_format is not None else _zarr_format()) == 2:
        logger.warning(
            "sharding requires zarr v3 (NGFF %s); writing unsharded.",
            _NGFF_VERSION_V3,
        )
        return None
    if shard is True:
        return _auto_shard(chunks, shape, dtype)
    return _effective_shard(tuple(shard), chunks, shape)


# Don't let deep pyramid levels fragment into thousands of tiny chunks.
_CHUNK_FLOOR = {"y": 128, "x": 128}


def _level_chunks(
    src_chunks: tuple[int, ...],
    strides: tuple[int, ...],
    axes: str,
    level_shape: tuple[int, ...],
) -> tuple[int, ...]:
    """Chunking for a strided level that keeps the per-task read bounded.

    Choosing ``ceil(src_chunk / stride)`` makes each output chunk the exact
    image of one source chunk, so a task reads one chunk and writes one chunk
    with nothing shared between tasks.

    The floor can override that at deep levels, where ``ceil(src / stride)``
    falls below it — an output chunk then covers a few source chunks instead
    of one. Still bounded by a small constant, which is the property that
    matters; those levels are a fraction of the volume anyway.
    """
    out = []
    for c, st, a, s in zip(src_chunks, strides, axes, level_shape):
        v = -(-c // st)
        v = min(v, _CHUNK_CAP.get(a, v))
        v = max(v, _CHUNK_FLOOR.get(a, 1))
        out.append(max(1, min(v, s)))
    return tuple(out)


def _create_level_array(
    group: "zarr.Group",
    name: str,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype,
) -> "zarr.Array":
    """Create (replacing any existing) a pyramid level array in *group*."""
    if name in group:
        del group[name]
    kwargs = zarr_compressor_kwargs(_group_zarr_format(group))
    return group.create_array(
        name, shape=shape, chunks=chunks, dtype=dtype, **kwargs
    )


DownsampleMethod = str  # "mean" or "nearest"
_DOWNSAMPLE_METHODS = ("mean", "mode", "nearest")


def _compression_scope(spec: Union[str, None]):
    """``compression(spec)``, or a no-op for None (keep the active codec)."""
    return _nullcontext() if spec is None else _compression(spec)


def _check_downsample(method: str) -> str:
    if method not in _DOWNSAMPLE_METHODS:
        raise ValueError(
            f"downsample must be one of {_DOWNSAMPLE_METHODS}, got {method!r}"
        )
    return method


def _downsample(
    region: np.ndarray, strides: tuple[int, ...], method: str
) -> np.ndarray:
    """Downsample *region* by *strides*: decimate, or exact block mean.

    The output has ``ceil(n / stride)`` voxels per axis either way. A block
    cut short by the array edge is averaged over the voxels it has, not
    padded, so edge pixels are not biased.

    Parameters
    ----------
    region : np.ndarray
        Source region, starting on a block boundary.
    strides : tuple of int
        Per-axis factor (1 = untouched).
    method : str
        ``"nearest"`` (every *stride*-th voxel), ``"mean"`` (block average,
        rounded back to an integer dtype; for images) or ``"mode"`` (the
        most frequent non-zero id in the block; for labels -- an object
        survives a coarser level as long as it wins some block, where
        ``"nearest"`` drops anything its sampled voxel misses).

    Returns
    -------
    np.ndarray
        The downsampled region, in *region*'s dtype.
    """
    if method == "nearest":
        return region[tuple(slice(None, None, st) for st in strides)]
    if method == "mode":
        return _block_mode(region, strides)
    if all(n % st == 0 for n, st in zip(region.shape, strides)):
        return _block_mean_whole(region, strides)
    out = region
    for ax, st in enumerate(strides):
        if st == 1:
            continue
        n = out.shape[ax]
        starts = np.arange(0, n, st)
        out = np.add.reduceat(out, starts, axis=ax, dtype=np.float64)
        counts = np.minimum(st, n - starts).astype(np.float64)
        shape = [1] * out.ndim
        shape[ax] = -1
        out = out / counts.reshape(shape)
    if out is region:
        return region
    if np.issubdtype(region.dtype, np.integer):
        info = np.iinfo(region.dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    return out.astype(region.dtype)


def _block_views(region: np.ndarray, strides: tuple[int, ...]) -> list:
    """One strided view per position inside a block (4 for a 2x2 block).

    ``views[k][i]`` is the k-th voxel of output voxel i's block. Plain
    slicing, no copy: summing or comparing these is far cheaper than
    reducing over the small block axes of a reshaped array.
    """
    return [
        region[tuple(slice(o, None, st) for o, st in zip(offs, strides))]
        for offs in _iproduct(*[range(st) for st in strides])
    ]


def _block_mean_whole(
    region: np.ndarray, strides: tuple[int, ...]
) -> np.ndarray:
    """Block mean when every axis is a whole number of blocks.

    The common case -- every chunk but the image's last. Integers are summed
    exactly and rounded half-to-even, as ``np.rint`` of the float mean
    would; a power-of-two block (2x2) divides by shifting.
    """
    views = _block_views(region, strides)
    count = len(views)
    if not np.issubdtype(region.dtype, np.integer):
        total_f = np.zeros(views[0].shape, dtype=np.float64)
        for v in views:
            total_f += v
        return (total_f / count).astype(region.dtype)
    # The narrowest accumulator that cannot overflow: 16-bit data summed over
    # a block of up to 2**15 voxels fits 32 bits, halving memory traffic.
    small = region.dtype.itemsize <= 2 and count <= 2**15
    acc = np.int32 if small else np.int64
    total = np.zeros(views[0].shape, dtype=acc)
    for v in views:
        total += v
    if count & (count - 1) == 0:  # power of two: floor-divide by shifting
        shift = count.bit_length() - 1
        q, r = total >> shift, total & (count - 1)
    else:
        q, r = np.divmod(total, count)
    # Round half to even: up past the midpoint, and at it only from odd q.
    q += (2 * r > count) | ((2 * r == count) & (q % 2 == 1))
    return q.astype(region.dtype)


def _block_mode(region: np.ndarray, strides: tuple[int, ...]) -> np.ndarray:
    """Most frequent non-zero value per block; 0 only for an empty block.

    Ties go to the first voxel in block order, so the result is
    deterministic. Edge blocks are padded with background, which never wins.
    """
    pads = [(0, (-n) % st) for n, st in zip(region.shape, strides)]
    padded = np.pad(region, pads) if any(p for _, p in pads) else region
    cols = _block_views(padded, strides)
    if len(cols) == 1:
        return cols[0].copy()
    # How often each voxel's value occurs in its block, counted pairwise
    # (k*(k-1)/2 compares; 6 for a 2x2 block). Background scores 0, so any
    # object present outvotes it.
    vote_t = np.uint8 if len(cols) < 256 else np.uint32
    votes = [np.ones(cols[0].shape, dtype=vote_t) for _ in cols]
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            same = cols[i] == cols[j]
            votes[i] += same
            votes[j] += same
    out = cols[0]
    best = votes[0] * (cols[0] != 0)
    for i in range(1, len(cols)):
        score = votes[i] * (cols[i] != 0)
        # np.where, not a boolean-mask assignment: ~25% faster here.
        out = np.where(score > best, cols[i], out)
        best = np.maximum(best, score)
    return np.ascontiguousarray(out)


def _stream_strided_level(
    src: "zarr.Array",
    dst: "zarr.Array",
    strides: tuple[int, ...],
    n_workers: int = 4,
    label: str = "level",
    progress: bool = True,
    method: str = "nearest",
) -> None:
    """Write *dst* as the downsampled *src*, one chunk at a time.

    Neither decimation nor a block mean needs a halo, so each output chunk
    depends only on the source region that maps onto it. Peak
    memory is therefore ``n_workers x (one source region + its subsample)``
    however large the image is -- unlike a dask ``rechunk``, whose threaded
    scheduler holds intermediates with no backpressure.

    Parameters
    ----------
    src, dst : zarr.Array
        Source level and the (already created) destination level.
    strides : tuple of int
        Per-axis downsampling factor.
    method : str
        ``"nearest"`` or ``"mean"``; see :func:`_downsample`.
    n_workers : int
        Threads used for the copy. zarr's codecs release the GIL, so threads
        are enough and avoid pickling a worker payload.
    """
    grid = [-(-s // c) for s, c in zip(dst.shape, dst.chunks)]

    def _one(idx: tuple[int, ...]) -> None:
        out_sl = tuple(
            slice(i * c, min((i + 1) * c, s))
            for i, c, s in zip(idx, dst.chunks, dst.shape)
        )
        src_sl = tuple(
            slice(o.start * st, min(o.stop * st, s))
            for o, st, s in zip(out_sl, strides, src.shape)
        )
        dst[out_sl] = _downsample(np.asarray(src[src_sl]), strides, method)

    indices = list(_iproduct(*[range(g) for g in grid]))
    total = len(indices)
    started = _time.monotonic()
    last = started

    def _tick(done: int) -> None:
        nonlocal last
        now = _time.monotonic()
        if progress and now - last >= _PROGRESS_INTERVAL_S:
            last = now
            log_progress(label, done, total, started)

    if n_workers <= 1:
        for done, idx in enumerate(indices, 1):
            _one(idx)
            _tick(done)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for done, _ in enumerate(pool.map(_one, indices), 1):
                _tick(done)
    if progress:
        log_progress(label, total, total, started)


def _stream_sharded_level(
    src: "zarr.Array",
    dst: "zarr.Array",
    strides: tuple[int, ...],
    label: str = "level",
    progress: bool = True,
    method: str = "nearest",
) -> None:
    """Write the sharded *dst* as the downsampled *src*, shard by shard.

    A shard must be written whole by one writer. Each shard's output chunks
    are computed in parallel -- each from the source region it maps onto --
    gathered, and the shard written in one go while the next one is being
    computed. That replaces a dask graph which, for an odd source chunk
    (839 -> rechunked to 838 to start blocks on a stride, then to the output
    chunks, then to the shards), grew to ~260k tasks for one level.

    A chunk whose source region holds no non-zero voxel is not computed, and
    a shard of only such chunks not written: it reads back as the fill value
    0, which is also what downsampling zeros gives. Neither decimation nor
    a block mean or mode needs a halo, and every region starts on a stride
    boundary, so the result is that of downsampling the whole level at once.
    """
    from concurrent.futures import Future

    from .._chunks import cpu_allocation, safe_worker_count

    shards = tuple(dst.shards or dst.chunks)
    chunks = tuple(dst.chunks)
    grid = [-(-s // c) for s, c in zip(dst.shape, shards)]
    itemsize = np.dtype(src.dtype).itemsize
    region_nbytes = (
        int(np.prod([c * st for c, st in zip(chunks, strides)])) * itemsize
    )
    shard_nbytes = int(np.prod(shards)) * np.dtype(dst.dtype).itemsize
    # Two shards held at most (one being written, one being filled).
    n_workers = max(
        1,
        min(
            cpu_allocation(),
            safe_worker_count(region_nbytes, fn_overhead=3),
        ),
    )

    def _chunk(out_sl: tuple[slice, ...]) -> "np.ndarray | None":
        src_sl = tuple(
            slice(o.start * st, min(o.stop * st, s))
            for o, st, s in zip(out_sl, strides, src.shape)
        )
        region = np.asarray(src[src_sl])
        if not region.any():
            return None
        return _downsample(region, strides, method)

    indices = list(_iproduct(*[range(g) for g in grid]))
    total = len(indices)
    if progress:
        logger.info(
            "writing %s: %d shard(s) of %s (%.0f MB) with %d worker(s), "
            "empty regions skipped",
            label,
            total,
            "x".join(map(str, shards)),
            shard_nbytes / 1e6,
            n_workers,
        )
    started = last = _time.monotonic()
    written = 0
    pending: "Future | None" = None
    with (
        ThreadPoolExecutor(max_workers=n_workers) as pool,
        ThreadPoolExecutor(max_workers=1) as writer,
    ):
        for done, idx in enumerate(indices, 1):
            shard_sl = tuple(
                slice(i * c, min((i + 1) * c, s))
                for i, c, s in zip(idx, shards, dst.shape)
            )
            inner = [
                tuple(
                    slice(o.start + j * c, min(o.start + (j + 1) * c, o.stop))
                    for o, j, c in zip(shard_sl, jdx, chunks)
                )
                for jdx in _iproduct(
                    *[
                        range(-(-(o.stop - o.start) // c))
                        for o, c in zip(shard_sl, chunks)
                    ]
                )
            ]
            parts = list(pool.map(_chunk, inner))
            if any(p is not None for p in parts):
                buf = np.zeros(
                    tuple(o.stop - o.start for o in shard_sl), dtype=dst.dtype
                )
                for sl, part in zip(inner, parts):
                    if part is not None:
                        buf[
                            tuple(
                                slice(c.start - o.start, c.stop - o.start)
                                for c, o in zip(sl, shard_sl)
                            )
                        ] = part
                if pending is not None:
                    pending.result()
                pending = writer.submit(dst.__setitem__, shard_sl, buf)
                written += 1
            now = _time.monotonic()
            if progress and now - last >= _PROGRESS_INTERVAL_S:
                last = now
                log_progress(label, done, total, started)
        if pending is not None:
            pending.result()
    if progress:
        log_progress(label, total, total, started)
        logger.info("%s: %d of %d shard(s) hold labels", label, written, total)


# One chunk is one file without sharding. A shared cluster filesystem starts
# to hurt well before a million of them, and the cost lands on every later
# read too, not just the write.
_CHUNK_COUNT_WARN = 200_000


def _warn_if_many_chunks(
    shape: tuple[int, ...], chunks: tuple[int, ...], shard: ShardSpec
) -> None:
    """Warn when a level will be written as a very large number of files.

    A coarse source forces fine output chunks (see :func:`_default_chunks`),
    and fine chunks mean many files. Sharding packs them into far fewer
    without changing the chunking, so the fix is cheap -- but only if someone
    notices before the store is written.
    """
    if shard:
        return
    count = int(np.prod([math.ceil(s / c) for s, c in zip(shape, chunks)]))
    if count < _CHUNK_COUNT_WARN:
        return
    logger.warning(
        "level 0 will be %s chunks, i.e. that many files: the source's own "
        "granularity forced chunks of %s. Set shard=True (config: "
        "shard: true) to pack them into far fewer files -- same chunking, "
        "same memory, and kinder to a shared filesystem.",
        f"{count:,}",
        chunks,
    )


def _bounded_scheduler(arr: da.Array):
    """Cap dask's threads by the memory one **source** chunk costs.

    Dask keeps a source chunk alive while every task that reads it runs, so
    peak memory tracks the *input* granularity, not the output's. A folder of
    stitched TIFFs gives one chunk per file -- gigabytes each -- and the
    default worker count is the machine's core count, so 32 threads x a 3.6 GB
    plane is over 100 GB before anything is written.

    Sizing from the source chunk against the job's real budget keeps that
    bounded. Returns a context manager; a no-op when a distributed client is
    driving, since it schedules with its own memory awareness.
    """
    import dask

    from .._chunks import cpu_allocation, safe_worker_count
    from .._cluster import _distributed_client

    if _distributed_client() is not None:
        return _nullcontext()
    chunk_nbytes = int(np.prod(arr.chunksize)) * arr.dtype.itemsize
    # Two live copies per worker: the source chunk plus the slice being written.
    workers = min(
        cpu_allocation(), safe_worker_count(chunk_nbytes, fn_overhead=2)
    )
    logger.info(
        "writing with %d dask worker(s): source chunks are %.2f GB each",
        workers,
        chunk_nbytes / 1024**3,
    )
    return dask.config.set(scheduler="threads", num_workers=workers)


def _progress_ctx(progress: bool, label: str):
    """Progress context for a long dask write; see patchworks._progress."""
    return dask_progress(label, progress)


# Source chunks this large (a stitched plane per TIFF) are written a slab at
# a time: see _store_in_slabs.
_SLAB_SOURCE_BYTES = 256 * 1024**2
# Copies of a source chunk alive while it is read and split: the bytes the
# reader returns, the decoded array, the slices being written.
_SOURCE_COPIES = 3


def _slab_starts(
    arr: da.Array, block: tuple, source_nbytes: int, copies: int = 0
) -> list:
    """Where each slab starts along the leading (non-y/x) axes.

    A slab is whole write blocks deep (no two slabs touch one chunk or
    shard), and holds as many source chunks as fit in the job's memory,
    counting *copies* of each (default :data:`_SOURCE_COPIES`).
    """
    from .._chunks import _get_available_memory

    lead = arr.ndim - 2
    budget = 0.6 * _get_available_memory()
    copies = copies or _SOURCE_COPIES
    planes = max(1, int(budget // max(1, source_nbytes * copies)))
    # Along the last leading axis (z), as many whole blocks as fit.
    step = list(block[:lead])
    if lead:
        step[-1] = block[lead - 1] * max(1, planes // max(1, block[lead - 1]))
    return list(
        _iproduct(*(range(0, arr.shape[i], step[i]) for i in range(lead)))
    ), step


def _store_in_slabs(
    arr: da.Array,
    target,
    block: tuple,
    source_nbytes: int,
    label: str,
    progress: bool,
) -> None:
    """Store *arr* into *target* one slab of leading-axis blocks at a time.

    Dask holds a source chunk until every output chunk cut from it is
    written, and starts reading the next ones meanwhile: with a whole
    stitched plane per chunk (3.6 GB), the planes in flight -- not the
    worker count -- set the peak, and a 64 or 128 GB job was OOM-killed.
    Writing slab by slab bounds them by what fits in memory.
    """
    starts, step = _slab_starts(arr, block, source_nbytes)
    lead = arr.ndim - 2
    logger.info(
        "writing %s in %d slab(s) of %s along the leading axes: source "
        "chunks are %.2f GB",
        label,
        len(starts),
        tuple(step),
        source_nbytes / 1024**3,
    )
    started = _time.monotonic()
    for n, start in enumerate(starts, 1):
        region = (
            tuple(
                slice(s, min(s + st, arr.shape[i]))
                for i, (s, st) in enumerate(zip(start, step[:lead]))
            )
            + (slice(None),) * 2
        )
        da.store(arr[region], target, regions=region, lock=False, compute=True)
        if progress:
            log_progress(label, n, len(starts), started)


def _to_zarr_level(
    arr: da.Array,
    group_path: str,
    component: str,
    shard: ShardSpec,
    progress: bool = True,
    source_nbytes: int = 0,
) -> None:
    """Write one array to ``group_path/component``, optionally sharded.

    Without sharding, ``da.to_zarr`` writes chunk by chunk. With sharding that
    is unsafe — many chunks share one shard file and per-chunk writes race —
    so we create the sharded array explicitly (inner *chunks* + *shards*) and
    store with the dask blocks rechunked to the **shard** size, so each task
    writes one whole shard atomically.

    Parameters
    ----------
    arr : da.Array
        Array to write (its chunk size becomes the inner chunk).
    group_path : str
        Path of the parent zarr group.
    component : str
        Array name within the group.
    shard : bool or tuple of int
        Sharding request; see :func:`_shard_for`.
    progress : bool
        Show a dask progress bar for the write.
    source_nbytes : int, optional
        Bytes of one chunk of the array *arr* was cut from. When large
        (:data:`_SLAB_SOURCE_BYTES`), the write goes slab by slab
        (:func:`_store_in_slabs`) to bound memory.

    Returns
    -------
    None
    """
    inner = arr.chunksize
    label = f"{Path(group_path).name}/{component}"
    slabs = source_nbytes >= _SLAB_SOURCE_BYTES and arr.ndim > 2
    fmt = _group_zarr_format(group_path)
    sh = _shard_for(shard, inner, arr.shape, arr.dtype, fmt)
    ctx = _progress_ctx(progress, f"{Path(group_path).name}/{component}")
    if not sh:
        # Created here (dask's array-creation keywords changed between
        # releases); regular chunks need no lock.
        if any(len(set(c[:-1])) > 1 or c[-1] > c[0] for c in arr.chunks):
            arr = arr.rechunk(arr.chunksize)
        z = _create_level_array(
            _open_group(group_path),
            component,
            arr.shape,
            arr.chunksize,
            arr.dtype,
        )
        if slabs:
            _store_in_slabs(
                arr, z, arr.chunksize, source_nbytes, label, progress
            )
            return
        with ctx:
            arr.store(z, lock=False, compute=True)  # type: ignore[call-arg]
        return
    grp = _open_group(group_path)
    if component in grp:
        del grp[component]
    z = grp.create_array(
        name=component,
        shape=arr.shape,
        chunks=inner,
        shards=sh,
        dtype=arr.dtype,
        **zarr_compressor_kwargs(fmt),
    )
    # The rechunk to shards is bounded by the job's CPUs, not the machine's.
    shard_nbytes = int(np.prod(sh)) * arr.dtype.itemsize
    n_workers = max(
        1,
        min(
            cpu_allocation(),
            safe_worker_count(shard_nbytes, fn_overhead=3),
        ),
    )
    logger.debug(
        "resharding with %d worker(s) for %.0f MB shards",
        n_workers,
        shard_nbytes / 1024**2,
    )
    if slabs:
        _store_in_slabs(arr.rechunk(sh), z, sh, source_nbytes, label, progress)
        return
    with ctx, dask.config.set(scheduler="threads", num_workers=n_workers):
        # No lock: every block is exactly one shard (rechunked to the shard
        # grid from the origin), so no two tasks touch the same file. A lock
        # would serialise the whole write, compression included.
        arr.rechunk(sh).store(z, lock=False, compute=True)


def reshard_level(
    group_path: Union[str, Path],
    component: str = "0",
    shard: ShardSpec = True,
    progress: bool = True,
) -> str:
    """Rewrite an existing array in place, sharded, via one dask pass.

    For an array that had to be written *unsharded* because many writers
    filled it a chunk at a time -- a label group's level 0, written tile by
    tile by concurrent segment jobs, or by the merge's own worker pool -- a
    shard cannot be produced while those writers are running: they would
    read-modify-write the same shard file and silently lose each other's
    chunks. Doing it afterwards, single-threaded through
    :func:`_to_zarr_level`, is safe because one writer owns every shard.

    The cost is one extra full read+write of the level, which is why this is
    opt-in rather than implied by ``shard``.

    Array attributes are carried across. That matters for a label level 0:
    the merge records how far it got there, and losing that would let a
    re-run merge already-merged ids and collide unrelated objects.

    Parameters
    ----------
    group_path : str or Path
        Path of the zarr group holding the array.
    component : str, optional
        Array name within the group (default ``"0"``).
    shard : bool or tuple of int, optional
        Sharding request; see :func:`_shard_for`. ``False`` is a no-op.
    progress : bool, optional
        Show a dask progress bar for the rewrite.

    Returns
    -------
    str
        The path of the array that was rewritten.
    """
    target = f"{group_path}/{component}"
    if not shard:
        return target
    if is_remote(group_path):
        raise ValueError(
            f"reshard_level swaps directories by rename, which a remote store "
            f"({group_path}) cannot do; reshard a local copy instead"
        )
    if _group_zarr_format(group_path) == 2:
        logger.warning(
            "%s is a zarr v2 store (NGFF %s), which has no sharding codec; "
            "leaving it unsharded.",
            target,
            _NGFF_VERSION,
        )
        return target

    grp = _open_group(group_path)
    attrs = dict(grp[component].attrs)

    # Written beside the original, then swapped: rewriting in place would
    # mean reading an array that is being overwritten underneath.
    tmp = f"{component}__resharding"
    src = da.from_zarr(str(group_path), component=component)
    _to_zarr_level(src, str(group_path), tmp, shard, progress)

    # Swap by renames only, deleting the original last: removing it first
    # left a window in which a crash or a scheduler kill lost the level
    # outright. Now the data exists under one of the two names throughout.
    old, new = Path(group_path) / component, Path(group_path) / tmp
    backup = Path(group_path) / f"{component}__pre_reshard"
    shutil.rmtree(backup, ignore_errors=True)
    old.rename(backup)
    new.rename(old)
    shutil.rmtree(backup)

    grp = _open_group(group_path)
    grp[component].attrs.update(attrs)
    logger.info("resharded %s (%d attr(s) carried over)", target, len(attrs))
    return target


def _normalize_pixel_size(
    pixel_size: Union[PixelSize, tuple, None], axes: str
) -> PixelSize:
    """Coerce a pixel-size dict/tuple into ``{axis: size}``.

    Parameters
    ----------
    pixel_size : dict, tuple or None
        Voxel size as a per-axis dict or a tuple
        aligned to the spatial axes.
    axes : str
        One letter per axis.

    Returns
    -------
    dict
        ``{axis: size}`` for the spatial axes present (empty if none given).
    """
    if not pixel_size:
        return {}
    if isinstance(pixel_size, dict):
        return {a: float(v) for a, v in pixel_size.items() if a in axes}
    # tuple/list: aligned to the spatial axes present, in axes order
    spatial = [a for a in axes if a in _SPATIAL_AXES]
    return {a: float(v) for a, v in zip(spatial, pixel_size)}


def _base_scale(axes: str, pixel_size: PixelSize) -> list[float]:
    """Build the level-0 NGFF scale vector.

    Parameters
    ----------
    axes : str
        One letter per axis.
    pixel_size : dict
        ``{axis: size}`` for spatial axes.

    Returns
    -------
    list of float
        Physical size per spatial axis, ``1.0`` for C/T.
    """
    return [float(pixel_size.get(a, 1.0)) for a in axes]


def _dataset(
    name: str, scale: list[float], translation: "list[float] | None" = None
) -> dict:
    """Build one NGFF ``multiscales`` dataset entry.

    Parameters
    ----------
    name : str
        Component path of the level (e.g. ``"0"``).
    scale : list of float
        Per-axis scale (physical size × downsample factor).
    translation : list of float, optional
        Physical offset of the level's first voxel centre. Omitted when all
        zero, which keeps the metadata of a decimated pyramid unchanged.

    Returns
    -------
    dict
        A dataset dict with its ``path`` and ``coordinateTransformations``.
    """
    transforms: list[dict] = [
        {"type": "scale", "scale": [float(s) for s in scale]}
    ]
    if translation is not None and any(t != 0 for t in translation):
        transforms.append(
            {
                "type": "translation",
                "translation": [float(t) for t in translation],
            }
        )
    return {"path": name, "coordinateTransformations": transforms}


def _level_translation(
    base_scale: list[float],
    base_translation: "list[float] | None",
    strides: tuple[int, ...],
    level: int,
    downsample: str,
) -> list[float]:
    """Where level *level*'s first voxel centre sits, physically.

    A decimated level keeps voxel 0 of the level below, so it inherits the
    base offset. A block method's (mean, mode) first voxel summarises the
    first ``f`` voxels (``f = stride**level``), so its centre is ``(f - 1) / 2``
    base voxels further in; without that offset, labels segmented at an
    averaged level would sit up to half a coarse voxel off the image.
    """
    base_t = list(base_translation or [0.0] * len(base_scale))
    if downsample == "nearest":
        return base_t
    return [
        t + (st**level - 1) / 2 * sc
        for t, st, sc in zip(base_t, strides, base_scale)
    ]


def _level_shapes(
    shape: tuple[int, ...], strides: tuple[int, ...], n_levels: int
) -> list[tuple[int, ...]]:
    """Each level's shape, stopping where :func:`_write_pyramid` does."""
    shapes = [tuple(shape)]
    for i in range(1, n_levels):
        nxt = tuple(-(-s // st) for s, st in zip(shapes[-1], strides))
        if min(nxt) < 1:
            logger.info("stopping pyramid at level %d (next too small)", i)
            break
        shapes.append(nxt)
    return shapes


def _downsample_bands(
    region: np.ndarray, strides: tuple[int, ...], method: str, workers: int
) -> np.ndarray:
    """:func:`_downsample` of *region*, in bands along y run in parallel.

    Only y and x are strided, and a band starts on a stride boundary, so
    each output voxel comes from the same input block as in one call: the
    result is identical, computed on every core.
    """
    ax = region.ndim - 2
    st = strides[ax]
    rows = region.shape[ax]
    band = max(st, (-(-rows // (4 * max(1, workers)))) // st * st)
    starts = list(range(0, rows, band))
    if len(starts) == 1:
        return _downsample(region, strides, method)

    def one(start: int) -> np.ndarray:
        sl = [slice(None)] * region.ndim
        sl[ax] = slice(start, start + band)
        return _downsample(region[tuple(sl)], strides, method)

    with ThreadPoolExecutor(workers) as pool:
        return np.concatenate(list(pool.map(one, starts)), axis=ax)


def _write_slab(
    data: np.ndarray,
    region: tuple,
    targets: list,
    blocks: list,
    strides: tuple[int, ...],
    downsample: str,
    workers: int,
    lead: int,
) -> None:
    """Write one slab into every level, each downsampled from the last."""
    for k, (target, block) in enumerate(zip(targets, blocks)):
        if k:
            data = _downsample_bands(data, strides, downsample, workers)
        da.from_array(
            data, chunks=tuple(data.shape[:lead]) + tuple(block[lead:])
        ).store(target, regions=region, lock=False, compute=True)


def _write_pyramid_one_pass(
    arr: da.Array,
    axes: str,
    group_path: str,
    *,
    base_chunks: tuple[int, ...],
    n_levels: int,
    strides: tuple[int, ...],
    chunks: Union[tuple[int, ...], None],
    base_scale: list[float],
    base_name: str,
    shard: ShardSpec,
    progress: bool,
    downsample: str,
    base_translation: "list[float] | None",
) -> Union[list[dict], None]:
    """Every level from one read of the source, slab by slab; or ``None``
    where that does not apply (the levels are then written one by one).

    For a source of huge chunks (a stitched plane per TIFF): each level is
    otherwise made by reading the one below back from disk -- level 1
    rereads all of level 0, terabytes for a large image, and took longer
    than level 0 itself. Only y and x are downsampled, so every level of a
    plane comes from that plane alone: a slab of planes is read once and
    each level computed from the previous one in memory, by the same
    :func:`_downsample` -- the values are those of the level-by-level path.

    Every level is chunked (or sharded) one block deep along the leading
    axes, as level 0 is, so that each slab writes whole chunks.
    """
    source_nbytes = int(np.prod(arr.chunksize)) * arr.dtype.itemsize
    lead = arr.ndim - 2
    if (
        source_nbytes < _SLAB_SOURCE_BYTES
        or lead < 1
        or any(st != 1 for st in strides[:lead])
        or downsample not in _DOWNSAMPLE_METHODS
    ):
        return None
    grp = _open_group(group_path)
    fmt = _group_zarr_format(group_path)
    shapes = _level_shapes(arr.shape, strides, n_levels)
    inners: list[tuple[int, ...]] = []
    blocks: list[tuple[int, ...]] = []
    for k, shape in enumerate(shapes):
        if k == 0:
            inner = tuple(base_chunks)
        elif shard or chunks:
            inner = tuple(chunks or _default_chunks(shape, axes))
        else:
            inner = _level_chunks(inners[-1], strides, axes, shape)
        # One block deep along the leading axes, as level 0.
        if k:
            inner = tuple(blocks[0][:lead]) + tuple(inner[lead:])
        inner = tuple(max(1, min(c, s)) for c, s in zip(inner, shape))
        sh = _shard_for(shard, inner, shape, arr.dtype, fmt)
        block = sh or inner
        if tuple(block[:lead]) != tuple((blocks or [block])[0][:lead]):
            return None  # an explicit shard deeper than level 0's blocks
        inners.append(inner)
        blocks.append(block)
    names = [base_name] + [str(k) for k in range(1, len(shapes))]
    targets = []
    for name, shape, inner, block in zip(names, shapes, inners, blocks):
        if name in grp:
            del grp[name]
        sharded = {} if block == inner else {"shards": block}
        targets.append(
            grp.create_array(
                name=name,
                shape=shape,
                chunks=inner,
                dtype=arr.dtype,
                **sharded,
                **zarr_compressor_kwargs(fmt),
            )
        )
    # A plane's whole pyramid: the plane, a quarter of it, ... (4/3). One
    # more copy when the next slab is read while this one is written -- if
    # slabs that small still fit; else one slab at a time.
    from .._chunks import _get_available_memory

    per_plane = source_nbytes * 4 // 3
    starts, step = _slab_starts(arr, blocks[0], per_plane, _SOURCE_COPIES + 1)
    slab_planes = int(np.prod(step[:lead]))
    read_ahead = len(starts) > 1 and (
        slab_planes * per_plane * (_SOURCE_COPIES + 1)
        <= 0.6 * _get_available_memory()
    )
    if not read_ahead:
        starts, step = _slab_starts(arr, blocks[0], per_plane)
    label = f"{Path(group_path).name}/{base_name}-{names[-1]}"
    logger.info(
        "writing %s in one pass, %d slab(s) of %s along the leading axes: "
        "source chunks are %.2f GB",
        label,
        len(starts),
        tuple(step),
        source_nbytes / 1024**3,
    )
    workers = max(1, cpu_allocation())
    started = _time.monotonic()

    def _region(start) -> tuple:
        return (
            tuple(
                slice(s, min(s + st, arr.shape[i]))
                for i, (s, st) in enumerate(zip(start, step[:lead]))
            )
            + (slice(None),) * 2
        )

    def _read(start) -> np.ndarray:
        return np.asarray(arr[_region(start)].compute())

    if read_ahead:
        logger.info(
            "%s: reading each slab while the one before is written", label
        )
    reader = ThreadPoolExecutor(max_workers=1) if read_ahead else None
    try:
        nxt = reader.submit(_read, starts[0]) if reader else None
        for n, start in enumerate(starts, 1):
            region = _region(start)
            if nxt is not None:
                data = nxt.result()
                nxt = (
                    reader.submit(_read, starts[n])
                    if reader and n < len(starts)
                    else None
                )
            else:
                data = _read(start)
            _write_slab(
                data,
                region,
                targets,
                blocks,
                strides,
                downsample,
                workers,
                lead,
            )
            del data
            if progress:
                log_progress(label, n, len(starts), started)
    finally:
        if reader is not None:
            reader.shutdown(wait=True, cancel_futures=True)
    datasets = []
    for i, (name, shape) in enumerate(zip(names, shapes)):
        scale = [base_scale[k] * (strides[k] ** i) for k in range(len(axes))]
        translation = (
            base_translation
            if i == 0
            else _level_translation(
                base_scale, base_translation, strides, i, downsample
            )
        )
        datasets.append(_dataset(name, scale, translation))
        if i:
            logger.info("pyramid level %d: shape=%s", i, shape)
    return datasets


def _write_pyramid(
    arr: da.Array,
    axes: str,
    group_path: str,
    *,
    n_levels: int,
    downscale: int,
    chunks: Union[tuple[int, ...], None],
    base_scale: list[float],
    base_name: str = "0",
    write_base: bool = True,
    shard: ShardSpec = False,
    progress: bool = True,
    downsample: str = "nearest",
    base_translation: "list[float] | None" = None,
) -> list[dict]:
    """Write pyramid levels into *group_path* and return NGFF datasets.

    Each deeper level is produced by reading the previous level **back from
    disk** and striding it, so the dask graph stays shallow and bounded — no
    whole-volume recomputation, no OOM. Level 0 is named *base_name*; when
    *write_base* is False it is assumed to already exist (used by
    :func:`add_pyramid`).

    Parameters
    ----------
    arr : da.Array
        Full-resolution array.
    axes : str
        One letter per axis.
    group_path : str
        Path of the zarr group to write into.
    n_levels : int
        Maximum number of levels including full resolution.
    downscale : int
        Per-level X/Y downsampling factor.
    chunks : tuple of int or None
        Chunk shape, or a bounded default.
    base_scale : list of float
        Level-0 physical scale per axis.
    base_name : str
        Component name of level 0.
    write_base : bool
        Write level 0, or assume it already exists.
    shard : bool or tuple of int
        Sharding request (see :func:`_shard_for`).
    progress : bool
        Show a per-level progress bar.
    downsample : str
        ``"mean"`` (images) or ``"nearest"`` (labels); see
        :func:`_downsample`.
    base_translation : list of float, optional
        Level-0 physical offset per axis (e.g. labels segmented from a
        coarser, averaged image level). Carried to every level.

    Returns
    -------
    list of dict
        One NGFF dataset entry per written level.
    """
    strides = _strides(axes, downscale)

    if write_base:
        # Cap by the source's own chunking: combining several source chunks
        # into one output chunk can mean holding gigabytes to write megabytes.
        base_chunks = chunks or _default_chunks(
            arr.shape, axes, source_chunks=arr.chunksize
        )
        _warn_if_many_chunks(arr.shape, base_chunks, shard)
        one_pass = _write_pyramid_one_pass(
            arr,
            axes,
            group_path,
            base_chunks=base_chunks,
            n_levels=n_levels,
            strides=strides,
            chunks=chunks,
            base_scale=base_scale,
            base_name=base_name,
            shard=shard,
            progress=progress,
            downsample=downsample,
            base_translation=base_translation,
        )
        if one_pass is not None:
            return one_pass
        _to_zarr_level(
            arr.rechunk(base_chunks),
            group_path,
            base_name,
            shard,
            progress,
            source_nbytes=int(np.prod(arr.chunksize)) * arr.dtype.itemsize,
        )
    datasets = [_dataset(base_name, base_scale, base_translation)]

    prev_name = base_name
    prev_shape = arr.shape
    for i in range(1, n_levels):
        next_shape = tuple(-(-s // st) for s, st in zip(prev_shape, strides))
        if min(next_shape) < 1:
            logger.info("stopping pyramid at level %d (next too small)", i)
            break
        grp = _open_group(group_path)
        sh = (
            _shard_for(
                shard,
                _level_chunks(
                    tuple(grp[prev_name].chunks), strides, axes, next_shape
                ),
                next_shape,
                grp[prev_name].dtype,
                _group_zarr_format(grp),
            )
            if shard and not chunks
            else None
        )
        if sh is not None:
            # Sharded: one task per output shard, written whole.
            src_arr = grp[prev_name]
            out_chunks = _level_chunks(
                tuple(src_arr.chunks), strides, axes, next_shape
            )
            if str(i) in grp:
                del grp[str(i)]
            dst_arr = grp.create_array(
                str(i),
                shape=next_shape,
                chunks=out_chunks,
                shards=sh,
                dtype=src_arr.dtype,
                **zarr_compressor_kwargs(_group_zarr_format(grp)),
            )
            _stream_sharded_level(
                src_arr,
                dst_arr,
                strides,
                label=f"{Path(group_path).name}/{i}",
                progress=progress,
                method=downsample,
            )
        elif shard or chunks:
            # An explicit chunking may not line up with the source chunks:
            # the dask path.
            src = da.from_zarr(group_path, component=prev_name)
            if downsample == "nearest":
                nxt = src[tuple(slice(None, None, st) for st in strides)]
            # mean and mode are both block reductions
            else:
                # Blocks must start on a stride boundary, so every chunk
                # (but the last) is a whole number of strides.
                src = src.rechunk(
                    tuple(
                        max(st, (c // st) * st)
                        for c, st in zip(src.chunksize, strides)
                    )
                )
                nxt = src.map_blocks(
                    _downsample,
                    strides,
                    downsample,
                    chunks=tuple(
                        tuple(-(-c // st) for c in dim)
                        for dim, st in zip(src.chunks, strides)
                    ),
                    dtype=src.dtype,
                )
            nxt = nxt.rechunk(chunks or _default_chunks(nxt.shape, axes))
            _to_zarr_level(nxt, group_path, str(i), shard, progress)
            next_shape = nxt.shape
        else:
            # One source chunk per task: decimation needs no halo.
            grp = _open_group(group_path)
            src_arr = grp[prev_name]
            out_chunks = _level_chunks(
                tuple(src_arr.chunks), strides, axes, next_shape
            )
            dst_arr = _create_level_array(
                grp, str(i), next_shape, out_chunks, src_arr.dtype
            )
            if progress:
                logger.info(
                    "writing %s/%s (streaming %d chunks)",
                    Path(group_path).name,
                    i,
                    int(
                        np.prod(
                            [-(-s // c) for s, c in zip(next_shape, out_chunks)]
                        )
                    ),
                )
            _stream_strided_level(
                src_arr,
                dst_arr,
                strides,
                n_workers=cpu_allocation(),
                label=f"{Path(group_path).name}/{i}",
                progress=progress,
                method=downsample,
            )
        scale = [base_scale[k] * (strides[k] ** i) for k in range(len(axes))]
        translation = _level_translation(
            base_scale, base_translation, strides, i, downsample
        )
        datasets.append(_dataset(str(i), scale, translation))
        logger.info("pyramid level %d: shape=%s", i, next_shape)
        prev_name = str(i)
        prev_shape = next_shape
    return datasets


def _write_multiscales(
    group_path: str,
    axes: str,
    datasets: list[dict],
    name: str,
    *,
    calibrated: bool,
) -> None:
    """Write NGFF ``multiscales`` metadata onto *group_path*.

    Parameters
    ----------
    group_path : str
        Path of the zarr group to annotate.
    axes : str
        One letter per axis.
    datasets : list of dict
        Per-level dataset entries (see :func:`_dataset`).
    name : str
        Multiscales name.
    calibrated : bool
        Whether spatial axes carry a micrometer unit.

    Returns
    -------
    None
    """
    entry = {
        "name": name,
        "axes": _axes_meta(axes, calibrated),
        "datasets": datasets,
    }
    version = _ngff_version_for(group_path)
    if _NGFF_ZARR_FORMAT[version] == 2:
        entry["version"] = version  # 0.5 carries it on the ome group instead
    group = _open_group(group_path)
    write_ngff_attrs(group, multiscales=[entry])
    if group.metadata.zarr_format == 3:
        _set_dimension_names(group, [d["path"] for d in datasets], axes)


def _set_dimension_names(group, paths, axes: str) -> None:
    """Name every level array's dimensions after the multiscales axes.

    OME-Zarr 0.5 requires each array's ``dimension_names`` to match the axes
    (validators reject a store without them). Set here, where the axes are
    written, rather than at each array's creation: a label image's level 0
    is created by the segment jobs or the merge, which know nothing of axes.
    """
    import dataclasses

    from zarr.core.sync import sync

    names = tuple(axes)
    for path in paths:
        arr = group[path]
        if tuple(arr.metadata.dimension_names or ()) == names:
            continue
        if len(arr.shape) != len(names):
            raise ValueError(
                f"{path}: {len(arr.shape)}-D array under axes {axes!r}"
            )
        inner = arr._async_array
        sync(
            inner._save_metadata(
                dataclasses.replace(inner.metadata, dimension_names=names)
            )
        )


def read_pixel_size(store: Union[str, Path], level: int = 0) -> PixelSize:
    """Physical voxel size recorded in an OME-ZARR's metadata for *level*.

    The calibration the conversion carried over from the source file, as
    ``{"z": .., "y": .., "x": ..}`` in micrometers. Axes left at scale 1.0
    (uncalibrated) are omitted, so an empty dict means the store carries no
    usable calibration.

    Use this instead of retyping voxel sizes into a config: a deconvolution
    told the wrong voxel size produces a plausible-looking but wrong result.

    Parameters
    ----------
    store : str or Path
        Path of the OME-ZARR group.
    level : int, optional
        Pyramid level (default 0, full resolution). Anything segmented or
        measured at a coarser level must use that level's voxel size.

    Returns
    -------
    dict
        ``{axis: size}`` for calibrated spatial axes.

    Examples
    --------
    >>> read_pixel_size("scan.zarr")  # doctest: +SKIP
    {'z': 0.2, 'y': 0.1, 'x': 0.1}
    >>> read_pixel_size("scan.zarr", level=1)  # doctest: +SKIP
    {'z': 0.2, 'y': 0.2, 'x': 0.2}
    """
    return _read_zarr_calibration(store, "", level=level)


def read_translation(store: Union[str, Path], level: int = 0) -> PixelSize:
    """Physical offset of *level*'s first voxel centre, per spatial axis.

    Zero (and so empty) for level 0 and for decimated levels; non-zero for a
    block-averaged level, whose first voxel centre sits between the base
    voxels it averages.

    Parameters
    ----------
    store : str or Path
        Path of the OME-ZARR group.
    level : int, optional
        Pyramid level (default 0).

    Returns
    -------
    dict
        ``{axis: offset}`` for spatial axes with a non-zero offset.
    """
    ms = _multiscale_meta(store)
    if ms is None:
        return {}
    try:
        ax = [a["name"] for a in ms["axes"]]
        transforms = ms["datasets"][level]["coordinateTransformations"]
    except (KeyError, IndexError, TypeError):
        return {}
    for t in transforms:
        if t.get("type") == "translation":
            return {
                a: float(v)
                for a, v in zip(ax, t["translation"])
                if a in _SPATIAL_AXES and float(v) != 0.0
            }
    return {}


def _multiscale_meta(store: Union[str, Path]) -> "dict | None":
    try:
        # open_group_any, not zarr.open_group: the store may be a .zip
        # bundle, possibly with a group path after it.
        return read_ngff_attr(open_group_any(store).attrs, "multiscales")[0]
    except (KeyError, IndexError, TypeError):
        return None


def _read_zarr_calibration(
    store: Union[str, Path], axes: str, level: int = 0
) -> PixelSize:
    """Read *level*'s spatial scale from an existing OME-ZARR, if any.

    Parameters
    ----------
    store : str or Path
        Path of the OME-ZARR group.
    axes : str
        One letter per axis (unused for parsing, kept for symmetry).
    level : int, optional
        Pyramid level whose dataset scale to read (default 0).

    Returns
    -------
    dict
        ``{axis: size}`` for spatial axes with a non-unit scale (empty if
        the store has no multiscales metadata).
    """
    ms = _multiscale_meta(store)
    if ms is None:
        return {}
    try:
        ax = [a["name"] for a in ms["axes"]]
        scale = ms["datasets"][level]["coordinateTransformations"][0]["scale"]
    except (KeyError, IndexError, TypeError):
        return {}
    return {
        a: float(s)
        for a, s in zip(ax, scale)
        if a in _SPATIAL_AXES and float(s) != 1.0
    }


def _open_bioio(path: str, scene: int) -> tuple[da.Array, str, PixelSize]:
    """Open *path* with bioio, lazily.

    Singleton non-spatial axes (T/C of size 1) are dropped.

    Parameters
    ----------
    path : str
        Image file path.
    scene : int
        Scene index for multi-scene files.

    Returns
    -------
    tuple
        ``(array, axes, pixel_size)`` — a lazy dask array, its axes string
        and a ``{axis: micrometers}`` calibration dict.
    """
    try:
        from bioio import BioImage
    except ImportError as exc:
        ext = Path(path).suffix or "this file type"
        raise ImportError(
            f"reading {ext} requires bioio. Install it with:\n"
            "    pip install 'patchworks[bioio]'\n"
            "plus the matching reader plugin, e.g. bioio-ome-tiff, "
            "bioio-czi, bioio-lif, bioio-nd2."
        ) from exc

    img = BioImage(path)
    img.set_scene(img.scenes[scene])
    order = img.dims.order  # e.g. "TCZYX"
    arr = img.get_image_dask_data(order)  # lazy dask array, no I/O yet

    # Drop leading singleton non-spatial axes for a tidy result.
    keep = [
        i
        for i, name in enumerate(order)
        if name.lower() in _SPATIAL_AXES or arr.shape[i] > 1
    ]
    index = tuple(slice(None) if i in keep else 0 for i in range(arr.ndim))
    arr = arr[index]
    axes = "".join(order[i].lower() for i in keep)

    pixel_size: PixelSize = {}
    pps = getattr(img, "physical_pixel_sizes", None)
    for axis in "zyx":
        val = getattr(pps, axis.upper(), None) if pps is not None else None
        if val:
            pixel_size[axis] = float(val)
    logger.info(
        "bioio opened %s as %s %s cal=%s", path, axes, arr.shape, pixel_size
    )
    return arr, axes, pixel_size


_UNIT_TO_UM = {
    "micron": 1.0,
    "um": 1.0,
    "µm": 1.0,
    "mm": 1000.0,
    "cm": 10000.0,
    "centimeter": 10000.0,
    "inch": 25400.0,
}
# TIFF ResolutionUnit tag values (RESUNIT): 2 == INCH, 3 == CENTIMETER.
_RESUNIT_TO_UM = {2: 25400.0, 3: 10000.0}


def _ome_xml_pixel_size(xml: str) -> PixelSize:
    """Voxel size from an OME-XML ``Pixels`` element.

    Stitching and acquisition software commonly writes OME-XML into
    ``ImageDescription`` and nothing else -- no ImageJ block, and resolution
    tags left at their defaults. Readers that only look at the TIFF tags then
    report an uncalibrated image even though the size is right there.

    Parameters
    ----------
    xml : str
        The OME-XML document.

    Returns
    -------
    dict
        ``{axis: micrometers}`` for whichever axes carry a size.
    """
    from xml.etree import ElementTree

    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError:
        return {}
    # The OME namespace is versioned, so match on the local tag name.
    pixels = next(
        (el for el in root.iter() if el.tag.rsplit("}", 1)[-1] == "Pixels"),
        None,
    )
    if pixels is None:
        return {}
    out: PixelSize = {}
    for axis in "zyx":
        value = pixels.get(f"PhysicalSize{axis.upper()}")
        if not value:
            continue
        unit = pixels.get(f"PhysicalSize{axis.upper()}Unit", "µm")
        factor = _UNIT_TO_UM.get(str(unit).lower())
        if factor is None:
            logger.warning(
                "OME-XML PhysicalSize%s has unit %r, which is not recognised; "
                "ignoring it.",
                axis.upper(),
                unit,
            )
            continue
        try:
            out[axis] = float(value) * factor
        except ValueError:
            continue
    return out


def _tiff_pixel_size(path: str) -> PixelSize:
    """Read physical voxel size from a TIFF file's own metadata.

    Checks every place the size is commonly written, most explicit first:

    1. **OME-XML** ``PhysicalSizeX/Y/Z`` (+ their units) in
       ``ImageDescription`` — what stitched and OME-TIFF output usually
       carries, and the only one of the three that can give ``z`` for a
       sequence of single planes.
    2. **ImageJ metadata** — ``spacing`` (+ ``unit``) for ``z``.
    3. **Resolution tags** — ``XResolution``/``YResolution`` in pixels per
       unit, the unit from ``ResolutionUnit`` (plain TIFFs) or from ImageJ's
       ``unit`` string (ImageJ writes ``ResolutionUnit=NONE``).

    Earlier sources win per axis, so a file carrying both keeps the explicit
    OME value. Unrecognized units are ignored rather than guessed at.

    Parameters
    ----------
    path : str
        Path of one TIFF file in the sequence.

    Returns
    -------
    dict
        ``{axis: micrometers}`` for whichever axes could be determined
        (empty if none).
    """
    import tifffile

    pixel_size: PixelSize = {}
    with tifffile.TiffFile(path) as tif:
        description = tif.pages[0].tags.get("ImageDescription")
        xml = tif.ome_metadata or (
            description.value
            if description and "<OME" in str(description.value)
            else None
        )
        if xml:
            pixel_size.update(_ome_xml_pixel_size(str(xml)))

        ij = tif.imagej_metadata or {}
        spacing = ij.get("spacing")
        if spacing and "z" not in pixel_size:
            factor = _UNIT_TO_UM.get(str(ij.get("unit", "micron")).lower(), 1.0)
            pixel_size["z"] = float(spacing) * factor

        page = tif.pages[0]
        unit_tag = page.tags.get("ResolutionUnit")
        um_per_unit = _RESUNIT_TO_UM.get(
            unit_tag.value if unit_tag else None
        ) or _UNIT_TO_UM.get(str(ij.get("unit", "")).lower())
        if um_per_unit is not None:
            for axis, tag_name in (("y", "YResolution"), ("x", "XResolution")):
                if axis in pixel_size:
                    continue
                tag = page.tags.get(tag_name)
                if tag and tag.value[0]:
                    num, den = tag.value
                    pixel_size[axis] = um_per_unit / (num / den)
    return pixel_size


def _open_tiff_sequence(
    pattern: str, sequence_pattern: str
) -> tuple[da.Array, str, PixelSize]:
    """Open a folder of single-plane TIFFs as one lazy dask array.

    Each file becomes exactly one chunk, decoded on access — no data is
    duplicated or eagerly loaded. See
    :func:`tifffile.TiffSequence`/``ZarrFileSequenceStore`` (the same
    mechanism used by Cellpose's own distributed pipeline,
    ``cellpose.contrib.distributed_segmentation.wrap_folder_of_tiffs``).

    Parameters
    ----------
    pattern : str
        Glob pattern matching all the files, e.g. ``"folder/*.tif"``.
    sequence_pattern : str
        Regular expression parsing each file name into axis labels and
        indices, e.g. ``r"_T(?P<T>\\d+)_Z(?P<Z>\\d+)_C(?P<C>\\d+)_V\\d+"``.
        Named groups become axis labels directly — the order they appear in
        the *pattern* (not the filename) doesn't matter, the result is
        always reordered to patchworks' ``tczyx`` convention and any
        singleton non-spatial axis (e.g. a constant ``T0``) is dropped, same
        as the bioio/Imaris readers, so a real channel axis always ends up
        first (required by :func:`patchworks.load_ome_zarr`).

    Returns
    -------
    tuple
        ``(array, axes, pixel_size)`` — a lazy dask array, its axes string
        (parsed axes, canonicalised, + trailing ``"yx"``) and a
        ``{axis: micrometers}`` calibration dict read from the first file.
    """
    try:
        import tifffile
    except ImportError as exc:
        raise ImportError(
            "reading a folder of TIFFs requires tifffile. Install it with:\n"
            "    pip install tifffile\n"
            "(already pulled in by patchworks[bioio] via bioio-tifffile)."
        ) from exc

    ts = tifffile.TiffSequence(pattern, pattern=sequence_pattern)

    # Each file should be a single 2D (y, x) plane, but tifffile.imread can
    # report extra singleton dims — leading, trailing, or both (e.g. a stray
    # page or samples-per-pixel axis) — even for a plain single-page TIFF.
    # Drop whichever ones are singleton so one file maps to exactly (y, x).
    # From the header, not by decoding the file: a stitched plane can be
    # gigabytes, and tifffile's store would decode it once more for the same
    # two facts unless it is told them.
    with tifffile.TiffFile(ts[0]) as tif:
        chunkshape = tuple(tif.series[0].shape)
        chunkdtype = tif.series[0].dtype
    keep_chunk = [i for i, s in enumerate(chunkshape) if s != 1]
    if len(keep_chunk) != 2:
        raise ValueError(
            f"expected single 2D (y, x) planes per file, got shape "
            f"{chunkshape} for {ts[0]!r}"
        )

    try:
        store = ts.aszarr(chunkshape=chunkshape, chunkdtype=chunkdtype)
    except TypeError:  # a tifffile without these arguments
        store = ts.aszarr()
    arr = da.from_zarr(zarr.open(store=store))
    n_seq = len(ts.axes)
    index = tuple(
        slice(None) if i in keep_chunk else 0 for i in range(len(chunkshape))
    )
    arr = arr[(slice(None),) * n_seq + index]
    full_axes = ts.axes.lower() + "yx"

    # Canonicalise to patchworks' tczyx axis order regardless of the order
    # sequence_pattern's groups happen to appear in (unrecognised axes sort
    # last, before y/x).
    order = sorted(
        range(len(full_axes)),
        key=lambda i: (
            _DEFAULT_ORDER.index(full_axes[i])
            if full_axes[i] in _DEFAULT_ORDER
            else len(_DEFAULT_ORDER)
        ),
    )
    arr = arr.transpose(order)
    ordered_axes = "".join(full_axes[i] for i in order)

    # Drop singleton non-spatial axes (e.g. a constant T), matching
    # _open_bioio/_open_imaris, so a real channel axis lands at position 0.
    keep = [
        i
        for i, a in enumerate(ordered_axes)
        if a in _SPATIAL_AXES or arr.shape[i] > 1
    ]
    index = tuple(slice(None) if i in keep else 0 for i in range(arr.ndim))
    arr = arr[index]
    axes = "".join(ordered_axes[i] for i in keep)

    pixel_size = _tiff_pixel_size(ts[0])
    logger.info(
        "tifffile sequence opened %s files as %s %s cal=%s",
        len(ts),
        axes,
        arr.shape,
        pixel_size,
    )
    missing = [a for a in axes if a in _SPATIAL_AXES and a not in pixel_size]
    if missing:
        logger.warning(
            "no voxel size for %s in %s -- checked OME-XML PhysicalSize*, "
            "ImageJ spacing/unit, and the resolution tags. The store will be "
            "uncalibrated on %s, which also leaves anything deriving physical "
            "units from it (e.g. deconvolution voxel sizes) with nothing to "
            "read. Pass pixel_size= to supply it.",
            "/".join(missing),
            Path(ts[0]).name,
            "/".join(missing),
        )
    return arr, axes, pixel_size


def _open_imaris(path: str, level: int = 0) -> tuple[da.Array, str, PixelSize]:
    """Open one Imaris ``.ims`` resolution level lazily.

    Reads the underlying HDF5 datasets directly (own handle, crops the Imaris
    chunk padding) and stacks the per-(timepoint, channel) 3-D arrays.

    Parameters
    ----------
    path : str
        Imaris ``.ims`` file path.
    level : int
        Resolution level to read (0 = full resolution).

    Returns
    -------
    tuple
        ``(array, axes, pixel_size)`` — a lazy dask array, its axes string
        and a ``{axis: micrometers}`` calibration dict.
    """
    try:
        from imaris_ims_file_reader.ims import ims
    except ImportError as exc:
        raise ImportError(
            "reading Imaris .ims files requires imaris-ims-file-reader. "
            "Install it with:\n    pip install 'patchworks[imaris]'"
        ) from exc

    # Read straight from the underlying HDF5 datasets. The reader's own
    # __getitem__ squeezes singletons and pads to chunk boundaries, which both
    # break da.from_array; the raw h5py datasets slice exactly and keep their
    # native chunking. Imaris stores one 3-D (Z, Y, X) Data array per
    # (timepoint, channel), padded to a chunk multiple — crop to the true
    # extent the reader reports.
    import h5py

    reader = ims(path, ResolutionLevelLock=level)
    n_t = int(getattr(reader, "TimePoints", 1) or 1)
    n_c = int(getattr(reader, "Channels", 1) or 1)
    z, y, x = (int(s) for s in reader.shape[-3:])

    # Open our own h5py handle (the reader closes its own on GC). The Dataset
    # objects keep this File alive for the lifetime of the dask graph, and
    # da.from_array's default read lock makes the (thread-unsafe) h5py reads
    # safe under the threaded scheduler.
    hf = h5py.File(path, "r")
    t_stack = []
    for t in range(n_t):
        c_stack = []
        for c in range(n_c):
            ds = hf[
                f"DataSet/ResolutionLevel {level}/TimePoint {t}/"
                f"Channel {c}/Data"
            ]
            c_stack.append(da.from_array(ds, chunks=ds.chunks)[:z, :y, :x])
        t_stack.append(da.stack(c_stack, axis=0))  # (c, z, y, x)
    arr = da.stack(t_stack, axis=0)  # (t, c, z, y, x)

    # Drop singleton non-spatial axes for a tidy result.
    order = "tczyx"
    keep = [
        i
        for i, name in enumerate(order)
        if name in _SPATIAL_AXES or arr.shape[i] > 1
    ]
    index = tuple(slice(None) if i in keep else 0 for i in range(arr.ndim))
    arr = arr[index]
    axes = "".join(order[i] for i in keep)

    pixel_size: PixelSize = {}
    res = getattr(reader, "resolution", None)  # (z, y, x) in micrometers
    if res is not None and len(res) >= 3:
        pixel_size = {
            "z": float(res[0]),
            "y": float(res[1]),
            "x": float(res[2]),
        }
    logger.info(
        "imaris opened %s as %s %s cal=%s", path, axes, arr.shape, pixel_size
    )
    return arr, axes, pixel_size


def _write_imaris_pyramid(
    path: str,
    out: str,
    *,
    chunks: Union[tuple[int, ...], None],
    overwrite: bool,
    shard: ShardSpec = False,
    progress: bool = True,
) -> str:
    """Copy an Imaris file's own resolution levels into an OME-ZARR.

    Each Imaris ``ResolutionLevel`` is written as a pyramid level with its own
    physical scale, so no downsampling is recomputed. Lazy (h5py-backed) reads
    stream straight to disk.

    Parameters
    ----------
    path : str
        Imaris ``.ims`` file path.
    out : str
        Destination ``.zarr`` store path.
    chunks : tuple of int or None
        Chunk shape, or a bounded default.
    overwrite : bool
        Overwrite an existing store.
    shard : bool or tuple of int
        Sharding request (see :func:`_shard_for`).
    progress : bool
        Show a per-level progress bar.

    Returns
    -------
    str
        The path to the written store.
    """
    from imaris_ims_file_reader.ims import ims

    base = ims(path, ResolutionLevelLock=0)
    n_levels = int(getattr(base, "ResolutionLevels", 1) or 1)

    _open_group(out, mode="w" if overwrite else "w-")
    datasets: list[dict] = []
    axes = ""
    calibrated = False
    for level in range(n_levels):
        arr, axes, ps = _open_imaris(path, level=level)
        scale = _base_scale(axes, ps)
        calibrated = calibrated or bool(ps)
        _to_zarr_level(
            arr.rechunk(chunks or _default_chunks(arr.shape, axes)),
            out,
            str(level),
            shard,
            progress,
        )
        datasets.append(_dataset(str(level), scale))
        logger.info("imaris level %d copied: shape=%s", level, arr.shape)
    _write_multiscales(
        out, axes, datasets, Path(out).stem, calibrated=calibrated
    )
    _name_channels_from(path, out)
    return out


def _to_dask(
    source: Union[da.Array, np.ndarray, str, Path],
    axes: Union[str, None],
    scene: int,
    sequence_pattern: Union[str, None] = None,
) -> tuple[da.Array, str, PixelSize]:
    """Resolve *source* into a lazy ``(array, axes, pixel_size)`` triple.

    Dispatches by type: dask/NumPy arrays pass through; ``.zarr`` paths use the
    OME-ZARR loader; ``.ims`` paths use the Imaris reader; a *sequence_pattern*
    treats *source* as a glob over a folder of single-plane TIFFs; anything
    else uses bioio.

    Parameters
    ----------
    source : da.Array, np.ndarray, str or Path
        Array or path to resolve.
    axes : str or None
        Explicit axes, or ``None`` to infer them.
    scene : int
        Scene index for bioio inputs.
    sequence_pattern : str or None
        When given, *source* is a glob pattern over a folder of single-plane
        TIFFs and this is the filename-parsing regex; see
        :func:`_open_tiff_sequence`.

    Returns
    -------
    tuple
        ``(array, axes, pixel_size)``.
    """
    if isinstance(source, da.Array):
        return source, axes or _default_axes(source.ndim), {}
    if isinstance(source, np.ndarray):
        return da.asarray(source), axes or _default_axes(source.ndim), {}

    path = str(source)
    if sequence_pattern is not None:
        arr, detected, ps = _open_tiff_sequence(path, sequence_pattern)
        return arr, axes or detected, ps
    if any(ch in path for ch in "*?[") or (
        not Path(path).exists() and glob.glob(path)
    ):
        # Without sequence_pattern this would be handed to bioio, which reads
        # single files and reports only that it cannot handle the "format" --
        # pointing at the glob rather than the missing setting.
        raise ValueError(
            f"{path!r} looks like a glob over several files, but "
            "sequence_pattern is not set, so there is no way to tell which "
            "part of each filename is Z, C or T.\n"
            "Set it to a regex with named groups, e.g.\n"
            "    sequence_pattern: '_Z(?P<Z>\\d+)_C(?P<C>\\d+)_V\\d+'\n"
            "for files like sample_Z001_C0_V0.tif. Pass a single file "
            "instead if you did not mean a sequence."
        )
    if path.endswith(".zarr"):
        arr = load_ome_zarr(source, channel=None)
        ax = axes or _default_axes(arr.ndim)
        return arr, ax, _read_zarr_calibration(source, ax)
    if path.lower().endswith(".ims"):
        arr, detected, ps = _open_imaris(path)
        return arr, axes or detected, ps

    arr, detected, ps = _open_bioio(path, scene)
    return arr, axes or detected, ps


def to_ome_zarr(
    source: Union[da.Array, np.ndarray, str, Path],
    out_path: Union[str, Path],
    *,
    axes: Union[str, None] = None,
    pixel_size: Union[PixelSize, tuple, None] = None,
    scene: int = 0,
    sequence_pattern: Union[str, None] = None,
    n_levels: int = 5,
    downscale: int = 2,
    chunks: Union[tuple[int, ...], None] = None,
    shard: ShardSpec = False,
    reuse_pyramid: bool = False,
    progress: bool = True,
    overwrite: bool = False,
    ngff_version: Union[str, None] = "auto",
    compression: Union[str, None] = None,
    downsample: str = "mean",
) -> str:
    """Write *source* as a pyramidal, calibrated OME-ZARR store.

    *source* may be a dask/NumPy array, a ``.zarr`` store, an Imaris ``.ims``
    file, any image format readable by bioio (CZI, LIF, ND2, OME-TIFF, …), or
    (with *sequence_pattern* set) a glob over a folder of single-plane TIFFs.
    File inputs are read lazily; the pyramid is built level-by-level from disk
    with bounded chunks, so the full volume never needs to fit in RAM. Only
    ``x``/``y`` are downsampled; ``z`` (and channel/time) stay full-resolution.

    Parameters
    ----------
    source : da.Array, np.ndarray, str or Path
        Array or path to convert.
    out_path : str or Path
        Destination ``.zarr`` store (a directory).
    axes : str, optional
        One character per array dimension, e.g. ``"zyx"`` or ``"cyx"``.
        ``None`` → inferred from the file metadata or the array dimensions.
    pixel_size : dict, tuple or None, optional
        Physical voxel size in micrometers, as ``{"z": .., "y": .., "x": ..}``
        or a tuple aligned to the spatial axes. ``None`` → read from the input
        (bioio/Imaris/OME-ZARR); falls back to 1.0 (uncalibrated) for bare
        arrays.
    scene : int, optional
        Scene index for multi-scene bioio files.
    sequence_pattern : str, optional
        When given, *source* is treated as a glob pattern over a folder of
        single-plane TIFFs (e.g. ``"folder/*.tif"``) instead of a single
        file, and this is the regex parsing each file name into axis labels
        and indices via named groups, e.g.
        ``r"_T(?P<T>\\d+)_Z(?P<Z>\\d+)_C(?P<C>\\d+)_V\\d+"``. Each file
        becomes exactly one chunk, read lazily on access (no data
        duplicated) — see :func:`tifffile.TiffSequence`. The result is
        always reordered to patchworks' ``czyx`` convention regardless of
        the order named groups appear in the pattern, and any singleton
        non-spatial axis (e.g. a constant ``T0``) is dropped, so a real
        channel axis always ends up first.
    n_levels : int, optional
        Maximum number of pyramid levels including full resolution.
    downscale : int, optional
        Per-level X/Y downsampling factor (default 2).
    chunks : tuple of int, optional
        Chunk shape for the written levels. ``None`` → a bounded default.
    shard : bool or tuple of int, optional
        Pack many chunks into one shard file (zarr v3), cutting the file count
        ~100× on huge arrays. ``False`` (default) → unsharded, maximum reader
        compatibility. ``True`` → auto-pick a ~512 MB shard. A tuple sets an
        explicit shard shape (clamped to a chunk multiple). Sharded writes hold
        ~one shard per worker in RAM. Requires zarr v3 (ignored otherwise).
    progress : bool, optional
        Show a per-level dask progress bar (default ``True``). Set ``False`` to
        silence it.
    reuse_pyramid : bool, optional
        *Imaris ``.ims`` only.* Copy the file's **own** resolution levels
        instead of rebuilding the pyramid (faster, no recompute), keeping each
        level's native scale. Ignored for other inputs; falls back to a
        rebuild if the Imaris levels can't be read. Default ``False`` (rebuild,
        for a consistent XY-only NGFF pyramid).
    overwrite : bool, optional
        Overwrite an existing store at *out_path*.
    ngff_version : str or None, optional
        NGFF version (and therefore zarr format) to write: ``"auto"``
        (default) writes 0.5 (zarr v3), while ``"0.4"`` pins the older, zarr-v2 layout for tools that
        cannot read 0.5 yet. 0.4 has no sharding codec, so ``shard`` is
        ignored there. See :func:`ngff_version`.
    downsample : {"mean", "mode", "nearest"}, optional
        How the pyramid's X/Y levels are made from the one above: ``"mean"``
        (default) block-averages, which is what an image wants;
        ``"nearest"`` keeps every *downscale*-th pixel (what this wrote
        before); ``"mode"`` takes each block's most frequent non-zero value,
        which is what a label image wants.

    compression : str, optional
        Codec for the arrays written (``"zstd"``, ``"zstd:3"``, ``"blosc"``,
        ``"blosc:lz4"``, ``"none"``, ...; see
        :func:`patchworks.compression`). ``None`` (default) uses the active
        setting, zstd level 1 unless changed.
    Returns
    -------
    str
        The path to the written store (``str(out_path)``).

    Examples
    --------
    >>> from patchworks.plugins.ome_zarr import to_ome_zarr  # doctest: +SKIP
    >>> to_ome_zarr("scan.ims", "scan.zarr", n_levels=4)  # doctest: +SKIP
    'scan.zarr'
    >>> to_ome_zarr(  # doctest: +SKIP
    ...     "ZT18_Male4_Left/*.tif",
    ...     "ZT18_Male4_Left.zarr",
    ...     sequence_pattern=r"_T(?P<T>\\d+)_Z(?P<Z>\\d+)_C(?P<C>\\d+)_V\\d+",
    ...     shard=True,
    ... )  # doctest: +SKIP
    'ZT18_Male4_Left.zarr'
    """
    with _writing_ngff(ngff_version), _compression_scope(compression):
        if downscale < 2:
            raise ValueError("downscale must be >= 2")
        if n_levels < 1:
            raise ValueError("n_levels must be >= 1")

        # Reuse an Imaris file's own resolution pyramid instead of rebuilding it.
        if (
            reuse_pyramid
            and isinstance(source, (str, Path))
            and str(source).lower().endswith(".ims")
        ):
            try:
                return _write_imaris_pyramid(
                    str(source),
                    str(out_path),
                    chunks=chunks,
                    overwrite=overwrite,
                    shard=shard,
                    progress=progress,
                )
            except Exception as exc:
                logger.warning(
                    "reuse_pyramid failed (%s); rebuilding the pyramid instead.",
                    exc,
                )

        _check_downsample(downsample)
        arr, axes, detected = _to_dask(source, axes, scene, sequence_pattern)
        if len(axes) != arr.ndim:
            raise ValueError(
                f"axes {axes!r} has {len(axes)} entries but array is {arr.ndim}-D"
            )

        ps = _normalize_pixel_size(pixel_size, axes) if pixel_size else detected
        base_scale = _base_scale(axes, ps)

        out = str(out_path)
        _open_group(out, mode="w" if overwrite else "w-")
        with _bounded_scheduler(arr):
            datasets = _write_pyramid(
                arr,
                axes,
                out,
                n_levels=n_levels,
                downscale=downscale,
                chunks=chunks,
                base_scale=base_scale,
                shard=shard,
                progress=progress,
                downsample=downsample,
            )
        _write_multiscales(
            out, axes, datasets, Path(out).stem, calibrated=bool(ps)
        )
        _name_channels_from(source, out, scene, sequence_pattern)
        return out


def add_pyramid(
    group_path: Union[str, Path],
    *,
    base: str = "0",
    axes: Union[str, None] = None,
    pixel_size: Union[PixelSize, tuple, None] = None,
    n_levels: int = 5,
    downscale: int = 2,
    chunks: Union[tuple[int, ...], None] = None,
    shard: ShardSpec = False,
    progress: bool = True,
    ngff_version: Union[str, None] = "auto",
    compression: Union[str, None] = None,
    downsample: Union[str, None] = None,
    translation: Union[PixelSize, None] = None,
) -> str:
    """Add downsampled pyramid levels to an existing single-resolution zarr.

    Reads the full-resolution array already at ``group_path/base``, writes the
    missing levels next to it (lazily, from disk), and (re)writes the NGFF
    ``multiscales`` metadata. Existing calibration is preserved; pass
    *pixel_size* to set it.

    Parameters
    ----------
    group_path : str or Path
        Zarr group containing the full-resolution array at *base*.
    base : str, optional
        Component name of the existing full-resolution level (default
        ``"0"``). Auto-detected from existing ``multiscales`` metadata if
        present, overriding this.
    axes : str, optional
        One letter per axis, e.g. ``"zyx"``. ``None`` → inferred from
        existing metadata, or from the array's dimensionality.
    pixel_size : dict, tuple or None, optional
        Physical voxel size in micrometers. ``None`` → read from the store's
        existing calibration, if any.
    n_levels : int, optional
        Maximum number of levels including the existing full-resolution one
        (default 5).
    downscale : int, optional
        Per-level X/Y downsampling factor (default 2).
    chunks : tuple of int, optional
        Chunk shape for the written levels. ``None`` → a bounded default.
    shard : bool or tuple of int, optional
        Sharding request (see :func:`to_ome_zarr`'s *shard*).
    progress : bool, optional
        Show a per-level dask progress bar (default ``True``).
    ngff_version : str or None, optional
        NGFF version (and therefore zarr format) to write: ``"auto"``
        (default) writes 0.5 (zarr v3), while ``"0.4"`` pins the older, zarr-v2 layout for tools that
        cannot read 0.5 yet. 0.4 has no sharding codec, so ``shard`` is
        ignored there. See :func:`ngff_version`.
    downsample : {"mean", "mode", "nearest"} or None, optional
        How each level is made from the one above (see :func:`to_ome_zarr`).
        ``None`` (default) picks ``"mode"`` for an NGFF label image (an
        ``image-label`` group, or one under ``labels/``) and ``"mean"``
        otherwise.
    translation : dict, optional
        Physical offset ``{axis: offset}`` of the base level's first voxel.
        ``None`` keeps whatever the store already records (zero if nothing).

    compression : str, optional
        Codec for the arrays written (``"zstd"``, ``"zstd:3"``, ``"blosc"``,
        ``"blosc:lz4"``, ``"none"``, ...; see
        :func:`patchworks.compression`). ``None`` (default) uses the active
        setting, zstd level 1 unless changed.
    Returns
    -------
    str
        The path to the updated group.

    Examples
    --------
    >>> add_pyramid("scan.zarr", n_levels=4)  # doctest: +SKIP
    'scan.zarr'
    """
    with _writing_ngff(ngff_version), _compression_scope(compression):
        if downscale < 2:
            raise ValueError("downscale must be >= 2")
        if n_levels < 1:
            raise ValueError("n_levels must be >= 1")

        gp = str(group_path)
        root = zarr.open_group(gp, mode="r")
        if downsample is None:
            is_label = (
                read_ngff_attr(root.attrs, "image-label") is not None
                or "labels" in Path(gp).parts[:-1]
            )
            downsample = "mode" if is_label else "mean"
        _check_downsample(downsample)
        multiscales = read_ngff_attr(root.attrs, "multiscales")
        if multiscales:
            base = multiscales[0]["datasets"][0]["path"]
            if axes is None:
                axes = "".join(a["name"] for a in multiscales[0]["axes"])

        base_arr = da.from_zarr(gp, component=base)
        if axes is None:
            axes = _default_axes(base_arr.ndim)
        if len(axes) != base_arr.ndim:
            raise ValueError(
                f"axes {axes!r} has {len(axes)} entries but array is "
                f"{base_arr.ndim}-D"
            )

        if pixel_size:
            ps = _normalize_pixel_size(pixel_size, axes)
        else:
            ps = _read_zarr_calibration(gp, axes)
        base_scale = _base_scale(axes, ps)
        if translation is None:
            translation = read_translation(gp)
        base_translation = [float(translation.get(a, 0.0)) for a in axes]

        datasets = _write_pyramid(
            base_arr,
            axes,
            gp,
            n_levels=n_levels,
            downscale=downscale,
            chunks=chunks,
            base_scale=base_scale,
            base_name=base,
            write_base=False,
            shard=shard,
            progress=progress,
            downsample=downsample,
            base_translation=base_translation,
        )
        _write_multiscales(
            gp, axes, datasets, Path(gp).stem, calibrated=bool(ps)
        )
        return gp


def _label_level_count(
    store: str, name: str, requested: Union[int, None]
) -> int:
    """Pyramid levels for a label image of *store*: the image's own count.

    "the JSON array associated with the datasets key MUST have the same
    number of entries (scale levels) as the original unlabeled image" (the
    OME-Zarr spec, image-label section). *requested* only applies when the
    image has no multiscales to follow.
    """
    try:
        multiscales = read_ngff_attr(
            zarr.open_group(store, mode="r").attrs, "multiscales"
        )
        count = len(multiscales[0]["datasets"]) if multiscales else 0
    except Exception:
        count = 0
    if not count:
        return 5 if requested is None else int(requested)
    if requested is not None and int(requested) != count:
        logger.info(
            "labels/%s: %d pyramid level(s), as many as the image (not %d): "
            "OME-Zarr requires a label image to match its image",
            name,
            count,
            int(requested),
        )
    return count


def fix_ngff_metadata(store: Union[str, Path]) -> list[str]:
    """Repair, in place, what older patchworks versions wrote wrongly.

    Metadata only, no voxel is rewritten:

    - OME-Zarr 0.5 (zarr v3): every pyramid array's ``dimension_names``,
      set to the multiscales axes, as the spec requires (patchworks wrote
      none before).
    - The ``labels`` list: entries without a label image (one deleted by
      hand) removed, label images missing from it added.
    - A label image with more pyramid levels than its image: the extra
      (coarsest) levels are dropped, as the spec requires the counts equal.
      One with fewer is reported, not changed: that needs levels computed
      (re-register it with :func:`register_labels`).

    Returns
    -------
    list of str
        What was changed or found, one line each.
    """
    store = str(store).rstrip("/")
    root = zarr.open_group(store, mode="r+")
    done: list[str] = []

    def fix(group, where: str) -> int:
        ms = read_ngff_attr(group.attrs, "multiscales") or []
        if not ms:
            return 0
        axes = "".join(a["name"] for a in ms[0]["axes"])
        paths = [d["path"] for d in ms[0]["datasets"]]
        if group.metadata.zarr_format == 3:
            missing = [
                p
                for p in paths
                if tuple(group[p].metadata.dimension_names or ()) != tuple(axes)
            ]
            if missing:
                _set_dimension_names(group, missing, axes)
                done.append(
                    f"{where}: dimension_names {list(axes)} set on "
                    f"{len(missing)} array(s)"
                )
        return len(paths)

    n_image = fix(root, "image")
    try:
        labels_grp = root["labels"]
    except KeyError:
        return done
    listed = list(read_ngff_attr(labels_grp.attrs, "labels", []) or [])
    images = {
        k
        for k, g in labels_grp.groups()
        if read_ngff_attr(g.attrs, "multiscales")
    }
    # The list must name label images that exist (readers open each one),
    # and should name all of them.
    names = [n for n in listed if n in images]
    names += sorted(images - set(names))
    if names != listed:
        write_ngff_attrs(labels_grp, labels=names)
        gone = [n for n in listed if n not in images]
        added = [n for n in names if n not in listed]
        if gone:
            done.append(
                f"labels: removed {', '.join(gone)} from the list "
                "(no such label image)"
            )
        if added:
            done.append(f"labels: listed {', '.join(added)}")
    for name in names:
        group = root["labels"][name]
        n_label = fix(group, f"labels/{name}")
        if n_image and n_label > n_image:
            ms = read_ngff_attr(group.attrs, "multiscales")
            extra = [d["path"] for d in ms[0]["datasets"][n_image:]]
            entry = dict(ms[0], datasets=ms[0]["datasets"][:n_image])
            write_ngff_attrs(group, multiscales=[entry, *ms[1:]])
            for path in extra:
                del group[path]
            done.append(
                f"labels/{name}: dropped {len(extra)} coarse level(s) "
                f"({', '.join(extra)}) to match the image's {n_image}"
            )
        elif n_image and n_label and n_label < n_image:
            done.append(
                f"labels/{name}: {n_label} pyramid level(s) for the image's "
                f"{n_image} -- OME-Zarr requires them equal; re-register it "
                "(register_labels) to rebuild its pyramid"
            )
    return done


def register_labels(
    image_store: Union[str, Path],
    name: str = "labels",
    *,
    axes: Union[str, None] = None,
    pixel_size: Union[PixelSize, tuple, None] = None,
    n_levels: Union[int, None] = None,
    downscale: int = 2,
    chunks: Union[tuple[int, ...], None] = None,
    shard: ShardSpec = False,
    progress: bool = True,
    n_objects: Union[int, None] = None,
    ngff_version: Union[str, None] = "auto",
    compression: Union[str, None] = None,
    level: int = 0,
    provenance: Union[dict, None] = None,
) -> str:
    """Pyramidalise and register an existing ``labels/<name>/0`` base level.

    Assumes the full-resolution label array already exists at
    ``image_store/labels/<name>/0``. Adds the downsampled levels, tags the
    group with NGFF ``image-label`` metadata, lists *name* in
    ``labels/.zattrs``, and inherits the parent image's pixel calibration
    (unless *pixel_size* is given).

    Parameters
    ----------
    image_store : str or Path
        OME-ZARR store path containing the image this label belongs to.
    name : str, optional
        Label image name under ``labels/`` (default ``"labels"``).
    axes : str, optional
        One letter per axis. ``None`` → inferred from the label array.
    pixel_size : dict, tuple or None, optional
        Physical voxel size in micrometers. ``None`` → inherited from the
        parent image's calibration at *level*.
    n_levels : int, optional
        Pyramid levels including full resolution, when the image has no
        pyramid to follow (default 5). With one, the labels get exactly as
        many levels as the image -- the OME-Zarr spec requires it of a
        label image -- whatever is asked here.
    downscale : int, optional
        Per-level X/Y downsampling factor (default 2).
    chunks : tuple of int, optional
        Chunk shape for the written levels. ``None`` → a bounded default.
    shard : bool or tuple of int, optional
        Sharding request (see :func:`to_ome_zarr`'s *shard*).
    progress : bool, optional
        Show a per-level dask progress bar (default ``True``).
    level : int, optional
        The image pyramid level the labels were segmented at (default 0).
        Their voxel size and offset are that level's, not level 0's: taking
        level 0's for labels made at level 1 draws them at half size,
        drifting further off the image the further from the origin.
    n_objects : int or None, optional
        Exact non-background object count, if known (e.g. from
        :func:`patchworks.merge_tile_labels`'s ``return_count=True`` after
        ``sequential_labels=True``, which means ``ids == range(1, n_objects
        + 1)`` by construction). When given, written into the label group's
        attrs as ``n_objects``/``sequential_labels`` so a downstream reader
        (e.g. napari-chunked-regionprops,
        https://github.com/imcf/napari-chunked-regionprops) can use the
        known id set instead of re-deriving it with a full-volume scan of
        its own.
    ngff_version : str or None, optional
        NGFF version (and therefore zarr format) to write: ``"auto"``
        (default) writes 0.5 (zarr v3), while ``"0.4"`` pins the older, zarr-v2 layout for tools that
        cannot read 0.5 yet. 0.4 has no sharding codec, so ``shard`` is
        ignored there. See :func:`ngff_version`.

    compression : str, optional
        Codec for the arrays written (``"zstd"``, ``"zstd:3"``, ``"blosc"``,
        ``"blosc:lz4"``, ``"none"``, ...; see
        :func:`patchworks.compression`). ``None`` (default) uses the active
        setting, zstd level 1 unless changed.
    provenance : dict, optional
        How the labels were made (see :func:`patchworks.provenance`),
        stored in the label group's attrs under ``"patchworks"``.
    Returns
    -------
    str
        Path to the label group (``image_store/labels/<name>``).

    Examples
    --------
    >>> register_labels("scan.zarr", "cells")  # doctest: +SKIP
    'scan.zarr/labels/cells'
    """
    with _writing_ngff(ngff_version), _compression_scope(compression):
        store = str(image_store)
        group = f"{store}/labels/{name}"
        n_levels = _label_level_count(store, name, n_levels)
        if not pixel_size:
            arr0 = da.from_zarr(group, component="0")
            lab_axes = axes or _default_axes(arr0.ndim)
            pixel_size = _read_zarr_calibration(store, lab_axes, level=level)
        add_pyramid(
            group,
            base="0",
            axes=axes,
            pixel_size=pixel_size,
            n_levels=n_levels,
            downscale=downscale,
            chunks=chunks,
            shard=shard,
            progress=progress,
            # Majority vote, not a mean (averaging ids invents objects) and not
            # decimation (which drops every object its sampled voxel misses).
            downsample="mode",
            translation=read_translation(store, level),
        )
        grp = _open_group(group)
        write_ngff_attrs(
            grp, **{"image-label": {"version": _ngff_version_for(grp)}}
        )
        if n_objects is not None:
            # patchworks' own hints, not NGFF keys, so they stay at the top level
            # where a consumer can find them without knowing the layout.
            grp.attrs["n_objects"] = int(n_objects)
            grp.attrs["sequential_labels"] = True
        write_provenance(grp, provenance)

        labels_grp = _open_group(f"{store}/labels")
        registered = list(read_ngff_attr(labels_grp.attrs, "labels", []) or [])
        # Entries whose group was deleted (an rm -r of a label image) would
        # send every reader to open something that is not there.
        present = {k for k, _ in labels_grp.groups()}
        registered = [n for n in registered if n in present]
        if name not in registered:
            registered.append(name)
        write_ngff_attrs(labels_grp, labels=registered)
        return group


def write_labels(
    image_store: Union[str, Path],
    labels: Union[da.Array, np.ndarray],
    *,
    name: str = "labels",
    axes: Union[str, None] = None,
    pixel_size: Union[PixelSize, tuple, None] = None,
    n_levels: Union[int, None] = None,
    downscale: int = 2,
    chunks: Union[tuple[int, ...], None] = None,
    shard: ShardSpec = False,
    progress: bool = True,
    overwrite: bool = False,
    n_objects: Union[int, None] = None,
    ngff_version: Union[str, None] = "auto",
    compression: Union[str, None] = None,
    level: int = 0,
    provenance: Union[dict, None] = None,
) -> str:
    """Store *labels* inside *image_store* under the NGFF ``labels/`` group.

    The labels are written as their own multi-scale pyramid at
    ``image_store/labels/<name>/`` and registered in
    ``image_store/labels/.zattrs``, so the image and its segmentation live in a
    single OME-ZARR store. Calibration is inherited from the parent image
    unless *pixel_size* is given.

    Parameters
    ----------
    image_store : str or Path
        OME-ZARR store this label image belongs to.
    labels : da.Array or np.ndarray
        Integer label array (0 = background), same spatial shape as the
        image.
    name : str, optional
        Label image name under ``labels/`` (default ``"labels"``).
    axes : str, optional
        One letter per axis. ``None`` → inferred from *labels*'
        dimensionality.
    pixel_size : dict, tuple or None, optional
        Physical voxel size in micrometers. ``None`` → inherited from the
        parent image's own calibration.
    n_levels : int, optional
        Pyramid levels when the image has no pyramid to follow (default
        5); otherwise exactly as many as the image, as the OME-Zarr spec
        requires of a label image.
    downscale : int, optional
        Per-level X/Y downsampling factor (default 2).
    chunks : tuple of int, optional
        Chunk shape for the written levels. ``None`` → a bounded default.
    shard : bool or tuple of int, optional
        Sharding request (see :func:`to_ome_zarr`'s *shard*).
    progress : bool, optional
        Show a per-level dask progress bar (default ``True``).
    overwrite : bool, optional
        Replace an existing label image of the same *name* (default
        ``False``).
    n_objects : int or None, optional
        Exact non-background object count, if known — forwarded to
        :func:`register_labels`; see its docstring for what this enables.
    level : int, optional
        The image pyramid level *labels* were segmented at (default 0);
        their calibration and offset are taken from that level. See
        :func:`register_labels`.
    ngff_version : str or None, optional
        NGFF version (and therefore zarr format) to write: ``"auto"``
        (default) writes 0.5 (zarr v3), while ``"0.4"`` pins the older, zarr-v2 layout for tools that
        cannot read 0.5 yet. 0.4 has no sharding codec, so ``shard`` is
        ignored there. See :func:`ngff_version`.

    compression : str, optional
        Codec for the arrays written (``"zstd"``, ``"zstd:3"``, ``"blosc"``,
        ``"blosc:lz4"``, ``"none"``, ...; see
        :func:`patchworks.compression`). ``None`` (default) uses the active
        setting, zstd level 1 unless changed.
    provenance : dict, optional
        How the labels were made (see :func:`patchworks.provenance`),
        stored in the label group's attrs under ``"patchworks"``.
    Returns
    -------
    str
        Path to the written label group (``image_store/labels/<name>``).

    Examples
    --------
    >>> from patchworks import merge_tile_labels
    >>> merged, n = merge_tile_labels(
    ...     "stage.zarr",
    ...     input_component="staged",
    ...     write_to="merged.zarr",
    ...     sequential_labels=True,
    ...     return_count=True,
    ... )  # doctest: +SKIP
    >>> write_labels(
    ...     "scan.zarr", merged, name="cells", n_objects=n
    ... )  # doctest: +SKIP
    'scan.zarr/labels/cells'
    """
    with _writing_ngff(ngff_version), _compression_scope(compression):
        arr = labels if isinstance(labels, da.Array) else da.asarray(labels)
        if axes is None:
            axes = _default_axes(arr.ndim)
        if len(axes) != arr.ndim:
            raise ValueError(
                f"axes {axes!r} has {len(axes)} entries but array is {arr.ndim}-D"
            )

        store = str(image_store)
        root = _open_group(store)
        parent = root.require_group("labels")
        if overwrite and name in parent:
            del parent[name]
        parent.require_group(name)

        label_group = f"{store}/labels/{name}"
        base = arr.rechunk(chunks or _default_chunks(arr.shape, axes))
        _to_zarr_level(base, label_group, "0", shard, progress)
        return register_labels(
            store,
            name,
            axes=axes,
            pixel_size=pixel_size,
            n_levels=n_levels,
            downscale=downscale,
            chunks=chunks,
            shard=shard,
            progress=progress,
            n_objects=n_objects,
            level=level,
            provenance=provenance,
        )
