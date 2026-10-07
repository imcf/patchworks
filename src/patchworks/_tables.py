"""Object tables: one row per labelled object, stored with the labels.

A label image says *where* the objects are; the table says *what* they are
-- size, position, bounding box, optional intensities, and which object of
another label image each one sits in (a cilium's cell, a nucleus' cell).
It is what the review panel works from, and what the result workbooks are
written from.

The table lives inside the label group it describes, as one 1-D zarr array
per column::

    image.zarr/labels/cilia_labels/
        0/ 1/ 2/ ...    the label pyramid
        table/          label, area_voxels, centroid_z, ..., cyto_labels_id

so it travels with the labels (bundles included), needs nothing beyond zarr
to read, and is replaced whenever the labels are: a table can never describe
a different segmentation than the one next to it. To make that hold even
when only the arrays are rewritten, the table records a fingerprint of the
labels it was computed from, and :func:`read_table` refuses a mismatch.
"""

from __future__ import annotations

import logging
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence, Union

import numpy as np
import scipy.ndimage as ndi
import zarr

from ._chunks import chunk_slices, cpu_allocation
from ._progress import PROGRESS_INTERVAL_S as _PROGRESS_INTERVAL_S
from ._progress import log_progress
from ._provenance import PROVENANCE_KEY

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

TABLE_GROUP = "table"
TABLE_KEY = "patchworks_table"
_SPATIAL = "tczyx"


class StaleTableError(ValueError):
    """The table was computed from different labels than the ones stored."""


# ---------------------------------------------------------------------------
# Locating things in a store
# ---------------------------------------------------------------------------


def _label_group(store: Union[str, Path], name: str) -> str:
    return f"{str(store).rstrip('/')}/labels/{name}"


def _level0(group: "zarr.Group") -> "zarr.Array":
    from .plugins.ome_zarr import read_ngff_attr

    try:
        path = read_ngff_attr(group.attrs, "multiscales")[0]["datasets"][0][
            "path"
        ]
    except (KeyError, IndexError, TypeError):
        path = "0"
    return group[path]


def _spatial_axes(group: "zarr.Group", ndim: int) -> str:
    from .plugins.ome_zarr import _default_axes, read_ngff_attr

    try:
        axes = [
            a["name"]
            for a in read_ngff_attr(group.attrs, "multiscales")[0]["axes"]
        ]
        return "".join(axes)[-ndim:]
    except (KeyError, IndexError, TypeError):
        return _default_axes(ndim)


def label_fingerprint(group: "zarr.Group") -> dict[str, Any]:
    """What identifies one particular label image, cheaply.

    The creation time of its provenance record (a re-segmentation writes a
    new one), its object count and its shape. Nothing that needs a read of
    the voxels.
    """
    attrs = dict(group.attrs)
    arr = _level0(group)
    prov = attrs.get(PROVENANCE_KEY) or {}
    return {
        "created": prov.get("created"),
        "n_objects": attrs.get("n_objects"),
        "shape": list(arr.shape),
    }


# ---------------------------------------------------------------------------
# Measuring, one chunk at a time
# ---------------------------------------------------------------------------


def _pairs(ndim: int) -> list[tuple[int, int]]:
    """Index pairs of a symmetric matrix's upper triangle, diagonal first."""
    return [(i, i) for i in range(ndim)] + [
        (i, j) for i in range(ndim) for j in range(i + 1, ndim)
    ]


def _block_partial(
    lab: np.ndarray,
    offset: tuple[int, ...],
    images: Sequence[np.ndarray],
) -> "dict[str, np.ndarray] | None":
    """Per-object partial sums for one block of labels.

    Everything here combines exactly across blocks (:func:`_merge`), so an
    object spanning several blocks is reassembled exactly. Second moments
    are central, in block-local coordinates: small numbers, so no precision
    is lost to coordinates in the tens of thousands.
    """
    nz = np.nonzero(lab)
    if not nz[0].size:
        return None
    ids, inv = np.unique(lab[nz], return_inverse=True)
    n = ids.size
    count = np.bincount(inv, minlength=n)
    ndim = lab.ndim
    coords = [c.astype(np.float64) for c in nz]
    mean = (
        np.stack([np.bincount(inv, weights=c, minlength=n) for c in coords], 1)
        / count[:, None]
    )
    dev = [c - mean[inv, ax] for ax, c in enumerate(coords)]
    m2 = np.stack(
        [
            np.bincount(inv, weights=dev[i] * dev[j], minlength=n)
            for i, j in _pairs(ndim)
        ],
        1,
    )
    lo = np.stack([np.full(n, np.iinfo(np.int64).max) for _ in range(ndim)], 1)
    hi = np.stack([np.full(n, -1) for _ in range(ndim)], 1).astype(np.int64)
    for ax, c in enumerate(nz):
        np.minimum.at(lo[:, ax], inv, c)
        np.maximum.at(hi[:, ax], inv, c)
    off = np.asarray(offset, dtype=np.int64)
    part = {
        "ids": ids.astype(np.int64),
        "count": count.astype(np.int64),
        "mean": mean + off,
        "m2": m2,
        "lo": lo + off,
        "hi": hi + off,
    }
    for i, img in enumerate(images):
        values = img[nz].astype(np.float64)
        part[f"sum{i}"] = np.bincount(inv, weights=values, minlength=n)
        part[f"sq{i}"] = np.bincount(inv, weights=values**2, minlength=n)
    return part


def _merge(parts: list[dict[str, np.ndarray]], n_images: int) -> dict:
    """Combine every block's partial sums into one row per object.

    Second moments by the parallel-axis theorem: each block's central
    moments plus its count times the offset of its mean from the object's.
    """
    ids = np.concatenate([p["ids"] for p in parts])
    order = np.argsort(ids, kind="stable")
    ids = ids[order]
    starts = np.flatnonzero(np.concatenate([[True], ids[1:] != ids[:-1]]))
    group = np.cumsum(np.concatenate([[False], ids[1:] != ids[:-1]]))

    def cat(key):
        return np.concatenate([p[key] for p in parts])[order]

    n_b = cat("count").astype(np.float64)
    mean_b = cat("mean")
    count = np.add.reduceat(cat("count"), starts)
    mean = (
        np.add.reduceat(mean_b * n_b[:, None], starts, axis=0) / count[:, None]
    )
    d = mean_b - mean[group]
    ndim = mean.shape[1]
    shift = np.stack([n_b * d[:, i] * d[:, j] for i, j in _pairs(ndim)], 1)
    out = {
        "label": ids[starts],
        "count": count,
        "mean": mean,
        "m2": np.add.reduceat(cat("m2") + shift, starts, axis=0),
        "lo": np.minimum.reduceat(cat("lo"), starts, axis=0),
        "hi": np.maximum.reduceat(cat("hi"), starts, axis=0),
    }
    for i in range(n_images):
        out[f"sum{i}"] = np.add.reduceat(cat(f"sum{i}"), starts)
        out[f"sq{i}"] = np.add.reduceat(cat(f"sq{i}"), starts)
    return out


#: Voxels measured at once by :func:`tile_partial`. Measuring holds about
#: a dozen 8-byte arrays of the foreground's size, so 4 M voxels stay near
#: 400 MB whatever the tile -- a fully labelled 126 x 516 x 516 tile at once
#: would need over 3 GB.
TILE_PARTIAL_BLOCK = 1 << 22


def tile_partial(
    labels: np.ndarray,
    offset: Sequence[int],
    *,
    block_voxels: int = TILE_PARTIAL_BLOCK,
) -> "dict[str, np.ndarray] | None":
    """Per-object sums of one tile of labels, for :func:`combine_partials`.

    Measured in slabs of at most *block_voxels* along the first axis and
    combined, so memory stays bounded by the slab, not the tile. *offset* is
    the tile's position in the full image: centroids and bounding boxes come
    out in its coordinates. ``None`` for a tile without labels.
    """
    labels = np.asarray(labels)
    if not labels.any():
        return None
    plane = int(np.prod(labels.shape[1:])) or 1
    step = max(1, int(block_voxels) // plane)
    off = [int(o) for o in offset]
    parts = []
    for z in range(0, labels.shape[0], step):
        part = _block_partial(labels[z : z + step], (off[0] + z, *off[1:]), [])
        if part is not None:
            parts.append(part)
    if len(parts) == 1:
        return parts[0]
    return _as_partial(_merge(parts, 0))


def _as_partial(m: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """A :func:`_merge` result back in partial form (``ids`` for ``label``)."""
    out = dict(m)
    out["ids"] = out.pop("label")
    return out


def combine_partials(
    parts: Sequence[dict[str, np.ndarray]],
) -> "dict[str, np.ndarray] | None":
    """One row per object from partial sums whose ``ids`` are already global
    (pieces of one object carry the same id): ``label``, ``count``, ``mean``,
    ``m2``, ``lo``, ``hi``, as :func:`_merge`. ``None`` when there are none."""
    parts = [p for p in parts if p is not None and len(p["ids"])]
    if not parts:
        return None
    return _merge(list(parts), 0)


def table_columns(
    m: "dict[str, np.ndarray] | None",
    axes: str,
    pixel_size: Mapping[str, float] | None = None,
    image_names: Sequence[str] = (),
) -> dict[str, np.ndarray]:
    """The table's columns from combined sums (:func:`_merge`)."""
    ndim = len(axes)
    cols: dict[str, np.ndarray] = {}
    if m is None or not len(m["label"]):
        cols["label"] = np.empty(0, np.int64)
        cols["area_voxels"] = np.empty(0, np.int64)
        for ax in axes:
            cols[f"centroid_{ax}"] = np.empty(0)
        for ax in axes:
            cols[f"bbox_min_{ax}"] = np.empty(0, np.int64)
            cols[f"bbox_max_{ax}"] = np.empty(0, np.int64)
        for i, j in _pairs(ndim):
            cols[f"cov_{axes[i]}{axes[j]}"] = np.empty(0)
        return cols
    count = m["count"]
    cols["label"] = m["label"]
    cols["area_voxels"] = count
    for i, ax in enumerate(axes):
        cols[f"centroid_{ax}"] = m["mean"][:, i]
    for i, ax in enumerate(axes):
        cols[f"bbox_min_{ax}"] = m["lo"][:, i]
        cols[f"bbox_max_{ax}"] = m["hi"][:, i]
    for k, (i, j) in enumerate(_pairs(ndim)):
        cols[f"cov_{axes[i]}{axes[j]}"] = m["m2"][:, k] / count
    for i, name in enumerate(image_names):
        mean = m[f"sum{i}"] / count
        cols[f"mean_intensity_{name}"] = mean
        cols[f"std_intensity_{name}"] = np.sqrt(
            np.maximum(m[f"sq{i}"] / count - mean**2, 0.0)
        )
    if pixel_size:
        size = [float(pixel_size.get(ax, 1.0)) for ax in axes]
        key = "area_um3" if ndim == 3 else f"area_um{ndim}"
        cols[key] = count * float(np.prod(size))
        for ax, sz in zip(axes, size):
            cols[f"centroid_{ax}_um"] = cols[f"centroid_{ax}"] * sz
    return cols


def measure_objects(
    labels: Any,
    *,
    images: Mapping[str, Any] | None = None,
    axes: str | None = None,
    pixel_size: Mapping[str, float] | None = None,
    n_workers: int | None = None,
) -> dict[str, np.ndarray]:
    """One row per object: size, centroid, bounding box, intensities.

    Reads *labels* once, block by block (its own chunk grid), and never holds
    more than one block per worker in memory; objects spanning blocks are
    reassembled exactly.

    Parameters
    ----------
    labels : zarr.Array or array-like
        Integer labels, 0 = background.
    images : mapping of str to array-like, optional
        Intensity images of the same shape, by name. Each adds
        ``mean_intensity_<name>`` and ``std_intensity_<name>`` columns --
        and one more full read of that image, so only ask for what you use.
    axes : str, optional
        One letter per axis of *labels* (default ``"zyx"``-style by ndim).
    pixel_size : mapping, optional
        Micrometres per voxel by axis letter. Adds ``area_um3`` (or
        ``area_um2``) and ``centroid_<axis>_um`` columns.
    n_workers : int, optional
        Blocks read in parallel (default: the CPUs this job was given).

    Returns
    -------
    dict of str to np.ndarray
        Columns, all aligned: ``label``, ``area_voxels``,
        ``centroid_<axis>``, ``bbox_min_<axis>``, ``bbox_max_<axis>``
        (inclusive, voxel indices), ``cov_<a><b>`` (the spread of the
        object's voxels, in voxels squared: its size and orientation, see
        :func:`shape_columns`), plus the optional ones above. Column
        names follow napari-chunked-regionprops, so either can read the
        other's tables.

    Examples
    --------
    >>> lab = np.zeros((1, 4, 6), "int32"); lab[0, 1:3, 1:3] = 1; lab[0, 0, 5] = 7
    >>> t = measure_objects(lab)
    >>> t["label"].tolist(), t["area_voxels"].tolist()
    ([1, 7], [4, 1])
    >>> t["bbox_max_x"].tolist(), t["centroid_y"].tolist()
    ([2, 5], [1.5, 0.0])
    """
    from .plugins.ome_zarr import _default_axes

    images = dict(images or {})
    ndim = len(labels.shape)
    axes = axes or _default_axes(ndim)
    chunk_shape = getattr(labels, "chunks", None)
    if not chunk_shape or not isinstance(chunk_shape[0], (int, np.integer)):
        chunk_shape = tuple(min(s, 256) for s in labels.shape)
    for name, img in images.items():
        if tuple(img.shape) != tuple(labels.shape):
            raise ValueError(
                f"image {name!r} has shape {tuple(img.shape)}, labels "
                f"{tuple(labels.shape)}: measure at the level the labels "
                "were segmented at"
            )
    blocks = list(chunk_slices(labels.shape, chunk_shape))
    nw = n_workers or cpu_allocation()
    img_list = list(images.values())

    def _one(sl):
        lab = np.asarray(labels[sl])
        if not lab.any():
            return None
        return _block_partial(
            lab,
            tuple(s.start for s in sl),
            [np.asarray(img[sl]) for img in img_list],
        )

    started = last = _time.monotonic()
    parts = []
    with ThreadPoolExecutor(max_workers=nw) as ex:
        futures = [ex.submit(_one, sl) for sl in blocks]
        for done, fut in enumerate(as_completed(futures), start=1):
            part = fut.result()
            if part is not None:
                parts.append(part)
            now = _time.monotonic()
            if now - last >= _PROGRESS_INTERVAL_S or done == len(blocks):
                log_progress("measure_objects", done, len(blocks), started)
                last = now

    return table_columns(
        _merge(parts, len(img_list)) if parts else None,
        axes,
        pixel_size,
        list(images),
    )


# ---------------------------------------------------------------------------
# Storing and reading
# ---------------------------------------------------------------------------


def write_table(
    group: Union[str, Path, "zarr.Group"],
    columns: Mapping[str, np.ndarray],
    *,
    attrs: Mapping[str, Any] | None = None,
) -> None:
    """Store *columns* as the table of label group *group*, replacing any.

    Parameters
    ----------
    group : str, Path or zarr.Group
        The label group (e.g. ``"scan.zarr/labels/cells"``).
    columns : mapping of str to np.ndarray
        Equal-length 1-D columns; must include ``label``.
    attrs : mapping, optional
        Extra table metadata (merged over the fingerprint and axes).
    """
    grp = (
        zarr.open_group(str(group), mode="r+")
        if not isinstance(group, zarr.Group)
        else group
    )
    fmt = grp.metadata.zarr_format
    n = len(columns["label"])
    if any(len(v) != n for v in columns.values()):
        raise ValueError("table columns must all have the same length")
    table = grp.create_group(TABLE_GROUP, overwrite=True)
    for name, values in columns.items():
        _write_column(table, name, np.asarray(values), fmt)
    meta = {
        "version": 1,
        "labels": label_fingerprint(grp),
        "columns": list(columns),
    }
    meta.update(attrs or {})
    table.attrs[TABLE_KEY] = meta


def _write_column(table: "zarr.Group", name: str, values: np.ndarray, fmt):
    from ._io import zarr_compressor_kwargs

    chunk = max(1, min(len(values), 1 << 20))
    table.create_array(
        name,
        shape=values.shape,
        dtype=values.dtype,
        chunks=(chunk,),
        overwrite=True,
        **zarr_compressor_kwargs(fmt),
    )[...] = values


def has_table(group: Union[str, Path]) -> bool:
    try:
        from ._io import open_group_any

        return TABLE_KEY in open_group_any(f"{group}/{TABLE_GROUP}").attrs
    except Exception:
        return False


def table_meta(group: Union[str, Path]) -> dict[str, Any]:
    from ._io import open_group_any

    return dict(open_group_any(f"{group}/{TABLE_GROUP}").attrs[TABLE_KEY])


def is_stale(group: Union[str, Path]) -> bool:
    """Whether the table no longer matches the labels next to it."""
    from ._io import open_group_any

    return table_meta(group)["labels"] != label_fingerprint(
        open_group_any(group)
    )


def read_columns(
    group: Union[str, Path], *, check: bool = True
) -> dict[str, np.ndarray]:
    """The table of label group *group*, as a dict of numpy columns.

    Raises
    ------
    StaleTableError
        When *check* and the labels changed since the table was computed.
    """
    from ._io import open_group_any

    if check and is_stale(group):
        raise StaleTableError(
            f"the table in {group} was computed from different labels; "
            "recompute it (patchworks tables <store>)"
        )
    table = open_group_any(f"{group}/{TABLE_GROUP}")
    listed = list(table.attrs[TABLE_KEY]["columns"])
    cols = {name: table[name][...] for name in listed}
    # Relation columns are added later, as arrays only (never an attrs
    # update), possibly by a concurrent relate job writing to this same
    # table: one caught mid-write is skipped, not fatal.
    for name in sorted(k for k in table.array_keys() if k not in listed):
        try:
            values = table[name][...]
        except Exception as exc:  # pragma: no cover - a race, by nature
            logger.warning("skipping column %s of %s: %s", name, group, exc)
            continue
        if len(values) == len(cols["label"]):
            cols[name] = values
    return cols


def read_table(
    group: Union[str, Path], *, check: bool = True
) -> "pd.DataFrame":
    """The table of label group *group* as a DataFrame indexed by ``label``.

    Examples
    --------
    >>> read_table("scan.zarr/labels/cilia_labels")  # doctest: +SKIP
    """
    import pandas as pd

    cols = read_columns(group, check=check)
    return pd.DataFrame(cols).set_index("label")


def add_columns(
    group: Union[str, Path], columns: Mapping[str, np.ndarray], **meta: Any
) -> None:
    """Add (or replace) columns of an existing table; *meta* merges into its
    metadata.

    Adding columns writes their arrays only -- no metadata update unless
    *meta* is given -- so jobs adding different columns to the same table
    concurrently (two relations with the same child) cannot lose each
    other's work.
    """
    grp = zarr.open_group(str(group), mode="r+")
    table = grp[TABLE_GROUP]
    fmt = grp.metadata.zarr_format
    n = table["label"].shape[0]
    for name, values in columns.items():
        values = np.asarray(values)
        if len(values) != n:
            raise ValueError(f"column {name!r}: {len(values)} rows, table {n}")
        _write_column(table, name, values, fmt)
    if meta:
        info = dict(table.attrs[TABLE_KEY])
        info.update(meta)
        table.attrs[TABLE_KEY] = info


# ---------------------------------------------------------------------------
# Whole stores
# ---------------------------------------------------------------------------


def compute_table(
    store: Union[str, Path],
    name: str,
    *,
    channels: Sequence[int] | None = None,
    n_workers: int | None = None,
) -> dict[str, np.ndarray]:
    """Measure ``labels/<name>`` of *store* and store the result as its table.

    Parameters
    ----------
    store : str or Path
        OME-ZARR image store holding ``labels/<name>``.
    name : str
        Label image name.
    channels : sequence of int, optional
        Image channels to add mean/std intensity columns for. Read at the
        pyramid level the labels were segmented at (from their provenance).
    n_workers : int, optional
        Parallel block reads.

    Returns
    -------
    dict
        The columns written.
    """
    from ._io import load_ome_zarr
    from .plugins.ome_zarr import read_pixel_size

    path = _label_group(store, name)
    grp = zarr.open_group(path, mode="r")
    arr = _level0(grp)
    axes = _spatial_axes(grp, arr.ndim)
    prov = dict(grp.attrs).get(PROVENANCE_KEY) or {}
    settings = prov.get("settings") or {}
    level = int(settings.get("level") or 0)
    images = {}
    for c in channels or ():
        images[f"ch{c}"] = load_ome_zarr(store, channel=int(c), level=level)
    logger.info(
        "measuring labels/%s (%s, %s channel(s))",
        name,
        "x".join(map(str, arr.shape)),
        len(images),
    )
    cols = measure_objects(
        arr,
        images=images,
        axes=axes,
        pixel_size=read_pixel_size(path),
        n_workers=n_workers,
    )
    tile = settings.get("tile_shape")
    write_table(
        path,
        cols,
        attrs={
            "axes": axes,
            "pixel_size": read_pixel_size(path),
            "tile_shape": list(tile) if tile else None,
            "channels": list(images),
        },
    )
    return cols


def write_table_from_sums(
    store: Union[str, Path], name: str, sums: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Store ``labels/<name>``'s table from combined per-tile sums (the
    merge's ``objects_out``), as :func:`compute_table` would from the labels.

    Call it once the label group is registered: the table records the
    labels' fingerprint, which the registration completes.
    """
    from .plugins.ome_zarr import read_pixel_size

    path = _label_group(store, name)
    grp = zarr.open_group(path, mode="r")
    arr = _level0(grp)
    axes = _spatial_axes(grp, arr.ndim)
    pixel_size = read_pixel_size(path)
    settings = (dict(grp.attrs).get(PROVENANCE_KEY) or {}).get("settings") or {}
    tile = settings.get("tile_shape")
    cols = table_columns(sums, axes, pixel_size)
    write_table(
        path,
        cols,
        attrs={
            "axes": axes,
            "pixel_size": pixel_size,
            "tile_shape": list(tile) if tile else None,
            "channels": [],
        },
    )
    return cols


def relate_tables(
    store: Union[str, Path],
    child: str,
    parent: str,
    *,
    matches: Mapping[int, Mapping[str, float]] | None = None,
    max_distance_um: float | None = None,
    n_workers: int | None = None,
) -> dict[int, dict[str, float]]:
    """Add *child*'s parent columns: which *parent* object each one is in.

    Adds ``<parent>_id`` (0 = in none), ``<parent>_overlap`` (fraction of
    the child inside it) and ``<parent>_overlap_voxels`` to the child's
    table, computing the child's table first if it has none.

    With *max_distance_um*, a child touching no parent gets the **nearest**
    one within that distance instead (a cilium beside its cell rather than
    on it), and a ``<parent>_distance_um`` column: 0 when overlapping, the
    gap for those assigned by distance, NaN for none within reach.

    Parameters
    ----------
    store : str or Path
        OME-ZARR store holding both label images.
    child, parent : str
        Label image names, e.g. ``"cilia_labels"`` in ``"cyto_labels"``.
    matches : mapping, optional
        :func:`~patchworks.label_relations` output for this pair, when
        already computed -- otherwise it is computed here (a read of both).
    n_workers : int, optional
        Parallel chunk reads.

    Returns
    -------
    dict
        The matches used.
    """
    import dask.array as da

    from ._relations import label_relations

    cpath, ppath = _label_group(store, child), _label_group(store, parent)
    if not has_table(cpath) or is_stale(cpath):
        compute_table(store, child, n_workers=n_workers)
    if matches is None:
        a = da.from_zarr(_level0(zarr.open_group(cpath, mode="r")))
        b = da.from_zarr(_level0(zarr.open_group(ppath, mode="r")))
        a, b = align_chunks(a, b)
        matches = label_relations(a, b, n_workers=n_workers)
    labels = zarr.open_group(f"{cpath}/{TABLE_GROUP}", mode="r")["label"][...]
    pid = np.zeros(labels.size, np.int64)
    frac = np.zeros(labels.size)
    vox = np.zeros(labels.size, np.int64)
    for i, lab in enumerate(labels.tolist()):
        m = matches.get(lab)
        if m is not None:
            pid[i], frac[i], vox[i] = (
                m["match"],
                m["overlap_fraction"],
                m["overlap_voxels"],
            )
    cols = {
        f"{parent}_id": pid,
        f"{parent}_overlap": frac,
        f"{parent}_overlap_voxels": vox,
    }
    if max_distance_um is not None:
        dist = np.where(pid != 0, 0.0, np.nan)
        orphans = np.flatnonzero(pid == 0)
        found = nearest_parents(
            store,
            child,
            parent,
            labels[orphans],
            max_distance_um=float(max_distance_um),
            n_workers=n_workers,
        )
        for i, lab in zip(orphans, labels[orphans].tolist()):
            if lab in found:
                pid[i], dist[i] = found[lab]
        cols[f"{parent}_distance_um"] = dist
    add_columns(cpath, cols)
    return dict(matches)


def nearest_parents(
    store: Union[str, Path],
    child: str,
    parent: str,
    labels: Sequence[int],
    *,
    max_distance_um: float,
    n_workers: int | None = None,
) -> dict[int, tuple[int, float]]:
    """The nearest *parent* object to each of *labels* (objects of *child*),
    within *max_distance_um*: ``{label: (parent_id, distance_um)}``.

    Exact surface-to-surface distance in µm (anisotropic voxels included),
    from the voxels around each object only: its bounding box grown by
    the distance. Meant for the few objects touching no parent, not all.
    """
    from ._io import open_group_any
    from .plugins.ome_zarr import read_pixel_size

    cpath, ppath = _label_group(store, child), _label_group(store, parent)
    cgroup = open_group_any(cpath)
    carr, parr = _level0(cgroup), _level0(open_group_any(ppath))
    axes = _spatial_axes(cgroup, carr.ndim)
    size = read_pixel_size(cpath)
    spacing = np.array([float(size.get(ax, 1.0)) for ax in axes])
    margin = np.ceil(max_distance_um / spacing).astype(int)
    table = zarr.open_group(f"{cpath}/{TABLE_GROUP}", mode="r")
    all_labels = table["label"][...]
    lo = np.stack([np.asarray(table[f"bbox_min_{ax}"][...]) for ax in axes], 1)
    hi = np.stack([np.asarray(table[f"bbox_max_{ax}"][...]) for ax in axes], 1)
    where = {int(v): i for i, v in enumerate(all_labels)}

    def one(label: int):
        i = where.get(int(label))
        if i is None:
            return None
        a = np.maximum(lo[i] - margin, 0)
        b = np.minimum(hi[i] + margin + 1, carr.shape)
        sl = tuple(slice(int(x), int(y)) for x, y in zip(a, b))
        mask = np.asarray(carr[sl]) == label
        region = np.asarray(parr[sl])
        if not region.any() or not mask.any():
            return None
        dist, idx = ndi.distance_transform_edt(
            region == 0, sampling=spacing, return_indices=True
        )
        pos = np.argmin(np.where(mask, dist, np.inf))
        d = float(dist.flat[pos])
        if d > max_distance_um:
            return None
        nearest = tuple(ix.flat[pos] for ix in idx)
        return int(label), (int(region[nearest]), d)

    with ThreadPoolExecutor(max_workers=n_workers or cpu_allocation()) as ex:
        return dict(r for r in ex.map(one, labels) if r is not None)


def physical_cov(
    df: "pd.DataFrame", axes: str, pixel_size: Mapping[str, float] | None
) -> np.ndarray:
    """Each object's spread as an (n, ndim, ndim) matrix in µm² (voxels²
    when uncalibrated), from its ``cov_<a><b>`` columns."""
    ndim = len(axes)
    size = np.array([float((pixel_size or {}).get(ax, 1.0)) for ax in axes])
    cov = np.zeros((len(df), ndim, ndim))
    for i, j in _pairs(ndim):
        col = f"cov_{axes[i]}{axes[j]}"
        if col not in df:
            return np.full((len(df), ndim, ndim), np.nan)
        v = df[col].to_numpy(dtype=float) * size[i] * size[j]
        cov[:, i, j] = cov[:, j, i] = v
    return cov


def physical_centroids(
    df: "pd.DataFrame", axes: str, pixel_size: Mapping[str, float] | None
) -> np.ndarray:
    size = np.array([float((pixel_size or {}).get(ax, 1.0)) for ax in axes])
    return np.stack([df[f"centroid_{ax}"].to_numpy() for ax in axes], 1) * size


def shape_columns(
    df: "pd.DataFrame", axes: str, pixel_size: Mapping[str, float] | None
) -> dict[str, np.ndarray]:
    """Length, elongation and main axis of each object, from its moments.

    ``length`` is the extent along the main axis of a uniform rod with the
    same spread (``sqrt(12 * largest variance)``): exact for a straight
    rod, shorter than the path of a curved one. ``elongation`` is the ratio
    of the two largest spreads (1: round, large: rod-like). ``axis_<a>``
    is the main axis as a unit vector (sign fixed so its first non-zero
    component is positive). In µm when *pixel_size* is known.

    Examples
    --------
    >>> import pandas as pd
    >>> rod = pd.DataFrame({"cov_zz": [0.0], "cov_yy": [0.0], "cov_xx": [12.0],
    ...     "cov_zy": [0.0], "cov_zx": [0.0], "cov_yx": [0.0]})
    >>> cols = shape_columns(rod, "zyx", {"z": 1, "y": 1, "x": 0.5})
    >>> round(float(cols["length_um"][0]), 3), float(cols["axis_x"][0])
    (6.0, 1.0)
    """
    cov = physical_cov(df, axes, pixel_size)
    unit = "um" if pixel_size else "voxels"
    if not len(df) or np.isnan(cov).all():
        return {}
    w, v = np.linalg.eigh(cov)  # ascending
    w = np.clip(w, 0, None)
    main = v[:, :, -1]
    first = np.argmax(np.abs(main) > 1e-9, axis=1)
    sign = np.sign(main[np.arange(len(main)), first])
    main = main * np.where(sign == 0, 1, sign)[:, None]
    out = {f"length_{unit}": np.sqrt(12 * w[:, -1])}
    with np.errstate(divide="ignore", invalid="ignore"):
        out["elongation"] = (
            np.sqrt(w[:, -1] / w[:, -2]) if w.shape[1] > 1 else np.ones(len(w))
        )
    for k, ax in enumerate(axes):
        out[f"axis_{ax}"] = main[:, k]
    return out


def align_chunks(a, b):
    """Rechunk the finer of two same-shape dask arrays to the coarser one.

    :func:`~patchworks.label_relations` walks both block by block; two
    segmentations may have been written with different chunks.
    """
    if a.chunks == b.chunks:
        return a, b

    def n(x):
        return int(np.prod([len(c) for c in x.chunks]))

    if n(a) <= n(b):
        return a, b.rechunk(a.chunks)
    return a.rechunk(b.chunks), b
