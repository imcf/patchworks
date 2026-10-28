"""Object sizes from the table, and a label image without the objects
outside a size range -- after the run, without segmenting again."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import numpy as np

logger = logging.getLogger(__name__)


def _group(store: Union[str, Path], name: str) -> str:
    return f"{str(store).rstrip('/')}/labels/{name}"


def _volumes(cols: dict[str, np.ndarray]) -> tuple[np.ndarray, str]:
    """Each object's volume and its unit: µm³ when calibrated, else voxels."""
    for key, unit in (("area_um3", "µm³"), ("area_um2", "µm²")):
        if key in cols:
            return np.asarray(cols[key], dtype=float), unit
    return np.asarray(cols["area_voxels"], dtype=float), "voxels"


def _lengths(store: Union[str, Path], name: str, cols: dict) -> np.ndarray:
    """Each object's length along its main axis (µm when calibrated), from
    the spread its table records; NaN without one."""
    import pandas as pd

    from ._io import open_group_any
    from ._tables import _level0, _spatial_axes, shape_columns
    from .plugins.ome_zarr import read_pixel_size

    path = _group(store, name)
    group = open_group_any(path)
    axes = _spatial_axes(group, _level0(group).ndim)
    shape = shape_columns(pd.DataFrame(cols), axes, read_pixel_size(path))
    for key in ("length_um", "length_voxels"):
        if key in shape:
            return np.asarray(shape[key], dtype=float)
    return np.full(len(cols["label"]), np.nan)


def two_humps(volumes: np.ndarray, cut: float) -> bool:
    """Whether the volumes fall in two groups with a dip at *cut*: the
    histogram there under half the lower of the two peaks either side."""
    v = np.log10(volumes[volumes > 0])
    counts, edges = np.histogram(v, 16)
    at = int(np.clip(np.searchsorted(edges, np.log10(cut)) - 1, 0, 15))
    below, above = counts[:at], counts[at + 1 :]
    if not below.size or not above.size:
        return False
    return counts[at] < 0.5 * min(below.max(), above.max())


def suggest_min_volume(volumes: np.ndarray) -> float:
    """The cut between small specks and the objects: Otsu's threshold on the
    log of the volumes, where specks and real objects form two humps.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> specks = 10 ** rng.normal(-2.0, 0.2, 9000)
    >>> cilia = 10 ** rng.normal(0.0, 0.2, 1000)
    >>> 0.02 < suggest_min_volume(np.r_[specks, cilia]) < 0.5
    True
    """
    from skimage.filters import threshold_otsu

    logv = np.log10(volumes[volumes > 0])
    if logv.size < 2 or logv.min() == logv.max():
        return float(volumes.min()) if volumes.size else 0.0
    t = threshold_otsu(logv)
    # Otsu's threshold is the top of the lower hump: the middle of the gap
    # above it (on the log scale) leaves room either side.
    below, above = logv[logv <= t], logv[logv > t]
    if below.size and above.size:
        t = (below.max() + above.min()) / 2
    return float(10**t)


def _histogram(volumes: np.ndarray, unit: str, cut: float) -> list[str]:
    """A text histogram of the volumes on a log scale, the cut marked."""
    v = volumes[volumes > 0]
    edges = np.logspace(np.log10(v.min()), np.log10(v.max()), 17)
    counts, _ = np.histogram(v, edges)
    width = max(counts.max(), 1)
    lines = []
    for lo, hi, n in zip(edges[:-1], edges[1:], counts):
        mark = "  <- cut" if lo <= cut < hi else ""
        bar = "#" * int(round(40 * n / width))
        lines.append(f"  {lo:10.4g} - {hi:<10.4g} {unit} {n:>10,} {bar}{mark}")
    return lines


def size_report(store: Union[str, Path], name: str) -> dict[str, Any]:
    """The sizes of *name*'s objects, and what a minimum volume would keep.

    Returns
    -------
    dict
        ``unit``, ``n``, ``quantiles`` ({percent: volume}), ``suggested``
        (the cut from :func:`suggest_min_volume`), ``histogram`` (lines),
        ``cuts``: for the suggestion and round values around it, the objects
        kept and, when the table relates them to a parent, kept per parent.
    """
    from ._tables import read_columns

    cols = read_columns(_group(store, name))
    volumes, unit = _volumes(cols)
    suggested = suggest_min_volume(volumes)
    lengths = _lengths(store, name, cols)
    parents = {}
    for key in cols:
        if key.endswith("_id"):
            parent = key[: -len("_id")]
            try:
                from ._io import open_group_any

                n = open_group_any(_group(store, parent)).attrs.get("n_objects")
            except Exception:  # noqa: BLE001 - no such label image
                n = None
            parents[parent] = (np.asarray(cols[key]), n)
    # The suggestion, and round values (1-2-5) from a quarter to 4x it.
    decades = np.floor(np.log10(max(suggested, 1e-12))) + np.arange(-1, 2)
    nice = [m * 10**d for d in decades for m in (1, 2, 5)]
    candidates = sorted(
        {suggested}
        | {float(c) for c in nice if suggested / 4 <= c <= suggested * 4}
    )

    def row_for(keep, **what):
        row = {**what, "kept": int(keep.sum())}
        for parent, (ids, n_parents) in parents.items():
            inside = keep & (ids > 0)
            row[f"per {parent}"] = (
                int(inside.sum()) / n_parents if n_parents else None
            )
            row[f"{parent} with one or more"] = (
                len(np.unique(ids[inside])) / n_parents if n_parents else None
            )
        return row

    cuts = [row_for(volumes >= cut, min_volume=cut) for cut in candidates]
    by_length = (
        []
        if np.isnan(lengths).all()
        else [
            row_for(lengths >= cut, min_length=cut)
            for cut in (0.5, 1.0, 1.5, 2.0, 3.0)
        ]
    )
    return {
        "unit": unit,
        "n": int(volumes.size),
        "quantiles": {
            q: float(np.percentile(volumes, q))
            for q in (1, 5, 25, 50, 75, 95, 99)
        },
        "suggested": suggested,
        "histogram": _histogram(volumes, unit, suggested),
        "two_humps": two_humps(volumes, suggested),
        "cuts": cuts,
        "length_quantiles": (
            {}
            if np.isnan(lengths).all()
            else {
                q: float(np.nanpercentile(lengths, q))
                for q in (5, 25, 50, 75, 95)
            }
        ),
        "by_length": by_length,
    }


def format_size_report(name: str, report: dict[str, Any]) -> str:
    unit = report["unit"]
    lines = [f"{name}: {report['n']:,} objects, volume in {unit}"]
    lines.append(
        "  percentiles: "
        + ", ".join(f"{q}%: {v:.4g}" for q, v in report["quantiles"].items())
    )
    lines.append("  histogram (log scale):")
    lines += report["histogram"]
    if report.get("two_humps", True):
        lines.append(
            f"  suggested min_volume: {report['suggested']:.3g} {unit} (the "
            "gap between the two humps; check a few objects either side of it)"
        )
    else:
        lines.append(
            "  one hump, no gap: volume alone does not separate two kinds of "
            "objects (any min_volume is a guess); see the lengths below"
        )

    def table(title, key, rows, fmt):
        lines.append(title)
        for row in rows:
            extra = "".join(
                f", {v:.2f} {k}" if v is not None else ""
                for k, v in row.items()
                if k not in (key, "kept")
            )
            lines.append(
                f"    {fmt(row[key]):<10} {row['kept']:>12,} objects{extra}"
            )

    table(
        "  what a min_volume keeps:",
        "min_volume",
        report["cuts"],
        lambda v: f"{v:.3g}",
    )
    if report.get("length_quantiles"):
        lines.append(
            "  length along the main axis (µm): "
            + ", ".join(
                f"{q}%: {v:.3g}" for q, v in report["length_quantiles"].items()
            )
        )
        table(
            "  what a min_length (µm) keeps -- a cilium is a rod >~1 µm, a "
            "speck as short as the optics allow (~0.3-0.5 µm):",
            "min_length",
            report["by_length"],
            lambda v: f"{v:g}",
        )
    return "\n".join(lines)


def filter_labels(
    store: Union[str, Path],
    name: str,
    *,
    min_volume: float | None = None,
    max_volume: float | None = None,
    min_length: float | None = None,
    out: str | None = None,
) -> tuple[str, int, int]:
    """Write ``labels/<out>``: *name* without the objects whose volume (µm³
    when calibrated, else voxels) is outside ``[min_volume, max_volume]``,
    numbered 1..N, with its table. *name* is left as it is. *min_length*
    (µm when calibrated) drops objects shorter along their main axis.

    Returns
    -------
    tuple
        ``(out, kept, dropped)``.
    """
    import dask.array as da

    from ._io import open_group_any
    from ._provenance import PROVENANCE_KEY
    from ._tables import _level0, read_columns, table_meta, write_table
    from .plugins.ome_zarr import write_labels

    group_path = _group(store, name)
    cols = read_columns(group_path)
    volumes, unit = _volumes(cols)
    keep = np.ones(volumes.size, bool)
    if min_volume is not None:
        keep &= volumes >= float(min_volume)
    if max_volume is not None:
        keep &= volumes <= float(max_volume)
    if min_length is not None:
        keep &= _lengths(store, name, cols) >= float(min_length)
    labels = np.asarray(cols["label"], dtype=np.int64)
    group = open_group_any(group_path)
    level0 = da.from_zarr(_level0(group))
    lut = np.zeros(int(labels.max(initial=0)) + 1, dtype=level0.dtype)
    lut[labels[keep]] = np.arange(1, int(keep.sum()) + 1, dtype=lut.dtype)
    filtered = level0.map_blocks(lambda b: lut[b], dtype=level0.dtype)

    out = out or f"{name}_filtered"
    prov = dict(dict(group.attrs).get(PROVENANCE_KEY) or {})
    settings = prov.get("settings") or {}
    prov["filter"] = {
        "from": name,
        "min_volume": min_volume,
        "max_volume": max_volume,
        "min_length": min_length,
        "unit": unit,
        "kept": int(keep.sum()),
        "dropped": int((~keep).sum()),
    }
    logger.info(
        "writing labels/%s: %d of %d objects of %s",
        out,
        keep.sum(),
        keep.size,
        name,
    )
    write_labels(
        str(store),
        filtered,
        name=out,
        overwrite=True,
        level=int(settings.get("level") or 0),
        n_objects=int(keep.sum()),
        progress=True,
        provenance=prov,
    )
    table = {k: np.asarray(v)[keep] for k, v in cols.items()}
    table["label"] = np.arange(1, int(keep.sum()) + 1, dtype=np.int64)
    meta = {
        k: v
        for k, v in table_meta(group_path).items()
        if k not in ("labels", "columns", "version")
    }
    write_table(
        _group(store, out), table, attrs={**meta, "filtered_from": name}
    )
    return out, int(keep.sum()), int((~keep).sum())
