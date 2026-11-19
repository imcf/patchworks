"""Per-object values shown beside a label image: volume, length, parent...

The object tables (:mod:`patchworks._tables`) hold what each object is; a
viewer hovering over an object should show it. Two ways out, both made
from the tables with the review's decisions applied:

* the OME-Zarr `image-label` `properties` (napari reads them), written
  into the label image's metadata by :func:`write_label_properties` --
  only up to :data:`PROPERTY_LIMIT` objects, as they are one JSON list
  every reader parses when it opens the label image;
* Neuroglancer's segment properties (:func:`neuroglancer_properties`),
  which `patchworks serve` offers beside each label image, for any number
  of objects.

Neither changes the labels or what identifies them, so the tables stay
current.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from ._review import Review

logger = logging.getLogger(__name__)

#: Most objects whose values go into a label image's metadata. Above it the
#: list would be megabytes of JSON parsed at every opening; Neuroglancer
#: (through `patchworks serve`) still shows them all.
PROPERTY_LIMIT = 50_000

#: Significant digits kept of a measured value.
_DIGITS = 4


def _columns(review: "Review", name: str) -> list[str]:
    """The columns of *name*'s corrected table worth showing, in order.

    Parameters
    ----------
    review : Review
        The store's review, holding the tables.
    name : str
        The label image.

    Returns
    -------
    list of str
        Volume, length and elongation, parents, children counted, position
        and review status, mean intensities -- those the table has.
    """
    eff = review.effective(name)
    volume = next(
        (c for c in ("area_um3", "area_um2", "area_voxels") if c in eff), None
    )
    cols = [volume] if volume else []
    cols += [
        c for c in ("length_um", "length_voxels", "elongation") if c in eff
    ]
    cols += [f"{p}_id" for p in review.parents[name]]
    cols += [f"n_{c}" for c in review.children[name]]
    cols += [c for c in ("position", "qc") if c in eff]
    cols += sorted(c for c in eff if str(c).startswith("mean_intensity_"))
    return [c for c in cols if c in eff]


def object_properties(
    store: Union[str, Path], name: str, review: "Review | None" = None
) -> "pd.DataFrame | None":
    """The values to show for each object of label image *name*.

    Parameters
    ----------
    store : str or Path
        The OME-Zarr image.
    name : str
        The label image.
    review : Review, optional
        The store's review, when already loaded (default: loaded here,
        with every table, as positions need their parents').

    Returns
    -------
    pandas.DataFrame or None
        Indexed by label: the objects the review kept, with the columns
        of :func:`_columns`. None when *name* has no current table.
    """
    from ._review import Review

    review = review or Review(store)
    if name not in review.names:
        return None
    eff = review.effective(name)
    return eff[_columns(review, name)]


def _rounded(value: Any) -> Any:
    """*value* for JSON: integers as they are, floats to a few digits.

    Parameters
    ----------
    value : Any
        One cell of the table.

    Returns
    -------
    Any
        An int, a float rounded to :data:`_DIGITS` significant digits, a
        str, or None for a missing number.

    Examples
    --------
    >>> _rounded(np.float64(12.345678)), _rounded(np.int64(3)), _rounded(np.nan)
    (12.35, 3, None)
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if not math.isfinite(v):
            return None
        if v == 0:
            return 0.0
        return round(v, _DIGITS - 1 - int(math.floor(math.log10(abs(v)))))
    return str(value)


def label_properties(df: "pd.DataFrame") -> list[dict[str, Any]]:
    """The OME-Zarr `image-label` `properties` list of a table.

    Parameters
    ----------
    df : pandas.DataFrame
        Indexed by label (:func:`object_properties`).

    Returns
    -------
    list of dict
        One `{"label-value": label, column: value, ...}` per object; a
        missing value is left out.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"area_um3": [1.23456, np.nan]}, index=[4, 7])
    >>> label_properties(df)
    [{'label-value': 4, 'area_um3': 1.235}, {'label-value': 7}]
    """
    out = []
    columns = list(df.columns)
    for label, row in zip(df.index, df.itertuples(index=False)):
        entry: dict[str, Any] = {"label-value": int(label)}
        for col, value in zip(columns, row):
            v = _rounded(value)
            if v is not None and v != "":
                entry[str(col)] = v
        out.append(entry)
    return out


def write_label_properties(
    store: Union[str, Path],
    name: str,
    *,
    review: "Review | None" = None,
    limit: int = PROPERTY_LIMIT,
) -> str:
    """Put each object's values into label image *name*'s metadata.

    As the `image-label` `properties` of OME-Zarr, which napari shows when
    hovering over an object. Metadata only: the labels, their record and
    so their tables are left as they are. A label image with more than
    *limit* objects gets none (any older list is removed, as it would be
    out of date).

    Run it where nothing else writes the label image's metadata: two
    processes rewriting it at once can leave it unreadable.

    Parameters
    ----------
    store : str or Path
        The OME-Zarr image, writable.
    name : str
        The label image.
    review : Review, optional
        The store's review, when already loaded.
    limit : int, optional
        Most objects to write values for (default :data:`PROPERTY_LIMIT`).

    Returns
    -------
    str
        What was done, one line.
    """
    import zarr

    from .plugins.ome_zarr import read_ngff_attr, write_ngff_attrs

    group = zarr.open_group(
        f"{str(store).rstrip('/')}/labels/{name}", mode="r+"
    )
    image_label = read_ngff_attr(group.attrs, "image-label")
    if not isinstance(image_label, dict):
        return f"labels/{name}: no image-label metadata, left as it is"
    df = object_properties(store, name, review)
    old = "properties" in image_label
    if df is None or len(df) > limit:
        if old:
            rest = {k: v for k, v in image_label.items() if k != "properties"}
            write_ngff_attrs(group, **{"image-label": rest})
        why = (
            "no current object table"
            if df is None
            else f"{len(df):,} objects, more than {limit:,}; Neuroglancer "
            "(patchworks serve) shows them"
        )
        return f"labels/{name}: no per-object values ({why})"
    entries = label_properties(df)
    write_ngff_attrs(
        group, **{"image-label": {**image_label, "properties": entries}}
    )
    return (
        f"labels/{name}: values of {len(entries):,} objects "
        f"({', '.join(map(str, df.columns))})"
    )


def _property_id(column: str) -> str:
    """A Neuroglancer property id for a column name.

    Parameters
    ----------
    column : str
        The table column.

    Returns
    -------
    str
        Letters, digits and underscores, starting with a letter.

    Examples
    --------
    >>> _property_id("cyto_labels_id"), _property_id("3d")
    ('cyto_labels_id', 'p_3d')
    """
    out = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in column)
    return out if out[:1].isalpha() else f"p_{out}"


def _summary(df: "pd.DataFrame") -> list[str]:
    """One short line per object, shown beside its id in Neuroglancer.

    Parameters
    ----------
    df : pandas.DataFrame
        :func:`object_properties`.

    Returns
    -------
    list of str
        e.g. `"12.3 µm³, 2.41 µm, in cyto_labels 45, apical"`.
    """
    parts: list[list[str]] = [[] for _ in range(len(df))]
    for col in df.columns:
        values = df[col].to_numpy()
        if col.startswith("area_"):
            unit = {"area_um3": " µm³", "area_um2": " µm²"}.get(col, " voxels")
            fmt = [
                f"{_rounded(v):g}{unit}" if _rounded(v) is not None else ""
                for v in values
            ]
        elif col.startswith("length_"):
            unit = " µm" if col == "length_um" else " voxels long"
            fmt = [
                f"{_rounded(v):g}{unit}" if _rounded(v) is not None else ""
                for v in values
            ]
        elif col.endswith("_id"):
            parent = col[: -len("_id")]
            fmt = [f"in {parent} {int(v)}" if v else "" for v in values]
        elif col in ("position", "qc"):
            fmt = [str(v) if v else "" for v in values]
        else:
            continue
        for row, text in zip(parts, fmt):
            if text:
                row.append(text)
    return [", ".join(row) for row in parts]


def neuroglancer_properties(df: "pd.DataFrame") -> dict[str, Any]:
    """Neuroglancer's segment properties (inline) for a table.

    A summary line per object (shown in the segment list), each number as a
    `number` property (to sort or filter by, e.g. `area_um3>10`), and the
    position and review status as tags (`#apical`).

    Parameters
    ----------
    df : pandas.DataFrame
        Indexed by label (:func:`object_properties`).

    Returns
    -------
    dict
        `{"@type": "neuroglancer_segment_properties", "inline": ...}`,
        the `info` of a `precomputed://` source.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"area_um3": [2.5, 7.0], "position": ["apical", ""]},
    ...                   index=[1, 2])
    >>> info = neuroglancer_properties(df)
    >>> info["inline"]["ids"], info["inline"]["properties"][0]["values"]
    (['1', '2'], ['2.5 µm³, apical', '7 µm³'])
    >>> info["inline"]["properties"][2]
    {'id': 'tags', 'type': 'tags', 'tags': ['apical'], 'values': [[0], []]}
    """
    props: list[dict[str, Any]] = [
        {"id": "label", "type": "label", "values": _summary(df)}
    ]
    tags: list[str] = []
    tag_values: list[list[int]] = [[] for _ in range(len(df))]
    for col in df.columns:
        values = df[col].to_numpy()
        if col in ("position", "qc"):
            for row, v in zip(tag_values, values):
                if v:
                    tag = _property_id(str(v))
                    if tag not in tags:
                        tags.append(tag)
                    row.append(tags.index(tag))
            continue
        if not np.issubdtype(values.dtype, np.number):
            continue
        integer = np.issubdtype(values.dtype, np.integer)
        if (
            integer
            and values.size
            and (values.min() < -(2**31) or values.max() >= 2**31)
        ):
            integer = False  # outside int32: as a float
        props.append(
            {
                "id": _property_id(str(col)),
                "type": "number",
                "data_type": "int32" if integer else "float32",
                # "NaN" (a string): JSON has no NaN, Neuroglancer reads it.
                "values": [
                    int(v)
                    if integer
                    else (float(v) if math.isfinite(float(v)) else "NaN")
                    for v in values
                ],
            }
        )
    if tags:
        props.append(
            {"id": "tags", "type": "tags", "tags": tags, "values": tag_values}
        )
    return {
        "@type": "neuroglancer_segment_properties",
        "inline": {"ids": [str(int(i)) for i in df.index], "properties": props},
    }
