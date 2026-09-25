"""Provenance: record how a label image was made, next to the labels.

Months after a run, "which threshold made these cilia?" or "were these two
segmentations tiled the same way?" should be answerable from the store
itself, not from whichever config file happened to be kept.
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
from typing import Any

#: Attribute key the record is stored under in a label group / array.
PROVENANCE_KEY = "patchworks"


def _version(dist: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def provenance(**settings: Any) -> dict[str, Any]:
    """A JSON-safe record of a run: versions, time, and its *settings*.

    Anything not JSON-serialisable (a function, a path object) is stored as
    its string form, so the record can always be written as zarr attrs.

    Examples
    --------
    >>> rec = provenance(tile_shape=(16, 1024, 1024), stitch="iou")
    >>> rec["settings"]["stitch"], "patchworks" in rec["versions"]
    ('iou', True)
    """
    record = {
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "versions": {
            name: _version(name)
            for name in ("patchworks", "zarr", "dask", "numpy", "scipy")
        },
        "python": platform.python_version(),
        "settings": settings,
    }
    record["versions"] = {k: v for k, v in record["versions"].items() if v}
    return json.loads(json.dumps(record, default=str))


def write_provenance(target: Any, record: dict[str, Any] | None) -> None:
    """Store *record* in *target*'s attrs (a zarr group or array)."""
    if record:
        target.attrs[PROVENANCE_KEY] = record


def read_provenance(store: Any, component: str | None = None) -> dict | None:
    """The provenance record of a label group (or array), if it has one.

    Parameters
    ----------
    store : str, Path, zarr.Group or zarr.Array
        A label group path such as ``"scan.zarr/labels/cells"``, or an
        opened zarr node.
    component : str, optional
        Array inside *store* to read instead (e.g. ``write_to`` stores keep
        it on the ``"labels"`` array).

    Returns
    -------
    dict or None
        The record written by :func:`provenance`, or None.
    """
    from ._io import open_group_any

    node = store
    if isinstance(store, str) or hasattr(store, "__fspath__"):
        node = open_group_any(str(store))
    if component is not None:
        node = node[component]
    return dict(node.attrs).get(PROVENANCE_KEY)
