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
    """The installed version of a distribution.

    Parameters
    ----------
    dist : str
        The distribution's name, as pip knows it (`scikit-image`).

    Returns
    -------
    str or None
        Its version, or None when it is not installed.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def provenance(**settings: Any) -> dict[str, Any]:
    """A JSON-safe record of a run: versions, time, and its *settings*.

    Anything not JSON-serialisable (a function, a path object) is stored as
    its string form, so the record can always be written as zarr attrs.

    Parameters
    ----------
    **settings
        The run's settings, stored as given under `"settings"`.

    Returns
    -------
    dict
        `{"created", "versions", "python", "settings"}`; `versions`
        lists only the packages installed.

    Examples
    --------
    >>> rec = provenance(tile_shape=(16, 1024, 1024), stitch="iou")
    >>> rec["settings"]["stitch"], "patchworks" in rec["versions"]
    ('iou', True)
    """
    record: dict[str, Any] = {
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "versions": {
            name: _version(name)
            for name in (
                "patchworks",
                "zarr",
                "dask",
                "numpy",
                "scipy",
                "scikit-image",
                # The segmentation tools, where installed: the same settings
                # can segment differently from one of their versions to the
                # next.
                "cellpose",
                "plantseg",
                "plant-seg",
                "careamics",
                "torch",
                "pycudadecon",
            )
        },
        "python": platform.python_version(),
        "settings": settings,
    }
    record["versions"] = {k: v for k, v in record["versions"].items() if v}
    # A JSON round trip turns anything unserialisable into its string form.
    return json.loads(json.dumps(record, default=str))


def write_provenance(target: Any, record: dict[str, Any] | None) -> None:
    """Store *record* in *target*'s attrs.

    Parameters
    ----------
    target : zarr group or array
        Where to write it, under `PROVENANCE_KEY`.
    record : dict or None
        The record from :func:`provenance`; None or empty writes nothing.
    """
    if record:
        target.attrs[PROVENANCE_KEY] = record


def read_provenance(store: Any, component: str | None = None) -> dict | None:
    """The provenance record of a label group (or array), if it has one.

    Parameters
    ----------
    store : str, Path, zarr.Group or zarr.Array
        A label group path such as `"scan.zarr/labels/cells"`, or an
        opened zarr node.
    component : str, optional
        Array inside *store* to read instead (e.g. `write_to` stores keep
        it on the `"labels"` array).

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


def summarize(record: dict[str, Any] | None) -> str | None:
    """One line on how a label image was made, from its record.

    Parameters
    ----------
    record : dict or None
        A record from :func:`provenance`, as stored on a label group.

    Returns
    -------
    str or None
        For example `"made 2026-11-12 by patchworks 3.4.0: cellpose
        (cpsam), channel 1, level 0"`; None without a record.

    Examples
    --------
    >>> rec = {
    ...     "created": "2026-11-12T08:00:00+00:00",
    ...     "versions": {"patchworks": "3.4.0"},
    ...     "settings": {"method": "dog", "channel": 2, "level": 0},
    ... }
    >>> summarize(rec)
    'made 2026-11-12 by patchworks 3.4.0: dog, channel 2, level 0'
    >>> summarize(None) is None
    True
    """
    if not record:
        return None
    settings = record.get("settings") or {}
    method = settings.get("method") or settings.get("fn")
    if isinstance(method, list) and method:
        # The library's key for a function: its dotted name first
        method = str(method[0]).rsplit(".", 1)[-1].strip("_")
    if method == "cellpose":
        model = (settings.get("cellpose") or {}).get("model")
        method = f"cellpose ({model})" if model else method
    parts = [str(method)] if method else []
    for key in ("channel", "level"):
        if settings.get(key) is not None:
            parts.append(f"{key} {settings[key]}")
    if record.get("filter"):
        parts.append(f"filtered from {record['filter'].get('from')}")
    created = str(record.get("created") or "")[:10]
    version = (record.get("versions") or {}).get("patchworks")
    head = "made" + (f" {created}" if created else "")
    head += f" by patchworks {version}" if version else ""
    return head + (": " + ", ".join(parts) if parts else "")
