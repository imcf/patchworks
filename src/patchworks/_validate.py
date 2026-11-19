"""`patchworks validate`: is a store what the spec and patchworks expect?

Two kinds of check, reported together:

* **the OME-Zarr spec**, by an independent validator (`yaozarrs`, in
  `patchworks[validate]`): what any reader may refuse;
* **patchworks' own**: what readers accept but patchworks writes
  differently (label images without their `source`, unnamed channels,
  object tables out of date...) and how to repair each.

Plus a summary of the store: its levels, chunks and shards, and each
label image's objects and how it was made.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Union


@dataclasses.dataclass
class Finding:
    """One thing found wrong.

    Parameters
    ----------
    where : str
        The node, e.g. `"image"` or `"labels/cells"`.
    message : str
        What is wrong, and how to repair it where patchworks can.
    code : str, optional
        The validator's code for the rule broken (`"spec"` checks), or
        patchworks' own (`"pw-..."`).
    error : bool, optional
        True for what readers may refuse; False for what they accept.
    """

    where: str
    message: str
    code: str = ""
    error: bool = True

    def __str__(self) -> str:
        """The finding as one line, `ERROR`/`warning`, place, code, text.

        Returns
        -------
        str
        """
        kind = "ERROR  " if self.error else "warning"
        code = f" [{self.code}]" if self.code else ""
        return f"{kind} {self.where}{code}: {self.message}"


def spec_findings(store: Union[str, Path]) -> list[Finding] | None:
    """What the OME-Zarr spec validator finds wrong with *store*.

    Parameters
    ----------
    store : str or Path
        The OME-Zarr image.

    Returns
    -------
    list of Finding or None
        Empty when valid; None when no validator is installed.
    """
    try:
        import yaozarrs
    except ImportError:
        return None
    try:
        yaozarrs.validate_zarr_store(str(store))
    except Exception as exc:  # noqa: BLE001 - every failure is a finding
        errors = getattr(exc, "errors", None)
        if not callable(errors):
            return [Finding("store", str(exc).splitlines()[0], "spec")]
        out = []
        for err in errors():
            loc = ".".join(str(p) for p in err.get("loc", ()))
            where = (err.get("ctx") or {}).get("path")
            out.append(
                Finding(
                    f"labels/{where}" if where else (loc or "store"),
                    str(err.get("msg", "")).splitlines()[0],
                    str(err.get("type", "spec")),
                )
            )
        return out
    return []


def _levels(group: Any) -> list[dict[str, Any]]:
    """Each pyramid level of a multiscale group: path, shape, chunks...

    Parameters
    ----------
    group : zarr.Group
        An image or label group.

    Returns
    -------
    list of dict
        `{"path", "shape", "chunks", "shards", "dtype"}`; empty without
        multiscales.
    """
    from .plugins.ome_zarr import read_ngff_attr

    ms = read_ngff_attr(group.attrs, "multiscales") or []
    out = []
    for ds in ms[0]["datasets"] if ms else []:
        arr = group[ds["path"]]
        out.append(
            {
                "path": ds["path"],
                "shape": tuple(arr.shape),
                "chunks": tuple(arr.chunks),
                "shards": tuple(getattr(arr, "shards", None) or ()) or None,
                "dtype": str(arr.dtype),
            }
        )
    return out


def _describe_levels(levels: list[dict[str, Any]], indent: str) -> list[str]:
    """Summary lines for the pyramid levels of one group.

    Parameters
    ----------
    levels : list of dict
        From :func:`_levels`.
    indent : str
        Prefix of each line.

    Returns
    -------
    list of str
    """
    lines = []
    for i, lv in enumerate(levels):
        shards = f" in shards {lv['shards']}" if lv["shards"] else ""
        lines.append(
            f"{indent}level {i}: {lv['shape']} {lv['dtype']}, chunks "
            f"{lv['chunks']}{shards}"
        )
    return lines


def patchworks_findings(
    store: Union[str, Path],
) -> tuple[list[Finding], list[str]]:
    """What patchworks writes that this store lacks, and its summary.

    Parameters
    ----------
    store : str or Path
        The OME-Zarr image.

    Returns
    -------
    findings : list of Finding
        Each with its repair.
    summary : list of str
        The image's levels; each label image's objects, how it was made,
        its table and its levels.
    """
    import zarr

    from ._provenance import PROVENANCE_KEY, summarize
    from ._tables import has_table, is_stale
    from .plugins.ome_zarr import channel_names, read_ngff_attr, read_pixel_size

    store = str(store).rstrip("/")
    root = zarr.open_group(store, mode="r")
    findings: list[Finding] = []
    summary = [f"image  {store}"]
    ms = read_ngff_attr(root.attrs, "multiscales") or []
    if not ms:
        findings.append(Finding("image", "no multiscales metadata", "pw-image"))
        return findings, summary
    axes = "".join(a["name"] for a in ms[0]["axes"])
    levels = _levels(root)
    size = read_pixel_size(store)
    summary.append(f"  axes {axes}, voxel size {size or 'none'}")
    summary += _describe_levels(levels, "  ")
    if not size:
        findings.append(
            Finding(
                "image",
                "no voxel size: volumes and lengths are in voxels "
                "(convert again with --pixel-size)",
                "pw-calibration",
                error=False,
            )
        )
    if "c" in axes and not channel_names(store):
        findings.append(
            Finding(
                "image",
                "channels have no names (patchworks fix-metadata "
                "--channel-names or --channels-from)",
                "pw-channel-names",
                error=False,
            )
        )
    try:
        labels_grp = root["labels"]
    except KeyError:
        return findings, summary
    listed = list(read_ngff_attr(labels_grp.attrs, "labels", []) or [])
    present = {
        k
        for k, g in labels_grp.groups()
        if read_ngff_attr(g.attrs, "multiscales")
    }
    for name in listed:
        if name not in present:
            findings.append(
                Finding(
                    f"labels/{name}",
                    "listed but missing (patchworks fix-metadata)",
                    "pw-labels-list",
                )
            )
    for name in sorted(present - set(listed)):
        findings.append(
            Finding(
                f"labels/{name}",
                "not in the labels list, so readers skip it (patchworks "
                "fix-metadata)",
                "pw-labels-list",
                error=False,
            )
        )
    for name in [n for n in listed if n in present]:
        where = f"labels/{name}"
        group = labels_grp[name]
        attrs = dict(group.attrs)
        n = attrs.get("n_objects")
        summary.append(
            f"{where}" + (f"  ({n:,} objects)" if isinstance(n, int) else "")
        )
        made = summarize(attrs.get(PROVENANCE_KEY))
        if made:
            summary.append(f"  {made}")
        path = f"{store}/labels/{name}"
        if not has_table(path):
            table = "no object table (patchworks tables)"
        elif is_stale(path):
            table = "object table out of date"
            findings.append(
                Finding(
                    where,
                    "its object table was computed from other labels "
                    f"(patchworks tables {store} --names {name})",
                    "pw-table",
                    error=False,
                )
            )
        else:
            table = "object table current"
        image_label = read_ngff_attr(group.attrs, "image-label")
        props = (
            len(image_label.get("properties") or [])
            if isinstance(image_label, dict)
            else 0
        )
        summary.append(
            f"  {table}"
            + (
                f"; values of {props:,} objects in its metadata"
                if props
                else ""
            )
        )
        label_levels = _levels(group)
        summary += _describe_levels(label_levels, "  ")
        if not isinstance(image_label, dict):
            findings.append(
                Finding(
                    where,
                    "no image-label metadata: readers take it for an image",
                    "pw-image-label",
                )
            )
        elif "source" not in image_label:
            findings.append(
                Finding(
                    where,
                    "does not name its image (source; patchworks fix-metadata)",
                    "pw-label-source",
                    error=False,
                )
            )
        if len(label_levels) != len(levels):
            findings.append(
                Finding(
                    where,
                    f"{len(label_levels)} pyramid level(s) for the image's "
                    f"{len(levels)} (patchworks fix-metadata)",
                    "pw-label-levels",
                )
            )
    return findings, summary


def validate(store: Union[str, Path]) -> tuple[list[Finding], list[str], bool]:
    """Check *store* against the spec and what patchworks writes.

    Parameters
    ----------
    store : str or Path
        The OME-Zarr image.

    Returns
    -------
    findings : list of Finding
        The spec validator's, then patchworks' own.
    summary : list of str
        What the store holds (:func:`patchworks_findings`).
    spec_checked : bool
        Whether a spec validator was installed and ran.
    """
    spec = spec_findings(store)
    own, summary = patchworks_findings(store)
    return (spec or []) + own, summary, spec is not None
