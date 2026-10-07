"""Review: look at the objects most likely to be wrong, and correct them.

Works from the object tables (:mod:`patchworks._tables`). Nothing here reads
voxels, so everything is instant however large the image:

* **flags** -- why an object deserves a look: not inside any parent object,
  only partly inside one, an unexpected number of children (a cell with no
  nucleus), an extreme size, or two objects meeting exactly at a tile seam
  (one object split by the tiling?);
* **decisions** -- what the reviewer said: correct as is, not a real object,
  belongs to another parent, or is one object with another. Stored as a
  small log next to the table (never by editing it), so every decision can
  be undone and the log shows what was checked;
* **the corrected view** -- the tables with the decisions applied: rejected
  objects gone, merged ones combined, parents reassigned, children counted
  again. Workbooks and exports are written from it, so a correction shows up
  in the numbers without rewriting any voxel;
* **an error estimate** -- review objects in a fixed random order and the
  fraction found wrong, with a 95% confidence interval, says how good the
  segmentation is.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Union

import numpy as np
import zarr

from ._tables import (
    TABLE_GROUP,
    _label_group,
    has_table,
    is_stale,
    label_fingerprint,
    read_table,
    table_meta,
)

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

REVIEW_KEY = "patchworks_review"
ACTIONS = ("ok", "wrong", "parent", "merge", "position")
POSITIONS = ("apical", "basal", "lateral", "central")
_AXES_SIGNS = {
    f"{s}{a}": (a, 1.0 if s == "+" else -1.0) for s in "+-" for a in "zyx"
}
QUEUES = ("flagged", "random", "all")
#: Excel's hard row limit, header included.
EXCEL_MAX_ROWS = 1_048_576


RULES_KEY = "patchworks_review_rules"


def write_rules(store: Union[str, Path], rules: Mapping[str, Any]) -> None:
    """Store review rules for *store* (``expect``, ``min_overlap``,
    ``position``), read by
    :class:`Review` when none are passed. The workflow writes the multi
    config's ``review:`` block here."""
    unknown = set(rules) - {"expect", "min_overlap", "position"}
    if unknown:
        raise ValueError(
            f"unknown review rule(s) {sorted(unknown)}; expected expect, "
            "min_overlap, position"
        )
    group = zarr.open_group(f"{str(store).rstrip('/')}/labels", mode="r+")
    new = {k: v for k, v in rules.items() if v is not None}
    # Unchanged rules are not rewritten: every run_multi writes them, and a
    # touched labels/zarr.json made the bundle look stale (hours of
    # re-packing for nothing).
    if json.loads(json.dumps(group.attrs.get(RULES_KEY))) != json.loads(
        json.dumps(new)
    ):
        group.attrs[RULES_KEY] = new


def read_rules(store: Union[str, Path]) -> dict[str, Any]:
    from ._io import open_group_any

    try:
        attrs = open_group_any(f"{str(store).rstrip('/')}/labels").attrs
        return dict(attrs.get(RULES_KEY) or {})
    except Exception:
        return {}


def review_updated(store: Union[str, Path], name: str) -> float | None:
    """When the review decisions about ``labels/<name>`` last changed
    (seconds since the epoch), or None if never."""
    from ._io import open_group_any

    try:
        log = open_group_any(
            f"{_label_group(store, name)}/{TABLE_GROUP}"
        ).attrs.get(REVIEW_KEY)
        return _dt.datetime.fromisoformat(log["updated"]).timestamp()
    except Exception:
        return None


def table_names(store: Union[str, Path]) -> list[str]:
    """Label images of *store* that carry a table: the registered ones in
    their registered order, then any other label group found."""
    from ._io import open_group_any
    from .plugins.napari import _inner_label_names

    names = list(_inner_label_names(store))
    try:
        group = open_group_any(f"{str(store).rstrip('/')}/labels")
        names += sorted(k for k, _ in group.groups() if k not in names)
    except Exception:
        pass
    return [n for n in names if has_table(_label_group(store, n))]


def wilson_interval(
    errors: int, n: int, z: float = 1.96
) -> tuple[float, float]:
    """95% Wilson score interval for a proportion *errors* / *n*.

    Examples
    --------
    >>> lo, hi = wilson_interval(3, 100)
    >>> round(lo, 3), round(hi, 3)
    (0.01, 0.085)
    """
    if n == 0:
        return 0.0, 1.0
    p = errors / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class Flag:
    """Why one object deserves a look."""

    label: int
    reasons: list[str] = field(default_factory=list)
    score: float = 0.0
    #: Another object of the same image it may be one object with.
    partner: int | None = None


class Review:
    """The objects of one store, their flags and the reviewer's decisions.

    Parameters
    ----------
    store : str or Path
        OME-ZARR image store whose label images carry tables.
    expect : mapping, optional
        Expected child counts per parent object:
        ``{"cyto_labels": {"nuclei_labels": (1, 1), "cilia_labels": (0, 2)}}``
        -- a cell with no nucleus or 3 cilia is then flagged. Falls back to
        what the tables' metadata recorded (the workflow's ``review:``).
    min_overlap : float or mapping, optional
        A child less than this fraction inside its parent is flagged
        (default 0.5; per ``{child: {parent: value}}`` if a mapping).
    seam_tolerance : int, optional
        Voxels of slack when matching two objects meeting at a tile seam.
    names : iterable of str, optional
        Only these label images (default: every one with a table).
    position : mapping, optional
        Classify objects by where they sit in their parent:
        ``{"cilia_labels": {"parent": "cyto_labels", "apical":
        "nuclei_labels"}}`` -- see :meth:`positions`. Falls back to the
        store's recorded rules.

    Examples
    --------
    >>> rv = Review("scan.zarr", expect={"cells": {"nuclei": (1, 1)}})  # doctest: +SKIP
    >>> rv.flags("nuclei")[:3]  # doctest: +SKIP
    >>> rv.decide("nuclei", 17, "wrong")  # doctest: +SKIP
    >>> rv.summary("nuclei")  # doctest: +SKIP
    """

    def __init__(
        self,
        store: Union[str, Path],
        *,
        expect: Mapping[str, Mapping[str, Iterable[int]]] | None = None,
        min_overlap: float | Mapping[str, Mapping[str, float]] | None = None,
        seam_tolerance: int = 1,
        names: Iterable[str] | None = None,
        position: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.store = str(store).rstrip("/")
        self.seam_tolerance = int(seam_tolerance)
        self.names: list[str] = []
        self.meta: dict[str, dict[str, Any]] = {}
        self.tables: dict[str, pd.DataFrame] = {}
        self.stale: list[str] = []
        rules = read_rules(self.store)
        wanted = None if names is None else set(names)
        if wanted is not None:
            # A position rule reads its parent's table and, for an apical
            # reference ("away from the nuclei"), that one's too: asked for
            # the cilia and cell tables only, the workbook then had no
            # nuclei to orient the cells by.
            for child, rule in {
                **(rules.get("position") or {}),
                **(position or {}),
            }.items():
                if child in wanted and rule:
                    spec = str(rule.get("apical", ""))
                    wanted |= {
                        str(rule.get("parent")),
                        spec.removeprefix("towards:"),
                    } - {""}
        for name in table_names(self.store):
            if wanted is not None and name not in wanted:
                continue
            group = _label_group(self.store, name)
            if not has_table(group):
                continue
            if is_stale(group):
                self.stale.append(name)
                continue
            self.names.append(name)
            self.meta[name] = table_meta(group)
            self.tables[name] = read_table(group, check=False)
        if self.stale:
            logger.warning(
                "tables of %s were computed from older labels and are "
                "ignored; recompute them (patchworks tables %s)",
                ", ".join(self.stale),
                self.store,
            )
        # A table's parents are the label images it has relation columns for.
        self.parents = {
            n: [
                p
                for p in self.names
                if p != n
                and f"{p}_id" in self.tables[n]
                and f"{p}_overlap" in self.tables[n]
            ]
            for n in self.names
        }
        self.children = {
            n: [c for c in self.names if n in self.parents[c]]
            for n in self.names
        }
        self.expect: dict[str, dict[str, tuple[int, int]]] = {}
        for n, rules_n in (rules.get("expect") or {}).items():
            for child, rng in rules_n.items():
                self.expect.setdefault(n, {})[child] = _range(rng)
        self._recorded_overlap = rules.get("min_overlap")
        for n, rules in (expect or {}).items():
            for child, rng in rules.items():
                self.expect.setdefault(n, {})[child] = _range(rng)
        self._min_overlap = min_overlap
        self.position: dict[str, dict[str, Any]] = {
            k: dict(v) for k, v in (rules.get("position") or {}).items()
        }
        for child, rule in (position or {}).items():
            self.position[child] = dict(rule)
        for child, rule in list(self.position.items()):
            if child not in self.tables or "parent" not in rule:
                # A rule for a label image left out on purpose (names=, as
                # each relate job loads just its pair) is no news.
                if "parent" not in rule or wanted is None or child in wanted:
                    logger.warning(
                        "position rule for %s ignored: %s", child, rule
                    )
                del self.position[child]
        self.decisions: dict[str, dict[int, dict[str, Any]]] = {}
        self.seed: dict[str, int] = {}
        for n in self.names:
            self._load_decisions(n)
        self._cache: dict[str, pd.DataFrame] = {}
        self._flag_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    # -- decisions ---------------------------------------------------------

    def _table_group(self, name: str, mode: str = "r"):
        from ._io import open_group_any

        return open_group_any(
            f"{_label_group(self.store, name)}/{TABLE_GROUP}", mode=mode
        )

    def _load_decisions(self, name: str) -> None:
        log = dict(self._table_group(name).attrs.get(REVIEW_KEY) or {})
        fingerprint = label_fingerprint(
            zarr.open_group(_label_group(self.store, name), mode="r")
        )
        decisions = log.get("decisions") or {}
        if decisions and log.get("labels") != fingerprint:
            logger.warning(
                "%s: the review decisions were made on older labels; "
                "starting a fresh review",
                name,
            )
            decisions = {}
        self.decisions[name] = {int(k): v for k, v in decisions.items()}
        self.seed[name] = int(log.get("seed", 0))

    def _save(self, name: str) -> None:
        try:
            group = self._table_group(name, mode="r+")
        except Exception as exc:
            raise PermissionError(
                f"cannot save review decisions into {self.store} (read-only, "
                "or a .zip bundle?): review a writable copy of the store"
            ) from exc
        group.attrs[REVIEW_KEY] = {
            "updated": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "labels": label_fingerprint(
                zarr.open_group(_label_group(self.store, name), mode="r")
            ),
            "seed": self.seed[name],
            "decisions": {
                str(k): v for k, v in sorted(self.decisions[name].items())
            },
        }
        self._cache.clear()
        self._flag_cache.clear()

    def decide(
        self,
        name: str,
        label: int,
        action: str,
        *,
        parent: str | None = None,
        parent_id: int | None = None,
        into: int | None = None,
        position: str | None = None,
        queue: str = "flagged",
        reviewer: str | None = None,
    ) -> None:
        """Record a decision about object *label* of label image *name*.

        Parameters
        ----------
        action : {"ok", "wrong", "parent", "merge"}
            ``ok``: correct as it is. ``wrong``: not a real object (dropped
            from the corrected tables). ``parent``: belongs to *parent_id*
            of label image *parent* (0 = to none). ``merge``: the same
            object as *into*. ``position``: its position class is
            *position* (one of :data:`POSITIONS`), whatever was computed.
        queue : str
            The queue it was reviewed from; ``"random"`` decisions feed the
            error estimate.
        """
        if name not in self.tables:
            raise KeyError(f"no table for labels/{name}")
        label = int(label)
        if label not in self.tables[name].index:
            raise KeyError(f"{name} has no object {label}")
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}, got {action!r}")
        entry: dict[str, Any] = {
            "action": action,
            "queue": queue,
            "time": _dt.datetime.now(_dt.timezone.utc).isoformat(
                timespec="seconds"
            ),
        }
        if reviewer:
            entry["by"] = reviewer
        if action == "parent":
            if parent not in self.parents[name]:
                raise ValueError(
                    f"{name} has no parent image {parent!r} "
                    f"(has {self.parents[name]})"
                )
            pid = int(parent_id or 0)
            if pid and pid not in self.tables[parent].index:
                raise KeyError(f"{parent} has no object {pid}")
            entry.update(parent=parent, parent_id=pid)
            previous = self.decisions[name].get(label, {})
            if previous.get("action") == "parent":
                entry["parents"] = {**previous.get("parents", {}), parent: pid}
            else:
                entry["parents"] = {parent: pid}
        elif action == "position":
            if name not in self.position:
                raise ValueError(f"{name} has no position rule")
            if position not in POSITIONS:
                raise ValueError(f"position must be one of {POSITIONS}")
            entry["position"] = position
        elif action == "merge":
            into = int(into or 0)
            if into == label or into not in self.tables[name].index:
                raise KeyError(f"cannot merge {name} {label} into {into}")
            entry["into"] = into
        self.decisions[name][label] = entry
        self._save(name)

    def relations(self) -> list[tuple[str, str]]:
        """Every (child, parent) pair the tables relate."""
        return [(c, p) for c in self.names for p in self.parents[c]]

    def undo(self, name: str, label: int) -> None:
        """Forget the decision about *label* (it goes back to unreviewed)."""
        if self.decisions[name].pop(int(label), None) is not None:
            self._save(name)

    # -- the corrected view --------------------------------------------------

    def roots(self, name: str) -> dict[int, int]:
        """Where each decided object ends up: 0 if rejected, the object it
        was merged into (followed to the end) otherwise. Undecided objects
        are absent (they map to themselves)."""
        dec = self.decisions[name]
        out: dict[int, int] = {}
        for label in dec:
            seen, cur = {label}, label
            while True:
                d = dec.get(cur)
                if d is None or d["action"] in ("ok", "parent", "position"):
                    break
                if d["action"] == "wrong":
                    cur = 0
                    break
                cur = int(d["into"])
                if cur in seen:  # a merge cycle: keep the smallest id
                    cur = min(seen)
                    break
                seen.add(cur)
            if cur != label:
                out[label] = cur
        return out

    def _corrected(self, name: str) -> "pd.DataFrame":
        """*name*'s rows with the decisions applied, before derived columns
        (counts, shape, position) -- which need other tables' rows, never
        their derived columns, so nothing here recurses."""
        key = ("rows", name)
        if key in self._cache:
            return self._cache[key]
        import pandas as pd

        df = self.tables[name].copy()
        dec = self.decisions[name]
        for p in self.parents[name]:
            col = f"{p}_id"
            for label, d in dec.items():
                if d["action"] == "parent" and p in d.get("parents", {}):
                    df.loc[label, col] = d["parents"][p]
                    df.loc[label, f"{p}_overlap"] = np.nan
            proots = self.roots(p)
            if proots:
                df[col] = _remap(df[col].to_numpy(dtype=np.int64), proots)
        roots = self.roots(name)
        if roots:
            target = pd.Series(df.index, index=df.index)
            target.update(pd.Series(roots))
            df = df[target != 0]
            target = target[target != 0]
            if (target != target.index).any():
                df = _combine(df, target)
        qc = pd.Series("", index=df.index)
        for label, d in dec.items():
            if label in qc.index:
                qc[label] = "ok" if d["action"] == "ok" else "fixed"
        for label, root in roots.items():
            if root and root in qc.index:
                qc[root] = "fixed"
        df["qc"] = qc
        self._cache[key] = df
        return df

    def effective(self, name: str) -> "pd.DataFrame":
        """*name*'s table with every decision applied.

        Rejected objects are dropped; merged objects become one row (sizes
        added, centroids, spreads and intensities combined exactly, boxes
        joined); parent columns follow reassignments and the parents' own
        merges; ``qc`` says ``ok``, ``fixed`` or ``""`` (not reviewed).
        Derived columns: ``n_<child>`` (children counted), ``length_um``,
        ``elongation``, ``axis_<a>`` (shape, see
        :func:`~patchworks._tables.shape_columns`) and, for label images
        with a position rule, ``position`` and its measures (see
        :meth:`positions`).
        """
        if name in self._cache:
            return self._cache[name]
        from ._tables import shape_columns

        df = self._corrected(name).copy()
        for child in self.children[name]:
            ce = self._corrected(child)
            counts = ce[f"{name}_id"].value_counts()
            df[f"n_{child}"] = (
                counts.reindex(df.index).fillna(0).astype(np.int64)
            )
        meta = self.meta[name]
        for col, values in shape_columns(
            df, meta.get("axes") or "", meta.get("pixel_size")
        ).items():
            df[col] = values
        if name in self.position:
            for col, values in self.positions(name, df).items():
                df[col] = values
        for child in self.children[name]:
            if (
                child in self.position
                and self.position[child]["parent"] == name
            ):
                pos = self.effective(child)
                table = (
                    pos[pos[f"{name}_id"] != 0]
                    .groupby([f"{name}_id", "position"])
                    .size()
                    .unstack(fill_value=0)
                )
                for cls in POSITIONS:
                    counts = table[cls] if cls in table else None
                    df[f"n_{child}_{cls}"] = (
                        counts.reindex(df.index).fillna(0).astype(np.int64)
                        if counts is not None
                        else 0
                    )
        self._cache[name] = df
        return df

    def positions(
        self, name: str, df: "pd.DataFrame | None" = None
    ) -> dict[str, Any]:
        """Where each object of *name* sits in its parent.

        Per parent object, an apical axis: a fixed direction (``"+z"``:
        apical is up the z axis), or pointing away from the parent's
        children of another label image (``"nuclei_labels"``: away from
        the nucleus, for epithelia whose nuclei sit basally), or towards
        them (``"towards:<name>"``). Each object's **base** is its end
        nearer the parent's centre (a cilium grows out from its base).

        The class is the parent surface the base is nearest to -- top
        (``apical``), bottom (``basal``) or side wall (``lateral``) --
        each depth measured relative to the parent's size in that
        direction; ``central`` when deeper than ``central_depth`` (default
        0.5, i.e. half-way) from all three. The parent's shape is taken
        from its moments (an equivalent cylinder), so this is a
        classification, not a surface distance.

        Returns columns ``position`` (the class; ``outside`` with no
        parent; ``unknown`` when the axis cannot be told, e.g. no nucleus),
        ``position_axial`` (-1 basal .. +1 apical, in half-heights of the
        parent along its axis), ``position_radial`` (0 on the axis .. 1 at
        the side) and ``angle_to_axis_deg`` (0: along the apical axis, 90:
        across it). A reviewer's ``position`` decision overrides the class.
        """
        from ._tables import physical_centroids, physical_cov

        rule = self.position[name]
        parent = rule["parent"]
        df = self._corrected(name) if df is None else df
        n = len(df)
        out: dict[str, Any] = {
            "position": np.full(n, "unknown", dtype=object),
            "position_axial": np.full(n, np.nan),
            "position_radial": np.full(n, np.nan),
            "angle_to_axis_deg": np.full(n, np.nan),
        }
        if parent not in self.parents[name] or not n:
            return out
        axes = self.meta[name].get("axes") or ""
        size = self.meta[name].get("pixel_size")
        pe = self._corrected(parent)
        pid = df[f"{parent}_id"].to_numpy(dtype=np.int64)
        spread = [f"cov_{a}{a}" for a in axes]
        missing = [
            t
            for t, d in ((name, df), (parent, pe))
            if not set(spread) <= set(d)
        ]
        if missing:
            logger.warning(
                "no spread (cov_*) columns in the table(s) of %s -- written "
                "by an older patchworks; recompute them (patchworks tables "
                "%s) to classify positions",
                ", ".join(missing),
                self.store,
            )
            out["position"][pid == 0] = "outside"
            return out
        out["position"][pid == 0] = "outside"
        inside = np.flatnonzero(np.isin(pid, pe.index.to_numpy()))
        if not inside.size:
            return out
        prow = pe.index.get_indexer(pid[inside])
        centre = physical_centroids(pe, axes, size)[prow]
        pcov = physical_cov(pe, axes, size)[prow]
        axis = self._apical_axes(name, rule, pe, axes, size)[prow]

        c = physical_centroids(df, axes, size)[inside]
        w, v = np.linalg.eigh(physical_cov(df, axes, size)[inside])
        main = v[:, :, -1]
        half = np.sqrt(12 * np.clip(w[:, -1], 0, None)) / 2
        ends = np.stack([c + half[:, None] * main, c - half[:, None] * main])
        near = np.argmin(np.linalg.norm(ends - centre, axis=2), axis=0)
        base = ends[near, np.arange(inside.size)]

        rel = base - centre
        along = np.einsum("ij,ij->i", rel, axis)
        var_a = np.einsum("ij,ijk,ik->i", axis, pcov, axis)
        axial = along / np.sqrt(3 * np.clip(var_a, 1e-12, None))
        perp = rel - along[:, None] * axis
        across = np.sqrt(
            2 * np.clip(np.trace(pcov, axis1=1, axis2=2) - var_a, 1e-12, None)
        )
        radial = np.linalg.norm(perp, axis=1) / across
        angle = np.degrees(
            np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", main, axis)), 0, 1))
        )
        # The surface the base is nearest to, each depth relative to the
        # parent's size in that direction: top, bottom or side wall. Deeper
        # than `central_depth` from all three: central.
        depth = np.stack([1 - axial, 1 + axial, 1 - radial], 1)
        cls = np.array(["apical", "basal", "lateral"], dtype=object)[
            np.argmin(depth, axis=1)
        ]
        cls[depth.min(axis=1) > float(rule.get("central_depth", 0.5))] = (
            "central"
        )
        known = np.isfinite(axis).all(axis=1)
        cls[~known] = "unknown"
        out["position"][inside] = cls
        out["position_axial"][inside] = np.where(known, axial, np.nan)
        out["position_radial"][inside] = np.where(known, radial, np.nan)
        out["angle_to_axis_deg"][inside] = np.where(known, angle, np.nan)
        for label, d in self.decisions[name].items():
            if d["action"] == "position" and label in df.index:
                out["position"][df.index.get_loc(label)] = d["position"]
        return out

    def _apical_axes(self, name, rule, pe, axes, size) -> np.ndarray:
        """Unit apical direction per parent object (NaN: undetermined)."""
        from ._tables import physical_centroids, physical_cov

        spec = str(rule.get("apical", ""))
        n = len(pe)
        if spec in _AXES_SIGNS:
            ax, sign = _AXES_SIGNS[spec]
            if ax not in axes:
                raise ValueError(f"apical {spec!r}: {name} has axes {axes!r}")
            vec = np.zeros(len(axes))
            vec[axes.index(ax)] = sign
            return np.tile(vec, (n, 1))
        towards = spec.startswith("towards:")
        ref = spec.split(":", 1)[1] if towards else spec
        parent = rule["parent"]
        if ref not in self.tables or parent not in self.parents.get(ref, []):
            # Not an error: the relation may simply not be computed yet (a
            # concurrent relate job). Positions stay "unknown"; the rest of
            # the workbook is still worth writing.
            logger.warning(
                "apical %r for %s: needs %r related to %r (its objects inside "
                "the parent objects), or a direction like '+z'; positions "
                "left unknown",
                spec,
                name,
                ref,
                parent,
            )
            return np.full((n, len(axes)), np.nan)
        re = self._corrected(ref)
        re = re[re[f"{parent}_id"].isin(pe.index)]
        weights = re["area_voxels"].to_numpy(dtype=float)
        rc = physical_centroids(re, axes, size) * weights[:, None]
        import pandas as pd

        sums = pd.DataFrame(rc).groupby(re[f"{parent}_id"].to_numpy()).sum()
        wsum = pd.Series(weights).groupby(re[f"{parent}_id"].to_numpy()).sum()
        refc = (sums.div(wsum, axis=0)).reindex(pe.index).to_numpy()
        centre = physical_centroids(pe, axes, size)
        vec = centre - refc if not towards else refc - centre
        norm = np.linalg.norm(vec, axis=1)
        scale = np.sqrt(
            np.trace(physical_cov(pe, axes, size), axis1=1, axis2=2)
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            vec = vec / norm[:, None]
        vec[~(norm > 0.1 * scale)] = np.nan  # reference at the centre: no axis
        return vec

    # -- flags and queues ----------------------------------------------------

    def min_overlap(self, child: str, parent: str) -> float:
        for rule in (self._min_overlap, self._recorded_overlap):
            if isinstance(rule, Mapping):
                value = (rule.get(child) or {}).get(parent)
                if value is not None:
                    return float(value)
            elif rule is not None:
                return float(rule)
        return 0.5

    def flag_table(self, name: str) -> "pd.DataFrame":
        """Flagged objects of *name*: ``score`` and ``partner`` (-1: none),
        indexed by label, most suspicious first.

        Cached until the next decision: the panel asks several times per
        key press, and a poor segmentation can flag a large share of
        hundreds of thousands of objects.
        """
        if name not in self._flag_cache:
            rows = self._flag_rows(name)
            g = rows.groupby("label")
            table = (g["score"].max() + 0.1 * (g.size() - 1)).to_frame("score")
            table["partner"] = g["partner"].max()
            table = table.sort_index().sort_values(
                "score", ascending=False, kind="stable"
            )
            self._flag_cache[name] = (table, rows)
        return self._flag_cache[name][0]

    def reasons(self, name: str, label: int) -> list[str]:
        """Why object *label* of *name* is flagged (empty if it isn't)."""
        self.flag_table(name)
        rows = self._flag_cache[name][1]
        return rows.loc[rows["label"] == int(label), "reason"].tolist()

    def flags(self, name: str) -> list[Flag]:
        """Every flagged object of *name* with its reasons, most suspicious
        first. See :meth:`flag_table` for the fast form."""
        table = self.flag_table(name)
        rows = self._flag_cache[name][1]
        by_label = rows.groupby("label")["reason"].apply(list)
        return [
            Flag(
                int(label),
                by_label[label],
                float(r.score),
                None if r.partner < 0 else int(r.partner),
            )
            for label, r in zip(table.index, table.itertuples())
        ]

    def _flag_rows(self, name: str) -> "pd.DataFrame":
        """One row per (object, reason): label, reason, score, partner."""
        import pandas as pd

        eff = self.effective(name)
        ids = eff.index.to_numpy()
        parts = []

        def rule_(mask, reasons, scores, partners=None):
            labels = ids[mask]
            if labels.size:
                parts.append(
                    pd.DataFrame(
                        {
                            "label": labels.astype(np.int64),
                            "reason": list(reasons),
                            "score": np.broadcast_to(
                                np.asarray(scores, float), labels.shape
                            ),
                            "partner": (
                                -1
                                if partners is None
                                else np.asarray(partners, np.int64)
                            ),
                        }
                    )
                )

        for p in self.parents[name]:
            pid = eff[f"{p}_id"].to_numpy()
            ov = eff[f"{p}_overlap"].to_numpy(dtype=float)
            limit = self.min_overlap(name, p)
            nice = _nice(p)
            orphan = pid == 0
            by_distance = f"{p}_distance_um" in eff
            rule_(
                orphan,
                [
                    f"not inside or near any {nice}"
                    if by_distance
                    else f"not inside any {nice}"
                ]
                * int(orphan.sum()),
                3.0,
            )
            gap = (
                eff[f"{p}_distance_um"].to_numpy(dtype=float)
                if by_distance
                else np.zeros(len(eff))
            )
            near = (pid != 0) & (gap > 0)
            rule_(
                near,
                (
                    f"outside {nice} #{q}, {g:.2g} µm away"
                    for q, g in zip(pid[near], gap[near])
                ),
                1.5,
            )
            weak = (pid != 0) & (ov < limit) & ~(gap > 0)
            rule_(
                weak,
                (
                    f"only {v:.0%} inside {nice} #{q}"
                    for v, q in zip(ov[weak], pid[weak])
                ),
                2.0 + (limit - ov[weak]),
            )
        for child, (lo, hi) in self.expect.get(name, {}).items():
            if f"n_{child}" not in eff:
                continue
            n = eff[f"n_{child}"].to_numpy()
            off = (n < lo) | (n > hi)
            want = f"{lo}" if lo == hi else f"{lo}-{hi}"
            rule_(
                off,
                (f"{k} {_nice(child)} (expected {want})" for k in n[off]),
                2.0,
            )
        if name in self.position and "position" in eff:
            rule = self.position[name]
            unknown = (eff["position"] == "unknown").to_numpy()
            ref = str(rule.get("apical", "")).removeprefix("towards:")
            rule_(
                unknown,
                [
                    f"position unclear: its {_nice(rule['parent'])} has no "
                    f"{_nice(ref)} to orient it"
                ]
                * int(unknown.sum()),
                1.2,
            )
        if len(eff) >= 20:
            logv = np.log(eff["area_voxels"].to_numpy(dtype=float))
            med = np.median(logv)
            mad = 1.4826 * np.median(np.abs(logv - med))
            if mad > 0:
                z = (logv - med) / mad
                odd = np.abs(z) > 3.5
                rule_(
                    odd,
                    (
                        f"unusually large ({math.exp(lv - med):.1f}x the median)"
                        if zz > 0
                        else f"unusually small ({math.exp(lv - med):.2f}x the median)"
                        for zz, lv in zip(z[odd], logv[odd])
                    ),
                    1.0 + np.minimum(np.abs(z[odd]) / 10, 1.0),
                )
        pairs = self._seam_pairs(name, eff)
        if pairs:
            a, b, axes = (np.array(x) for x in zip(*pairs))
            for one, other in ((a, b), (b, a)):
                mask = np.isin(ids, one)
                order = pd.Series(other, index=one)
                order = order[~order.index.duplicated()]
                partners = order.reindex(ids[mask]).to_numpy()
                ax = pd.Series(axes, index=one)
                ax = ax[~ax.index.duplicated()].reindex(ids[mask]).to_numpy()
                rule_(
                    mask,
                    (
                        f"meets #{q} exactly at a tile seam ({x}): "
                        "one object split?"
                        for q, x in zip(partners, ax)
                    ),
                    2.5,
                    partners,
                )
        if not parts:
            return pd.DataFrame(
                {
                    "label": np.empty(0, np.int64),
                    "reason": [],
                    "score": np.empty(0),
                    "partner": np.empty(0, np.int64),
                }
            )
        return pd.concat(parts, ignore_index=True)

    def _seam_pairs(self, name: str, eff: "pd.DataFrame"):
        """Pairs of objects that meet face to face on a tile boundary.

        Two objects, one ending on the seam and the other starting there,
        whose footprints across the seam coincide: either two neighbours
        pressed together, or one object the tiling cut in two. Worth a look
        either way; from bounding boxes alone, so no voxels are read.
        """
        from scipy.spatial import cKDTree

        meta = self.meta[name]
        tile = meta.get("tile_shape")
        axes = meta.get("axes") or ""
        if not tile or len(eff) < 2 or not axes:
            return []
        shape = meta["labels"]["shape"]
        tile = list(tile)[-len(axes) :]
        tol = self.seam_tolerance
        lo = {ax: eff[f"bbox_min_{ax}"].to_numpy() for ax in axes}
        hi = {ax: eff[f"bbox_max_{ax}"].to_numpy() + 1 for ax in axes}
        ids = eff.index.to_numpy()
        pairs = []
        for k, ax in enumerate(axes):
            t = int(tile[k])
            if t <= 0 or t >= shape[k]:
                continue
            others = [a for a in axes if a != ax]
            end_s = np.round(hi[ax] / t) * t
            beg_s = np.round(lo[ax] / t) * t
            ok_m = (
                (np.abs(hi[ax] - end_s) <= tol)
                & (end_s > 0)
                & (end_s < shape[k])
            )
            ok_p = (
                (np.abs(lo[ax] - beg_s) <= tol)
                & (beg_s > 0)
                & (beg_s < shape[k])
            )
            for s in np.intersect1d(end_s[ok_m], beg_s[ok_p]):
                m = np.flatnonzero(ok_m & (end_s == s))
                p = np.flatnonzero(ok_p & (beg_s == s))
                if not m.size or not p.size:
                    continue

                def box(idx):
                    b0 = np.stack([lo[a][idx] for a in others], 1).astype(float)
                    b1 = np.stack([hi[a][idx] for a in others], 1).astype(float)
                    return b0, b1

                m0, m1 = box(m)
                p0, p1 = box(p)
                pc = (p0 + p1) / 2
                reach = np.linalg.norm(m1 - m0, axis=1) / 2 + (
                    np.linalg.norm(p1 - p0, axis=1).max() / 2
                )
                tree = cKDTree(pc)
                for i, js in enumerate(
                    tree.query_ball_point((m0 + m1) / 2, reach)
                ):
                    if not js:
                        continue
                    js = np.asarray(js)
                    inter = np.prod(
                        np.clip(
                            np.minimum(m1[i], p1[js])
                            - np.maximum(m0[i], p0[js]),
                            0,
                            None,
                        ),
                        axis=1,
                    )
                    small = np.minimum(
                        np.prod(m1[i] - m0[i]), np.prod(p1[js] - p0[js], axis=1)
                    )
                    exact = (hi[ax][m[i]] == s) | (lo[ax][p[js]] == s)
                    for j in js[(inter >= 0.5 * small) & exact]:
                        if ids[m[i]] != ids[p[j]]:
                            pairs.append((int(ids[m[i]]), int(ids[p[j]]), ax))
        return pairs

    def queue(self, name: str, mode: str = "flagged") -> list[int]:
        """Objects still to review, in review order.

        ``flagged``: flagged objects, most suspicious first. ``random``: a
        fixed random order over every object (the error estimate). ``all``:
        every object by id.
        """
        done = self.decisions[name]
        if mode == "flagged":
            return [
                int(x) for x in self.flag_table(name).index if x not in done
            ]
        if mode == "random":
            return [x for x in self.random_order(name) if x not in done]
        if mode == "all":
            return [int(x) for x in self.effective(name).index if x not in done]
        raise ValueError(f"queue must be one of {QUEUES}, got {mode!r}")

    def random_order(self, name: str) -> list[int]:
        ids = self.tables[name].index.to_numpy()
        rng = np.random.default_rng(self.seed[name])
        return [int(x) for x in ids[rng.permutation(ids.size)]]

    def summary(self, name: str) -> dict[str, Any]:
        """Counts of objects, flags and decisions, and the error estimate.

        The estimate uses the longest run of the random order that has been
        fully reviewed (from any queue: a decision is the truth about that
        object however it came up), so it stays unbiased.
        """
        dec = self.decisions[name]
        # A segmentation error: rejected, joined or re-parented. A position
        # correction is a classification fix, counted on its own.
        wrong = {
            k
            for k, d in dec.items()
            if d["action"] in ("wrong", "merge", "parent")
        }
        sample = 0
        errors = 0
        for label in self.random_order(name):
            if label not in dec:
                break
            sample += 1
            errors += label in wrong
        flagged = self.flag_table(name).index
        lo, hi = wilson_interval(errors, sample)
        return {
            "objects": int(len(self.tables[name])),
            "flagged": len(flagged),
            "flagged_open": sum(int(x) not in dec for x in flagged),
            "reviewed": len(dec),
            "ok": sum(d["action"] == "ok" for d in dec.values()),
            "wrong": sum(d["action"] == "wrong" for d in dec.values()),
            "fixed": sum(
                d["action"] in ("parent", "merge") for d in dec.values()
            ),
            "position_fixed": sum(
                d["action"] == "position" for d in dec.values()
            ),
            "sample": sample,
            "sample_errors": errors,
            "error_rate": errors / sample if sample else None,
            "error_ci": (lo, hi) if sample else None,
        }

    # -- outputs -------------------------------------------------------------

    def export(self, out_dir: Union[str, Path], fmt: str = "csv") -> list[Path]:
        """Write every corrected table to *out_dir*, one file per label image.

        ``csv`` files load straight into napari-chunked-regionprops
        ("Reload previous results"). ``xlsx`` falls back to csv for a table
        longer than Excel allows. ``parquet`` needs pyarrow.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        written = []
        for name in self.names:
            df = self.effective(name)
            kind = fmt
            if fmt == "xlsx" and len(df) + 1 > EXCEL_MAX_ROWS:
                logger.warning(
                    "%s has %d objects, more than an Excel sheet holds; "
                    "writing csv instead",
                    name,
                    len(df),
                )
                kind = "csv"
            path = out / f"{name}.{kind}"
            if kind == "csv":
                df.to_csv(path)
            elif kind == "xlsx":
                df.to_excel(path, sheet_name=name[:31])
            elif kind == "parquet":
                df.to_parquet(path)
            else:
                raise ValueError(
                    f"format must be csv, xlsx or parquet: {fmt!r}"
                )
            written.append(path)
        return written

    def relation_workbook(
        self, child: str, parent: str, path: Union[str, Path]
    ) -> Path:
        """Write the *child* -> *parent* workbook from the corrected tables.

        Sheet *child*: each object, its parent, overlap and review status.
        Sheet *parent*: each parent object and how many children it holds.
        Written as two csv files (``<stem>_<name>.csv``) when a sheet would
        exceed Excel's row limit.
        """
        path = Path(path)
        ce = self.effective(child)
        pe = self.effective(parent)
        a = ce[
            [f"{parent}_id", f"{parent}_overlap_voxels", f"{parent}_overlap"]
        ]
        a = a.rename(
            columns={
                f"{parent}_id": f"{parent}_id",
                f"{parent}_overlap_voxels": "overlap_voxels",
                f"{parent}_overlap": "overlap_fraction",
            }
        ).copy()
        a[f"{parent}_id"] = a[f"{parent}_id"].where(a[f"{parent}_id"] != 0)
        if (
            "position" in ce
            and self.position.get(child, {}).get("parent") == parent
        ):
            a["position"] = ce["position"]
        a["qc"] = ce["qc"]
        a.index.name = f"{child}_id"
        grouped = ce[ce[f"{parent}_id"] != 0].groupby(f"{parent}_id")
        b = pe[[]].copy()
        b[f"{child}_count"] = (
            grouped.size().reindex(pe.index).fillna(0).astype(np.int64)
        )
        b["total_overlap_voxels"] = (
            grouped[f"{parent}_overlap_voxels"]
            .sum()
            .reindex(pe.index)
            .fillna(0)
            .astype(np.int64)
        )
        for cls in POSITIONS:
            col = f"n_{child}_{cls}"
            if col in pe:
                b[f"{child}_{cls}"] = pe[col]
        b["qc"] = pe["qc"]
        b.index.name = f"{parent}_id"
        if max(len(a), len(b)) + 1 > EXCEL_MAX_ROWS:
            logger.warning(
                "%s -> %s exceeds Excel's row limit; writing csv files",
                child,
                parent,
            )
            a.to_csv(path.with_name(f"{path.stem}_{child}.csv"))
            b.to_csv(path.with_name(f"{path.stem}_{parent}.csv"))
            return path.with_name(f"{path.stem}_{child}.csv")
        import pandas as pd

        with pd.ExcelWriter(path) as xl:
            a.to_excel(xl, sheet_name=child[:31])
            b.to_excel(xl, sheet_name=parent[:31])
        return path

    def write_reviewed_labels(
        self, name: str, *, suffix: str = "_reviewed"
    ) -> str:
        """Write ``labels/<name><suffix>``: the labels with the decisions
        applied to the voxels (rejected objects erased, merged objects one
        id), plus its corrected table.

        The corrected tables and workbooks never need this; it is for figures
        and for tools that read label images only.
        """
        import dask.array as da

        from ._provenance import PROVENANCE_KEY
        from ._tables import _level0, write_table
        from .plugins.ome_zarr import write_labels

        group = zarr.open_group(_label_group(self.store, name), mode="r")
        level0 = da.from_zarr(_level0(group))
        roots = self.roots(name)
        top = int(max(self.tables[name].index.max(), max(roots, default=0)))
        lut = np.arange(top + 1, dtype=level0.dtype)
        for label, root in roots.items():
            lut[label] = root
        fixed = level0.map_blocks(lambda b: lut[b], dtype=level0.dtype)
        prov = dict(group.attrs).get(PROVENANCE_KEY) or {}
        settings = prov.get("settings") or {}
        record = dict(prov)
        record["review"] = {
            "from": name,
            "decisions": len(self.decisions[name]),
        }
        out = f"{name}{suffix}"
        write_labels(
            self.store,
            fixed,
            name=out,
            overwrite=True,
            level=int(settings.get("level") or 0),
            progress=False,
            provenance=record,
        )
        eff = self.effective(name).drop(columns=["qc"])
        cols = {"label": eff.index.to_numpy()}
        cols.update({c: eff[c].to_numpy() for c in eff.columns})
        meta = {
            k: v
            for k, v in self.meta[name].items()
            if k not in ("labels", "columns", "parents")
        }
        write_table(
            _label_group(self.store, out),
            cols,
            attrs={**meta, "reviewed_from": name},
        )
        return out


def _remap(values: np.ndarray, mapping: Mapping[int, int]) -> np.ndarray:
    """*values* with every key of *mapping* replaced by its value."""
    keys = np.fromiter(mapping, np.int64, len(mapping))
    new = np.fromiter(mapping.values(), np.int64, len(mapping))
    order = np.argsort(keys)
    keys, new = keys[order], new[order]
    pos = np.clip(np.searchsorted(keys, values), 0, keys.size - 1)
    hit = keys[pos] == values
    out = values.copy()
    out[hit] = new[pos[hit]]
    return out


def _range(rng: Any) -> tuple[int, int]:
    if isinstance(rng, (int, np.integer)):
        return int(rng), int(rng)
    lo, hi = list(rng)
    return int(lo), int(hi)


def _nice(name: str) -> str:
    """``cyto_labels`` -> ``cyto`` for messages."""
    return name[: -len("_labels")] if name.endswith("_labels") else name


def _combine(df: "pd.DataFrame", target: "pd.Series") -> "pd.DataFrame":
    """Collapse rows sharing a *target* id into one, exactly where possible."""
    import pandas as pd

    n = df["area_voxels"].astype(float)
    groups = target.to_numpy()
    work = df.copy()
    work["_g"] = groups
    agg: dict[str, Any] = {}
    for col in df.columns:
        if col.startswith(("area_", "n_")) or col.endswith("_overlap_voxels"):
            agg[col] = "sum"
        elif col.startswith("bbox_min_"):
            agg[col] = "min"
        elif col.startswith("bbox_max_"):
            agg[col] = "max"
        elif col.startswith("cov_"):
            agg[col] = "first"  # replaced below, once centroids are known
        elif col.startswith(("centroid_", "mean_intensity_")):
            work[col] = df[col] * n
            agg[col] = "sum"
        elif col.startswith("std_intensity_"):
            mean = df["mean_intensity_" + col[len("std_intensity_") :]]
            work[col] = (df[col] ** 2 + mean**2) * n
            agg[col] = "sum"
        else:  # parent ids, overlaps: the surviving object's own
            agg[col] = "first"
    work["_n"] = n
    agg["_n"] = "sum"
    # "first" must be the target's own row: put targets first.
    work["_own"] = work.index.to_numpy() == groups
    work = work.sort_values("_own", ascending=False, kind="stable")
    out = work.groupby("_g").agg(agg)
    total = out.pop("_n")
    for col in df.columns:
        if col.startswith(("centroid_", "mean_intensity_")):
            out[col] = out[col] / total
    for col in df.columns:
        if col.startswith("std_intensity_"):
            mean = out["mean_intensity_" + col[len("std_intensity_") :]]
            out[col] = np.sqrt(np.maximum(out[col] / total - mean**2, 0))
    covs = [c for c in df.columns if c.startswith("cov_")]
    if covs:
        # Parallel-axis theorem: each part's spread plus its offset from the
        # merged centroid (deviations, not raw products: no cancellation).
        dev = {
            ax: df[f"centroid_{ax}"].to_numpy()
            - out.loc[groups, f"centroid_{ax}"].to_numpy()
            for ax in {c[4] for c in covs} | {c[5] for c in covs}
        }
        for col in covs:
            a, b = col[4], col[5]
            part = n.to_numpy() * (df[col].to_numpy() + dev[a] * dev[b])
            out[col] = (
                pd.Series(part, index=df.index).groupby(groups).sum() / total
            )
    merged = target.value_counts()
    for col in df.columns:
        if col.endswith("_overlap") and not col.endswith("_overlap_voxels"):
            out.loc[merged[merged > 1].index, col] = np.nan
    out.index.name = df.index.name
    return out[df.columns]
