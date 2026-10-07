"""Relate configured label pairs: object tables, then an .xlsx per pair.

Split out of run_multi.py so this step can be submitted as its own SLURM job
instead of running in-process on the login node. It streams every chunk of
two full-resolution label volumes -- real CPU/IO work, not orchestration --
same reasoning as the occupancy-map fix (see run_multi.py's phase A comment).

Usage (called by run_multi.py under --profile, but also runnable standalone,
e.g. under srun):
    python scripts/relate.py --work-dir /path/to/work_dir \
        --image-store /path/to/work_dir/image.zarr \
        --relations '[{"a": "nuclei_labels", "b": "cyto_labels", "output": "nuclei_to_cyto.xlsx"}]'

Unlike prepare/segment/merge, this doesn't run as a Snakemake rule, so nothing
wires up its own logs/<rule>/*.log by default -- srun just streams output to
whatever invoked it. main() calls start_log() itself instead, writing to
<work_dir>/logs/relate.log (override with --log), the same tee-to-file-and-
stdout behaviour the Snakemake-driven scripts get via _pw.start_log.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path


def _num_chunks(arr: "da.Array") -> int:  # noqa: F821 - dask imported lazily by callers
    """Total chunk count of a dask array, for picking the coarser side to rechunk to."""
    return math.prod(len(c) for c in arr.chunks)


def _n_objects(image_store: str, name: str) -> str:
    """Object count from the label group's attrs, for a log line.

    Read from the metadata the merge already wrote, never by scanning -- a
    progress line that costs a full volume read would defeat its own purpose.
    """
    import zarr

    try:
        attrs = dict(zarr.open_group(f"{image_store}/labels/{name}").attrs)
    except Exception:  # pragma: no cover - a log line must never fail the run
        return "object count unknown"
    n = attrs.get("n_objects")
    return f"{int(n):,} objects" if n is not None else "object count unknown"


def written_outputs(out_path: Path, a_name: str, b_name: str) -> list[Path]:
    """What a relation's workbook was written as: the ``.xlsx``, or -- a
    sheet too long for Excel -- its two csv files (``<stem>_<name>.csv``).

    Only the .xlsx used to count, so every relation of more than a million
    objects was recomputed from scratch on each run: hours each.
    """
    if out_path.exists():
        return [out_path]
    csvs = [
        out_path.with_name(f"{out_path.stem}_{n}.csv") for n in (a_name, b_name)
    ]
    return csvs if all(p.exists() for p in csvs) else []


def _relation_up_to_date(
    work_dir: str, a_name: str, b_name: str, out_path: Path
) -> bool:
    """Whether *out_path* already reflects the current *a_name*/*b_name* labels.

    Mirrors the workflow's existing "delete to force" convention (see
    ``image.zarr`` not being reconverted once it exists): a relation is
    considered current when its workbook is newer than both labels'
    ``labels.done`` merge marker, and stale (or never computed) otherwise.
    Missing markers -- a label group written before merge started recording
    one, or a nonstandard ``image_store`` layout -- are treated as "unknown,
    recompute" rather than raise, since staleness can't be judged without them.
    """
    outputs = written_outputs(out_path, a_name, b_name)
    if not outputs:
        return False
    try:
        out_mtime = min(p.stat().st_mtime for p in outputs)
        for name in (a_name, b_name):
            marker = Path(work_dir) / name / "labels.done"
            if not marker.exists() or marker.stat().st_mtime > out_mtime:
                return False
    except OSError:
        return False
    return True


def _tables_ready(
    image_store: str, a_name: str, b_name: str, max_distance=None
) -> bool:
    """Both tables current, and *a*'s already relating it to *b* (by
    distance too, when asked)."""
    from patchworks._tables import has_table, is_stale, read_columns

    groups = [f"{image_store}/labels/{n}" for n in (a_name, b_name)]
    try:
        if not all(has_table(g) and not is_stale(g) for g in groups):
            return False
        cols = read_columns(groups[0], check=False)
        return f"{b_name}_id" in cols and (
            max_distance is None or f"{b_name}_distance_um" in cols
        )
    except Exception:
        return False


def run_relations(
    work_dir: str, image_store: str, relations: list[dict]
) -> None:
    """Compute and write every configured relation pair as an .xlsx workbook.

    A relation already reflected by an up-to-date workbook (see
    :func:`_relation_up_to_date`) is skipped rather than recomputed -- so
    retrying a partially-failed run (this step has no Snakemake rule of its
    own to track that for it) only redoes what's actually missing or stale.
    Delete the ``.xlsx`` yourself to force a specific relation to recompute.

    Parameters
    ----------
    work_dir : str
        Directory relation workbooks are written into (a relation's
        ``output``, when relative, resolves against this).
    image_store : str
        The shared ``image.zarr`` holding every config's ``labels/<name>``.
    relations : list of dict
        Each ``{"a": ..., "b": ..., "output": ...}`` (``output`` optional,
        defaults to ``<a>_to_<b>.xlsx``), matching ``multi.yaml``'s
        ``relations:`` list.
    """
    from patchworks._review import Review, review_updated
    from patchworks._tables import (
        compute_table,
        has_table,
        is_stale,
        relate_tables,
        relation_current,
    )

    pending: list = []
    for rel in relations:
        a_name, b_name = rel["a"], rel["b"]
        out_path = Path(work_dir) / rel.get(
            "output", f"{a_name}_to_{b_name}.xlsx"
        )
        if _relation_up_to_date(work_dir, a_name, b_name, out_path):
            reviewed = max(
                review_updated(image_store, n) or 0 for n in (a_name, b_name)
            )
            written_at = min(
                p.stat().st_mtime
                for p in written_outputs(out_path, a_name, b_name)
            )
            if reviewed <= written_at:
                print(
                    f"[relate] {out_path} is already up to date with "
                    f"{a_name}/{b_name}; skipping",
                    flush=True,
                )
                continue
            if _tables_ready(
                image_store, a_name, b_name, rel.get("max_distance_um")
            ):
                # Only review decisions changed: the overlaps are already in
                # the tables, so rewrite the workbook without re-reading the
                # labels.
                written = Review(
                    image_store, names=[a_name, b_name]
                ).relation_workbook(a_name, b_name, out_path)
                print(
                    f"[relate] rewrote {written} with the review decisions",
                    flush=True,
                )
                continue
        elif relation_current(
            image_store, a_name, b_name, rel.get("max_distance_um")
        ):
            # The tables already hold this relation for the labels there
            # now (recorded with it): the workbook was deleted, or never
            # written. Rewrite it; the labels need not be read again.
            written = Review(
                image_store, names=[a_name, b_name]
            ).relation_workbook(a_name, b_name, out_path)
            print(
                f"[relate] {a_name} -> {b_name} is already in the tables; "
                f"wrote {written} from them",
                flush=True,
            )
            continue
        pending.append((rel, out_path))

    # One pass per parent: a parent several label images are related to is
    # read once for all of them, and only where some child has labels.
    by_parent: dict[str, list] = {}
    for rel, out_path in pending:
        by_parent.setdefault(rel["b"], []).append((rel, out_path))
    for b_name, group in by_parent.items():
        started = time.monotonic()
        matches = _group_matches(
            image_store, [rel["a"] for rel, _ in group], b_name
        )
        print(
            f"[relate] {', '.join(rel['a'] for rel, _ in group)} -> {b_name}: "
            f"related in {(time.monotonic() - started) / 60:.1f}m",
            flush=True,
        )
        # The object tables carry every object -- unmatched ones included --
        # and the review decisions; the workbooks are written from them, so
        # a correction made in `patchworks review` shows up in them.
        for name in {b_name, *(rel["a"] for rel, _ in group)}:
            g = f"{image_store}/labels/{name}"
            if not has_table(g) or is_stale(g):
                print(
                    f"[relate] measuring {name} for its object table",
                    flush=True,
                )
                compute_table(image_store, name)
        for rel, _ in group:
            relate_tables(
                image_store,
                rel["a"],
                b_name,
                matches=matches[rel["a"]],
                max_distance_um=rel.get("max_distance_um"),
            )
    # Workbooks last: a position rule (cilia placed by the nuclei of their
    # cell) needs its sibling relation in the tables first.
    for rel, out_path in pending:
        a_name, b_name = rel["a"], rel["b"]
        # Only the tables these need: another relate job may write others.
        written = Review(image_store, names=[a_name, b_name]).relation_workbook(
            a_name, b_name, out_path
        )
        print(f"[relate] wrote {written}", flush=True)


def _group_matches(
    image_store: str, children: list[str], b_name: str
) -> dict[str, dict]:
    """label_relations of every child against *b_name*, reading it once.

    A child chunked differently from the parent (segmented at another
    tile_shape) is related on its own, the finer side rechunked to the
    coarser: same shape, different chunking is extra I/O, not an error.
    """
    import dask.array as da
    import zarr

    from patchworks import label_relations
    from patchworks._relations import label_relations_many

    def array(name):
        return zarr.open_array(image_store, path=f"labels/{name}/0", mode="r")

    b = array(b_name)
    print(
        f"[relate]   {b_name}: shape={b.shape} chunks={b.chunks} "
        f"({_n_objects(image_store, b_name)})",
        flush=True,
    )
    same, other = [], []
    for name in children:
        a = array(name)
        print(
            f"[relate]   {name}: chunks={a.chunks} "
            f"({_n_objects(image_store, name)})",
            flush=True,
        )
        (same if a.chunks == b.chunks else other).append(name)
    out = label_relations_many({n: array(n) for n in same}, b) if same else {}
    for name in other:
        a = da.from_zarr(image_store, component=f"labels/{name}/0")
        bb = da.from_zarr(image_store, component=f"labels/{b_name}/0")
        if _num_chunks(a) <= _num_chunks(bb):
            print(
                f"[relate] {name} chunks differ from {b_name}'s; rechunking "
                f"{b_name} to match (fewer chunks)",
                flush=True,
            )
            bb = bb.rechunk(a.chunks)
        else:
            print(
                f"[relate] {name} chunks differ from {b_name}'s; rechunking "
                f"{name} to match (fewer chunks)",
                flush=True,
            )
            a = a.rechunk(bb.chunks)
        out[name] = label_relations(a, bb)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--image-store", required=True)
    parser.add_argument(
        "--relations",
        required=True,
        help=(
            "JSON list of {a, b, output} dicts, matching multi.yaml's "
            "relations:"
        ),
    )
    parser.add_argument(
        "--log",
        default=None,
        help="log file path (default: <work-dir>/logs/relate.log)",
    )
    args = parser.parse_args()

    from _pw import start_log

    log_path = args.log or str(Path(args.work_dir) / "logs" / "relate.log")
    start_log(log_path)
    print(f"[relate] logging to {log_path}", flush=True)

    run_relations(args.work_dir, args.image_store, json.loads(args.relations))


if __name__ == "__main__":
    main()
