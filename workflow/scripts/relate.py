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
    if not out_path.exists():
        return False
    try:
        out_mtime = out_path.stat().st_mtime
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
    import dask.array as da

    from patchworks import label_relations
    from patchworks._review import Review, review_updated
    from patchworks._tables import (
        compute_table,
        has_table,
        is_stale,
        relate_tables,
    )

    for rel in relations:
        a_name, b_name = rel["a"], rel["b"]
        out_path = Path(work_dir) / rel.get(
            "output", f"{a_name}_to_{b_name}.xlsx"
        )
        if _relation_up_to_date(work_dir, a_name, b_name, out_path):
            reviewed = max(
                review_updated(image_store, n) or 0 for n in (a_name, b_name)
            )
            if reviewed <= out_path.stat().st_mtime:
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
        print(f"[relate] relating {a_name} -> {b_name} …", flush=True)
        started = time.monotonic()
        a = da.from_zarr(image_store, component=f"labels/{a_name}/0")
        b = da.from_zarr(image_store, component=f"labels/{b_name}/0")

        # What this pair actually costs, before it starts costing it. A
        # relation runs for hours on a real dataset, and until now the log
        # said only "relating a -> b" and then nothing at all until it
        # finished -- indistinguishable from a hang.
        for name, arr in ((a_name, a), (b_name, b)):
            print(
                f"[relate]   {name}: shape={arr.shape} chunks="
                f"{tuple(c[0] for c in arr.chunks)} "
                f"({_num_chunks(arr):,} chunks, {_n_objects(image_store, name)})",
                flush=True,
            )

        # label_relations() requires matching chunk layouts (it walks both
        # arrays block-by-block at the same index) but two configs are free
        # to have segmented at different tile_shape -- e.g. one already
        # published before the other's config changed, or a cheaper method
        # naturally sized its tile differently. Same shape, different
        # chunking is a normal dask op (extra I/O reading across misaligned
        # source chunks, not a correctness issue), so rechunk the finer side
        # to the coarser one here rather than require identical tile_shape
        # across every config up front.
        if a.chunks != b.chunks:
            a_n, b_n = _num_chunks(a), _num_chunks(b)
            if a_n <= b_n:
                print(
                    f"[relate] {a_name} chunks {a.chunks} != {b_name} "
                    f"chunks {b.chunks}; rechunking {b_name} to match "
                    f"{a_name} (fewer chunks)",
                    flush=True,
                )
                b = b.rechunk(a.chunks)
            else:
                print(
                    f"[relate] {a_name} chunks {a.chunks} != {b_name} "
                    f"chunks {b.chunks}; rechunking {a_name} to match "
                    f"{b_name} (fewer chunks)",
                    flush=True,
                )
                a = a.rechunk(b.chunks)

        table = label_relations(a, b)
        print(
            f"[relate] {a_name} -> {b_name}: matched {len(table):,} object(s) "
            f"in {(time.monotonic() - started) / 60:.1f}m",
            flush=True,
        )

        # The object tables carry every object -- unmatched ones included --
        # and the review decisions; the workbook is written from them, so a
        # correction made in `patchworks review` shows up in it.
        for name in (a_name, b_name):
            group = f"{image_store}/labels/{name}"
            if not has_table(group) or is_stale(group):
                print(
                    f"[relate] measuring {name} for its object table",
                    flush=True,
                )
                compute_table(image_store, name)
        relate_tables(
            image_store,
            a_name,
            b_name,
            matches=table,
            max_distance_um=rel.get("max_distance_um"),
        )
        # Only these two tables: another relate job may be writing others.
        written = Review(image_store, names=[a_name, b_name]).relation_workbook(
            a_name, b_name, out_path
        )
        print(f"[relate] wrote {written}", flush=True)


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
