"""Compute label_relations for configured pairs and write .xlsx.

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
from pathlib import Path


def _num_chunks(arr: "da.Array") -> int:  # noqa: F821 - dask imported lazily by callers
    """Total chunk count of a dask array, for picking the coarser side to rechunk to."""
    return math.prod(len(c) for c in arr.chunks)


def _label_ids(image_store: str, name: str) -> list[int]:
    """Ids present in a label image, without scanning the volume.

    The merge writes n_objects/sequential_labels into the label group's
    attrs precisely so consumers don't have to re-derive the id set; the ids
    are 1..n_objects by construction. Fall back to a full scan only for a
    label group written before those attrs existed.
    """
    import dask.array as da
    import zarr

    attrs = dict(zarr.open_group(f"{image_store}/labels/{name}").attrs)
    if attrs.get("sequential_labels") and attrs.get("n_objects") is not None:
        return list(range(1, int(attrs["n_objects"]) + 1))
    print(
        f"[relate] {name}: no n_objects attr, falling back to a full scan "
        "for its id set",
        flush=True,
    )
    arr = da.from_zarr(image_store, component=f"labels/{name}/0")
    return sorted(int(x) for x in da.unique(arr[arr > 0]).compute())


def run_relations(
    work_dir: str, image_store: str, relations: list[dict]
) -> None:
    """Compute and write every configured relation pair as an .xlsx workbook.

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
    import openpyxl

    from patchworks import label_relations

    for rel in relations:
        a_name, b_name = rel["a"], rel["b"]
        out_path = Path(work_dir) / rel.get(
            "output", f"{a_name}_to_{b_name}.xlsx"
        )
        print(f"[relate] relating {a_name} -> {b_name} …", flush=True)
        a = da.from_zarr(image_store, component=f"labels/{a_name}/0")
        b = da.from_zarr(image_store, component=f"labels/{b_name}/0")

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

        # label_relations() only returns a-objects that touch a b-object.
        # Pull the full id sets so unmatched a-objects (zero overlap) and
        # b-objects with no matches at all still get a row -- otherwise
        # they'd silently vanish instead of counting as zero.
        a_ids = _label_ids(image_store, a_name)
        b_ids = _label_ids(image_store, b_name)

        per_b = {b_id: {"count": 0, "overlap_voxels": 0} for b_id in b_ids}
        for m in table.values():
            agg = per_b.get(m["match"])
            if agg is not None:
                agg["count"] += 1
                agg["overlap_voxels"] += m["overlap_voxels"]

        wb = openpyxl.Workbook()
        ws_a = wb.active
        ws_a.title = a_name[:31]  # Excel sheet-name length limit
        ws_a.append(
            [
                f"{a_name}_id",
                f"{b_name}_id",
                "overlap_voxels",
                "overlap_fraction",
            ]
        )
        for a_id in a_ids:
            m = table.get(a_id)
            if m is None:
                ws_a.append([a_id, None, 0, 0])  # no overlap -- still counted
            else:
                ws_a.append(
                    [
                        a_id,
                        m["match"],
                        m["overlap_voxels"],
                        m["overlap_fraction"],
                    ]
                )

        ws_b = wb.create_sheet(title=b_name[:31])
        ws_b.append([f"{b_name}_id", f"{a_name}_count", "total_overlap_voxels"])
        for b_id in b_ids:
            agg = per_b[b_id]
            ws_b.append([b_id, agg["count"], agg["overlap_voxels"]])

        wb.save(out_path)
        print(
            f"[relate] wrote {out_path} "
            f"({len(a_ids)} {a_name}, {len(b_ids)} {b_name})",
            flush=True,
        )


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
