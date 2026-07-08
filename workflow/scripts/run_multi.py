"""Run several segmentation configs, then relate their labels by overlap.

Usage:
    python scripts/run_multi.py --config config/multi.yaml
    python scripts/run_multi.py --config config/multi.yaml --profile profile/slurm
    python scripts/run_multi.py --config config/multi.yaml -n   # dry-run only

See config/multi.yaml and docs/guide/snakemake.md "Running two segmentations"
for the config format. Each listed segmentation config is run as an ordinary
`snakemake --configfile ...` invocation (this script is a thin sequencer, not
a Snakemake rule — the segmentations already namespace their own paths under
work_dir/<label_name>/, so running them one after another here is exactly
equivalent to running each snakemake command by hand). Once all segmentations
finish, each configured relation pair is computed via
patchworks.label_relations and written as an Excel workbook in work_dir,
with two sheets: one row per a-object (unmatched ones included, with an
empty b-id and zeros) and one row per b-object (a-object count + total
overlap, including b-objects with zero matches).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _run_snakemake(
    configfile: Path,
    *,
    workflow_dir: Path,
    profile: str | None,
    cores: int,
    dry_run: bool,
) -> None:
    cmd = [
        "snakemake",
        "-s",
        str(workflow_dir / "Snakefile"),
        "--configfile",
        str(configfile),
    ]
    if profile:
        cmd += ["--workflow-profile", profile]
    else:
        cmd += ["--cores", str(cores), "--rerun-triggers", "mtime"]
    if dry_run:
        cmd += ["-n", "-p"]
    print(f"[run_multi] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=workflow_dir)


def _resolve(workflow_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else workflow_dir / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True, help="multi-segmentation config YAML"
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Snakemake --workflow-profile (e.g. profile/slurm); omit to run locally",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=8,
        help="local run: --cores (ignored with --profile)",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="pass -n -p to every Snakemake run; skips relations",
    )
    args = parser.parse_args()

    workflow_dir = Path(__file__).resolve().parent.parent
    multi_cfg_path = _resolve(workflow_dir, args.config)
    multi_cfg = _load_yaml(multi_cfg_path)

    seg_config_paths = [
        _resolve(workflow_dir, c) for c in multi_cfg["segmentations"]
    ]
    for cfg_path in seg_config_paths:
        _run_snakemake(
            cfg_path,
            workflow_dir=workflow_dir,
            profile=args.profile,
            cores=args.cores,
            dry_run=args.dry_run,
        )

    relations = multi_cfg.get("relations", [])
    if args.dry_run or not relations:
        return

    seg_cfgs = [_load_yaml(p) for p in seg_config_paths]
    work_dirs = {cfg["work_dir"] for cfg in seg_cfgs}
    if len(work_dirs) != 1:
        print(
            f"[run_multi] ERROR: segmentation configs use different work_dir "
            f"({sorted(work_dirs)}); label_relations needs one shared "
            "image.zarr to compare against.",
            file=sys.stderr,
        )
        sys.exit(1)
    work_dir = work_dirs.pop()
    image_store = f"{work_dir}/image.zarr"

    import dask.array as da
    import openpyxl

    from patchworks import label_relations

    for rel in relations:
        a_name, b_name = rel["a"], rel["b"]
        out_path = Path(work_dir) / rel.get(
            "output", f"{a_name}_to_{b_name}.xlsx"
        )
        print(f"[run_multi] relating {a_name} -> {b_name} …", flush=True)
        a = da.from_zarr(image_store, component=f"labels/{a_name}/0")
        b = da.from_zarr(image_store, component=f"labels/{b_name}/0")
        table = label_relations(a, b)

        # label_relations() only returns a-objects that touch a b-object.
        # Pull the full id sets so unmatched a-objects (zero overlap) and
        # b-objects with no matches at all still get a row -- otherwise
        # they'd silently vanish instead of counting as zero.
        a_ids = sorted(int(x) for x in da.unique(a[a > 0]).compute())
        b_ids = sorted(int(x) for x in da.unique(b[b > 0]).compute())

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
            f"[run_multi] wrote {out_path} "
            f"({len(a_ids)} {a_name}, {len(b_ids)} {b_name})",
            flush=True,
        )


if __name__ == "__main__":
    main()
