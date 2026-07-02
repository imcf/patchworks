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
patchworks.label_relations and written as a CSV in work_dir.
"""

from __future__ import annotations

import argparse
import csv
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

    from patchworks import label_relations

    for rel in relations:
        a_name, b_name = rel["a"], rel["b"]
        out_path = Path(work_dir) / rel.get(
            "output", f"{a_name}_to_{b_name}.csv"
        )
        print(f"[run_multi] relating {a_name} -> {b_name} …", flush=True)
        a = da.from_zarr(image_store, component=f"labels/{a_name}/0")
        b = da.from_zarr(image_store, component=f"labels/{b_name}/0")
        table = label_relations(a, b)

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    f"{a_name}_id",
                    f"{b_name}_id",
                    "overlap_voxels",
                    "overlap_fraction",
                ]
            )
            for a_id, m in table.items():
                writer.writerow(
                    [
                        a_id,
                        m["match"],
                        m["overlap_voxels"],
                        m["overlap_fraction"],
                    ]
                )
        print(f"[run_multi] wrote {out_path} ({len(table)} rows)", flush=True)


if __name__ == "__main__":
    main()
