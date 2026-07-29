"""Run several segmentation configs, then relate their labels by overlap.

Usage:
    python scripts/run_multi.py --config config/multi.yaml
    python scripts/run_multi.py --config config/multi.yaml --profile profile/slurm
    python scripts/run_multi.py --config config/multi.yaml -n   # dry-run only

See config/multi.yaml and docs/guide/snakemake.md "Running two segmentations"
for the config format.

The conversion runs once up front, then every segmentation config runs
**concurrently** as its own `snakemake --configfile ...` invocation. They
namespace their paths under work_dir/<label_name>/ and so touch disjoint
files; running them together keeps the GPU partition busy instead of idling
through each config's prepare and multi-hour merge in turn. Each gets its own
`--directory` because Snakemake's lock lives in the working directory, not in
the config. A config that fails does not abort its siblings.

Once all segmentations succeed, each configured relation pair is computed via
patchworks.label_relations and written as an Excel workbook in work_dir,
with two sheets: one row per a-object (unmatched ones included, with an
empty b-id and zeros) and one row per b-object (a-object count + total
overlap, including b-objects with zero matches).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _snakemake_cmd(
    configfile: Path,
    *,
    workflow_dir: Path,
    profile: str | None,
    cores: int,
    dry_run: bool,
    state_dir: Path | None = None,
    targets: list[str] | None = None,
    extra: list[str] | None = None,
    jobname_prefix: str | None = None,
    common: Path | None = None,
) -> list[str]:
    """Build one snakemake invocation.

    Every path is absolutised because ``--directory`` moves the working
    directory: each config needs its own ``.snakemake`` state directory, or
    concurrent runs would contend for the same ``.snakemake/locks/``.

    *common*, when given, is passed as the first of two ``--configfile``
    values. Snakemake merges them in order with the later winning, so the
    settings every config shares -- the input, the work_dir, everything
    ``convert`` reads -- live in one file and the per-config file carries only
    what actually differs.
    """
    configfiles = [str(configfile.resolve())]
    if common is not None:
        configfiles.insert(0, str(common.resolve()))
    cmd = [
        "snakemake",
        "-s",
        str(workflow_dir / "Snakefile"),
        "--configfile",
        *configfiles,
    ]
    if state_dir is not None:
        state_dir.mkdir(parents=True, exist_ok=True)
        cmd += ["--directory", str(state_dir.resolve())]
    if profile:
        cmd += ["--workflow-profile", str((workflow_dir / profile).resolve())]
        if jobname_prefix:
            # A SLURM-executor setting, so only valid alongside the profile.
            cmd += ["--slurm-jobname-prefix", jobname_prefix]
    else:
        cmd += ["--cores", str(cores), "--rerun-triggers", "mtime"]
    if dry_run:
        cmd += ["-n", "-p"]
    if extra:
        cmd += extra
    if targets:
        # "--" ends option parsing: --rerun-triggers takes a variable number
        # of values and would otherwise swallow the target path.
        cmd += ["--", *targets]
    return cmd


def slurm_jobname_prefix(label: str) -> str:
    """Sanitise *label* into a SLURM job-name prefix the executor accepts.

    The SLURM executor names every job after its run UUID and refuses a
    ``--job-name`` in ``slurm_extra``, so a prefix is the only way to get
    something recognisable into ``squeue``. It becomes ``<prefix>_<uuid>``,
    which puts the readable part first -- the part that survives truncation
    in a queue listing.

    The executor requires alphanumerics, underscores and hyphens only, at
    most 50 characters, and rejects the whole run otherwise.

    Examples
    --------
    >>> slurm_jobname_prefix("nuclei_labels")
    'pw-nuclei_labels'
    >>> slurm_jobname_prefix("cilia/v2 (test)")
    'pw-cilia-v2--test-'
    """
    safe = re.sub(r"[^A-Za-z0-9_-]", "-", label)
    return f"pw-{safe}"[:50]


def _run(cmd: list[str], workflow_dir: Path) -> int:
    print(f"[run_multi] $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=workflow_dir).returncode


# Exactly the config keys scripts/convert.py reads -- keep the two in step.
# Conversion runs once, in phase A, from the first config, so these have to
# agree across all of them or the disagreement is invisible.
#
# Deliberately NOT here: pyramid_levels / pyramid_downscale. Those are read by
# merge.py, which runs once per config and builds that config's own label
# pyramid, so they may legitimately differ.
_CONVERT_KEYS = (
    "input",
    "sequence_pattern",
    "convert_chunks",
    "shard",
    "reuse_pyramid",
)


def _validate_configs(paths: list[Path], cfgs: list[dict]) -> str:
    """Check the cross-config invariants before anything is submitted.

    These used to surface hours later -- as a shape mismatch from
    label_relations, or not at all when two configs quietly overwrote each
    other's label group. Only ``work_dir`` was checked, and only when
    relations were configured and it was not a dry run.

    Returns
    -------
    str
        The shared ``work_dir``.
    """
    problems = []

    def _spread(key):
        return {p.name: cfg.get(key) for p, cfg in zip(paths, cfgs)}

    work_dirs = {cfg.get("work_dir") for cfg in cfgs}
    if len(work_dirs) != 1:
        problems.append(
            f"configs must share one work_dir (label_relations compares "
            f"against a single image.zarr); got {_spread('work_dir')}"
        )

    for key in ("tile_shape", "level"):
        values = {repr(cfg.get(key)) for cfg in cfgs}
        if len(values) != 1:
            problems.append(
                f"{key} must be identical across configs so the label arrays "
                f"share a chunk layout; got {_spread(key)}"
            )

    # Phase A converts once, from the first config. Anything `convert` reads
    # out of a later config is therefore silently ignored -- someone setting
    # `shard: true` on the second config and watching a million files appear
    # anyway has no way to see why. Refuse instead, and point at common.yaml.
    for key in _CONVERT_KEYS:
        values = {repr(cfg.get(key)) for cfg in cfgs}
        if len(values) != 1:
            problems.append(
                f"{key} affects `convert`, which runs once from the first "
                f"config, so the other values would be silently ignored; got "
                f"{_spread(key)}. Put the settings every config shares in one "
                f"file and point `common:` in multi.yaml at it."
            )

    for path, cfg in zip(paths, cfgs):
        source = str(cfg.get("input", ""))
        if any(ch in source for ch in "*?[") and not cfg.get(
            "sequence_pattern"
        ):
            problems.append(
                f"{path.name}: input {source!r} is a glob over several files "
                "but sequence_pattern is unset, so nothing says which part of "
                "each filename is Z/C/T. Set e.g. "
                r"sequence_pattern: '_Z(?P<Z>\d+)_C(?P<C>\d+)_V\d+'"
            )

    names = [cfg.get("label_name") for cfg in cfgs]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        problems.append(
            f"label_name must be unique per config -- duplicates silently "
            f"overwrite each other's work_dir/<label_name>/ and "
            f"image.zarr/labels/<name>/; repeated: {sorted(duplicates)}"
        )

    if problems:
        for p in problems:
            print(f"[run_multi] ERROR: {p}", file=sys.stderr)
        sys.exit(1)
    return work_dirs.pop()


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
    parser.add_argument(
        "--unlock",
        action="store_true",
        help=(
            "release stale Snakemake locks in every state directory this "
            "script manages, then exit. Needed after a run was killed or "
            "died: the lock is only released on a clean exit."
        ),
    )
    args = parser.parse_args()

    workflow_dir = Path(__file__).resolve().parent.parent
    multi_cfg_path = _resolve(workflow_dir, args.config)
    multi_cfg = _load_yaml(multi_cfg_path)

    seg_config_paths = [
        _resolve(workflow_dir, c) for c in multi_cfg["segmentations"]
    ]
    # Optional shared config: Snakemake merges --configfile values in order,
    # so `common` holds what every segmentation agrees on and each per-config
    # file overrides only what differs. Validation has to see the same merged
    # view Snakemake will, or it would report a missing work_dir that is
    # simply defined one file over.
    common_path = multi_cfg.get("common")
    common_path = _resolve(workflow_dir, common_path) if common_path else None
    common_cfg = _load_yaml(common_path) if common_path else {}
    seg_cfgs = [{**common_cfg, **_load_yaml(p)} for p in seg_config_paths]
    work_dir = _validate_configs(seg_config_paths, seg_cfgs)
    image_store = f"{work_dir}/image.zarr"
    # Shared by every config, hence keyed on the image and level, not on a
    # label_name. Levels are validated identical across configs below.
    _level = int(seg_cfgs[0].get("level", 0))
    occupancy_store = f"{work_dir}/image.occupancy.zarr/{_level}"

    # Each phase gets its own Snakemake state directory (the lock lives in the
    # working directory, not the config), so unlocking has to cover all of
    # them -- and nobody should have to reconstruct these paths by hand.
    state_dirs = [Path(work_dir) / ".snakemake_convert"] + [
        Path(cfg["work_dir"]) / cfg["label_name"] / ".snakemake"
        for cfg in seg_cfgs
    ]
    if args.unlock:
        for state_dir in state_dirs:
            if not state_dir.exists():
                continue
            _run(
                _snakemake_cmd(
                    seg_config_paths[0],
                    workflow_dir=workflow_dir,
                    profile=args.profile,
                    cores=args.cores,
                    dry_run=False,
                    state_dir=state_dir,
                    extra=["--unlock"],
                    common=common_path,
                ),
                workflow_dir,
            )
        print("[run_multi] unlocked; re-run without --unlock", flush=True)
        return

    # Phase A: convert exactly once. The three runs are about to go concurrent
    # and `convert` writes with overwrite=True, so letting them race on it
    # would have them clobbering one store. Ask for its marker explicitly.
    rc = _run(
        _snakemake_cmd(
            seg_config_paths[0],
            workflow_dir=workflow_dir,
            profile=args.profile,
            cores=args.cores,
            dry_run=args.dry_run,
            state_dir=Path(work_dir) / ".snakemake_convert",
            # Both in one phase-A call so they run as SLURM jobs. The
            # occupancy map streams the whole image; building it here in the
            # driver ran it on the login node, where the read is killed
            # without a traceback. It is shared by every config, so it must
            # not be left to the concurrent `prepare` steps either.
            targets=[
                f"{image_store}/zarr.json",
                f"{occupancy_store}/zarr.json",
            ],
            jobname_prefix=slurm_jobname_prefix("convert"),
            common=common_path,
        ),
        workflow_dir,
    )
    if rc != 0:
        print(
            "[run_multi] ERROR: conversion failed.\n"
            "  If the log says the directory cannot be locked, a previous run "
            "was killed rather than exiting cleanly; release it with:\n"
            f"      {Path(sys.argv[0]).name} --config {args.config} --unlock",
            file=sys.stderr,
        )
        sys.exit(rc)

    # Phase B: the segmentations touch disjoint files under
    # work_dir/<label_name>/, so run them together and let the GPU partition
    # stay busy instead of idling through each config's prepare and merge.
    # Each needs its own state directory: .snakemake/locks/ is per working
    # directory, not per config.
    procs = []
    for cfg_path, cfg in zip(seg_config_paths, seg_cfgs):
        cmd = _snakemake_cmd(
            cfg_path,
            workflow_dir=workflow_dir,
            profile=args.profile,
            cores=args.cores,
            dry_run=args.dry_run,
            state_dir=Path(cfg["work_dir"]) / cfg["label_name"] / ".snakemake",
            # Names the config in squeue, so concurrent runs are tellable apart.
            jobname_prefix=slurm_jobname_prefix(cfg["label_name"]),
            common=common_path,
        )
        print(f"[run_multi] $ {' '.join(cmd)}", flush=True)
        procs.append((cfg_path.name, subprocess.Popen(cmd, cwd=workflow_dir)))

    # Don't abort the siblings when one config fails: the others are
    # independent, and killing them would throw away hours of finished GPU
    # work over an unrelated failure.
    failed = [name for name, p in procs if p.wait() != 0]
    for name, p in procs:
        status = "FAILED" if p.returncode else "ok"
        print(f"[run_multi] {name}: {status}", flush=True)
    if failed:
        print(
            f"[run_multi] ERROR: {len(failed)} config(s) failed: "
            f"{', '.join(failed)}; skipping relations.",
            file=sys.stderr,
        )
        sys.exit(1)

    relations = multi_cfg.get("relations", [])
    if args.dry_run or not relations:
        return

    import dask.array as da
    import openpyxl
    import zarr

    from patchworks import label_relations

    def _label_ids(name: str) -> list[int]:
        """Ids present in a label image, without scanning the volume.

        The merge writes n_objects/sequential_labels into the label group's
        attrs precisely so consumers don't have to re-derive the id set; the
        ids are 1..n_objects by construction. Fall back to the full scan only
        for a label group written before those attrs existed -- that scan runs
        here on the login node, so it is worth avoiding.
        """
        attrs = dict(zarr.open_group(f"{image_store}/labels/{name}").attrs)
        if (
            attrs.get("sequential_labels")
            and attrs.get("n_objects") is not None
        ):
            return list(range(1, int(attrs["n_objects"]) + 1))
        print(
            f"[run_multi] {name}: no n_objects attr, falling back to a full "
            "scan for its id set",
            flush=True,
        )
        arr = da.from_zarr(image_store, component=f"labels/{name}/0")
        return sorted(int(x) for x in da.unique(arr[arr > 0]).compute())

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
        a_ids = _label_ids(a_name)
        b_ids = _label_ids(b_name)

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
