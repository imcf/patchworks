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
overlap, including b-objects with zero matches). Under --profile, this runs
as a submitted SLURM job (see scripts/relate.py) rather than in-process here
-- same reasoning as the occupancy-map fix: it streams entire label volumes,
which is real work, not orchestration, and does not belong on the login node.
"""

from __future__ import annotations

import argparse
import json
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
    extra_configfiles: list[Path] | None = None,
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

    *extra_configfiles*, when given, are appended after *configfile* and so
    win over both it and *common* -- used to pin a driver-computed value
    (e.g. a resolved ``tile_shape``) across every config without editing any
    config file on disk.
    """
    configfiles = [str(configfile.resolve())]
    if common is not None:
        configfiles.insert(0, str(common.resolve()))
    if extra_configfiles:
        configfiles += [str(p.resolve()) for p in extra_configfiles]
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


def _test_email(cfg: dict) -> int:
    """Send one test notification and report the outcome. Returns an exit code.

    "No email arrived" has two very different causes that look identical from
    the outside: the address never made it into the merged config, or it did
    and the message was dropped somewhere downstream. This distinguishes them
    without waiting for a multi-hour step to finish.
    """
    from patchworks._notify import send, slurm_mail_extra

    email = cfg.get("notify_email") or ""
    events = cfg.get("notify_events") or ["finish", "error"]
    if not email:
        print(
            "[run_multi] notify_email is empty in the merged config, so no "
            "mail is sent by design.\n"
            "  Set it in the file `common:` points at (the per-config files "
            "no longer carry it), then re-run this check.",
            file=sys.stderr,
        )
        return 1

    print(f"[run_multi] notify_email  : {email}")
    print(f"[run_multi] notify_events : {events}")
    print(
        f"[run_multi] SLURM per-job : sbatch {slurm_mail_extra(email, events)}"
    )
    print(
        "[run_multi] NOTE: convert, occupancy and merge send per-job mail; "
        "segment does not (one job per tile batch would mean hundreds)."
    )
    ok = send(
        email,
        "[patchworks] test notification",
        "This is a patchworks test message.\n\n"
        "If you received it, the workflow's own success/failure mail will "
        "reach you too.\n\n"
        "Per-job start/finish mail is sent by SLURM itself, not by this "
        "path, so it can still be blocked separately -- verify with:\n"
        "    scontrol show job <jobid> | grep -i mail\n",
    )
    if ok:
        print(
            "[run_multi] handed to a local mail transport. If nothing "
            "arrives, the message was accepted and then dropped further "
            "along -- ask the cluster admins about outbound mail."
        )
        return 0
    print(
        "[run_multi] no local mail transport accepted the message (see the "
        "warning above). SLURM's own per-job mail may still work, since the "
        "controller sends that, not this host.",
        file=sys.stderr,
    )
    return 1


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

    # `tile_shape: "auto"` is identical as a *value* across configs while
    # producing different tiles: the sizer charges per channel, so a config
    # with nuclei_channel gets a smaller one. The label groups would then
    # disagree on chunk layout and label_relations would raise -- after every
    # segmentation had run. That used to be a hard error asking for a manual
    # explicit tile_shape; main() now resolves and pins one automatically
    # (see _resolve_shared_tile_shape), once the converted image exists to
    # size against, so there is nothing to check here anymore.

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


def _resolve_shared_tile_shape(
    seg_cfgs: list[dict], image_store: str, work_dir: str
) -> Path:
    """Auto-size ``tile_shape`` once, shared across every config.

    ``tile_shape: "auto"`` resolves differently per config when
    ``nuclei_channel`` differs -- the sizer charges per channel, so a
    two-channel config gets a smaller tile. Left alone, the label groups
    would end up with different chunk layouts and ``label_relations`` would
    raise, after every segmentation had already run.

    Computes what each config's own settings (channel count, ``do_3D``,
    ``diameter``, GPU budget) would actually produce, then pins every config
    to the *smallest* of them by voxel count -- the most memory-constrained
    case, and safe for every config since a smaller tile only ever asks for
    less memory than that config's own budget allows, never more. Configs
    can therefore use different ``do_3D``/``diameter``/channel-count
    settings and still end up with one shared, valid ``tile_shape``.

    Only called once the converted image exists (the sizer needs its real
    shape/dtype), so this runs from ``main()`` after phase A, not from
    ``_validate_configs()``.

    Returns
    -------
    Path
        A generated one-key YAML file (``tile_shape: [...]``), meant to be
        passed as an ``extra_configfiles`` entry to ``_snakemake_cmd`` so it
        overrides every config's own (or common's) ``tile_shape`` value.
    """
    from functools import partial

    import numpy as np
    from patchworks import (
        auto_tile_shape,
        auto_tile_shape_cellpose,
        load_ome_zarr,
    )

    candidates = []
    for cfg in seg_cfgs:
        image = load_ome_zarr(
            image_store, channel=cfg["channel"], level=int(cfg.get("level", 0))
        )
        gpu_gb = cfg.get("gpu_memory_gb")
        gpu_bytes = int(gpu_gb * 1024**3) if gpu_gb else None
        n_channels = 2 if cfg.get("nuclei_channel") is not None else 1
        method = cfg.get("method", "cellpose")
        if method == "cellpose":
            cp = cfg["cellpose"]
            sizer = partial(
                auto_tile_shape_cellpose,
                do_3D=cp.get("do_3D", False),
                use_gpu=cp.get("gpu", True),
                diameter=cp.get("diameter"),
                gpu_memory=gpu_bytes,
                n_channels=n_channels,
            )
        else:
            sizer = partial(
                auto_tile_shape,
                use_gpu=gpu_bytes is not None,
                gpu_memory=gpu_bytes,
                n_channels=n_channels,
            )
        candidates.append(
            tuple(int(x) for x in sizer(image.shape, image.dtype))
        )

    tile_shape = min(candidates, key=lambda t: int(np.prod(t)))
    print(
        f'[run_multi] tile_shape: "auto" resolves differently across '
        f"configs (nuclei_channel differs); pinning every config to the "
        f"smallest computed tile {list(tile_shape)} so the label arrays "
        f"share a chunk layout. Candidates were {[list(c) for c in candidates]}.",
        flush=True,
    )

    override_path = Path(work_dir) / ".multi_tile_shape.generated.yaml"
    override_path.write_text(
        "# Generated by run_multi.py -- pins tile_shape across configs so\n"
        "# label_relations sees matching chunk layouts. Safe to delete; it\n"
        "# is regenerated on the next multi-config run.\n"
        f"tile_shape: {list(tile_shape)}\n"
    )
    return override_path


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
    parser.add_argument(
        "--test-email",
        action="store_true",
        help=(
            "send one test message to the configured notify_email and report "
            "what happened, then exit. Separates 'the address never reached "
            "the config' from 'the mail was rejected downstream', which "
            "otherwise look identical: no email either way."
        ),
    )
    parser.add_argument(
        "--relate-partition",
        default="scicore",
        help="SLURM partition for the relate step under --profile (default: scicore)",
    )
    parser.add_argument(
        "--relate-mem",
        default="32G",
        help="srun --mem for the relate step under --profile (default: 32G)",
    )
    parser.add_argument(
        "--relate-cpus",
        type=int,
        default=8,
        help="srun --cpus-per-task for the relate step under --profile (default: 8)",
    )
    parser.add_argument(
        "--relate-time",
        type=int,
        default=180,
        help="srun --time in minutes for the relate step under --profile (default: 180)",
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
    if args.test_email:
        sys.exit(_test_email(seg_cfgs[0]))

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

    # tile_shape: "auto" resolves differently per config when nuclei_channel
    # differs (the sizer charges per channel) -- pin every config to one
    # shared, computed value so the label arrays end up with matching chunk
    # layouts. Needs the just-converted image, so this can only happen here,
    # not in _validate_configs(). Dry runs never reach a real image.zarr.
    tile_override = None
    if not args.dry_run and {
        cfg.get("tile_shape", "auto") for cfg in seg_cfgs
    } == {"auto"}:
        channel_counts = {
            2 if cfg.get("nuclei_channel") is not None else 1
            for cfg in seg_cfgs
        }
        if len(channel_counts) > 1:
            tile_override = _resolve_shared_tile_shape(
                seg_cfgs, image_store, work_dir
            )

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
            extra_configfiles=[tile_override] if tile_override else None,
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

    if args.profile:
        # Real CPU/IO work -- tens of thousands of zarr chunk reads for a
        # full-resolution label volume -- not orchestration, so (like the
        # occupancy map) it does not belong in this driver process on the
        # login node. Submit it as its own job instead.
        cmd = [
            "srun",
            "--partition",
            args.relate_partition,
            "--mem",
            args.relate_mem,
            "--cpus-per-task",
            str(args.relate_cpus),
            "--time",
            str(args.relate_time),
            "--job-name",
            "pw-relate",
            sys.executable,
            str(workflow_dir / "scripts" / "relate.py"),
            "--work-dir",
            work_dir,
            "--image-store",
            image_store,
            "--relations",
            json.dumps(relations),
        ]
        rc = _run(cmd, workflow_dir)
        if rc != 0:
            print(
                f"[run_multi] ERROR: relate step failed (exit {rc}). "
                "Segmentations already succeeded -- only the relation "
                "workbook(s) are missing. Re-run with the same --config to "
                "retry just this step.",
                file=sys.stderr,
            )
            sys.exit(rc)
        return

    from relate import run_relations

    run_relations(work_dir, image_store, relations)


if __name__ == "__main__":
    main()
