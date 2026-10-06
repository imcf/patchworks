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
import os
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


def _safe_filename(name: str) -> str:
    """Sanitise a label name for use as (part of) a log filename."""
    return re.sub(r"[^A-Za-z0-9_-]", "-", name)


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


# Fallbacks for the relate step, used when neither a --relate-* flag nor a
# `relate:` block in the multi config supplies a value. Wide-margin guesses,
# not measured numbers -- see docs/guide/snakemake.md.
RELATE_DEFAULTS = {
    "partition": "scicore",
    "mem": "32G",
    "cpus": 8,
    "time": 180,
    "qos": None,
}


def _relate_settings(multi_cfg: dict, args) -> dict:
    """Resolve the relate step's SLURM settings.

    Precedence is flag > config > default. Putting them in the config file
    matters because the shipped `pixi run multi-slurm` task is a fixed
    command: a cluster whose default QOS caps the wall time below
    ``time`` has no way to say so without either editing that task or
    abandoning it for a hand-written command line.

    Parameters
    ----------
    multi_cfg : dict
        The parsed multi-segmentation config, whose optional ``relate:``
        block may carry any of ``partition``, ``mem``, ``cpus``, ``time``,
        ``qos``.
    args : argparse.Namespace
        Parsed CLI arguments; each ``relate_*`` is None when not passed.

    Returns
    -------
    dict
        One value per key of :data:`RELATE_DEFAULTS`.

    Raises
    ------
    ValueError
        If ``relate:`` is not a mapping, or carries an unknown key -- a
        typo there would otherwise be silently ignored and the job would
        run with the default that the user thought they had replaced.
    """
    block = multi_cfg.get("relate", {}) or {}
    if not isinstance(block, dict):
        raise ValueError(
            f"`relate:` in the multi config must be a mapping of "
            f"{'/'.join(RELATE_DEFAULTS)}; got {type(block).__name__}"
        )
    unknown = set(block) - set(RELATE_DEFAULTS)
    if unknown:
        raise ValueError(
            f"unknown key(s) in `relate:`: {', '.join(sorted(unknown))}; "
            f"expected any of {', '.join(sorted(RELATE_DEFAULTS))}"
        )
    resolved = {}
    for key, fallback in RELATE_DEFAULTS.items():
        flag = getattr(args, f"relate_{key}", None)
        resolved[key] = flag if flag is not None else block.get(key, fallback)
    return resolved


# Packing the finished store into one file. Off by default: it is a full
# read of everything the run produced, which only makes sense when the
# result is about to leave the cluster.
BUNDLE_DEFAULTS = {
    "format": None,  # None = don't bundle; "zip" or "iso"
    "output": None,  # None = <store>.<format> beside it
    "partition": "scicore",
    "mem": "8G",
    "cpus": 2,
    "time": 720,
    "qos": None,
}


def _bundle_settings(multi_cfg: dict, args) -> dict:
    """Resolve the final packing step's settings.

    Same precedence as :func:`_relate_settings` -- flag > config > default
    -- so `pixi run multi-slurm`, a fixed command, can still produce a
    bundle by way of the config alone.

    Raises
    ------
    ValueError
        For an unknown key or an unsupported format, rather than silently
        not bundling after a run that took hours.
    """
    block = multi_cfg.get("bundle", {}) or {}
    if not isinstance(block, dict):
        raise ValueError(
            "`bundle:` in the multi config must be a mapping of "
            f"{'/'.join(BUNDLE_DEFAULTS)}; got {type(block).__name__}"
        )
    unknown = set(block) - set(BUNDLE_DEFAULTS)
    if unknown:
        raise ValueError(
            f"unknown key(s) in `bundle:`: {', '.join(sorted(unknown))}; "
            f"expected any of {', '.join(sorted(BUNDLE_DEFAULTS))}"
        )
    resolved = {}
    for key, fallback in BUNDLE_DEFAULTS.items():
        flag = getattr(args, f"bundle_{key}", None)
        resolved[key] = flag if flag is not None else block.get(key, fallback)
    if resolved["format"] not in (None, "zip", "iso"):
        raise ValueError(
            f'bundle format must be "zip" or "iso"; got {resolved["format"]!r}'
        )
    return resolved


def _bundle_cmd(
    image_store: str, workflow_dir: Path, bundle: dict, profile: bool
) -> list[str]:
    """Build the invocation that packs the finished store into one file.

    Under ``--profile`` this is an ``srun``, for the same reason the
    occupancy and relate steps are: it reads every file the run produced,
    which a login node kills without a message.
    """
    script = str(workflow_dir / "scripts" / "export_iso.py")
    # Absolute: the command runs with cwd=workflow_dir, so a relative
    # work_dir would otherwise be resolved against the workflow directory
    # rather than against where the driver was invoked.
    inner = [
        sys.executable,
        script,
        "--store",
        str(Path(image_store).resolve()),
        "--format",
        bundle["format"],
        "--overwrite",
    ]
    if bundle["output"]:
        inner += ["--output", str(Path(bundle["output"]).resolve())]
    if not profile:
        return inner
    cmd = ["srun", "--partition", bundle["partition"]]
    if bundle["qos"]:
        cmd += ["--qos", bundle["qos"]]
    cmd += [
        "--mem",
        bundle["mem"],
        "--cpus-per-task",
        str(bundle["cpus"]),
        "--time",
        str(bundle["time"]),
        "--job-name",
        slurm_jobname_prefix(f"bundle-{bundle['format']}"),
    ]
    return cmd + inner


def _run_bundle(
    image_store: str, workflow_dir: Path, bundle: dict, profile: bool
) -> int:
    """Pack the store, returning the exit code (0 when not configured)."""
    if not bundle["format"]:
        return 0
    cmd = _bundle_cmd(image_store, workflow_dir, bundle, profile)
    print(f"[run_multi] $ {' '.join(cmd)}", flush=True)
    code = subprocess.run(cmd, cwd=workflow_dir).returncode
    if code:
        print(
            f"[run_multi] ERROR: bundling failed (exit {code}). Everything "
            "the run produced is already on disk -- the store is complete, "
            "only the single-file copy is missing. Re-run "
            "`pixi run zip --store <image.zarr>` to retry just this step.",
            file=sys.stderr,
        )
    else:
        print("[run_multi] bundle: ok", flush=True)
    return code


def _relate_cmd(
    rel: dict,
    *,
    work_dir: str,
    image_store: str,
    workflow_dir: Path,
    relate_partition: str,
    relate_mem: str,
    relate_cpus: int,
    relate_time: int,
    relate_qos: str | None,
) -> list[str]:
    """Build one ``srun`` invocation of ``relate.py`` for a single relation pair.

    One job per pair, not one job for the whole ``relations:`` list: a
    single shared ``srun`` time budget lets a slow pair (e.g. one needing a
    chunk-layout rechunk first) starve the others out of a fixed
    ``--relate-time``, and a kill that way loses everything not yet written
    even though earlier pairs already finished. Separate jobs also run
    concurrently instead of one after another, and ``relate.py`` itself
    skips a pair whose workbook is already up to date, so retrying with the
    same relations only redoes what actually failed.
    """
    a_name, b_name = rel["a"], rel["b"]
    cmd = [
        "srun",
        "--partition",
        relate_partition,
    ]
    if relate_qos:
        cmd += ["--qos", relate_qos]
    cmd += [
        "--mem",
        relate_mem,
        "--cpus-per-task",
        str(relate_cpus),
        "--time",
        str(relate_time),
        "--job-name",
        slurm_jobname_prefix(f"relate-{a_name}-to-{b_name}"),
        sys.executable,
        str(workflow_dir / "scripts" / "relate.py"),
        "--work-dir",
        work_dir,
        "--image-store",
        image_store,
        "--relations",
        json.dumps([rel]),
        # Concurrent per-pair jobs sharing the default <work_dir>/logs/
        # relate.log would interleave -- give each pair its own file.
        "--log",
        str(
            Path(work_dir)
            / "logs"
            / "relate"
            / f"{_safe_filename(a_name)}_to_{_safe_filename(b_name)}.log"
        ),
    ]
    return cmd


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
    "ngff_version",
)


# Prefix used by every path placeholder in the shipped config templates.
_PLACEHOLDER_PREFIX = "/path/to"


def _review_rules(multi_cfg: dict, label_names: list[str]) -> dict:
    """The ``review:`` block of the multi config, checked.

    ``expect`` maps a parent label image to the number of each child it
    should hold (``1`` or ``[min, max]``); ``min_overlap`` is the fraction
    of a child that must be inside its parent (a number, or per
    ``{child: {parent: value}}``); ``position`` classifies children by where
    they sit in their parent (apical, basal, lateral, central). ``patchworks
    review`` flags whatever breaks them. Relations may also carry
    ``max_distance_um`` (the nearest parent for a child touching none).
    """
    block = multi_cfg.get("review") or {}
    if not isinstance(block, dict):
        sys.exit("[run_multi] ERROR: `review:` must be a mapping")
    problems = []
    unknown = set(block) - {"expect", "min_overlap", "position"}
    if unknown:
        problems.append(
            f"unknown key(s) in `review:`: {', '.join(sorted(unknown))}; "
            "expected expect, min_overlap, position"
        )
    for parent, children in (block.get("expect") or {}).items():
        for name in [parent, *(children or {})]:
            if name not in label_names:
                problems.append(
                    f"`review: expect:` names {name!r}, which is not a "
                    f"segmentation's label_name ({', '.join(label_names)})"
                )
        for child, rng in (children or {}).items():
            ok = isinstance(rng, int) or (
                isinstance(rng, list)
                and len(rng) == 2
                and all(isinstance(v, int) for v in rng)
                and rng[0] <= rng[1]
            )
            if not ok:
                problems.append(
                    f"`review: expect: {parent}: {child}:` must be a count "
                    f"or [min, max]; got {rng!r}"
                )
    overlap = block.get("min_overlap")
    if overlap is not None and not isinstance(overlap, (int, float, dict)):
        problems.append("`review: min_overlap:` must be a number or a mapping")
    pairs = {(r.get("a"), r.get("b")) for r in multi_cfg.get("relations") or []}
    for rel in multi_cfg.get("relations") or []:
        dist = rel.get("max_distance_um")
        if dist is not None and (
            isinstance(dist, bool)
            or not isinstance(dist, (int, float))
            or dist <= 0
        ):
            problems.append(
                f"relation {rel.get('a')} -> {rel.get('b')}: max_distance_um "
                f"must be a positive number of micrometres; got {dist!r}"
            )
    for child, rule in (block.get("position") or {}).items():
        rule = rule or {}
        parent = rule.get("parent")
        apical = str(rule.get("apical", ""))
        ref = apical.removeprefix("towards:")
        if child not in label_names or parent not in label_names:
            problems.append(
                f"`review: position: {child}:` needs a segmentation's "
                f"label_name and a `parent:` one ({', '.join(label_names)})"
            )
            continue
        if (child, parent) not in pairs:
            problems.append(
                f"`review: position: {child}:` needs the relation "
                f"{child} -> {parent} under `relations:`"
            )
        directions = {f"{s}{a}" for s in "+-" for a in "zyx"}
        if apical not in directions:
            if ref not in label_names:
                problems.append(
                    f"`review: position: {child}: apical:` must be a "
                    f"direction like +z, or a label_name (away from its "
                    f"objects, e.g. the nuclei); got {apical!r}"
                )
            elif (ref, parent) not in pairs:
                problems.append(
                    f"`review: position: {child}: apical: {apical}` needs "
                    f"the relation {ref} -> {parent} under `relations:`"
                )
        unknown = set(rule) - {"parent", "apical", "central_depth"}
        if unknown:
            problems.append(
                f"`review: position: {child}:` unknown key(s) "
                f"{', '.join(sorted(unknown))}"
            )
    if problems:
        for p in problems:
            print(f"[run_multi] ERROR: {p}", file=sys.stderr)
        sys.exit(1)
    return block


def _tile_channels(cfg: dict) -> int:
    """Channels a segment tile carries (as ``_pw.tile_channels``): the
    nuclei channel or the seed labels are stacked onto the image."""
    two = cfg.get("nuclei_channel") is not None or bool(cfg.get("seed_labels"))
    return 2 if two else 1


def seed_dependencies(cfgs: list[dict]) -> dict[int, int]:
    """``{config index: index of the config whose labels it grows from}``.

    A config with ``seed_labels: <name>`` reads that label image in every
    tile, so the config producing ``<name>`` must have finished first.
    Labels no listed config produces are taken from the store as they are
    (an earlier run); ``prepare`` checks they exist.
    """
    by_name = {cfg.get("label_name"): i for i, cfg in enumerate(cfgs)}
    return {
        i: by_name[cfg["seed_labels"]]
        for i, cfg in enumerate(cfgs)
        if cfg.get("seed_labels") and cfg["seed_labels"] in by_name
    }


def seed_plan(cfgs: list[dict], image_store: str) -> list[str]:
    """What each ``seed_labels`` config will grow from, said up front.

    A name no listed config produces is taken from the store as it is --
    typically left by an earlier run under that name, so the cells would
    grow from old seeds while the current nuclei are re-segmented next to
    them, with nothing to show for it. Say so loudly, and name the configs'
    own label images, since a near-miss (``nuclei_labels`` for
    ``nuclei_labels_cpsam``) is the usual cause.
    """
    names = [cfg.get("label_name") for cfg in cfgs]
    lines = []
    for cfg in cfgs:
        seed = cfg.get("seed_labels")
        if not seed:
            continue
        me = cfg.get("label_name")
        if seed in names:
            lines.append(f"{me} grows from {seed}: starts once {seed} is done")
            continue
        here = Path(image_store, "labels", seed).exists()
        lines.append(
            f"WARNING: {me} grows from seed_labels {seed!r}, which no config "
            "in this run produces -- "
            + (
                f"using the {seed} already in the store, NOT re-made by "
                "this run"
                if here
                else f"and {seed} is not in the store either (prepare will "
                "stop)"
            )
            + f". This run's label images: {', '.join(map(str, names))}"
        )
    return lines


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

    # The shipped config/ files are templates. Running them unedited used to
    # get all the way to creating the state directory and die as a four-frame
    # pathlib traceback ending in `PermissionError: '/path'`, which names
    # neither the setting nor the file it came from.
    for key in ("work_dir", "input"):
        for path, cfg in zip(paths, cfgs):
            value = str(cfg.get(key, ""))
            if value.startswith(_PLACEHOLDER_PREFIX):
                problems.append(
                    f"{path.name} still has the template's placeholder "
                    f"{key}: {value!r}. The files under workflow/config/ are "
                    "examples -- point `--config` at your own copies, or "
                    "edit them for this dataset."
                )

    # A work_dir that cannot be created fails much later, after the first
    # Snakemake process is already being launched.
    work_dir = next(iter(work_dirs), None) if len(work_dirs) == 1 else None
    if work_dir and not str(work_dir).startswith(_PLACEHOLDER_PREFIX):
        target = Path(work_dir)
        existing = target
        while not existing.exists() and existing != existing.parent:
            existing = existing.parent
        if not existing.exists():
            problems.append(f"work_dir {work_dir} has no reachable parent")
        elif not os.access(existing, os.W_OK):
            problems.append(
                f"work_dir {work_dir} is not creatable: no write permission "
                f"on {existing}"
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

    deps = seed_dependencies(cfgs)
    for i, j in deps.items():
        if deps.get(j) == i or i == j:
            problems.append(
                f"{paths[i].name} and {paths[j].name} take their seed_labels "
                "from each other; one of them has to come first"
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
        n_channels = _tile_channels(cfg)
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


def run_after(
    names: list[str],
    deps: dict[int, int],
    launch,
    poll_seconds: float = 5.0,
) -> list[tuple[str, str]]:
    """Run every job, each as soon as the one it depends on has succeeded.

    *launch(i)* starts job *i* and returns its ``Popen``. Jobs without a
    dependency start at once, side by side. One whose dependency failed is
    not started (``"skipped"``); a failure does not stop the others, which
    are independent and may hold hours of finished GPU work.

    Returns
    -------
    list of (str, str)
        ``(name, "ok" | "FAILED" | "skipped (<dependency> failed)")``, in
        *names* order.
    """
    import time

    status: dict[int, str] = {}
    running: dict[int, subprocess.Popen] = {}
    pending = list(range(len(names)))
    while pending or running:
        for i in list(pending):
            dep = deps.get(i)
            if dep is None or status.get(dep) == "ok":
                running[i] = launch(i)
                pending.remove(i)
            elif dep in status:
                status[i] = f"skipped ({names[dep]} failed)"
                pending.remove(i)
        finished = [i for i, p in running.items() if p.poll() is not None]
        for i in finished:
            status[i] = "ok" if running.pop(i).returncode == 0 else "FAILED"
        if running and not finished:
            time.sleep(poll_seconds)
    return [(names[i], status[i]) for i in range(len(names))]


#: Marks the run_multi driving a work_dir: {"host", "pid", "started", "config"}.
DRIVER_FILE = ".run_multi.pid"


def _driver_alive(pid: int) -> bool:
    """Whether *pid* is a live run_multi on this machine."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # alive, someone else's
    cmdline = Path(f"/proc/{pid}/cmdline")
    if cmdline.exists():  # a recycled pid is some other program
        try:
            args = cmdline.read_bytes().split(b"\0")
        except OSError:
            return True
        return any(
            Path(a.decode(errors="replace")).name == "run_multi.py"
            for a in args
        )
    return True


def claim_driver(work_dir: str | Path, config: Path) -> Path:
    """Make this the only run_multi driving *work_dir*, or exit saying who is.

    Two drivers of one work_dir run every config twice: two Snakemakes per
    config, each submitting its own jobs and merges into the same label
    group, one of them holding the lock the other then fails on. A driver
    left running from an earlier attempt did exactly that. The marker of a
    driver that died is taken over; it is removed at exit.
    """
    import atexit
    import socket
    import time

    marker = Path(work_dir) / DRIVER_FILE
    host, me = socket.gethostname(), os.getpid()
    if marker.exists():
        try:
            other = json.loads(marker.read_text())
        except (OSError, ValueError):
            other = {}
        pid, where = other.get("pid"), other.get("host")
        if pid and pid != me and where == host and _driver_alive(int(pid)):
            sys.exit(
                f"[run_multi] ERROR: another run_multi (pid {pid}, started "
                f"{other.get('started', '?')}, --config {other.get('config')}) "
                f"is already driving {work_dir}. Wait for it, or stop it "
                f"first: kill {pid}"
            )
        if pid and where and where != host:
            sys.exit(
                f"[run_multi] ERROR: a run_multi on {where} (pid {pid}, "
                f"started {other.get('started', '?')}) is driving {work_dir}. "
                f"If it is not running any more (ssh {where} ps -p {pid}), "
                f"delete {marker} and start again."
            )
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "host": host,
                "pid": me,
                "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config": str(config),
            }
        )
    )

    def _release():
        try:
            if json.loads(marker.read_text()).get("pid") == me:
                marker.unlink()
        except (OSError, ValueError):
            pass

    atexit.register(_release)
    return marker


def held_locks(state_dir: Path) -> list[Path]:
    """Snakemake's lock files for a run whose --directory is *state_dir*."""
    locks = Path(state_dir) / ".snakemake" / "locks"
    return sorted(locks.glob("*")) if locks.is_dir() else []


def _resolve(path_str: str, *bases: Path) -> Path:
    """*path_str* as given if absolute, else under the first of *bases*
    holding it (the first base when none does, so the error names it)."""
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    for base in bases:
        if (base / path).exists():
            return base / path
    return bases[0] / path


def _config_bases(workflow_dir: Path) -> list[Path]:
    """Where a relative ``--config`` is looked for, in order.

    pixi runs a task from the workspace root (``workflow/``) whatever
    directory it was called from, and says which in ``INIT_CWD``: so
    ``pixi run multi-slurm --config my_multi.yaml`` from a project folder
    finds the file there, while the shipped ``config/multi.yaml`` still
    resolves against ``workflow/``.
    """
    bases = []
    for base in (os.environ.get("INIT_CWD"), os.getcwd(), workflow_dir):
        if base and Path(base) not in bases:
            bases.append(Path(base))
    return bases


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config/multi.yaml",
        help="multi-segmentation config YAML (default: config/multi.yaml). "
        "A relative path is looked for in the directory pixi was run from, "
        "then in workflow/; the configs it lists, next to it, then in "
        "workflow/. E.g. pixi run multi-slurm --config my_multi.yaml",
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
    # Defaults are None here, not the real values: main() has to tell "not
    # passed" from "passed the default" so a `relate:` block in multi.yaml can
    # fill the gap while an explicit flag still wins. RELATE_DEFAULTS holds
    # the actual fallbacks.
    parser.add_argument(
        "--relate-partition",
        help=(
            "SLURM partition for the relate step under --profile "
            f"(default: {RELATE_DEFAULTS['partition']}). Overrides `relate:` "
            "in the multi config."
        ),
    )
    parser.add_argument(
        "--relate-mem",
        help=(
            "srun --mem for the relate step under --profile "
            f"(default: {RELATE_DEFAULTS['mem']}). Overrides `relate:` in "
            "the multi config."
        ),
    )
    parser.add_argument(
        "--relate-cpus",
        type=int,
        help=(
            "srun --cpus-per-task for the relate step under --profile "
            f"(default: {RELATE_DEFAULTS['cpus']}). Overrides `relate:` in "
            "the multi config."
        ),
    )
    parser.add_argument(
        "--relate-time",
        type=int,
        help=(
            "srun --time in minutes for the relate step under --profile "
            f"(default: {RELATE_DEFAULTS['time']}). Each relation pair is "
            "its own SLURM job, so this bounds one relation, not the whole "
            "relations: list. Overrides `relate:` in the multi config."
        ),
    )
    parser.add_argument(
        "--bundle",
        dest="bundle_format",
        choices=("zip", "iso"),
        help=(
            "after everything succeeds, pack the finished store into one "
            "file. Overrides `bundle:` in the multi config. zip needs "
            "nothing extra; iso needs xorriso/genisoimage/mkisofs and "
            "mounts as a drive."
        ),
    )
    parser.add_argument(
        "--relate-qos",
        help=(
            "srun --qos for the relate step under --profile. Omit to let "
            "SLURM pick your account's default QOS for the partition -- set "
            "this explicitly if that default's MaxWall is shorter than "
            "--relate-time (sacctmgr -p show assoc/qos shows what's "
            "available, e.g. '1day', '1week'). Overrides `relate:` in the "
            "multi config."
        ),
    )
    args = parser.parse_args()

    workflow_dir = Path(__file__).resolve().parent.parent
    bases = _config_bases(workflow_dir)
    multi_cfg_path = _resolve(args.config, *bases)
    if not multi_cfg_path.is_file():
        looked = ", ".join(str(b / args.config) for b in bases)
        sys.exit(
            f"[run_multi] ERROR: --config {args.config} not found (looked at {looked})"
        )
    multi_cfg_path = multi_cfg_path.resolve()
    multi_cfg = _load_yaml(multi_cfg_path)
    # Paths inside the multi config: next to it first (a copy kept with the
    # data, listing its own configs), then workflow/ (the shipped one).
    inner = (multi_cfg_path.parent, workflow_dir)

    seg_config_paths = [_resolve(c, *inner) for c in multi_cfg["segmentations"]]
    # Optional shared config: Snakemake merges --configfile values in order,
    # so `common` holds what every segmentation agrees on and each per-config
    # file overrides only what differs. Validation has to see the same merged
    # view Snakemake will, or it would report a missing work_dir that is
    # simply defined one file over.
    common_path = multi_cfg.get("common")
    common_path = _resolve(common_path, *inner) if common_path else None
    common_cfg = _load_yaml(common_path) if common_path else {}
    seg_cfgs = [{**common_cfg, **_load_yaml(p)} for p in seg_config_paths]
    if args.test_email:
        sys.exit(_test_email(seg_cfgs[0]))

    work_dir = _validate_configs(seg_config_paths, seg_cfgs)
    # Every package a config needs, checked in the environment that will run
    # it (this one: the jobs re-launch from its interpreter) before anything
    # is converted or submitted.
    from _pw import environment_problems

    missing = [p for cfg in seg_cfgs for p in environment_problems(cfg)]
    if missing:
        for p in missing:
            print(f"[run_multi] ERROR: {p}", file=sys.stderr)
        sys.exit(1)
    # Checked now, not after hours of segmentation: a typo'd label name here
    # would otherwise only surface once the relations start.
    review_rules = _review_rules(
        multi_cfg, [str(c.get("label_name")) for c in seg_cfgs]
    )
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
    # One driver per work_dir -- --unlock included: releasing the locks of
    # a run that is still going is exactly how two runs end up in one store.
    if not args.dry_run:
        claim_driver(work_dir, multi_cfg_path)

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
            f"      {Path(sys.argv[0]).name} --config {multi_cfg_path} --unlock",
            file=sys.stderr,
        )
        sys.exit(rc)

    # tile_shape: "auto" resolves differently per config when nuclei_channel
    # differs (the sizer charges per channel) -- pin every config to one
    # shared, computed value so the label arrays end up with matching chunk
    # layouts. Needs the just-converted image, so this can only happen here,
    # not in _validate_configs(). Dry runs never reach a real image.zarr.
    tile_override = None
    # all(), not a set of the values: an explicit tile_shape is a list,
    # which a set cannot hold -- every real run with one (the shipped
    # common.yaml has one) died here, right after the conversion.
    if not args.dry_run and all(
        cfg.get("tile_shape", "auto") == "auto" for cfg in seg_cfgs
    ):
        channel_counts = {_tile_channels(cfg) for cfg in seg_cfgs}
        if len(channel_counts) > 1:
            tile_override = _resolve_shared_tile_shape(
                seg_cfgs, image_store, work_dir
            )

    # Phase B: the segmentations touch disjoint files under
    # work_dir/<label_name>/, so run them together and let the GPU partition
    # stay busy instead of idling through each config's prepare and merge.
    # Each needs its own state directory: .snakemake/locks/ is per working
    # directory, not per config.
    def _launch(cfg_path, cfg):
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
        return subprocess.Popen(cmd, cwd=workflow_dir)

    # A config growing from another's labels (seed_labels) waits for it;
    # everything else starts at once. A dry run never writes labels, so
    # nothing waits there.
    deps = {} if args.dry_run else seed_dependencies(seg_cfgs)
    for line in seed_plan(seg_cfgs, image_store):
        print(f"[run_multi] {line}", flush=True)
    status = run_after(
        [p.name for p in seg_config_paths],
        deps,
        lambda i: _launch(seg_config_paths[i], seg_cfgs[i]),
    )
    failed = [name for name, st in status if st != "ok"]
    for name, st in status:
        print(f"[run_multi] {name}: {st}", flush=True)
    if failed:
        print(
            f"[run_multi] ERROR: {len(failed)} config(s) failed: "
            f"{', '.join(failed)}; skipping relations.",
            file=sys.stderr,
        )
        locked = [
            p.name
            for p, cfg in zip(seg_config_paths, seg_cfgs)
            if p.name in failed
            and held_locks(
                Path(cfg["work_dir"]) / cfg["label_name"] / ".snakemake"
            )
        ]
        if locked:
            print(
                f"[run_multi] {', '.join(locked)}: its Snakemake directory is "
                "locked. If no other run of these configs is still going "
                "(pgrep -af snakemake; squeue -u $USER), the lock is left over "
                "from an interrupted run; release it with the same command "
                "plus --unlock:\n"
                f"      {Path(sys.argv[0]).name} --config {multi_cfg_path} --unlock",
                file=sys.stderr,
            )
        sys.exit(1)

    relations = multi_cfg.get("relations", [])
    relate = _relate_settings(multi_cfg, args)
    bundle = _bundle_settings(multi_cfg, args)
    if args.dry_run:
        if bundle["format"]:
            print(
                f"[run_multi] would bundle the store as .{bundle['format']} "
                "once everything is done"
            )
        return
    if review_rules:
        from patchworks._review import write_rules

        # Once, here, before the concurrent relate jobs: `patchworks review`
        # reads them from the store instead of being told again.
        write_rules(image_store, review_rules)
    if not relations:
        # No relations to compute, but the store is finished, so the
        # bundling step still applies.
        sys.exit(_run_bundle(image_store, workflow_dir, bundle, args.profile))

    if args.profile:
        # Real CPU/IO work -- tens of thousands of zarr chunk reads for a
        # full-resolution label volume -- not orchestration, so (like the
        # occupancy map) it does not belong in this driver process on the
        # login node. Submit it as its own job.
        #
        # One job *per relation pair*, not one job for the whole list: a
        # single shared srun budget lets a slow pair (e.g. one needing a
        # chunk-layout rechunk first) starve the others' time out of a fixed
        # --relate-time, and killed that way loses everything not yet
        # written even though earlier pairs already finished. Separate jobs
        # also run concurrently rather than one after another, and
        # relate.py itself now skips a pair whose workbook is already
        # up to date, so retrying this exact command only redoes what
        # actually failed.
        procs = []
        for rel in relations:
            cmd = _relate_cmd(
                rel,
                work_dir=work_dir,
                image_store=image_store,
                workflow_dir=workflow_dir,
                relate_partition=relate["partition"],
                relate_mem=relate["mem"],
                relate_cpus=relate["cpus"],
                relate_time=relate["time"],
                relate_qos=relate["qos"],
            )
            print(f"[run_multi] $ {' '.join(cmd)}", flush=True)
            procs.append(
                (
                    f"{rel['a']} -> {rel['b']}",
                    subprocess.Popen(cmd, cwd=workflow_dir),
                )
            )

        failed = [name for name, p in procs if p.wait() != 0]
        for name, p in procs:
            status = "FAILED" if p.returncode else "ok"
            print(f"[run_multi] relate {name}: {status}", flush=True)
        if failed:
            print(
                f"[run_multi] ERROR: {len(failed)} relation(s) failed: "
                f"{', '.join(failed)}. Segmentations already succeeded, and "
                "any relation that did finish is written -- re-run with the "
                "same --config to retry just the ones still missing (already "
                "up-to-date workbooks are skipped, not recomputed).",
                file=sys.stderr,
            )
            # Not bundled: a bundle of a run whose relations failed would
            # look complete and quietly be missing workbooks.
            sys.exit(1)
        sys.exit(_run_bundle(image_store, workflow_dir, bundle, args.profile))

    from relate import run_relations

    run_relations(work_dir, image_store, relations)
    sys.exit(_run_bundle(image_store, workflow_dir, bundle, args.profile))


if __name__ == "__main__":
    main()
