"""Check a run before starting it: its config, and what it will do.

    pixi run check config/local/my_run.yaml          # one segmentation
    pixi run check my_multi.yaml                     # several (a multi config)
    pixi run check my_run.yaml --sample 3            # + seconds per tile
    pixi run check my_run.yaml --json                # for the web launcher

Every config is checked the way the run checks it (unknown keys, wrong
values, packages this environment lacks; for a multi config, what its
configs must share). Once the image is converted (`--convert-only` in the
launcher, or any run's first step), it also plans each segmentation with
the workflow's own tile sizing: tiles, how many hold signal, GPU jobs,
memory per tile and the size of the result. `--sample N` times the method
on N real tiles -- run that where the jobs run (`srun` on a GPU node for a
GPU method); planning alone is light enough for the login node.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_multi import (  # noqa: E402
    _config_bases,
    _load_yaml,
    _resolve,
    config_set_problems,
)
from status import converted  # noqa: E402


def load(path: Path) -> tuple[list[Path], list[dict], bool]:
    """The segmentation configs of *path*, defaults applied.

    Parameters
    ----------
    path : Path
        A run config, or a multi config listing several.

    Returns
    -------
    tuple
        `(paths, configs, multi)`: each config's file, the configs, and
        whether *path* is a multi config.
    """
    from _pw import with_defaults

    cfg = _load_yaml(path)
    if "segmentations" not in cfg:
        return [path], [with_defaults(cfg)], False
    inner = (path.parent, Path(__file__).resolve().parent.parent)
    common = cfg.get("common")
    common_cfg = _load_yaml(_resolve(common, *inner)) if common else {}
    paths = [_resolve(p, *inner) for p in cfg["segmentations"]]
    cfgs = [with_defaults({**common_cfg, **_load_yaml(p)}) for p in paths]
    return paths, cfgs, True


def _pinned(cfgs: list[dict], multi: bool) -> dict:
    """The tile shape a multi run pins every config to, if it does.

    `run_multi` pins `tile_shape: "auto"` configs that would size their
    tiles differently to one shape, written to
    `.multi_tile_shape.generated.yaml`; their runs are prepared with that.

    Parameters
    ----------
    cfgs : list of dict
        The segmentation configs.
    multi : bool
        Whether they are a multi config's.

    Returns
    -------
    dict
        `{"tile_shape": [...]}` from the last run's file; empty when none.
    """
    if not multi or any(c.get("tile_shape", "auto") != "auto" for c in cfgs):
        return {}
    import yaml

    path = Path(cfgs[0]["work_dir"]) / ".multi_tile_shape.generated.yaml"
    try:
        return dict(yaml.safe_load(path.read_text()) or {})
    except (OSError, ValueError, yaml.YAMLError):
        return {}


def launch_effect(cfg: dict, plan: dict | None, pinned: dict) -> dict:
    """What launching does to one segmentation, from its markers.

    Parameters
    ----------
    cfg : dict
        The segmentation's config, defaults applied.
    plan : dict or None
        Its plan (`_pw.plan_run`), for the GPU jobs; None before
        conversion.
    pinned : dict
        A multi run's pinned tile shape (:func:`_pinned`).

    Returns
    -------
    dict
        `state` (`"new"`, `"resume"`, `"redo"`, `"done"` or
        `"done_other_settings"`), `text` (one line) and `changes` (the
        settings that changed since its tiles were prepared).
    """
    import json

    from _pw import preparation_changes
    from status import segmentation_status

    status = segmentation_status(cfg)
    run = Path(cfg["work_dir"]) / cfg.get("label_name", "labels")
    try:
        manifest = json.loads((run / "tiles.json").read_text())
    except (OSError, ValueError):
        manifest = None
    changes = (
        preparation_changes(manifest, {**cfg, **pinned}) if manifest else []
    )
    jobs = (plan or {}).get("gpu_jobs")
    jobs_text = f"{jobs} GPU jobs" if jobs is not None else "its GPU jobs"
    if status["merged"]:
        if changes:
            state = "done_other_settings"
            text = (
                f"finished with other settings ({'; '.join(changes)}): left "
                f"as it is. To redo it with these, delete {run}"
            )
        else:
            state, text = "done", "finished: nothing to redo"
    elif changes:
        state = "redo"
        done = status["batches_done"]
        text = (
            f"settings changed since its tiles were prepared "
            f"({'; '.join(changes)}): tiled again, and {jobs_text} run"
            + (f" (the {done} done so far are redone)" if done else "")
        )
    elif manifest is not None and status["batches"]:
        state = "resume"
        text = (
            f"continues: {status['batches_done']} of {status['batches']} "
            "GPU jobs done"
            + (
                f", {status['tiles_done']} of {status['tiles']} tiles"
                if status["tiles"]
                else ""
            )
        )
    else:
        state = "new"
        text = (
            f"tiled and segmented: {jobs_text}"
            if plan
            else "tiled and segmented once the image is converted"
        )
    if status.get("waits_for"):
        text += f"; starts once {status['waits_for']} is merged"
    return {"state": state, "text": text, "changes": changes}


def check(
    path: Path, sample: int = 0, preview: str | Path | None = None
) -> dict:
    """Validate a run's configs and plan each segmentation, without running.

    Parameters
    ----------
    path : Path
        A run config, or a multi config.
    sample : int, optional
        Time the method on this many real tiles (default 0: none).
    preview : str or Path, optional
        With *sample*, a directory for a PNG per segmentation: one tile with
        its objects outlined.

    Returns
    -------
    dict
        `shared_problems`, `converted`, and per segmentation its
        `problems`, `plan` and `launch` (what launching does to it:
        :func:`launch_effect`).
    """
    from _pw import plan_run, validate_config

    paths, cfgs, multi = load(path)
    report: dict = {
        "shared_problems": config_set_problems(paths, cfgs) if multi else [],
        "segmentations": [],
    }
    store = Path(cfgs[0]["work_dir"]) / "image.zarr"
    report["converted"] = converted(store)
    pinned = _pinned(cfgs, multi)
    for p, cfg in zip(paths, cfgs):
        entry: dict = {
            "label_name": cfg.get("label_name", "labels"),
            "config": str(p),
            "problems": [],
            "plan": None,
        }
        try:
            validate_config(cfg)
        except ValueError as exc:
            entry["problems"] = [
                line.removeprefix("  - ") for line in str(exc).splitlines()[1:]
            ] or [str(exc)]
        if report["converted"] and not entry["problems"]:
            try:
                entry["plan"] = plan_run(
                    cfg,
                    sample=sample,
                    preview=(
                        str(Path(preview) / f"{entry['label_name']}.png")
                        if preview and sample
                        else None
                    ),
                )
            except Exception as exc:  # reported, the other configs go on
                entry["plan_error"] = f"{type(exc).__name__}: {exc}"
        try:
            entry["launch"] = launch_effect(cfg, entry["plan"], pinned)
        except Exception as exc:  # reported, the check goes on
            entry["launch"] = {
                "state": "unknown",
                "text": f"could not tell ({type(exc).__name__}: {exc})",
                "changes": [],
            }
        report["segmentations"].append(entry)
    # A plan that fails (an overlap as wide as the tile) fails prepare too.
    report["valid"] = not report["shared_problems"] and not any(
        e["problems"] or e.get("plan_error") for e in report["segmentations"]
    )
    return report


def _gib(n) -> str:
    """Bytes as GiB, for the report.

    Parameters
    ----------
    n : int or float
        Bytes.

    Returns
    -------
    str
    """
    return f"{n / 1024**3:.2f} GiB"


def format_report(report: dict) -> str:
    """The report from :func:`check` as text for the terminal.

    Parameters
    ----------
    report : dict
        What :func:`check` returned.

    Returns
    -------
    str
    """
    lines = []
    for problem in report["shared_problems"]:
        lines.append(f"PROBLEM (all configs): {problem}")
    for e in report["segmentations"]:
        if e["problems"]:
            lines.append(f"{e['label_name']}: PROBLEMS in {e['config']}")
            lines += [f"  - {p}" for p in e["problems"]]
            continue
        lines.append(f"{e['label_name']}: config OK")
        if e.get("launch"):
            lines.append(f"  on launch: {e['launch']['text']}")
        plan = e.get("plan")
        if e.get("plan_error"):
            lines.append(f"  plan failed: {e['plan_error']}")
        if not plan:
            continue
        signal = plan["tiles_with_signal"]
        lines.append(
            f"  tiles {plan['tile_shape']} (grid {plan['grid']}): "
            f"{plan['tiles']} tiles, "
            + (
                "signal not known yet (the occupancy map is built by convert)"
                if signal is None
                else f"{signal} with signal -> {plan['gpu_jobs']} GPU jobs "
                f"of up to {plan['tiles_per_job']}"
            )
        )
        if plan.get("gpu_fallback"):
            lines.append(
                '  tile_shape "auto" sized for an 8 GB GPU: set gpu_memory_gb '
                "to the segment GPU's memory for larger tiles, fewer jobs"
            )
        lines.append(
            f"  {_gib(plan['tile_read_bytes'])} read per tile; labels "
            f"{_gib(plan['labels_bytes_uncompressed'])} before compression"
        )
        if plan.get("seconds_per_tile") is not None:
            line = (
                f"  {plan['seconds_per_tile']:.1f} s per tile "
                f"({plan['sampled_tiles']} sampled)"
            )
            if plan.get("gpu_hours") is not None:
                line += (
                    f" -> {plan['hours_per_job']:.1f} h per job, "
                    f"{plan['gpu_hours']:.1f} GPU hours in all"
                )
            lines.append(line)
        if plan.get("preview"):
            lines.append(
                f"  tile {plan['preview_tile']}: {plan['preview_objects']} "
                f"objects, drawn in {plan['preview']}"
            )
    if report["valid"] and not report["converted"]:
        lines.append(
            "Not converted yet: convert first (the run's first step) to see "
            "the tiles."
        )
    return "\n".join(lines)


def main() -> None:
    """Check a run's configs from the command line (see `--help`)."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", help="a run config, or a multi config")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        metavar="N",
        help="time the method on N real tiles (where the jobs would run)",
    )
    parser.add_argument(
        "--preview",
        metavar="DIR",
        help="with --sample: a PNG per segmentation in DIR, one tile with "
        "its objects outlined",
    )
    parser.add_argument("--json", action="store_true", help="print JSON")
    args = parser.parse_args()
    workflow_dir = Path(__file__).resolve().parent.parent
    path = _resolve(args.config, *_config_bases(workflow_dir))
    if not path.is_file():
        sys.exit(f"[check] ERROR: {args.config} not found")
    report = check(path, sample=args.sample, preview=args.preview)
    print(
        json.dumps(report, default=str) if args.json else format_report(report)
    )
    sys.exit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
