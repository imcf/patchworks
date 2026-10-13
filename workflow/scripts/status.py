"""How far a run has got: converted, tiles segmented, merged, related.

    pixi run status config/config.yaml        # one segmentation
    pixi run status my_multi.yaml             # several (a multi config)
    pixi run status my_multi.yaml --json      # for the web launcher

Reads only the markers the workflow leaves in work_dir (no Snakemake, no
image data), so it answers at once, while a run is going and after it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_multi import _config_bases, _load_yaml, _resolve  # noqa: E402


def _configs(path: Path) -> tuple[list[dict], dict | None]:
    """The segmentation configs of *path*, and the multi config if it is one."""
    cfg = _load_yaml(path)
    if "segmentations" not in cfg:
        return [cfg], None
    inner = (path.parent, Path(__file__).resolve().parent.parent)
    common = cfg.get("common")
    common_cfg = _load_yaml(_resolve(common, *inner)) if common else {}
    return [
        {**common_cfg, **_load_yaml(_resolve(p, *inner))}
        for p in cfg["segmentations"]
    ], cfg


def _attrs(path: Path) -> dict:
    """Attributes of a zarr group or array (v3 zarr.json, or v2 .zattrs)."""
    for name, inner in (("zarr.json", "attributes"), (".zattrs", None)):
        try:
            data = json.loads((path / name).read_text())
        except (OSError, ValueError):
            continue
        return data.get(inner, {}) if inner else data
    return {}


# What a failed step's log says, and what to do about it. First match wins.
HINTS = (
    (
        r"CUDA out of memory|OutOfMemoryError|cudaErrorMemoryAllocation",
        "the GPU ran out of memory: smaller tiles (a tile_shape, or a lower "
        "gpu_memory_gb)",
    ),
    (
        r"oom[_-]kill|OUT_OF_MEMORY|MemoryError|\bKilled\b",
        "the job ran out of memory: more memory for this step (mem_mb in "
        "the profile, Cluster resources in the launcher)",
    ),
    (
        r"DUE TO TIME LIMIT|TIMEOUT",
        "the job hit its time limit: a longer runtime for this step, or "
        "fewer tiles_per_job",
    ),
    (
        r"No module named|ModuleNotFoundError|which this environment does "
        r"not have",
        "a package is missing: run from the pixi environment that has it "
        "(the launcher's Environment)",
    ),
    (
        r"HTTPError|URLError|urlopen|Name or service not known|Temporary "
        r"failure in name resolution|Connection (refused|reset)",
        "a download failed, likely on an offline node: pixi run fetch "
        "<config> on the login node first",
    ),
    (
        r"No space left on device|Disk quota exceeded",
        "the disk is full: free space in work_dir, or ask for more quota",
    ),
    (
        r"cannot be locked|LockException",
        "a killed run left its lock: unlock (--unlock, or Unlock in the "
        "launcher), then start again",
    ),
)
_ERROR = re.compile(r"Traceback|Error|error:|Killed|CANCELLED|oom", re.I)


def explain(text: str) -> tuple[str, str | None]:
    """The line of a log that says what went wrong, and a fix if known."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for pattern, hint in HINTS:
        for line in reversed(lines):
            if re.search(pattern, line):
                return line[:300], hint
    errors = [
        line for line in lines if re.search(r"Error|Exception|error:", line)
    ]
    return (errors or lines or [""])[-1][:300], None


def _failure(step: str, log: Path) -> dict | None:
    """A failure record for *log*, if it says something went wrong."""
    try:
        text = log.read_text(errors="replace")[-200_000:]
    except OSError:
        return None
    if not _ERROR.search(text):
        return None
    error, hint = explain(text)
    return {"step": step, "log": str(log), "error": error, "hint": hint}


def _slurm_logs(dirs: list[Path]) -> list[Path]:
    """The SLURM executor's job logs (``.snakemake/slurm_logs``) there."""
    logs = []
    for d in dirs:
        root = d / ".snakemake" / "slurm_logs"
        if root.is_dir():
            logs += sorted(root.rglob("*.log"), key=lambda p: p.stat().st_mtime)
    return logs


def _killed(log: Path) -> bool:
    """A SLURM log of a job SLURM itself ended (time, memory)."""
    try:
        text = log.read_text(errors="replace")[-20_000:]
    except OSError:
        return False
    return bool(
        re.search(
            r"DUE TO TIME LIMIT|oom[_-]kill|OUT_OF_MEMORY|CANCELLED", text
        )
    )


def failures(cfg: dict, workflow_dir: Path | None = None) -> list[dict]:
    """The steps of one segmentation that failed and are not done since.

    From each step's own log (a traceback), and the SLURM executor's job
    logs (a job SLURM ended: time limit, out of memory), with what to do.
    """
    work = Path(cfg["work_dir"])
    label = cfg.get("label_name", "labels")
    run = work / label
    found = []
    steps = [("prepare", run / "logs" / "prepare.log", run / "tiles.json")]
    seg_logs = run / "logs" / "segment"
    if seg_logs.is_dir():
        for log in sorted(seg_logs.glob("*.log")):
            steps.append(
                (
                    f"segment job {log.stem}",
                    log,
                    run / "seg" / f"{log.stem}.done",
                )
            )
    steps.append(("merge", run / "logs" / "merge.log", run / "labels.done"))
    for step, log, done in steps:
        if log.exists() and not done.exists():
            failure = _failure(step, log)
            if failure:
                found.append(failure)
    # The state directories a run's Snakemake works in: the workflow folder
    # (pixi run slurm), or run_multi's per segmentation.
    dirs = [run / ".snakemake"]
    if workflow_dir is not None:
        dirs.append(workflow_dir)
    for log in _slurm_logs(dirs):
        # slurm_logs/rule_<rule>/<wildcards>/<job id>.log
        rule = next(
            (
                p.removeprefix("rule_")
                for p in log.parts
                if p.startswith("rule_")
            ),
            "",
        )
        batch = re.search(r"batch[_=]?(\d+)", str(log))
        if rule == "segment" and batch:
            if (run / "seg" / f"{batch.group(1)}.done").exists():
                continue
            step = f"segment job {batch.group(1)}"
        elif rule == "merge" and (run / "labels.done").exists():
            continue
        elif rule == "prepare" and (run / "tiles.json").exists():
            continue
        elif rule not in ("segment", "merge", "prepare"):
            continue
        else:
            step = rule
        if _killed(log) and not any(f["step"] == step for f in found):
            failure = _failure(f"{step} (SLURM)", log)
            if failure:
                found.append(failure)
    return found


def shared_failures(cfg: dict) -> list[dict]:
    """The conversion steps, shared by every segmentation of a work_dir."""
    work = Path(cfg["work_dir"])
    level = int(cfg.get("level", 0))
    store = work / "image.zarr"
    found = []
    for step, log, done in (
        (
            "convert",
            work / "logs" / "convert.log",
            [store / "zarr.json", store / ".zgroup"],
        ),
        (
            "occupancy",
            work / "logs" / "occupancy.log",
            [work / "image.occupancy.zarr" / str(level) / "zarr.json"],
        ),
    ):
        if log.exists() and not any(d.exists() for d in done):
            failure = _failure(step, log)
            if failure:
                found.append(failure)
    return found


def segmentation_status(cfg: dict) -> dict:
    """Where one segmentation stands, from its markers in work_dir."""
    work = Path(cfg["work_dir"])
    label = cfg.get("label_name", "labels")
    run = work / label
    store = work / "image.zarr"
    out: dict = {
        "label_name": label,
        "converted": (store / "zarr.json").exists()
        or (store / ".zgroup").exists(),
        "batches": None,
        "batches_done": 0,
        "merged": (run / "labels.done").exists(),
        "objects": None,
        "seams": None,
    }
    try:
        manifest = json.loads((run / "tiles.json").read_text())
        out["batches"] = len(manifest["batches"])
    except (OSError, ValueError, KeyError):
        pass
    seg = run / "seg"
    if seg.is_dir():
        out["batches_done"] = sum(1 for _ in seg.glob("*.done"))
    if out["merged"]:
        level0 = _attrs(store / "labels" / label / "0")
        out["objects"] = level0.get("patchworks_n_objects")
        try:
            report = json.loads((run / "seams.json").read_text())
            # The report keys axes by index into the (z, y, x) labels.
            out["seams"] = {
                "zyx"[int(ax)] if str(ax) in "012" else str(ax): {
                    k: row.get(k) for k in ("seam_rate", "interior_rate")
                }
                for ax, row in report["axes"].items()
            }
        except (OSError, ValueError, KeyError):
            pass
    if out["merged"]:
        out["stage"] = "done"
    elif out["batches"] is not None:
        out["stage"] = "segmenting"
    elif out["converted"]:
        out["stage"] = "converted"
    else:
        out["stage"] = "not started"
    return out


def run_status(cfgs: list[dict], multi: dict | None) -> dict:
    """Every segmentation, and for a multi run its relations and bundle."""
    workflow_dir = Path(__file__).resolve().parent.parent
    status = {
        "work_dir": str(cfgs[0]["work_dir"]),
        "segmentations": [segmentation_status(c) for c in cfgs],
        "failures": shared_failures(cfgs[0]),
    }
    for seg, cfg in zip(status["segmentations"], cfgs):
        # A single run's Snakemake works in the workflow folder.
        seg["failures"] = failures(cfg, workflow_dir if multi is None else None)
    if multi is None:
        return status
    work = Path(cfgs[0]["work_dir"])
    relations = []
    for rel in multi.get("relations") or []:
        out = work / rel.get("output", f"{rel['a']}_to_{rel['b']}.xlsx")
        # A sheet too long for Excel is written as two csv files instead.
        csvs = [
            out.with_name(f"{out.stem}_{n}.csv") for n in (rel["a"], rel["b"])
        ]
        files = [out] if out.exists() else [p for p in csvs if p.exists()]
        relations.append(
            {"a": rel["a"], "b": rel["b"], "files": [str(p) for p in files]}
        )
    status["relations"] = relations
    bundle = multi.get("bundle", {})
    if bundle is not False:
        fmt = (bundle or {}).get("format") or "zip"
        path = Path((bundle or {}).get("output") or f"{work}/image.zarr.{fmt}")
        status["bundle"] = str(path) if path.exists() else None
    return status


def format_status(status: dict) -> str:
    lines = [f"work_dir: {status['work_dir']}"]
    for s in status["segmentations"]:
        if s["batches"] is not None:
            tiles = f"{s['batches_done']}/{s['batches']} batches"
        else:
            tiles = "not prepared"
        line = f"  {s['label_name']:<24} {s['stage']:<12} {tiles}"
        if s["objects"] is not None:
            line += f", {s['objects']:,} objects"
        lines.append(line)
        for ax, row in (s["seams"] or {}).items():
            inner = row["interior_rate"]
            lines.append(
                f"    seams {ax}: {100 * row['seam_rate']:.1f}% of objects end "
                "on a seam vs "
                f"{'n/a' if inner is None else f'{100 * inner:.1f}%'} inside"
            )
    for f in status.get("failures", []) + [
        dict(f, step=f"{s['label_name']}: {f['step']}")
        for s in status["segmentations"]
        for f in s.get("failures", [])
    ]:
        lines.append(f"  FAILED {f['step']}: {f['error']}")
        if f["hint"]:
            lines.append(f"    -> {f['hint']}")
        lines.append(f"    log: {f['log']}")
    for rel in status.get("relations", []):
        files = ", ".join(rel["files"]) or "not yet"
        lines.append(f"  relation {rel['a']} -> {rel['b']}: {files}")
    if "bundle" in status:
        lines.append(f"  bundle: {status['bundle'] or 'not yet'}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="a run config, or a multi config")
    parser.add_argument("--json", action="store_true", help="print JSON")
    args = parser.parse_args()
    workflow_dir = Path(__file__).resolve().parent.parent
    path = _resolve(args.config, *_config_bases(workflow_dir))
    if not path.is_file():
        sys.exit(f"[status] ERROR: {args.config} not found")
    status = run_status(*_configs(path))
    print(json.dumps(status) if args.json else format_status(status))


if __name__ == "__main__":
    main()
