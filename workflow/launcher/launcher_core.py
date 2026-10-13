"""Launcher logic without the UI: config, commands, host keys, job records.

Kept free of Streamlit (and of paramiko at import time) so it can be tested
on its own and reused by other front ends. ``app.py`` is the UI on top.
"""

from __future__ import annotations

import base64
import hashlib
import json
import shlex
from pathlib import Path
from typing import Any

import yaml

# Run modes, in the order the UI offers them.
DRY_RUN = "Dry run"
SLURM_JOB = "Submit — controller as a SLURM job (recommended)"
LOGIN_NODE = "Submit — controller on the login node"
RUN_MODES = (DRY_RUN, SLURM_JOB, LOGIN_NODE)

LOG_DIR = ".streamlit_launcher_logs"
REGISTRY = f"{LOG_DIR}/jobs.jsonl"


# ---------------------------------------------------------------------------
# Cluster presets and host keys
# ---------------------------------------------------------------------------


def load_clusters(path: str | Path) -> dict[str, dict[str, Any]]:
    """Cluster presets, with fingerprints normalised to a list.

    ``host_key_fingerprints`` (a list, one per key type the server may
    offer) is the field; the older single ``host_key_fingerprint`` string is
    still accepted.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) if path.exists() else None
    clusters = {}
    for name, preset in (raw or {}).items():
        preset = dict(preset or {})
        fps = preset.pop("host_key_fingerprints", None) or []
        single = preset.pop("host_key_fingerprint", None)
        if isinstance(fps, str):
            fps = [fps]
        if single:
            fps = [*fps, single]
        preset["host_key_fingerprints"] = [f.strip() for f in fps if f]
        clusters[name] = preset
    return clusters


def fingerprint(key_bytes: bytes) -> str:
    """OpenSSH-style ``SHA256:...`` fingerprint of a public key blob."""
    digest = hashlib.sha256(key_bytes).digest()
    return "SHA256:" + base64.b64encode(digest).decode().rstrip("=")


def check_host_key(
    hostname: str, got: str, expected: list[str], allow_unverified: bool
) -> None:
    """Accept or refuse the host key the server presented.

    With pinned fingerprints, the key must match one of them (a list, since
    the server may present its ed25519, ECDSA or RSA key depending on what
    the client negotiates). With none, the connection is refused unless the
    user explicitly allowed an unverified one -- an in-memory "trust on first
    use" re-trusts on *every* session, since nothing is persisted, so on a
    shared deployment it would never verify anything.

    Raises
    ------
    ValueError
        With a message fit for the UI.
    """
    if expected:
        if got not in expected:
            raise ValueError(
                f"host key for {hostname} is {got}, which matches none of "
                f"the pinned fingerprints ({', '.join(expected)}). Refusing "
                "to connect: the pin may be stale, or the connection is "
                "being intercepted."
            )
        return
    if not allow_unverified:
        raise ValueError(
            f"{hostname} has no pinned host-key fingerprint, so its identity "
            f"cannot be verified (its key is {got}). Pin it in "
            "clusters.yaml, or tick 'allow an unverified host key' if you "
            "are running this app yourself and accept the risk."
        )


def make_host_key_policy(expected: list[str], allow_unverified: bool):
    """A paramiko MissingHostKeyPolicy applying :func:`check_host_key`."""
    import paramiko

    class _Policy(paramiko.MissingHostKeyPolicy):
        def missing_host_key(self, client, hostname, key):
            try:
                check_host_key(
                    hostname,
                    fingerprint(key.asbytes()),
                    expected,
                    allow_unverified,
                )
            except ValueError as exc:
                raise paramiko.SSHException(str(exc)) from None
            client.get_host_keys().add(hostname, key.get_name(), key)

    return _Policy()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _yaml_or(text: str, default: Any) -> Any:
    text = (text or "").strip()
    return yaml.safe_load(text) if text else default


def build_config(v: dict[str, Any]) -> dict[str, Any]:
    """The complete run config for the form values *v*.

    Complete on purpose: Snakemake overlays this file on the cluster copy's
    ``config/config.yaml``, so any key left out here would silently take
    whatever that file holds.
    """
    cfg: dict[str, Any] = {
        "input": v["input"],
        "work_dir": v["work_dir"],
        "label_name": v["label_name"],
        # conversion
        "reuse_pyramid": v.get("reuse_pyramid", False),
        "convert_chunks": _yaml_or(v.get("convert_chunks", ""), None),
        "shard": v.get("shard", False),
        "ngff_version": v.get("ngff_version", "auto"),
        "sequence_pattern": v.get("sequence_pattern") or None,
        "compression": v.get("compression", "zstd"),
        # tiling
        "channel": v.get("channel", 0),
        "nuclei_channel": v.get("nuclei_channel"),
        "level": v.get("level", 0),
        "tile_shape": _yaml_or(v.get("tile_shape", "auto"), "auto"),
        "gpu_memory_gb": v.get("gpu_memory_gb") or None,
        "overlap": _yaml_or(v.get("overlap", "30"), 30),
        "skip_empty": v.get("skip_empty", True),
        "empty_threshold": v.get("empty_threshold"),
        "tiles_per_job": v.get("tiles_per_job", 4),
        # segmentation + post-processing
        "method": v["method"],
        "fill_holes": v.get("fill_holes") or False,
        "open_radius": v.get("open_radius", 0),
        "dilate": v.get("dilate", 0),
        "min_volume": v.get("min_volume"),
        "max_volume": v.get("max_volume"),
        # merge
        "stitch": v.get("stitch", "touch"),
        "iou_threshold": v.get("iou_threshold", 0.5),
        "sequential_labels": v.get("sequential_labels", True),
        "merge_workers": v.get("merge_workers"),
        "seam_report": v.get("seam_report", True),
        # object table (patchworks review, relation workbooks)
        "object_table": v.get("object_table", True),
        "table_channels": list(v.get("table_channels") or []),
        # label pyramid
        "pyramid_levels": v.get("pyramid_levels", 5),
        "pyramid_downscale": v.get("pyramid_downscale", 2),
        "shard_labels": v.get("shard_labels", False),
        # notifications
        "notify_email": v.get("notify_email", ""),
        "notify_events": v.get("notify_events", ["finish", "error"]),
    }
    method = v["method"]
    if method == "cellpose":
        cellpose = {
            "model": v.get("cp_model", "cyto3"),
            "diameter": v.get("cp_diameter") or None,
            "do_3D": v.get("cp_do_3d", False),
            "gpu": v.get("cp_gpu", True),
        }
        cellpose.update(_yaml_or(v.get("cp_extra_kwargs", ""), {}) or {})
        cfg["cellpose"] = cellpose
    elif method == "dog":
        # The DoG plugin runs through the workflow's "custom" method.
        kwargs: dict[str, Any] = {
            "low_sigma": v.get("dog_low_sigma", 1.0),
            "high_sigma": v.get("dog_high_sigma", 3.0),
            "threshold": v.get("dog_threshold", 0.02),
            "sigma_units": v.get("dog_sigma_units", "px"),
            "use_gpu": v.get("dog_use_gpu", False),
        }
        if v.get("dog_psf"):
            decon: dict[str, Any] = {"psf": v["dog_psf"]}
            for key in ("wavelength", "na", "nimm"):
                if v.get(f"dog_{key}") is not None:
                    decon[key] = v[f"dog_{key}"]
            if v.get("dog_dup_rev_z") not in (None, "off"):
                decon["dup_rev_z"] = (
                    "auto" if v["dog_dup_rev_z"] == "auto" else True
                )
            kwargs["decon_kwargs"] = decon
        cfg["method"] = "custom"
        cfg["custom"] = {
            "module": "patchworks.plugins.dog",
            "function": "segment",
            "kwargs": kwargs,
        }
    elif method == "custom":
        cfg["custom"] = {
            "module": v["custom_module"],
            "function": v.get("custom_function") or "segment",
            "kwargs": _yaml_or(v.get("custom_kwargs", ""), {}) or {},
        }
    return cfg


def missing_fields(cfg: dict[str, Any]) -> list[str]:
    """Settings a run cannot do without that the form left empty.

    An empty ``work_dir`` would put the converted image at ``/image.zarr``
    (and every result next to it), so it is refused before anything runs.
    """
    missing = []
    for key in ("input", "work_dir", "label_name"):
        value = str(cfg.get(key) or "").strip()
        if not value or value.startswith("/path/to/"):
            missing.append(key)
    if cfg.get("method") == "custom" and not (cfg.get("custom") or {}).get(
        "module"
    ):
        missing.append("custom.module")
    return missing


def config_problems(cfg: dict[str, Any]) -> list[str]:
    """What the form holds that the workflow would get wrong."""
    work_dir = str(cfg.get("work_dir") or "").strip()
    if work_dir and not work_dir.startswith("/"):
        return [
            f"work_dir must be an absolute path on the cluster: {work_dir!r}"
        ]
    return []


# ---------------------------------------------------------------------------
# pixi environment
# ---------------------------------------------------------------------------

# What each environment of workflow/pixi.toml adds to the default one, in
# the order to prefer them: the smallest that has everything wins.
ENVIRONMENTS = {
    "default": set(),
    "cuda12": {"cupy"},
    "cuda13": {"cupy"},
    "plantseg": {"plantseg", "cupy"},
    "careamics": {"careamics"},
    "plantseg-careamics": {"plantseg", "cupy", "careamics"},
    "cellpose4-cuda12": {"cupy"},
    "cellpose4-cuda13": {"cupy"},
}


def needed_packages(cfg: dict[str, Any]) -> dict[str, str]:
    """Optional packages a config needs, each with why.

    Mirrors the workflow's own check (``_pw.environment_problems``), which
    runs on the cluster and refuses to start without them.
    """
    needs: dict[str, str] = {}
    custom = cfg.get("custom") or {}
    if cfg.get("method") == "custom":
        if str(custom.get("module", "")).endswith("plugins.plantseg"):
            needs["plantseg"] = "PlantSeg"
        if (custom.get("kwargs") or {}).get("use_gpu"):
            needs["cupy"] = "use_gpu"
    if cfg.get("dilate_gpu"):
        needs["cupy"] = "dilate_gpu"
    if cfg.get("denoise"):
        needs["careamics"] = "denoise"
    return needs


def pixi_environments(toml_text: str | None) -> list[str]:
    """The environments a workflow ``pixi.toml`` defines, default first."""
    import tomllib

    try:
        data = tomllib.loads(toml_text or "")
    except tomllib.TOMLDecodeError:
        return ["default"]
    return ["default", *(data.get("environments") or {})]


def suggest_environment(
    cfgs: list[dict[str, Any]], available: list[str]
) -> tuple[str | None, list[str]]:
    """The smallest environment with every package the run needs.

    Returns it (None if no environment has them all) and one reason per
    segmentation that needs something.
    """
    needs: set[str] = set()
    reasons = []
    for cfg in cfgs:
        mine = needed_packages(cfg)
        needs |= set(mine)
        if mine:
            what = ", ".join(f"{m} ({why})" for m, why in mine.items())
            reasons.append(f"{cfg.get('label_name')} needs {what}")
    for env, has in ENVIRONMENTS.items():
        if env in available and needs <= has:
            return env, reasons
    return None, reasons


def setup_with_environment(setup_cmd: str, env: str) -> str | None:
    """*setup_cmd* activating pixi environment *env*.

    None when the line does not use ``pixi shell-hook``, so the environment
    cannot be swapped in.
    """
    import re

    pattern = r"pixi shell-hook(\s+(-e|--environment)[\s=]+[\w.-]+)?"
    if not re.search(pattern, setup_cmd):
        return None
    hook = (
        "pixi shell-hook" if env == "default" else f"pixi shell-hook -e {env}"
    )
    line = re.sub(pattern, hook, setup_cmd, count=1)
    if "plantseg" in env and "CONDA_OVERRIDE_CUDA" not in line:
        # Its PyTorch asks for CUDA, which the login node and the
        # controller's CPU node do not have (see workflow/pixi.toml).
        line = f"export CONDA_OVERRIDE_CUDA=12.0 && {line}"
    return line


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def snakemake_command(
    config_path: str, mode: str, *, profile: str = "profile/slurm"
) -> str:
    args = [
        "snakemake",
        "-s",
        "Snakefile",
        "--configfile",
        config_path,
        "--rerun-triggers",
        "mtime",
    ]
    if mode == DRY_RUN:
        args += ["--cores", "1", "-n", "-p"]
    else:
        args += ["--workflow-profile", profile]
    return " ".join(shlex.quote(a) for a in args)


def convert_command(
    config_path: str,
    cfg: dict[str, Any],
    mode: str,
    *,
    workflow_dir: str,
    profile: str = "profile/slurm",
) -> str:
    """Snakemake for the conversion alone, as ``run_multi.py`` runs it.

    The same targets and state directory as run_multi's first phase, so a
    later run -- one segmentation or several -- finds the image converted
    and goes straight on. *Plan* needs that image.
    """
    work = cfg["work_dir"].rstrip("/")
    root = ".zgroup" if str(cfg.get("ngff_version")) == "0.4" else "zarr.json"
    level = int(cfg.get("level", 0))
    if not profile.startswith("/"):
        profile = f"{workflow_dir}/{profile}"
    args = [
        "snakemake",
        "-s",
        f"{workflow_dir}/Snakefile",
        "--configfile",
        config_path,
        "--directory",
        f"{work}/.snakemake_convert",
    ]
    if mode == DRY_RUN:
        args += ["--cores", "1", "-n", "-p"]
    else:
        args += ["--workflow-profile", profile]
    args += [
        "--",
        f"{work}/image.zarr/{root}",
        f"{work}/image.occupancy.zarr/{level}/zarr.json",
    ]
    return " ".join(shlex.quote(a) for a in args)


def multi_command(
    multi_path: str, mode: str, *, profile: str = "profile/slurm"
) -> str:
    """``run_multi.py`` for a multi-segmentation run."""
    args = ["python", "scripts/run_multi.py", "--config", multi_path]
    args += ["-n"] if mode == DRY_RUN else ["--profile", profile]
    return " ".join(shlex.quote(a) for a in args)


def safe_name(text: str) -> str:
    """*text* reduced to characters safe in a file or job name."""
    return (
        "".join(c if c.isalnum() or c in "_." else "-" for c in text) or "run"
    )


def build_multi(
    shared: dict[str, Any],
    segmentations: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    *,
    directory: str,
    relate: dict[str, Any] | None = None,
    bundle: dict[str, Any] | None = None,
    review: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """The files of a multi-segmentation run, keyed by their remote path.

    One complete config per segmentation (shared form values overlaid with
    that segmentation's own), and the ``multi.yaml`` listing them and the
    relations, for ``scripts/run_multi.py``. Each segmentation config is
    complete rather than split into a ``common:`` file: nothing then depends
    on what the cluster copy's config files happen to contain.
    """
    files: dict[str, dict[str, Any]] = {}
    paths = []
    for seg in segmentations:
        cfg = build_config({**shared, **seg})
        path = f"{directory}/seg_{safe_name(cfg['label_name'])}.yaml"
        files[path] = cfg
        paths.append(path)
    multi: dict[str, Any] = {
        "segmentations": paths,
        "relations": [
            {"a": r["a"], "b": r["b"], "output": r["output"]}
            | (
                {"max_distance_um": float(r["max_distance_um"])}
                if r.get("max_distance_um")
                else {}
            )
            for r in relations
        ],
    }
    relate = {k: v for k, v in (relate or {}).items() if v not in (None, "")}
    if relate:
        multi["relate"] = relate
    if bundle is not None and not bundle.get("format"):
        multi["bundle"] = False  # asked for none; the default is a zip
    else:
        bundle = {
            k: v for k, v in (bundle or {}).items() if v not in (None, "")
        }
        if bundle:
            multi["bundle"] = bundle
    review = {
        k: v for k, v in (review or {}).items() if v not in (None, {}, "")
    }
    if review:
        multi["review"] = review
    files[f"{directory}/multi.yaml"] = multi
    return files


def multi_problems(files: dict[str, dict[str, Any]]) -> list[str]:
    """What run_multi would reject, found before anything is uploaded."""
    multi_path = next(p for p in files if p.endswith("/multi.yaml"))
    multi = files[multi_path]
    cfgs = [files[p] for p in multi["segmentations"]]
    labels = [c["label_name"] for c in cfgs]
    problems = []
    if len(set(labels)) != len(labels):
        problems.append(f"label_name must differ per segmentation: {labels}")
    # run_multi converts once, from the first config, so everything
    # `convert` reads must agree too (its _CONVERT_KEYS).
    for key in (
        "work_dir",
        "input",
        "tile_shape",
        "level",
        "sequence_pattern",
        "convert_chunks",
        "shard",
        "reuse_pyramid",
        "ngff_version",
    ):
        values = {str(c.get(key)) for c in cfgs}
        if len(values) > 1:
            problems.append(f"segmentations must share {key}: {values}")
    outputs = []
    for rel in multi["relations"]:
        for side in ("a", "b"):
            if rel[side] not in labels:
                problems.append(
                    f"relation {rel['a']} -> {rel['b']}: {rel[side]!r} is "
                    f"not a segmentation's label_name ({labels})"
                )
        if rel["a"] == rel["b"]:
            problems.append(f"relation relates {rel['a']} to itself")
        if not str(rel["output"]).endswith(".xlsx"):
            problems.append(f"relation output must be .xlsx: {rel['output']}")
        outputs.append(rel["output"])
    if len(set(outputs)) != len(outputs):
        problems.append(f"relation outputs must differ: {outputs}")
    for parent, children in (
        (multi.get("review") or {}).get("expect") or {}
    ).items():
        for name in [parent, *children]:
            if name not in labels:
                problems.append(
                    f"review rule names {name!r}, not a segmentation's "
                    f"label_name ({labels})"
                )
    return problems


def inner_command(workflow_dir: str, setup_cmd: str, snakemake: str) -> str:
    """``cd`` + environment setup + snakemake, as one shell line."""
    parts = [f"cd {shlex.quote(workflow_dir)}"]
    if setup_cmd.strip():
        # The user's own shell snippet (module load, pixi shell-hook ...),
        # run as themselves on their own account.
        parts.append(setup_cmd.strip())
    parts.append(snakemake)
    return " && ".join(parts)


def controller_job_script(
    inner: str,
    *,
    time: str = "3-00:00:00",
    mem: str = "4G",
    partition: str = "",
    qos: str = "",
    account: str = "",
) -> str:
    """An sbatch script running the Snakemake *controller* as a small job.

    The controller only submits and watches the real jobs, so it needs one
    CPU and little memory -- but it must outlive every job it submits, and
    login nodes are commonly rebooted, or reap long-running processes, and
    are not meant for multi-day processes at all.

    The job name and log path go on the sbatch command line
    (:func:`sbatch_command`), where shell quoting applies: ``#SBATCH``
    lines are not shell-parsed, so a quoted path there would be taken
    literally.
    """
    lines = [
        "#!/bin/bash -l",
        "#SBATCH --cpus-per-task=1",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time}",
    ]
    for flag, value in (
        ("partition", partition),
        ("qos", qos),
        ("account", account),
    ):
        if value:
            lines.append(f"#SBATCH --{flag}={value}")
    lines += [
        "set -euo pipefail",
        # Forget this job's own SLURM context. Everything the controller
        # launches must be a job of its own: an `srun` (run_multi's relate
        # and bundle steps) would otherwise become a step inside this
        # 1-CPU allocation and fail to get its memory, and a submitted job
        # would inherit variables such as SLURM_MEM_PER_NODE that
        # patchworks reads to size itself.
        "for v in $(compgen -e | grep -E '^(SLURM|SBATCH)_'); do "
        'unset "$v"; done',
        inner,
        "",
    ]
    return "\n".join(lines)


def sbatch_command(script: str, *, name: str, log: str) -> str:
    """``sbatch --parsable`` for *script*, logging to *log*."""
    return " ".join(
        shlex.quote(a)
        for a in (
            "sbatch",
            "--parsable",
            f"--job-name={name}",
            f"--output={log}",
            script,
        )
    )


def detached_command(inner: str, log: str) -> str:
    """Run *inner* detached from the SSH session; prints ``PID:<pid>``."""
    return (
        f"setsid nohup bash -lc {shlex.quote(inner)} "
        f"> {shlex.quote(log)} 2>&1 < /dev/null & echo PID:$!"
    )


def parse_tag(output: str, tag: str) -> str | None:
    """Value after ``TAG:`` in command output (``PID:123``), if any."""
    for line in output.splitlines():
        if line.startswith(f"{tag}:"):
            return line.split(":", 1)[1].strip() or None
    return None


def parse_sbatch(output: str) -> str | None:
    """Job id from ``sbatch --parsable`` (``123`` or ``123;cluster``)."""
    for line in output.strip().splitlines()[::-1]:
        head = line.split(";")[0].strip()
        if head.isdigit():
            return head
    return None


def status_command(job: dict[str, Any]) -> str:
    """Shell line printing RUNNING/PENDING/DONE for a registered job."""
    if job.get("slurm_job"):
        jid = shlex.quote(str(job["slurm_job"]))
        return f's=$(squeue -h -j {jid} -o %T 2>/dev/null); echo "${{s:-DONE}}"'
    pid = shlex.quote(str(job["pid"]))
    return f"kill -0 {pid} 2>/dev/null && echo RUNNING || echo DONE"


def image_store(cfg: dict[str, Any]) -> str:
    """The converted image the workflow writes (``<work_dir>/image.zarr``)."""
    return f"{cfg['work_dir'].rstrip('/')}/image.zarr"


def store_exists_command(store: str) -> str:
    """Shell line printing YES when *store* holds a converted image."""
    q = shlex.quote(store)
    return (
        f"if [ -e {q}/zarr.json ] || [ -e {q}/.zgroup ]; "
        "then echo YES; else echo NO; fi"
    )


def plan_command(cfg: dict[str, Any]) -> str | None:
    """``patchworks segment --plan`` for the run's converted image.

    Planning reads the converted store, so it needs ``convert`` to have run
    once; it runs no segmentation (no sampling), so it is safe on a login
    node. None when the method has no CLI equivalent.
    """
    args = ["patchworks", "segment", image_store(cfg), "--plan"]
    tile = cfg.get("tile_shape")
    if isinstance(tile, (list, tuple)):
        args += ["--tile-shape", ",".join(str(t) for t in tile)]
    elif tile == "auto":
        args += ["--tile-shape", "auto"]
    ov = cfg.get("overlap")
    if ov is not None:
        args += [
            "--overlap",
            ",".join(str(o) for o in ov) if isinstance(ov, list) else str(ov),
        ]
    args += ["--level", str(cfg.get("level", 0))]
    ch = cfg.get("channel")
    args += ["--channel", "none" if ch is None else str(ch)]
    if cfg.get("skip_empty"):
        args.append("--skip-empty")
    args += ["--stitch", cfg.get("stitch", "touch")]
    if cfg["method"] == "threshold":
        args += ["--method", "threshold"]
    elif cfg["method"] == "cellpose":
        cp = cfg.get("cellpose", {})
        args += ["--method", "cellpose", "--model", str(cp.get("model"))]
        if cp.get("diameter"):
            args += ["--diameter", str(cp["diameter"])]
        if cp.get("do_3D"):
            args.append("--do-3d")
    else:
        custom = cfg.get("custom", {})
        args += [
            "--method",
            "custom",
            "--fn",
            f"{custom.get('module')}:{custom.get('function', 'segment')}",
            "--fn-kwargs",
            json.dumps(custom.get("kwargs") or {}),
        ]
    return " ".join(shlex.quote(a) for a in args)


def parse_plan(output: str) -> dict[str, Any] | None:
    """The JSON plan ``patchworks segment --plan`` printed, if any."""
    start = output.find("{")
    end = output.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        plan = json.loads(output[start : end + 1])
    except ValueError:
        return None
    return plan if isinstance(plan, dict) else None


def plan_row(label: str, plan: dict[str, Any]) -> dict[str, Any]:
    """One readable table row for a plan."""

    def axes(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            return " × ".join(str(v) for v in value)
        return "" if value is None else str(value)

    gib = 1024**3
    row = {
        "segmentation": label,
        "image shape": axes(plan.get("shape")),
        "tile shape": axes(plan.get("tile_shape")),
        "grid": axes(plan.get("grid")),
        "tiles": plan.get("tiles"),
        "with signal": plan.get("tiles_with_signal"),
        "read per tile (GiB)": round(
            (plan.get("tile_read_bytes") or 0) / gib, 2
        ),
        "labels (GiB, raw)": round(
            (plan.get("labels_bytes_uncompressed") or 0) / gib, 1
        ),
    }
    if plan.get("estimated_seconds") is not None:
        row["estimate (h)"] = round(plan["estimated_seconds"] / 3600, 1)
    return row


def last_error_line(text: str) -> str:
    """The line of a traceback worth showing first (its last non-empty one)."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


# ---------------------------------------------------------------------------
# Remote file browser
# ---------------------------------------------------------------------------

# What the input of a run can be (a directory store counts as a file).
IMAGE_SUFFIXES = (
    ".ims",
    ".czi",
    ".lif",
    ".nd2",
    ".tif",
    ".tiff",
    ".zarr",
    ".zip",
)


def is_store(name: str) -> bool:
    """A directory that is one dataset rather than a folder to open."""
    return name.lower().endswith((".zarr", ".n5"))


def browse_entries(
    entries: list[tuple[str, bool]], *, dirs_only: bool = False
) -> list[tuple[str, str]]:
    """``(name, kind)`` for a remote listing, in the order to show them.

    *entries* are ``(name, is_dir)``. Kinds: ``"dir"`` (a folder to open),
    ``"store"`` (a ``.zarr`` folder, picked like a file), ``"image"`` (a
    file with an image suffix) and ``"file"``. Hidden entries are left out;
    folders come first, then stores, images and other files, each by name.
    With *dirs_only*, only folders (and stores, which are folders too).
    """
    rank = {"dir": 0, "store": 1, "image": 2, "file": 3}
    out = []
    for name, is_dir in entries:
        if name.startswith("."):
            continue
        if is_dir:
            kind = "store" if is_store(name) else "dir"
        elif dirs_only:
            continue
        else:
            kind = "image" if name.lower().endswith(IMAGE_SUFFIXES) else "file"
        out.append((name, kind))
    return sorted(out, key=lambda e: (rank[e[1]], e[0].lower()))


def parent_dir(path: str) -> str:
    """The folder above *path* (``/`` stays ``/``)."""
    path = path.rstrip("/")
    if not path:
        return "/"
    head = path.rsplit("/", 1)[0]
    return head or "/"


def join_remote(folder: str, name: str) -> str:
    return f"{folder.rstrip('/')}/{name}" if folder != "/" else f"/{name}"


# ---------------------------------------------------------------------------
# Job registry (on the cluster, so a reload can find its jobs again)
# ---------------------------------------------------------------------------


def registry_line(job: dict[str, Any]) -> str:
    return json.dumps(job, sort_keys=True) + "\n"


def parse_registry(text: str) -> list[dict[str, Any]]:
    """Jobs recorded in the registry, newest first; bad lines are skipped."""
    jobs = []
    for line in text.splitlines():
        try:
            job = json.loads(line)
        except ValueError:
            continue
        if isinstance(job, dict):
            jobs.append(job)
    return jobs[::-1]
