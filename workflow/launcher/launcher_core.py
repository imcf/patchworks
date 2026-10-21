"""Launcher logic without the UI: config, commands, host keys, job records.

Kept free of Streamlit (and of paramiko at import time) so it can be tested
on its own and reused by other front ends. ``app.py`` is the UI on top.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shlex
from pathlib import Path
from typing import Any

import yaml

# Run modes, in the order the UI offers them.
DRY_RUN = "Dry run"
SLURM_JOB = "Submit — controller as a SLURM job (recommended)"
LOGIN_NODE = "Submit — controller on the login node"
RUN_MODES = (DRY_RUN, SLURM_JOB, LOGIN_NODE)

# The methods the form has fields for that run a plugin: they fill space
# from a membrane stain, PlantSeg with a U-Net, the watershed from nuclei.
SEEDED = {
    "plantseg": "patchworks.plugins.plantseg",
    "watershed": "patchworks.plugins.watershed",
}
# PlantSeg's zoo models for cells from a membrane stain; any other can be
# typed (patchworks.plugins.plantseg.available_models() lists them all).
PLANTSEG_MODELS = (
    "generic_confocal_3D_unet",
    "generic_light_sheet_3D_unet",
    "confocal_3D_unet_ovules_ds2x",
    "lightsheet_3D_unet_root_ds1x",
)
PLANTSEG_SEGMENTATIONS = (
    "gasp",
    "mutex_ws",
    "multicut",
    "dt_watershed",
    "lifted_multicut",
    "nuclei_watershed",
)
# The PlantSeg modes that use nuclei: from a channel or another segmentation.
PLANTSEG_NUCLEI = ("lifted_multicut", "nuclei_watershed")

LOG_DIR = ".streamlit_launcher_logs"
REGISTRY = f"{LOG_DIR}/jobs.jsonl"


# ---------------------------------------------------------------------------
# Cluster presets and host keys
# ---------------------------------------------------------------------------


# What the launcher remembers between sessions, on this computer: how to
# connect and where the workflow is -- never a password.
SETTINGS_KEYS = ("preset", "host", "port", "username", "auth", "wf_dir")


def settings_path() -> Path:
    """``~/.config/patchworks/launcher.yaml`` (``$PATCHWORKS_LAUNCHER_SETTINGS``
    to put it elsewhere)."""
    import os

    override = os.environ.get("PATCHWORKS_LAUNCHER_SETTINGS")
    if override:
        return Path(override)
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "patchworks" / "launcher.yaml"


def load_settings(path: Path | None = None) -> dict[str, Any]:
    """The remembered settings; empty if none (or unreadable)."""
    try:
        data = yaml.safe_load((path or settings_path()).read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: data[k] for k in SETTINGS_KEYS if k in data}


def save_settings(changes: dict[str, Any], path: Path | None = None) -> None:
    """Remember *changes* (only :data:`SETTINGS_KEYS`; a password never).

    Best effort: a read-only home must not break connecting.
    """
    path = path or settings_path()
    data = {**load_settings(path)}
    data.update({k: v for k, v in changes.items() if k in SETTINGS_KEYS})
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False))
    except OSError:
        pass


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
        # Blank or "auto": sized from the objects when the tiles are planned
        # (an earlier run's table, or tiles tried with check --sample).
        "overlap": _yaml_or(v.get("overlap", "auto"), "auto"),
        "skip_empty": v.get("skip_empty", True),
        "empty_threshold": v.get("empty_threshold"),
        "tiles_per_job": v.get("tiles_per_job", 4),
        # segmentation + post-processing
        "method": v["method"],
        "fill_holes": v.get("fill_holes") or False,
        "open_radius": v.get("open_radius", 0),
        "dilate": v.get("dilate", 0),
        "dilate_gpu": bool(v.get("dilate_gpu", False)),
        "min_volume": v.get("min_volume"),
        # Grow from another segmentation's objects (one cell per nucleus).
        "seed_labels": v.get("seed_labels") or None,
        # Cellpose: one intensity range for every tile, or each its own.
        "normalize": v.get("normalize", "image"),
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
    elif method in SEEDED:
        kwargs: dict[str, Any] = {}
        if method == "plantseg":
            kwargs = {
                "model": v.get("ps_model") or PLANTSEG_MODELS[0],
                "segmentation": v.get("ps_segmentation", "gasp"),
                "beta": float(v.get("ps_beta", 0.6)),
                "post_minsize": int(v.get("ps_post_minsize", 100)),
            }
        else:
            kwargs = {"nuclei_min_size": int(v.get("ws_nuclei_min_size", 50))}
        foreground = v.get("foreground", "otsu")
        kwargs["foreground"] = None if foreground == "none" else foreground
        if v.get("max_radius_um"):
            kwargs["max_radius_um"] = float(v["max_radius_um"])
        kwargs.update(_yaml_or(v.get("extra_kwargs", ""), {}) or {})
        cfg["method"] = "custom"
        cfg["custom"] = {
            "module": SEEDED[method],
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


# Methods giving every voxel to some cell (their function says so with
# ``patchworks_stitch = "iou"``): cells touch at every tile seam.
FILLS_SPACE = ("patchworks.plugins.plantseg", "patchworks.plugins.watershed")


def fills_space(v: dict[str, Any]) -> bool:
    """Whether the form values *v* pick a method that needs iou stitching."""
    return v.get("method") in SEEDED or (
        v.get("method") == "custom"
        and str(v.get("custom_module", "")).strip() in FILLS_SPACE
    )


def config_problems(cfg: dict[str, Any]) -> list[str]:
    """What the form holds that the workflow would get wrong."""
    problems = []
    module = (cfg.get("custom") or {}).get("module")
    kwargs = (cfg.get("custom") or {}).get("kwargs") or {}
    seeds = cfg.get("seed_labels")
    if seeds and cfg.get("nuclei_channel") is not None:
        problems.append(
            f"{cfg['label_name']}: nuclei from a channel or from "
            f"{seeds!r}, not both (clear nuclei_channel)"
        )
    if seeds and seeds == cfg.get("label_name"):
        problems.append(f"{cfg['label_name']}: cannot grow from itself")
    if (
        module == SEEDED["watershed"]
        and not seeds
        and (cfg.get("nuclei_channel") is None)
    ):
        problems.append(
            f"{cfg['label_name']}: the watershed grows cells from nuclei: give "
            "a nuclei channel, or another segmentation's nuclei"
        )
    if module == SEEDED["plantseg"]:
        needs = kwargs.get("segmentation") in PLANTSEG_NUCLEI
        has = seeds or cfg.get("nuclei_channel") is not None
        if needs and not has:
            problems.append(
                f"{cfg['label_name']}: segmentation "
                f"{kwargs.get('segmentation')!r} uses nuclei: give a nuclei "
                "channel, or another segmentation's nuclei"
            )
        if seeds and not needs:
            problems.append(
                f"{cfg['label_name']}: only segmentation "
                f"{' or '.join(PLANTSEG_NUCLEI)} grows from "
                f"{seeds!r}; {kwargs.get('segmentation')!r} would ignore it"
            )
    if (
        cfg.get("method") == "custom"
        and module in FILLS_SPACE
        and cfg.get("stitch") != "iou"
    ):
        problems.append(
            f"{cfg['label_name']}: {module} gives every voxel to a cell, so "
            'cells touch at every tile seam and stitch "touch" would join '
            'them: use stitch "iou"'
        )
    work_dir = str(cfg.get("work_dir") or "").strip()
    if work_dir and not work_dir.startswith("/"):
        problems.append(
            f"work_dir must be an absolute path on the cluster: {work_dir!r}"
        )
    if is_glob(str(cfg.get("input") or "")) and not cfg.get("sequence_pattern"):
        problems.append(
            "a folder of TIFFs needs a sequence_pattern: which part of each "
            "file name is the z plane (and channel, time point)"
        )
    return problems


# ---------------------------------------------------------------------------
# A folder of single-plane TIFFs
# ---------------------------------------------------------------------------

TIFF_SUFFIXES = (".tif", ".tiff")


def is_glob(path: str) -> bool:
    return any(c in path for c in "*?[")


def tiff_names(entries: list[tuple[str, bool]]) -> list[str]:
    """The TIFF files of a folder listing (``(name, is_dir)`` pairs)."""
    return sorted(
        name
        for name, is_dir in entries
        if not is_dir
        and not name.startswith(".")
        and name.lower().endswith(TIFF_SUFFIXES)
    )


def tiff_glob(folder: str, names: list[str]) -> str:
    """A glob matching the TIFFs *names* of *folder*.

    A glob is case-sensitive, so it uses the suffix the files actually have
    (the most common one, if they are mixed).
    """
    suffixes = [n[n.rfind(".") :] for n in names]
    suffix = max(set(suffixes), key=suffixes.count) if suffixes else ".tif"
    return join_remote(folder, f"*{suffix}")


def guess_sequence_pattern(names: list[str]) -> str | None:
    """A ``sequence_pattern`` for file names such as ``x_Z012_C1.tif``.

    Looks for an axis letter (T, Z or C, either case) followed by digits,
    not preceded by another letter, that is in every name and changes
    between them -- a constant one (``T0``) is no axis. None if no z or
    channel index is found, or the names do not agree.
    """
    import re

    token = re.compile(r"(?<![A-Za-z])([TZCtzc])(\d+)")

    def tokens(name: str) -> list[tuple[str, str, str]]:
        # (letter, digits, the character before it)
        return [
            (m.group(1), m.group(2), name[m.start() - 1] if m.start() else "")
            for m in token.finditer(name)
        ]

    def numbered() -> str | None:
        # img0001.tif, img0002.tif ...: the number is the z plane.
        ends = [re.search(r"(\d+)\.tiff?$", n, re.IGNORECASE) for n in names]
        if all(ends) and len({m.group(1) for m in ends}) == len(names) > 1:
            return r"(?P<Z>\d+)\.(?i:tiff?)$"
        return None

    parsed = [tokens(n) for n in names]
    if not parsed or not parsed[0]:
        return numbered()
    first = parsed[0]
    if any([t[0] for t in p] != [t[0] for t in first] for p in parsed):
        return None  # not one naming scheme
    parts, seen = [], set()
    for i, (letter, _, before) in enumerate(first):
        values = {p[i][1] for p in parsed}
        axis = letter.upper()
        if len(values) < 2 or axis in seen:
            continue
        seen.add(axis)
        prefix = re.escape(before) if not before.isalnum() else ""
        parts.append(f"{prefix}{re.escape(letter)}(?P<{axis}>\\d+)")
    if not seen & {"Z", "C"}:
        return numbered()
    return ".*?".join(parts)


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


def jobname_prefix(label: str) -> str:
    """What the run's SLURM jobs are named after: ``pw-<label>_<run id>``.

    The SLURM executor names every job of a run after the run's id and
    refuses a ``--job-name``; a prefix is what can make them recognisable
    (in squeue and in SLURM's mails). As run_multi's
    ``slurm_jobname_prefix``: alphanumerics, ``_`` and ``-``, 50 at most.

    >>> jobname_prefix("nuclei_labels_cpsam")
    'pw-nuclei_labels_cpsam'
    """
    return f"pw-{re.sub(r'[^A-Za-z0-9_-]', '-', label)}"[:50]


def snakemake_command(
    config_path: str,
    mode: str,
    *,
    profile: str = "profile/slurm",
    label: str | None = None,
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
        if label:
            args += ["--slurm-jobname-prefix", jobname_prefix(label)]
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
        # pw-convert_<run id>, as run_multi's conversion.
        args += [
            "--workflow-profile",
            profile,
            "--slurm-jobname-prefix",
            jobname_prefix("convert"),
        ]
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
    gres: str = "",
    cpus: int = 1,
) -> str:
    """An sbatch script running the Snakemake *controller* as a small job.

    (Or, with *gres*, a short GPU job: ``check --sample`` timing tiles where
    the segment jobs run.)

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
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time}",
    ]
    for flag, value in (
        ("partition", partition),
        ("qos", qos),
        ("account", account),
        ("gres", gres),
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


def stop_command(job: dict[str, Any], *, force: bool = False) -> str:
    """Stop a launched run, letting Snakemake cancel the jobs it submitted.

    SIGINT is Ctrl-C: Snakemake then cancels its SLURM jobs before it
    exits, which a plain kill does not give it the chance to. *force* kills
    outright, for a run that does not stop.
    """
    if job.get("slurm_job"):
        jid = shlex.quote(str(job["slurm_job"]))
        if force:
            return f"scancel {jid}"
        # --full: the batch script's processes too (the controller).
        return f"scancel --full --signal=INT {jid}"
    # setsid made the run a process group of its own: signal all of it.
    pid = shlex.quote(str(job["pid"]))
    return f"kill -{'KILL' if force else 'INT'} -- -{pid}"


def run_jobs_in_queue(
    job: dict[str, Any],
    steps: list[dict[str, Any]],
    queue: list[dict[str, str]],
) -> list[str]:
    """The SLURM jobs of this run still in the queue: those its Snakemake
    log says it submitted, and its controller. Nobody else's: the ids come
    from the run's own log, not from job names."""
    queued = {row["job"] for row in queue}
    ids = [
        s["slurm_job"]
        for s in steps
        if s.get("slurm_job") and s["slurm_job"] in queued
    ]
    controller = str(job.get("slurm_job") or "")
    if controller and controller in queued:
        ids.append(controller)
    return list(dict.fromkeys(ids))


def cancel_command(ids: list[str]) -> str:
    """``scancel`` for *ids* (job numbers only)."""
    clean = [i for i in ids if str(i).isdigit()]
    return "scancel " + " ".join(clean) if clean else "true"


def unlock_command(job: dict[str, Any], workflow_dir: str) -> str | None:
    """Release the Snakemake lock a killed run left behind."""
    config = job.get("config")
    if not config:
        return None
    kind = job.get("kind")
    if kind == "multi":
        args = ["python", "scripts/run_multi.py", "--config", config]
    elif kind == "convert":
        if not job.get("work_dir"):
            return None
        args = [
            "snakemake",
            "-s",
            f"{workflow_dir}/Snakefile",
            "--configfile",
            config,
            "--directory",
            f"{job['work_dir'].rstrip('/')}/.snakemake_convert",
        ]
    elif kind == "single":
        args = ["snakemake", "-s", "Snakefile", "--configfile", config]
    else:
        return None
    return " ".join(shlex.quote(a) for a in [*args, "--unlock"])


def check_command(
    config: str, *, sample: int = 0, preview: str | None = None
) -> str:
    """``scripts/check.py`` on a run's config (or multi config), as JSON;
    with *preview*, a picture of one segmented tile per segmentation."""
    args = ["python", "scripts/check.py", config, "--json"]
    if sample:
        args += ["--sample", str(sample)]
    if preview:
        args += ["--preview", preview]
    return " ".join(shlex.quote(a) for a in args)


def fetch_command(config: str) -> str:
    """``scripts/fetch_models.py``: the run's models, downloaded now."""
    return " ".join(
        shlex.quote(a) for a in ("python", "scripts/fetch_models.py", config)
    )


def channels_command(config: str) -> str:
    """``scripts/channels.py``: a picture per channel of the converted image."""
    return " ".join(
        shlex.quote(a)
        for a in ("python", "scripts/channels.py", config, "--json")
    )


# The pretrained models the cluster's Cellpose accepts (v3: cyto3, nuclei
# ...; v4: cpsam ...), printed after a tag so log lines cannot confuse it.
CELLPOSE_MODELS_COMMAND = (
    "python -c 'import json; "
    "from patchworks.plugins.cellpose import available_models; "
    'print("MODELS:" + json.dumps(available_models()))\''
)
# When the cluster cannot say: Cellpose 4's model, then Cellpose 3's.
CELLPOSE_FALLBACK_MODELS = ["cpsam", "cyto3", "nuclei", "cyto2"]


def parse_models(output: str) -> list[str]:
    """The model names CELLPOSE_MODELS_COMMAND printed ([] if none)."""
    tag = parse_tag(output, "MODELS")
    try:
        names = json.loads(tag) if tag else []
    except ValueError:
        return []
    return [str(n) for n in names] if isinstance(names, list) else []


def run_status_command(config: str, since: float | None = None) -> str:
    """``scripts/status.py`` on a run's config (or multi config), as JSON.

    *since* (seconds since the epoch): only failures logged after it --
    the run's own, not an earlier run's (an environment variable, which
    an older status.py simply ignores).
    """
    cmd = " ".join(
        shlex.quote(a)
        for a in ("python", "scripts/status.py", config, "--json")
    )
    return f"PATCHWORKS_STATUS_SINCE={int(since)} {cmd}" if since else cmd


def started_epoch(job: dict[str, Any]) -> float | None:
    """When the job was launched (its ``started`` stamp, this computer's
    local time), as seconds since the epoch; None if unknown."""
    import datetime as _dt

    try:
        stamp = _dt.datetime.strptime(str(job.get("started")), "%Y%m%d-%H%M%S")
    except ValueError:
        return None
    return stamp.timestamp()


def parse_json_output(output: str) -> dict[str, Any] | None:
    """The last JSON object a script printed (logs may come before it)."""
    for line in reversed(output.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                data = json.loads(line)
            except ValueError:
                continue
            if isinstance(data, dict):
                return data
    return None


# ---------------------------------------------------------------------------
# SLURM resources: the workflow profile, and what the account may use
# ---------------------------------------------------------------------------

# The resources the launcher lets you change, per rule.
RESOURCE_KEYS = ("slurm_partition", "qos", "gres", "runtime", "mem_mb")


def profile_resources(text: str | None) -> dict[str, dict[str, Any]]:
    """Each rule's resources in a workflow profile, defaults filled in."""
    profile = yaml.safe_load(text or "") or {}
    defaults = profile.get("default-resources") or {}
    rules = profile.get("set-resources") or {}
    return {
        rule: {**defaults, **(values or {})}
        for rule, values in {"default": {}, **rules}.items()
    }


def first_attempt(value: Any) -> int | None:
    """A profile memory or time for the first attempt (``"attempt * 64000"``)."""
    import re

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    numbers = re.findall(r"\d+", str(value))
    return int(numbers[0]) if len(numbers) == 1 else None


def profile_with_overrides(
    text: str, overrides: dict[str, dict[str, Any]]
) -> str:
    """The profile *text* with *overrides* (``{rule: {key: value}}``).

    A memory set for ``segment`` grows with each attempt like the profile's
    own (``attempt * N``), and goes to ``prepare`` too: prepare sizes
    ``tile_shape: "auto"`` from its own allocation, standing in for the
    segment job's.
    """
    profile = yaml.safe_load(text or "") or {}
    rules = profile.setdefault("set-resources", {}) or {}
    profile["set-resources"] = rules
    overrides = {r: dict(v) for r, v in overrides.items()}
    seg_mem = (overrides.get("segment") or {}).get("mem_mb")
    if seg_mem is not None:
        overrides.setdefault("prepare", {})["mem_mb"] = seg_mem
    for rule, values in overrides.items():
        target = rules.setdefault(rule, {}) or {}
        rules[rule] = target
        for key, value in values.items():
            if value in (None, ""):
                target.pop(key, None)
            elif key == "mem_mb":
                target[key] = f"attempt * {int(value)}"
            else:
                target[key] = value
    return yaml.safe_dump(profile, sort_keys=False)


def slurm_minutes(text: str) -> int | None:
    """SLURM's ``[D-]HH:MM:SS`` (or ``MM:SS``, ``MM``) as minutes."""
    text = (text or "").strip()
    if not text or text.upper() in ("UNLIMITED", "INFINITE"):
        return None
    days = 0
    try:
        if "-" in text:
            d, text = text.split("-", 1)
            days = int(d)
        parts = [int(p) for p in text.split(":")]
    except ValueError:
        return None  # not a time: nothing to check against
    if len(parts) == 3:
        h, m, sec = parts
    elif len(parts) == 2:
        h, m, sec = 0, *parts
    else:
        h, m, sec = 0, parts[0], 0
    return days * 1440 + h * 60 + m + (1 if sec else 0)


# What the cluster offers and this account may use, in one round trip:
# partitions (sinfo: availability, time limit, GPUs, CPUs
# allocated/idle/other/total), associations (account, partition, QOS,
# default QOS) and each QOS's MaxWall.
CLUSTER_INFO_COMMAND = (
    'echo "#partitions"; sinfo -h -o "%P|%a|%l|%G|%C" 2>/dev/null; '
    'echo "#assoc"; sacctmgr -nP show assoc user="$USER" '
    "format=account,partition,qos,defaultqos 2>/dev/null; "
    'echo "#qos"; sacctmgr -nP show qos format=name,maxwall 2>/dev/null'
)


def _gres_options(gres: str) -> set[str]:
    """What a job may ask for of a node's ``%G`` (``gpu:rtx4090:8(S:0-1)``):
    one GPU, of any type and of this type."""
    options = set()
    for item in gres.split(","):
        item = item.split("(")[0].strip()
        parts = item.split(":")
        if len(parts) >= 2 and parts[0] and parts[0] != "(null)":
            options.add(f"{parts[0]}:1")
            if len(parts) == 3 and not parts[1].isdigit():
                options.add(f"{parts[0]}:{parts[1]}:1")
    return options


def parse_cluster_info(output: str) -> dict[str, Any]:
    """Partitions, the QOS and accounts this account may use, QOS limits."""
    partitions: dict[str, dict[str, Any]] = {}
    assoc: dict[str, set[str]] = {}
    accounts: list[str] = []
    maxwall: dict[str, int | None] = {}
    default_qos: dict[str, str] = {}
    default = None
    part = None
    for line in output.splitlines():
        line = line.strip()
        if line in ("#partitions", "#assoc", "#qos"):
            part = line
            continue
        if not line or "|" not in line:
            continue
        fields = line.split("|")
        if part == "#partitions" and len(fields) >= 5:
            name, avail, limit, gres, cpus = fields[:5]
            if name.endswith("*"):  # sinfo marks the default partition
                name = name[:-1]
                default = name
            entry = partitions.setdefault(
                name,
                {
                    "up": avail == "up",
                    "timelimit": slurm_minutes(limit),
                    "gres": set(),
                    "idle_cpus": 0,
                    "cpus": 0,
                },
            )
            entry["gres"] |= _gres_options(gres)
            # %C is allocated/idle/other/total; a partition may span lines.
            counts = cpus.split("/")
            if len(counts) == 4 and all(c.isdigit() for c in counts):
                entry["idle_cpus"] += int(counts[1])
                entry["cpus"] += int(counts[3])
        elif part == "#assoc":
            if len(fields) >= 3:
                account, partition, qos = fields[:3]
                if account and account not in accounts:
                    accounts.append(account)
                # The QOS a job gets with none asked for (the first
                # association's, the account's default, comes first).
                if len(fields) >= 4 and fields[3]:
                    default_qos.setdefault(partition, fields[3])
            else:
                partition, qos = fields[0], fields[-1]
            assoc.setdefault(partition, set()).update(
                q for q in qos.split(",") if q
            )
        elif part == "#qos":
            maxwall[fields[0]] = slurm_minutes(fields[-1])
    for entry in partitions.values():
        entry["gres"] = sorted(entry["gres"])
    return {
        "partitions": partitions,
        "default_partition": default,
        "assoc": assoc,
        "accounts": accounts,
        "maxwall": maxwall,
        "default_qos": default_qos,
    }


def default_qos(info: dict[str, Any], partition: str | None = None) -> str:
    """The QOS a job gets on *partition* when it asks for none ("" when
    not known)."""
    defaults = info.get("default_qos") or {}
    part = partition or info.get("default_partition") or ""
    return defaults.get(part) or defaults.get("") or ""


def fitting_qos(
    info: dict[str, Any], partition: str | None, minutes: int | None
) -> str:
    """The QOS this account may use on *partition* whose MaxWall allows
    *minutes*, the shortest such (shorter ones usually start sooner); ""
    when none is known to."""
    maxwall = info.get("maxwall") or {}
    fitting = [
        q
        for q in usable_qos(info, partition)
        if q in maxwall
        and (not minutes or maxwall[q] is None or maxwall[q] >= minutes)
    ]
    return min(
        fitting, key=lambda q: (maxwall[q] is None, maxwall[q] or 0), default=""
    )


def qos_for(
    info: dict[str, Any], partition: str | None, minutes: int | None
) -> str:
    """The QOS to ask for so a job of *minutes* is accepted: "" when the
    default QOS allows it, else :func:`fitting_qos`."""
    if not minutes:
        return ""
    default = default_qos(info, partition)
    if not default:
        # Not known (sacctmgr shows no default when it is inherited from
        # the account): name one that surely allows the time.
        return fitting_qos(info, partition, minutes)
    limit = (info.get("maxwall") or {}).get(default, None)
    if limit is None or limit >= minutes:
        return ""
    return fitting_qos(info, partition, minutes)


def better_qos(
    info: dict[str, Any],
    partition: str | None,
    qos: str | None,
    minutes: int | None,
) -> str:
    """The QOS to switch to when *qos* (blank: the default QOS) is known
    to be too short for *minutes*; "" when it is not, or none fits."""
    effective = qos or default_qos(info, partition)
    limit = (info.get("maxwall") or {}).get(effective) if effective else None
    if not minutes or limit is None or limit >= minutes:
        return ""
    return fitting_qos(info, partition, minutes)


def usable_qos(info: dict[str, Any], partition: str | None = None) -> list[str]:
    """The QOS this account may use, on *partition* when known."""
    assoc = info.get("assoc") or {}
    if not assoc:
        return []
    if partition is None:
        return sorted(set().union(*assoc.values()))
    # An association without a partition holds on every partition.
    return sorted(assoc.get(partition, set()) | assoc.get("", set()))


def partition_rows(info: dict[str, Any]) -> list[dict[str, Any]]:
    """The cluster's partitions as a readable table, GPU ones first."""
    rows = []
    for name, p in info.get("partitions", {}).items():
        limit = p["timelimit"]
        rows.append(
            {
                "partition": name
                + (
                    " (default)"
                    if name == info.get("default_partition")
                    else ""
                ),
                "state": "up" if p["up"] else "DOWN",
                "time limit": "none"
                if limit is None
                else f"{limit // 1440}-{limit % 1440 // 60:02d}:{limit % 60:02d}"
                if limit >= 1440
                else f"{limit // 60}:{limit % 60:02d}",
                "GPUs": ", ".join(p["gres"]) or "-",
                "idle CPUs": f"{p['idle_cpus']} / {p['cpus']}",
                "your QOS": ", ".join(usable_qos(info, name)) or "-",
            }
        )
    return sorted(rows, key=lambda r: (r["GPUs"] == "-", r["partition"]))


# Memory of common GPU types (GB), as SLURM names them in a node's gres.
# Where a type comes in two sizes (A100 40/80 GB, V100 16/32 GB) the
# smaller: a tile sized for it fits both.
GPU_MEMORY_GB = {
    "a100": 40,
    "a10": 24,
    "a30": 24,
    "a40": 48,
    "a6000": 48,
    "rtx_a6000": 48,
    "a5000": 24,
    "rtx_a5000": 24,
    "a4000": 16,
    "h100": 80,
    "h200": 141,
    "l4": 24,
    "l40": 48,
    "l40s": 48,
    "rtx2080ti": 11,
    "rtx3090": 24,
    "rtx4090": 24,
    "rtx6000": 24,
    "rtx8000": 48,
    "titanrtx": 24,
    "gtx1080ti": 11,
    "p100": 16,
    "t4": 16,
    "v100": 16,
}


def gpu_memory_gb(
    resources: dict[str, Any], info: dict[str, Any] | None
) -> tuple[int | None, str]:
    """The segment GPU's memory, from its gres or its partition's GPUs.

    Returns ``(GB, which GPU)``; ``(None, why)`` when not known. With
    several types on the partition, the smallest: the job may land on any.
    """
    gres = str(resources.get("gres") or "")
    parts = gres.split(":")
    if len(parts) == 3 and not parts[1].isdigit():
        types = [parts[1]]
    else:
        part = resources.get("slurm_partition") or (info or {}).get(
            "default_partition"
        )
        offered = ((info or {}).get("partitions") or {}).get(part, {})
        types = [
            g.split(":")[1]
            for g in offered.get("gres", [])
            if g.count(":") == 2
        ]
    known = {
        t: GPU_MEMORY_GB[t.lower()] for t in types if t.lower() in GPU_MEMORY_GB
    }
    if not known:
        return None, (
            f"GPU type {', '.join(types)} unknown" if types else "no GPU type"
        )
    gpu = min(known, key=known.get)
    return known[gpu], gpu


def needs_gpu_memory(cfg: dict[str, Any]) -> bool:
    """Cellpose on the GPU with ``tile_shape: "auto"`` and no
    ``gpu_memory_gb``: prepare would size its tiles for an 8 GB GPU (the
    workflow's ``_pw.sized_for_fallback_gpu``)."""
    return (
        cfg.get("tile_shape", "auto") == "auto"
        and not cfg.get("gpu_memory_gb")
        and cfg.get("method") == "cellpose"
        and bool((cfg.get("cellpose") or {}).get("gpu", True))
    )


def with_gpu_memory(files: dict[str, dict], gb: int) -> dict[str, dict]:
    """*files* with ``gpu_memory_gb`` set where tiles would be sized for
    the 8 GB fallback."""
    return {
        path: {**cfg, "gpu_memory_gb": gb}
        if "label_name" in cfg and needs_gpu_memory(cfg)
        else cfg
        for path, cfg in files.items()
    }


def resource_problems(
    resources: dict[str, dict[str, Any]], info: dict[str, Any]
) -> list[str]:
    """What sbatch would refuse, or what would never start.

    A partition that does not exist or is down, a QOS the account lacks,
    a runtime over the QOS's MaxWall or the partition's time limit: sbatch
    refuses such a job (QOSMaxWallDurationPerJobLimit ...) or it waits
    forever, with an empty log either way.
    """
    assoc, maxwall = info.get("assoc") or {}, info.get("maxwall") or {}
    partitions = info.get("partitions") or {}
    allowed = set().union(*assoc.values()) if assoc else set()
    problems = []
    for rule, res in resources.items():
        if rule == "default":
            continue
        qos = res.get("qos")
        runtime = first_attempt(res.get("runtime"))
        part = res.get("slurm_partition") or info.get("default_partition")
        if part and partitions:
            if part not in partitions:
                problems.append(
                    f"{rule}: there is no partition {part!r} on this cluster "
                    f"({', '.join(sorted(partitions))})"
                )
            elif not partitions[part]["up"]:
                problems.append(f"{rule}: partition {part!r} is down")
            else:
                limit = partitions[part]["timelimit"]
                if runtime and limit and runtime > limit:
                    problems.append(
                        f"{rule}: {_duration(runtime)} is longer than "
                        f"partition {part!r} allows ({_duration(limit)}): "
                        "the job would never start"
                    )
                if res.get("gres") and not partitions[part]["gres"]:
                    problems.append(
                        f"{rule}: asks for {res['gres']} on {part!r}, which "
                        "has no GPUs"
                    )
        if part and assoc and part not in assoc and "" not in assoc:
            problems.append(
                f"{rule}: your account has no association with partition "
                f"{part!r} (it has: {', '.join(sorted(p for p in assoc if p))})"
            )
        # The QOS allowed with this partition, or on any if unknown.
        usable = assoc.get(part) or assoc.get("") or allowed
        if qos and assoc and qos not in usable:
            problems.append(
                f"{rule}: QOS {qos!r} is not one your account may use "
                f"{'on ' + repr(part) + ' ' if part in assoc else ''}"
                f"({', '.join(sorted(usable))})"
            )
        # No QOS asked for: the account's default QOS applies.
        effective = qos or default_qos(info, part)
        limit = maxwall.get(effective) if effective else None
        if runtime and limit and runtime > limit:
            better = better_qos(info, part, qos, runtime)
            problems.append(
                f"{rule}: {_duration(runtime)} is longer than "
                + (
                    f"QOS {qos!r}"
                    if qos
                    else f"your default QOS {effective!r} (QOS left blank)"
                )
                + f" allows ({_duration(limit)}), so sbatch would refuse the "
                "job: "
                + (
                    f"choose QOS {better!r}"
                    if better
                    else "no QOS of yours allows it, lower the time"
                )
            )
    return problems


# What sbatch's refusals mean, and what to change.
SBATCH_ERRORS = {
    "QOSMaxWallDurationPerJobLimit": "the time asked for is over what the "
    "QOS allows (with QOS left blank, your default QOS)",
    "InvalidQOS": "the QOS is not one your account may use on this partition",
    "Invalid qos specification": "the QOS is not one your account may use "
    "on this partition",
    "Invalid account": "the account is not one you belong to, or not on "
    "this partition",
    "invalid partition": "there is no such partition",
    "PartitionTimeLimit": "the time asked for is over the partition's limit",
    "AssocMaxSubmitJobLimit": "your account already has as many jobs "
    "queued as it may",
    "QOSMaxSubmitJobPerUserLimit": "you already have as many jobs queued "
    "as this QOS allows",
    "Requested node configuration is not available": "no node of the "
    "partition has the memory, CPUs or GPUs asked for",
}


def explain_sbatch_error(
    text: str, info: dict[str, Any] | None, slurm: dict[str, Any]
) -> str | None:
    """Why sbatch refused the controller job, and what to choose instead."""
    lowered = text.lower()
    reason = next(
        (why for key, why in SBATCH_ERRORS.items() if key.lower() in lowered),
        None,
    )
    if reason is None:
        return None
    info = info or {}
    part = slurm.get("partition") or None
    lines = [reason[0].upper() + reason[1:] + "."]
    if "time" in reason or "qos" in reason.lower():
        minutes = slurm_minutes(str(slurm.get("time") or ""))
        better = fitting_qos(info, part, minutes) if minutes else ""
        if better and better != slurm.get("qos"):
            lines.append(
                f"Choose QOS {better!r}, which allows {slurm['time']}."
            )
        maxwall = info.get("maxwall") or {}
        offered = [
            f"{q} ({'no limit' if maxwall.get(q) is None else _duration(maxwall[q])})"
            if q in maxwall
            else q
            for q in usable_qos(info, part)
        ]
        if offered:
            lines.append("Your QOS: " + ", ".join(offered) + ".")
    return " ".join(lines)


def _duration(minutes: int) -> str:
    """*minutes* as SLURM writes a time: D-HH:MM:SS or HH:MM:SS."""
    days, rest = divmod(minutes, 1440)
    clock = f"{rest // 60:02d}:{rest % 60:02d}:00"
    return f"{days}-{clock}" if days else clock


# Your patchworks jobs in the queue: the run's controller (pw-<label>), its
# segmentations' jobs (pw-<label>_<uuid>) and a single run's (patchworks_).
QUEUE_COMMAND = 'squeue --me -h -o "%i|%j|%T|%r|%M|%l|%P" 2>/dev/null'

# Why a job waits, in words, and whether it ever starts.
PENDING_REASONS = {
    "Priority": "waiting its turn: other jobs come first",
    "Resources": "next in line, waiting for a node to free up",
    "Dependency": "waiting for the job it depends on",
    "QOSMaxJobsPerUserLimit": "you already run as many jobs as this QOS allows",
    "QOSMaxGRESPerUser": "you already use as many GPUs as this QOS allows",
    "AssocGrpGRES": "your group already uses all its GPUs",
    "AssocMaxJobsLimit": "your account already runs its most jobs",
    "QOSGrpCpuLimit": "the QOS's CPUs are all in use",
    "ReqNodeNotAvail": "the nodes it needs are down or reserved",
    "BeginTime": "starts later, as asked",
    "QOSMaxWallDurationPerJobLimit": "NEVER STARTS: its time is over the "
    "QOS's limit; cancel it, raise the QOS or lower the runtime",
    "PartitionTimeLimit": "NEVER STARTS: its time is over the partition's "
    "limit; cancel it and lower the runtime",
    "PartitionNodeLimit": "NEVER STARTS: asks for more nodes than the "
    "partition has",
    "InvalidQOS": "NEVER STARTS: a QOS the account may not use",
}


_RUN_ID = re.compile(
    r"_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def parse_queue(output: str) -> list[dict[str, str]]:
    """Your patchworks jobs in the queue, with why each waits."""
    rows = []
    for line in output.splitlines():
        fields = line.strip().split("|")
        if len(fields) < 7:
            continue
        jid, name, state, reason, elapsed, limit, partition = fields[:7]
        if not name.startswith(("pw-", "patchworks")):
            continue
        rows.append(
            {
                "job": jid,
                # Without the run id the SLURM executor appends to every
                # job's name (it finds its jobs by it): pw-convert, pw-cells.
                "name": _RUN_ID.sub("", name),
                "state": state,
                "partition": partition,
                "running for": elapsed if state == "RUNNING" else "",
                "time limit": limit,
                "why it waits": ""
                if state == "RUNNING"
                else PENDING_REASONS.get(reason, reason),
            }
        )
    return rows


# The lines of Snakemake's log that say which step ran where: each job's
# rule, wildcards, log and SLURM job, and whether it failed or finished.
STEPS_FILTER = (
    r"^(local)?(rule|checkpoint) |^\s+(log|jobid|wildcards|resources):|"
    r"has been submitted|"
    r"^Error in rule|Finished job"
)


def steps_command(log: str) -> str:
    """The step lines of the controller's log *log* (see :data:`STEPS_FILTER`)."""
    return (
        f"grep -E {shlex.quote(STEPS_FILTER)} {shlex.quote(log)} "
        "2>/dev/null | tail -n 20000"
    )


# Why a controller ended without running anything, from its log.
CONTROLLER_PROBLEMS = (
    (
        r"LockException|cannot be locked|Directory cannot be locked",
        "Snakemake's lock from a run that was killed: press Unlock, then "
        "Relaunch",
    ),
    (
        r"IncompleteFilesException",
        "files of a killed step are incomplete: Relaunch (the profile reruns "
        "them)",
    ),
    (
        r"DUE TO TIME LIMIT",
        "the controller's own time ran out: Relaunch, it resumes",
    ),
    (r"oom[_-]kill|Out Of Memory", "the controller ran out of memory"),
    (
        r"run_multi on \S+ \(pid \d+.*is driving",
        "an earlier run that was killed left its marker: if none of your "
        "jobs runs (Cancel jobs shows none), delete the .run_multi.pid file "
        "named here, then Relaunch",
    ),
)


def controller_problem(text: str) -> tuple[str, str | None] | None:
    """The error that ended a controller, and what to do; None if its log
    shows none."""
    for pattern, hint in CONTROLLER_PROBLEMS:
        # The last line that matches: after an exception's bare name comes
        # its message.
        found = re.findall(f".*(?:{pattern}).*", text)
        if found:
            return found[-1].strip()[:300], hint
    errors = [
        line.strip()
        for line in text.splitlines()
        if re.search(r"Error|Exception|error:", line)
    ]
    return (errors[-1][:300], None) if errors else None


_SUBMITTED = re.compile(
    r"Job (\d+) has been submitted with SLURM jobid (\d+) \(log: ([^)]+)\)"
)
_FINISHED = re.compile(r"Finished job(?:id:)? (\d+)")


def parse_steps(text: str) -> list[dict[str, Any]]:
    """The steps Snakemake ran, from its log, the latest attempt of each.

    Each: ``rule``, ``wildcards``, ``log`` (the step's own log),
    ``slurm_job`` and ``slurm_log`` (when submitted to SLURM), ``mem_mb``,
    ``attempt``, ``state`` ("submitted", "failed" or "done") and
    ``earlier``: the attempts before, each with its SLURM job, log and
    memory.
    """
    # Keyed by job id, rule and log: run_multi's Snakemake processes share
    # one log, each numbering its jobs from 0.
    steps: dict[tuple, dict[str, Any]] = {}

    def latest(jid: str, rule: str | None = None) -> dict[str, Any] | None:
        return next(
            (
                v
                for k, v in reversed(steps.items())
                if k[0] == jid and rule in (None, k[1])
            ),
            None,
        )

    block: dict[str, Any] | None = None  # a "rule" or "Error in rule" block
    for line in text.splitlines():
        stripped = line.strip()
        # A checkpoint (prepare) is announced as "checkpoint prepare:".
        start = re.match(
            r"(?:(?:local)?(?:rule|checkpoint)|Error in rule) (\w+):$", stripped
        )
        if start:
            block = {
                "rule": start.group(1),
                "wildcards": "",
                "log": None,
                "jid": None,
                "error": stripped.startswith("Error"),
            }
            continue
        if block is not None and line[:1].isspace() and ":" in stripped:
            key, value = (x.strip() for x in stripped.split(":", 1))
            step = block.get("step")
            if key == "jobid":
                block["jid"] = jid = value.split()[0]
                if block["error"]:
                    failed = latest(jid, block["rule"])
                    if failed:
                        failed["state"] = "failed"
                    continue
                key = (jid, block["rule"], block["log"])
                previous = steps.pop(key, None)  # re-inserted: latest last
                steps[key] = block["step"] = {
                    "rule": block["rule"],
                    "wildcards": block["wildcards"],
                    "log": block["log"],
                    "slurm_job": None,
                    "slurm_log": None,
                    "mem_mb": None,
                    "attempt": previous["attempt"] + 1 if previous else 1,
                    "state": "submitted",
                    "earlier": (
                        previous["earlier"]
                        + [
                            {
                                k: previous[k]
                                for k in (
                                    "attempt",
                                    "slurm_job",
                                    "slurm_log",
                                    "mem_mb",
                                )
                            }
                        ]
                        if previous
                        else []
                    ),
                }
            elif key == "resources" and step and not block["error"]:
                mem = re.search(r"\bmem_mb=(\d+)", value)
                step["mem_mb"] = int(mem.group(1)) if mem else None
            elif key == "log" and not block["error"]:
                block["log"] = value.split(",")[0].split(" (")[0].strip()
            elif key == "wildcards" and not block["error"]:
                block["wildcards"] = value
                if step:
                    step["wildcards"] = value
            continue
        block = None
        submitted = _SUBMITTED.search(line)
        if submitted:
            # run_multi's processes share this log and all number from 0:
            # the latest step with this number not yet submitted.
            jid = submitted.group(1)
            step = next(
                (
                    v
                    for k, v in reversed(steps.items())
                    if k[0] == jid and v["slurm_job"] is None
                ),
                None,
            ) or latest(jid)
            if step:
                step.update(
                    slurm_job=submitted.group(2),
                    slurm_log=submitted.group(3).strip(),
                )
            continue
        finished = _FINISHED.search(line)
        if finished:
            rule = re.search(r"\(Rule: (\w+)\)", line)
            step = latest(finished.group(1), rule.group(1) if rule else None)
            if step:
                step["state"] = "done"
    return list(steps.values())


# How SLURM itself ended a job, in its log (the executor's slurm_logs).
KILLED = (
    (r"oom[_-]kill|Out Of Memory|OUT_OF_MEMORY", "out of memory"),
    (r"DUE TO TIME LIMIT", "out of time"),
    (r"CANCELLED", "cancelled"),
)


def why_killed(text: str) -> str | None:
    """Why SLURM ended a job, from the end of its log; None if it did not."""
    return next((why for pat, why in KILLED if re.search(pat, text)), None)


_PROGRESS = re.compile(
    r"(?P<label>[^\s:][^:]*): (?P<done>[\d,]+)/(?P<total>[\d,]+) "
    r"\((?P<pct>\d+)%\) after (?P<elapsed>\S+), ~(?P<left>\S+) left"
)


TAIL_MARK = "@@patchworks-tail@@"


def tails_command(lines: int) -> str:
    """A command printing the last *lines* of each file named on its
    standard input, one path per line, each after a :data:`TAIL_MARK` line.

    The paths go on standard input, not the command line: a multi run has
    hundreds of logs with long paths, and a command that long exceeds the
    largest message an SSH server accepts -- it drops the whole connection.
    """
    loop = (
        "while IFS= read -r p; do "
        f'echo "{TAIL_MARK}"; tail -n {int(lines)} -- "$p" 2>/dev/null; '
        "done"
    )
    return f"bash -c {shlex.quote(loop)}"  # whatever the login shell is


def parse_tails(output: str, paths: list[str]) -> dict[str, str]:
    """:func:`tails_command`'s output for *paths*, back to ``{path: text}``."""
    found: dict[str, list[str]] = {p: [] for p in paths}
    i = -1
    for line in output.splitlines():
        if line == TAIL_MARK:
            i += 1
        elif 0 <= i < len(paths):
            found[paths[i]].append(line)
    return {p: "\n".join(lines) for p, lines in found.items()}


def last_progress(text: str) -> dict[str, Any] | None:
    """The last progress line a step logged (``patchworks._progress``):
    ``label``, ``fraction``, ``elapsed``, ``left``."""
    for line in reversed(text.splitlines()):
        m = _PROGRESS.search(line)
        if m:
            done = int(m["done"].replace(",", ""))
            total = int(m["total"].replace(",", ""))
            return {
                "label": m["label"].split(" - ")[-1].strip(),
                "fraction": min(1.0, done / max(1, total)),
                "done": done,
                "total": total,
                "elapsed": m["elapsed"],
                "left": m["left"],
            }
    return None


def status_command(job: dict[str, Any]) -> str:
    """Shell line printing RUNNING/PENDING/DONE for a registered job, or
    ``UNKNOWN <why>`` when SLURM did not answer -- never DONE for that.

    All your jobs, not ``-j <id>``: squeue errors out for an id it has
    already forgotten, which is a finished job, not a failure.
    """
    if job.get("slurm_job"):
        jid = str(job["slurm_job"])
        if not jid.isdigit():
            return "echo DONE"
        return (
            'out=$(squeue --me -h -o "%i %T" 2>&1) || '
            '{ echo "UNKNOWN $out" | head -c 300; exit 0; }; '
            f"s=$(printf '%s\\n' \"$out\" | awk '$1 == {jid} {{print $2}}'); "
            'echo "${s:-DONE}"'
        )
    pid = shlex.quote(str(job["pid"]))
    return f"kill -0 {pid} 2>/dev/null && echo RUNNING || echo DONE"


def plan_row(label: str, plan: dict[str, Any]) -> dict[str, Any]:
    """One readable table row for a plan (``_pw.plan_run``)."""

    def axes(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            return " × ".join(str(v) for v in value)
        return "" if value is None else str(value)

    def known(value: Any) -> Any:
        return "?" if value is None else value

    gib = 1024**3
    row = {
        "segmentation": label,
        "image shape": axes(plan.get("shape")),
        "tile shape": axes(plan.get("tile_shape")),
        "grid": axes(plan.get("grid")),
        "tiles": plan.get("tiles"),
        "with signal": known(plan.get("tiles_with_signal")),
        "GPU jobs": known(plan.get("gpu_jobs")),
        "read per tile (GiB)": round(
            (plan.get("tile_read_bytes") or 0) / gib, 2
        ),
        "labels (GiB, raw)": round(
            (plan.get("labels_bytes_uncompressed") or 0) / gib, 1
        ),
        "overlap": axes(plan.get("overlap")),
    }
    if plan.get("overlap_source"):
        row["overlap from"] = plan["overlap_source"]
    if plan.get("seconds_per_tile") is not None:
        row["s / tile"] = round(plan["seconds_per_tile"], 1)
    if plan.get("hours_per_job") is not None:
        row["h / job"] = round(plan["hours_per_job"], 2)
        row["GPU hours"] = round(plan["gpu_hours"], 1)
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


def registry_without(text: str, names: set[str]) -> str:
    """The registry *text* without the jobs named *names* (the list only:
    their logs and results stay)."""
    kept = []
    for line in text.splitlines():
        try:
            job = json.loads(line)
        except ValueError:
            continue
        if isinstance(job, dict) and job.get("name") not in names:
            kept.append(line + "\n")
    return "".join(kept)


# Which of the registry's jobs still run: all your SLURM jobs (squeue with
# a list of ids fails as a whole once one id is purged) and each PID.
def running_command(jobs: list[dict[str, Any]]) -> str:
    pids = [str(j["pid"]) for j in jobs if str(j.get("pid", "")).isdigit()]
    return "; ".join(
        ["squeue --me -h -o %i 2>/dev/null"]
        + [f"kill -0 {p} 2>/dev/null && echo PID{p}" for p in pids]
    )


def ended_jobs(jobs: list[dict[str, Any]], output: str) -> set[str]:
    """The names of *jobs* no longer running, from :func:`running_command`'s
    *output*."""
    running = {line.strip() for line in output.splitlines()}
    return {
        j["name"]
        for j in jobs
        if str(j.get("slurm_job") or "") not in running
        and f"PID{j.get('pid')}" not in running
    }


DISMISSED = f"{LOG_DIR}/dismissed_failures.json"


def failure_key(failure: dict[str, Any]) -> str:
    """What identifies a failure: its log and its error. A step that fails
    again differently is a new failure, shown again."""
    return f"{failure.get('log')}|{failure.get('error')}"


# ---------------------------------------------------------------------------
# A config back into the form (reopen a past run)
# ---------------------------------------------------------------------------

DOG_MODULE = "patchworks.plugins.dog"
COMPRESSIONS = ("zstd", "zstd:3", "blosc", "blosc:lz4", "none")
RUN_KINDS = ("One segmentation", "Several segmentations + relations")


def _flow(value: Any) -> str:
    """*value* as the one-line YAML a text field holds (``[4, 30, 30]``)."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict)):
        return yaml.safe_dump(value, default_flow_style=True).strip()
    return str(value)


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def shared_state(cfg: dict[str, Any]) -> dict[str, Any]:
    """Form values of what a run's segmentations share."""
    compression = str(cfg.get("compression", "zstd"))
    return {
        "input": _text(cfg.get("input")),
        "work_dir": _text(cfg.get("work_dir")),
        "sequence_pattern": _text(cfg.get("sequence_pattern")),
        "reuse_pyramid": bool(cfg.get("reuse_pyramid", False)),
        "shard": bool(cfg.get("shard", False)),
        "ngff_version": str(cfg.get("ngff_version", "auto")),
        "compression": compression if compression in COMPRESSIONS else "zstd",
        "convert_chunks": _flow(cfg.get("convert_chunks")),
        "level": int(cfg.get("level", 0)),
        "tile_shape": _flow(cfg.get("tile_shape", "auto")),
        "tiles_per_job": int(cfg.get("tiles_per_job", 4)),
        "gpu_memory_gb": int(cfg.get("gpu_memory_gb") or 0),
        "skip_empty": bool(cfg.get("skip_empty", True)),
        "empty_threshold": _text(cfg.get("empty_threshold")),
        "sequential_labels": bool(cfg.get("sequential_labels", True)),
        "pyramid_levels": int(cfg.get("pyramid_levels", 5)),
        "pyramid_downscale": int(cfg.get("pyramid_downscale", 2)),
        "shard_labels": bool(cfg.get("shard_labels", False)),
        "seam_report": bool(cfg.get("seam_report", True)),
        "merge_workers": _text(cfg.get("merge_workers")),
        "object_table": bool(cfg.get("object_table", True)),
        "table_channels": ",".join(
            str(c) for c in cfg.get("table_channels") or []
        ),
        "notify_email": _text(cfg.get("notify_email")),
        "notify_events": list(cfg.get("notify_events") or ["finish", "error"]),
    }


def segmentation_state(cfg: dict[str, Any], k: str = "") -> dict[str, Any]:
    """Form values of one segmentation; *k* prefixes the keys."""
    custom = cfg.get("custom") or {}
    method = cfg.get("method", "cellpose")
    if method == "custom" and custom.get("module") == DOG_MODULE:
        method = "dog"
    for name, module in SEEDED.items():
        if method == "custom" and custom.get("module") == module:
            method = name
    v: dict[str, Any] = {
        "lbl": _text(cfg.get("label_name", "labels")),
        "ch": int(cfg.get("channel") or 0),
        "nuc": _text(cfg.get("nuclei_channel")),
        "ov": _flow(cfg.get("overlap", 0)),
        "method": method,
    }
    if method == "cellpose":
        cp = dict(cfg.get("cellpose") or {})
        v["cpm"] = _text(cp.pop("model", "cyto3"))
        v["cpd"] = float(cp.pop("diameter", None) or 0.0)
        v["cp3"] = bool(cp.pop("do_3D", False))
        v["cpg"] = bool(cp.pop("gpu", True))
        v["cpx"] = yaml.safe_dump(cp, sort_keys=False).strip() if cp else ""
        v["norm"] = cfg.get("normalize", "image")
    elif method in SEEDED:
        kw = dict(custom.get("kwargs") or {})
        if method == "plantseg":
            v["psm"] = _text(kw.pop("model", PLANTSEG_MODELS[0]))
            v["pss"] = kw.pop("segmentation", "gasp")
            v["psb"] = float(kw.pop("beta", 0.6))
            v["psmin"] = int(kw.pop("post_minsize", 100))
        else:
            v["nmin"] = int(kw.pop("nuclei_min_size", 50))
        # The plugins' own default when left out: PlantSeg none, the
        # watershed Otsu.
        fg = kw.pop("foreground", None if method == "plantseg" else "otsu")
        v["fg"] = "none" if fg is None else str(fg)
        v["rad"] = _text(kw.pop("max_radius_um", None))
        kw.pop("seeds", None)  # the workflow sets it from seed_labels
        v["px"] = yaml.safe_dump(kw, sort_keys=False).strip() if kw else ""
        if cfg.get("seed_labels"):
            v["seed_from"] = "labels"
            v["seed"] = cfg["seed_labels"]
    elif method == "dog":
        kw = dict(custom.get("kwargs") or {})
        v["dl"] = float(kw.get("low_sigma", 1.0))
        v["dh"] = float(kw.get("high_sigma", 3.0))
        v["dt"] = float(kw.get("threshold", 0.02))
        v["du"] = kw.get("sigma_units", "px")
        v["dg"] = bool(kw.get("use_gpu", False))
        decon = kw.get("decon_kwargs") or {}
        v["psf"] = _text(decon.get("psf"))
        if decon:
            # Left out, these are the plugin's (and pycudadecon's) defaults,
            # not the form's: keep what the config meant.
            v["wl"] = int(decon.get("wavelength", 520))
            v["na"] = float(decon.get("na", 1.25))
            v["ni"] = float(decon.get("nimm", 1.3))
            dup = decon.get("dup_rev_z")
            v["dup"] = "auto" if dup == "auto" else "on" if dup else "off"
    elif method == "custom":
        v["cm"] = _text(custom.get("module"))
        v["cf"] = _text(custom.get("function", "segment"))
        kwargs = custom.get("kwargs") or {}
        v["ck"] = (
            yaml.safe_dump(kwargs, sort_keys=False).strip() if kwargs else ""
        )
    fills = fills_space(
        {"method": cfg.get("method"), "custom_module": custom.get("module", "")}
    )
    v[f"stitch_{fills}"] = cfg.get("stitch", "touch")
    v["iou"] = float(cfg.get("iou_threshold", 0.5))
    holes = cfg.get("fill_holes", False)
    v["holes"] = {False: "off", True: "3-D", "per_plane": "per plane"}.get(
        holes, "off"
    )
    v["open"] = int(cfg.get("open_radius") or 0)
    v["dil"] = int(cfg.get("dilate") or 0)
    v["dgpu"] = bool(cfg.get("dilate_gpu", False))
    v["minv"] = _text(cfg.get("min_volume"))
    v["maxv"] = _text(cfg.get("max_volume"))
    return {f"{k}{key}": value for key, value in v.items()}


def form_state(
    seg_cfgs: list[dict[str, Any]], multi: dict[str, Any] | None = None
) -> tuple[dict[str, Any], list[str]]:
    """Every form value for these configs, and the settings it cannot hold.

    The form holds what :func:`build_config` writes; a config written by
    hand may hold more (``seed_labels``, ``denoise:`` ...), which would be
    lost: those keys are returned, to be said.
    """
    state = shared_state(seg_cfgs[0])
    state["run_kind"] = RUN_KINDS[0 if multi is None else 1]
    if multi is None:
        state.update(segmentation_state(seg_cfgs[0]))
    else:
        state["n_segs"] = len(seg_cfgs)
        for i, cfg in enumerate(seg_cfgs):
            state.update(segmentation_state(cfg, f"s{i}_"))
        relate = multi.get("relate") or {}
        state.update(
            {
                "relate_partition": _text(relate.get("partition")),
                "relate_mem": _text(relate.get("mem")),
                "relate_cpus": int(relate.get("cpus") or 0),
                "relate_time": int(relate.get("time") or 0),
                "relate_qos": _text(relate.get("qos")),
            }
        )
        bundle = multi.get("bundle")
        state["bundle_format"] = (
            "none"
            if bundle is False
            else ((bundle or {}).get("format") or "zip")
        )
        review = multi.get("review") or {}
        state["min_overlap"] = float(review.get("min_overlap") or 0.5)
        state["loaded_relations"] = [
            {
                "a": r.get("a"),
                "b": r.get("b"),
                "output": r.get("output"),
                **(
                    {"max_distance_um": r["max_distance_um"]}
                    if r.get("max_distance_um") is not None
                    else {}
                ),
            }
            for r in multi.get("relations") or []
        ]
        rules = []
        for parent, children in (review.get("expect") or {}).items():
            for child, count in children.items():
                lo, hi = (count, count) if isinstance(count, int) else count
                rules.append(
                    {"parent": parent, "child": child, "min": lo, "max": hi}
                )
        state["loaded_rules"] = rules
        for child, rule in (review.get("position") or {}).items():
            state["pos_child"] = child
            state["pos_parent"] = rule.get("parent")
            apical = str(rule.get("apical", ""))
            state["pos_apical"] = (
                apical if apical in ("+z", "-z") else f"away from {apical}"
            )
            break
    known = {
        key
        for method in ("cellpose", "custom")
        for key in build_config(
            {
                "input": "",
                "work_dir": "",
                "label_name": "",
                "method": method,
                "custom_module": "m",
            }
        )
    }
    ignored = sorted(
        {key for cfg in seg_cfgs for key in cfg if key not in known}
    )
    return state, ignored


# ---------------------------------------------------------------------------
# Templates: a config kept under a name, to start the next run from
# ---------------------------------------------------------------------------

# What makes a config one image's run rather than a way of segmenting.
IMAGE_KEYS = ("input", "work_dir", "sequence_pattern")


def template_files(
    files: dict[str, dict], name: str, directory: str, *, with_image=False
) -> dict[str, dict]:
    """The run's files as a template named *name* in *directory*.

    One config is ``<name>.yaml``; a multi run a folder ``<name>/`` with
    its ``multi.yaml`` naming the segmentations next to it (relative, so
    the folder can be copied or shared). Without *with_image*, input and
    work_dir are left out: loading the template keeps the form's.
    """
    name = safe_name(name)

    def strip(cfg: dict) -> dict:
        return (
            dict(cfg)
            if with_image
            else {k: v for k, v in cfg.items() if k not in IMAGE_KEYS}
        )

    multi = next(
        (c for p, c in files.items() if p.endswith("/multi.yaml")), None
    )
    if multi is None:
        (cfg,) = files.values()
        return {f"{directory}/{name}.yaml": strip(cfg)}
    folder = f"{directory}/{name}"
    out = {}
    segs = []
    for path in multi["segmentations"]:
        file = path.rsplit("/", 1)[-1]
        out[f"{folder}/{file}"] = strip(files[path])
        segs.append(file)
    out[f"{folder}/multi.yaml"] = {**multi, "segmentations": segs}
    return out


def template_choices(entries: list[tuple[str, bool]], directory: str) -> dict:
    """Template name -> the file to load, from a listing of *directory*."""
    out = {}
    for name, is_dir in sorted(entries):
        if is_dir:
            out[name] = f"{directory}/{name}/multi.yaml"
        elif name.endswith((".yaml", ".yml")):
            out[name.rsplit(".", 1)[0]] = f"{directory}/{name}"
    return out


# ---------------------------------------------------------------------------
# The workflow on the cluster, against this launcher
# ---------------------------------------------------------------------------

# Scripts of the workflow this launcher runs on the cluster. A clone older
# than the launcher (pulled on your computer, not on the cluster) lacks some.
REQUIRED_SCRIPTS = (
    "scripts/run_multi.py",
    "scripts/check.py",
    "scripts/status.py",
    "scripts/fetch_models.py",
    "scripts/channels.py",
)


def update_command(workflow_dir: str) -> str:
    """``git pull`` the cluster's clone, fast-forward only (never a merge)."""
    return f"cd {shlex.quote(workflow_dir)} && git pull --ff-only 2>&1"


def explain_remote_error(text: str) -> str:
    """The line worth showing of a failed remote command: for a script the
    cluster's workflow does not have yet, what to do about it."""
    import re

    missing = re.search(
        r"can't open file '([^']*/(scripts/[\w.]+))'", text or ""
    )
    if missing:
        return (
            f"the workflow on the cluster has no {missing.group(2)}: it is "
            "older than this launcher. Update it: `git pull` in the clone "
            "on the cluster, or *Update it* in the sidebar."
        )
    return last_error_line(text)
