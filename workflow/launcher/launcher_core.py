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


def check_command(config: str, *, sample: int = 0) -> str:
    """``scripts/check.py`` on a run's config (or multi config), as JSON."""
    args = ["python", "scripts/check.py", config, "--json"]
    if sample:
        args += ["--sample", str(sample)]
    return " ".join(shlex.quote(a) for a in args)


def run_status_command(config: str) -> str:
    """``scripts/status.py`` on a run's config (or multi config), as JSON."""
    return " ".join(
        shlex.quote(a)
        for a in ("python", "scripts/status.py", config, "--json")
    )


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
# allocated/idle/other/total), associations (account, partition, QOS) and
# each QOS's MaxWall.
CLUSTER_INFO_COMMAND = (
    'echo "#partitions"; sinfo -h -o "%P|%a|%l|%G|%C" 2>/dev/null; '
    'echo "#assoc"; sacctmgr -nP show assoc user="$USER" '
    "format=account,partition,qos 2>/dev/null; "
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
            if len(fields) == 3:
                account, partition, qos = fields
                if account and account not in accounts:
                    accounts.append(account)
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
    }


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
                        f"{rule}: runtime {runtime} min is over partition "
                        f"{part!r}'s time limit of {limit} min"
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
        limit = maxwall.get(qos) if qos else None
        if runtime and limit and runtime > limit:
            problems.append(
                f"{rule}: runtime {runtime} min is over QOS {qos!r}'s MaxWall "
                f"of {limit} min; sbatch would refuse the job"
            )
    return problems


def status_command(job: dict[str, Any]) -> str:
    """Shell line printing RUNNING/PENDING/DONE for a registered job."""
    if job.get("slurm_job"):
        jid = shlex.quote(str(job["slurm_job"]))
        return f's=$(squeue -h -j {jid} -o %T 2>/dev/null); echo "${{s:-DONE}}"'
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
    }
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
