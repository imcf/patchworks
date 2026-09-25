"""Tests for the Streamlit launcher's logic (workflow/launcher)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workflow" / "launcher"))
sys.path.insert(0, str(ROOT / "workflow" / "scripts"))

import launcher_core as core  # noqa: E402

BASE = {
    "input": "/data/scan.czi",
    "work_dir": "/scratch/run",
    "label_name": "cells",
}


@pytest.mark.parametrize(
    "extra",
    [
        {
            "method": "cellpose",
            "cp_model": "cyto3",
            "cp_diameter": 30.0,
            "cp_extra_kwargs": "flow_threshold: 0.4",
        },
        {
            "method": "threshold",
            "stitch": "iou",
            "fill_holes": "per_plane",
            "compression": "zstd:3",
        },
        {
            "method": "dog",
            "dog_psf": "/psf.tif",
            "dog_dup_rev_z": "auto",
            "dog_sigma_units": "um",
            "dog_wavelength": 525,
            "dog_na": 1.4,
            "dog_nimm": 1.515,
        },
        {
            "method": "custom",
            "custom_module": "patchworks.plugins.dog",
            "custom_kwargs": "{low_sigma: 1, high_sigma: 3, threshold: 0.1}",
        },
    ],
)
def test_generated_configs_pass_the_workflows_own_validation(extra):
    """Whatever the form builds, the pipeline must accept."""
    from _pw import validate_config

    cfg = core.build_config({**BASE, **extra})
    validate_config(cfg)
    for key in (
        "stitch",
        "compression",
        "fill_holes",
        "open_radius",
        "seam_report",
        "min_volume",
        "nuclei_channel",
    ):
        assert key in cfg  # complete, so nothing leaks from the remote file


def test_dog_method_becomes_the_custom_plugin_with_decon():
    cfg = core.build_config(
        {**BASE, "method": "dog", "dog_psf": "/p.tif", "dog_dup_rev_z": "auto"}
    )
    assert cfg["method"] == "custom"
    assert cfg["custom"]["module"] == "patchworks.plugins.dog"
    decon = cfg["custom"]["kwargs"]["decon_kwargs"]
    assert decon == {"psf": "/p.tif", "dup_rev_z": "auto"}


def test_effective_config_flags_keys_inherited_from_the_cluster_file():
    base = {
        "work_dir": "/old",
        "cellpose": {"model": "nuclei", "flow_threshold": 0.9},
        "legacy_key": 1,
    }
    run = {"work_dir": "/new", "cellpose": {"model": "cyto3"}}
    merged, inherited = core.effective_config(base, run)
    assert merged["work_dir"] == "/new"
    assert merged["cellpose"] == {"model": "cyto3", "flow_threshold": 0.9}
    assert sorted(inherited) == ["cellpose.flow_threshold", "legacy_key"]
    assert core.effective_config(None, run) == (run, [])


def test_host_key_rules():
    fp = core.fingerprint(b"server-key")
    core.check_host_key("h", fp, [fp, "SHA256:other"], False)  # pinned ok
    with pytest.raises(ValueError, match="matches none"):
        core.check_host_key("h", fp, ["SHA256:other"], True)
    with pytest.raises(ValueError, match="no pinned"):
        core.check_host_key("h", fp, [], False)
    core.check_host_key("h", fp, [], True)  # explicit local opt-in


def test_clusters_accept_old_and_new_fingerprint_fields(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text(
        "a: {host: x, host_key_fingerprint: 'SHA256:one'}\n"
        "b: {host: y, host_key_fingerprints: ['SHA256:1', 'SHA256:2']}\n"
        "c: {host: z}\n"
    )
    c = core.load_clusters(f)
    assert c["a"]["host_key_fingerprints"] == ["SHA256:one"]
    assert c["b"]["host_key_fingerprints"] == ["SHA256:1", "SHA256:2"]
    assert c["c"]["host_key_fingerprints"] == []
    shipped = core.load_clusters(ROOT / "workflow/launcher/clusters.yaml")
    assert "scicore" in shipped


def test_commands_and_parsing():
    snake = core.snakemake_command("/w/config/x.yaml", core.SLURM_JOB)
    assert "--workflow-profile profile/slurm" in snake
    assert "-n" in core.snakemake_command("c.yaml", core.DRY_RUN).split()
    inner = core.inner_command("/w dir", 'eval "$(pixi shell-hook)"', snake)
    assert inner.startswith("cd '/w dir' && eval")
    script = core.controller_job_script(inner, partition="long", qos="1week")
    assert "#SBATCH --cpus-per-task=1" in script
    assert "#SBATCH --partition=long" in script and inner in script
    assert "--output" not in script  # not shell-parsed there
    sb = core.sbatch_command("/w/j.sbatch", name="pw x", log="/w/my log.txt")
    assert sb == (
        "sbatch --parsable '--job-name=pw x' '--output=/w/my log.txt' "
        "/w/j.sbatch"
    )
    assert core.parse_sbatch("12345\n") == "12345"
    assert core.parse_sbatch("678;cluster1") == "678"
    assert core.parse_tag("noise\nPID:4321\n", "PID") == "4321"
    assert "squeue" in core.status_command({"slurm_job": "12"})
    assert "kill -0" in core.status_command({"pid": "9"})
    jobs = core.parse_registry(
        core.registry_line({"name": "a"})
        + "garbage\n"
        + core.registry_line({"name": "b"})
    )
    assert [j["name"] for j in jobs] == ["b", "a"]


@pytest.mark.parametrize("method", ["threshold", "cellpose", "dog"])
def test_plan_command_parses_with_the_real_cli(method):
    import shlex

    from patchworks.cli import build_parser

    cfg = core.build_config(
        {
            **BASE,
            "method": method,
            "tile_shape": "[16, 512, 512]",
            "overlap": "[4, 30, 30]",
            "stitch": "iou",
        }
    )
    argv = shlex.split(core.plan_command(cfg))[1:]
    args = build_parser().parse_args(argv)
    assert args.plan and args.image == "/scratch/run/image.zarr"
    assert args.overlap == (4, 30, 30) and args.stitch == "iou"


def test_app_renders_the_login_screen():
    """Headless smoke test: the page loads and asks to connect."""
    pytest.importorskip("paramiko")
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(
        str(ROOT / "workflow/launcher/app.py"), default_timeout=30
    )
    at.run()
    assert not at.exception
    assert any("Connect to a cluster" in i.value for i in at.info)


def _multi_files(tmp_path, relations=None):
    shared = {
        "input": "/data/scan.czi",
        "work_dir": str(tmp_path / "run"),
        "tile_shape": "[16, 512, 512]",
        "compression": "zstd:1",
    }
    segs = [
        {"label_name": "nuclei_labels", "method": "threshold", "channel": 0},
        {
            "label_name": "cyto_labels",
            "method": "cellpose",
            "cp_model": "cyto3",
            "channel": 1,
            "nuclei_channel": 0,
        },
        {"label_name": "cilia_labels", "method": "dog", "channel": 2},
    ]
    if relations is None:
        relations = [
            {"a": "nuclei_labels", "b": "cyto_labels", "output": "n_c.xlsx"},
            {"a": "cilia_labels", "b": "cyto_labels", "output": "ci_c.xlsx"},
        ]
    return core.build_multi(
        shared,
        segs,
        relations,
        directory="/w/config/launcher/run1",
        relate={"partition": "rtx4090", "mem": "", "time": None},
        bundle={"format": ""},
    )


def test_build_multi_writes_what_run_multi_accepts(tmp_path):
    import run_multi
    from _pw import validate_config

    files = _multi_files(tmp_path)
    multi = files["/w/config/launcher/run1/multi.yaml"]
    assert multi["segmentations"] == [
        f"/w/config/launcher/run1/seg_{n}.yaml"
        for n in ("nuclei_labels", "cyto_labels", "cilia_labels")
    ]
    assert multi["relate"] == {"partition": "rtx4090"}  # blanks dropped
    assert "bundle" not in multi and "common" not in multi
    assert len(multi["relations"]) == 2
    assert run_multi._relate_settings(multi, argparse_ns()) is not None

    cfgs = [files[p] for p in multi["segmentations"]]
    for cfg in cfgs:
        validate_config(cfg)
    assert cfgs[1]["nuclei_channel"] == 0 and cfgs[2]["method"] == "custom"
    paths = [Path(p) for p in multi["segmentations"]]
    assert run_multi._validate_configs(paths, cfgs) == str(tmp_path / "run")
    assert core.multi_problems(files) == []


def argparse_ns():
    import argparse

    return argparse.Namespace(
        relate_partition=None, relate_mem=None, relate_cpus=None,
        relate_time=None, relate_qos=None,
    )  # fmt: skip


def test_multi_problems_catch_bad_relations(tmp_path):
    files = _multi_files(
        tmp_path,
        relations=[
            {"a": "nuclei_labels", "b": "nope", "output": "x.xlsx"},
            {"a": "cyto_labels", "b": "cyto_labels", "output": "x.xlsx"},
            {"a": "nuclei_labels", "b": "cyto_labels", "output": "y.csv"},
        ],
    )
    problems = "\n".join(core.multi_problems(files))
    assert "'nope' is not a segmentation" in problems
    assert "to itself" in problems
    assert "must be .xlsx" in problems
    assert "outputs must differ" in problems


def test_multi_command():
    cmd = core.multi_command("/w/config/m.yaml", core.SLURM_JOB)
    assert cmd.startswith(
        "python scripts/run_multi.py --config /w/config/m.yaml"
    )
    assert "--profile" in cmd
    assert core.multi_command("m.yaml", core.DRY_RUN).split()[-1] == "-n"


def test_controller_job_forgets_its_own_slurm_context():
    script = core.controller_job_script("run")
    unset = script.index("unset")
    assert "SLURM|SBATCH" in script and unset < script.index("\nrun\n")
