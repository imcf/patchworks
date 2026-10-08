"""Tests for the Streamlit launcher's logic (workflow/launcher)."""

import sys
from pathlib import Path
from typing import ClassVar

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


def test_nothing_is_inherited_from_the_cluster_config_yaml():
    """The Snakefile reads only the --configfile given (and its defaults).

    The launcher used to warn about keys "taken from config/config.yaml";
    if the Snakefile ever reads that file again, the warning is due back.
    """
    snakefile = (ROOT / "workflow" / "Snakefile").read_text()
    assert "configfile:" not in snakefile


PLANTSEG = {
    "method": "custom",
    "custom_module": "patchworks.plugins.plantseg",
    "label_name": "cyto",
}
DOG_GPU = {"method": "dog", "dog_use_gpu": True, "label_name": "cilia"}


def test_environment_matches_what_the_workflow_checks():
    """needed_packages names what _pw.environment_problems would refuse."""
    from _pw import environment_problems, has_module

    for extra in (PLANTSEG, DOG_GPU):
        cfg = core.build_config({**BASE, **extra})
        needs = core.needed_packages(cfg)
        assert needs
        refused = " ".join(environment_problems(cfg))
        for module in needs:
            assert has_module(module) or repr(module) in refused


def test_suggested_environment():
    toml = (ROOT / "workflow" / "pixi.toml").read_text()
    envs = core.pixi_environments(toml)
    assert envs[0] == "default" and "plantseg" in envs and "cuda12" in envs
    cfg = lambda extra: core.build_config({**BASE, **extra})  # noqa: E731
    plain = cfg({"method": "threshold"})
    assert core.suggest_environment([plain], envs) == ("default", [])
    assert core.suggest_environment([cfg(DOG_GPU)], envs)[0] == "cuda12"
    env, reasons = core.suggest_environment(
        [plain, cfg(PLANTSEG), cfg(DOG_GPU)], envs
    )
    assert env == "plantseg" and len(reasons) == 2
    assert core.suggest_environment([cfg(PLANTSEG)], ["default"])[0] is None


def test_setup_line_gets_the_environment():
    hook = 'eval "$(pixi shell-hook)"'
    assert core.setup_with_environment(hook, "default") == hook
    assert core.setup_with_environment(hook, "cuda12") == (
        'eval "$(pixi shell-hook -e cuda12)"'
    )
    swapped = core.setup_with_environment(
        'module load X && eval "$(pixi shell-hook -e cuda12)"', "plantseg"
    )
    assert swapped == (
        "export CONDA_OVERRIDE_CUDA=12.0 && module load X && "
        'eval "$(pixi shell-hook -e plantseg)"'
    )
    assert core.setup_with_environment("conda activate pw", "cuda12") is None


def test_convert_only_command_matches_run_multi_phase_a():
    import shlex

    cfg = core.build_config({**BASE, "method": "threshold", "level": 1})
    cmd = shlex.split(
        core.convert_command(
            "/w/c.yaml", cfg, core.SLURM_JOB, workflow_dir="/w"
        )
    )
    assert (
        cmd[cmd.index("--directory") + 1] == "/scratch/run/.snakemake_convert"
    )
    assert cmd[cmd.index("--workflow-profile") + 1] == "/w/profile/slurm"
    assert cmd[cmd.index("--") + 1 :] == [
        "/scratch/run/image.zarr/zarr.json",
        "/scratch/run/image.occupancy.zarr/1/zarr.json",
    ]
    dry = core.convert_command(
        "/w/c.yaml", cfg, core.DRY_RUN, workflow_dir="/w"
    )
    assert "-n" in shlex.split(dry)
    assert core.config_problems({**cfg, "work_dir": "results"})
    assert not core.config_problems(cfg)


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
            {
                "a": "cilia_labels",
                "b": "cyto_labels",
                "output": "ci_c.xlsx",
                "max_distance_um": 1.5,
            },
        ]
    return core.build_multi(
        shared,
        segs,
        relations,
        directory="/w/config/launcher/run1",
        relate={"partition": "rtx4090", "mem": "", "time": None},
        bundle={"format": ""},
        review={
            "expect": {
                "cyto_labels": {"nuclei_labels": 1, "cilia_labels": [0, 2]}
            },
            "min_overlap": None,
            "position": {
                "cilia_labels": {
                    "parent": "cyto_labels",
                    "apical": "nuclei_labels",
                }
            },
        },
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
    # "none" picked: written out, since a run zips by default.
    assert multi["bundle"] is False and "common" not in multi
    assert len(multi["relations"]) == 2
    assert run_multi._relate_settings(multi, argparse_ns()) is not None

    cfgs = [files[p] for p in multi["segmentations"]]
    for cfg in cfgs:
        validate_config(cfg)
    assert cfgs[1]["nuclei_channel"] == 0 and cfgs[2]["method"] == "custom"
    paths = [Path(p) for p in multi["segmentations"]]
    assert run_multi._validate_configs(paths, cfgs) == str(tmp_path / "run")
    assert core.multi_problems(files) == []
    labels = [c["label_name"] for c in cfgs]
    rules = run_multi._review_rules(multi, labels)
    assert rules["expect"] == {
        "cyto_labels": {"nuclei_labels": 1, "cilia_labels": [0, 2]}
    }
    assert rules["position"]["cilia_labels"]["apical"] == "nuclei_labels"
    assert multi["relations"][1]["max_distance_um"] == 1.5
    assert all(c["object_table"] is True for c in cfgs)


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


def test_required_fields_and_plan_helpers():
    cfg = core.build_config({**BASE, "method": "threshold"})
    assert core.missing_fields(cfg) == []
    empty = core.build_config(
        {**BASE, "input": "", "work_dir": " ", "method": "threshold"}
    )
    # An empty work_dir is what planned /image.zarr.
    assert core.missing_fields(empty) == ["input", "work_dir"]
    assert core.image_store(cfg) == "/scratch/run/image.zarr"
    assert "zarr.json" in core.store_exists_command("/s/image.zarr")

    out = 'INFO plan: ...\n{"shape": [10, 2048, 2048], "tile_shape": [10, '
    out += '1024, 1024], "grid": [1, 2, 2], "tiles": 4, '
    out += '"tiles_with_signal": 3, "tile_read_bytes": 1073741824, '
    out += '"labels_bytes_uncompressed": 0, "estimated_seconds": null}\n'
    plan = core.parse_plan(out)
    row = core.plan_row("cells", plan)
    assert row["tiles"] == 4 and row["grid"] == "1 × 2 × 2"
    assert row["read per tile (GiB)"] == 1.0
    assert core.parse_plan("Traceback ...") is None
    assert core.last_error_line("a\nFileNotFoundError: x\n\n") == (
        "FileNotFoundError: x"
    )


def test_browse_entries_and_paths():
    entries = [
        ("b.czi", False),
        (".hidden", True),
        ("notes.txt", False),
        ("image.zarr", True),
        ("Data", True),
        ("a.ims", False),
    ]
    assert core.browse_entries(entries) == [
        ("Data", "dir"),
        ("image.zarr", "store"),
        ("a.ims", "image"),
        ("b.czi", "image"),
        ("notes.txt", "file"),
    ]
    assert core.browse_entries(entries, dirs_only=True) == [
        ("Data", "dir"),
        ("image.zarr", "store"),
    ]
    assert core.parent_dir("/a/b/") == "/a" and core.parent_dir("/a") == "/"
    assert core.parent_dir("/") == "/"
    assert core.join_remote("/", "x") == "/x"
    assert core.join_remote("/a/", "x") == "/a/x"


class _FakeAttr:
    def __init__(self, name, is_dir):
        import stat

        self.filename = name
        self.st_mode = stat.S_IFDIR if is_dir else stat.S_IFREG


class _FakeSFTP:
    """Just enough of paramiko's SFTPClient over a dict of folders."""

    tree: ClassVar[dict[str, list[str]]] = {
        "/home/u": ["patchworks", "data"],
        "/home/u/patchworks": ["workflow"],
        "/home/u/patchworks/workflow": ["Snakefile", "config"],
        "/home/u/data": ["scan.czi", "old.zarr"],
    }

    class sock:
        closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self):
        pass

    def normalize(self, path):
        return "/home/u"

    def _is_dir(self, path):
        return path in self.tree or path.endswith(".zarr")

    def stat(self, path):
        parent, name = path.rsplit("/", 1)
        if path in self.tree or name in self.tree.get(parent, []):
            return _FakeAttr(name, self._is_dir(path))
        raise FileNotFoundError(path)

    def listdir_attr(self, path):
        if path not in self.tree:
            raise FileNotFoundError(path)
        return [
            _FakeAttr(n, self._is_dir(f"{path}/{n}")) for n in self.tree[path]
        ]

    def open(self, path, mode="r"):
        raise FileNotFoundError(path)


class _FakeSSH:
    """An SSH client whose commands report a missing converted image."""

    def __init__(self):
        self.commands = []

    def open_sftp(self):
        return _FakeSFTP()

    def exec_command(self, command, timeout=None):
        import io as _io
        from types import SimpleNamespace

        self.commands.append(command)
        out = b"NO\n" if "zarr.json" in command else b""
        chan = SimpleNamespace(recv_exit_status=lambda: 0)
        stdout = SimpleNamespace(read=_io.BytesIO(out).read, channel=chan)
        return None, stdout, SimpleNamespace(read=lambda: b"")

    def close(self):
        pass


def test_app_connected_flow_validates_and_explains_a_missing_image():
    """With a (fake) connection: required fields, then the plan's check."""
    pytest.importorskip("paramiko")
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(
        str(ROOT / "workflow/launcher/app.py"), default_timeout=30
    )
    ssh = _FakeSSH()
    at.session_state["ssh_client"] = ssh
    at.session_state["ssh_target"] = "u@host:22"
    at.session_state["wf_dir"] = "/home/u/patchworks/workflow"
    at.run()
    assert not at.exception
    assert any(
        "still to fill in: input, work_dir" in w.value for w in at.warning
    )

    at.text_input(key="input").set_value("/home/u/data/scan.czi")
    at.text_input(key="work_dir").set_value("/scratch/run")
    at.run()
    assert not at.exception
    assert any("config is complete" in s.value for s in at.success)

    plan = next(b for b in at.button if b.label == "Plan")
    plan.click().run()
    assert not at.exception
    assert any("/scratch/run/image.zarr" in w.value for w in at.warning)
    # Only the existence check ran: no plan on a missing image.
    assert not any("--plan" in c for c in ssh.commands)


class _FakeSSHWithImage(_FakeSSH):
    """The converted image exists, and every plan succeeds."""

    plan = (
        '{"shape": [10, 2048, 2048], "tile_shape": [10, 1024, 1024], '
        '"grid": [1, 2, 2], "tiles": 4, "tiles_with_signal": 3}'
    )

    def exec_command(self, command, timeout=None):
        import io as _io
        from types import SimpleNamespace

        self.commands.append(command)
        if "zarr.json" in command:
            out = b"YES\n"
        elif "--plan" in command:
            out = self.plan.encode()
        else:
            out = b""
        chan = SimpleNamespace(recv_exit_status=lambda: 0)
        stdout = SimpleNamespace(read=_io.BytesIO(out).read, channel=chan)
        return None, stdout, SimpleNamespace(read=lambda: b"")


def test_app_plans_every_segmentation_of_a_multi_run():
    pytest.importorskip("paramiko")
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(
        str(ROOT / "workflow/launcher/app.py"), default_timeout=30
    )
    ssh = _FakeSSHWithImage()
    at.session_state["ssh_client"] = ssh
    at.session_state["ssh_target"] = "u@host:22"
    at.session_state["wf_dir"] = "/home/u/patchworks/workflow"
    at.session_state["input"] = "/home/u/data/scan.czi"
    at.session_state["work_dir"] = "/scratch/run"
    at.run()
    at.button_group[0].set_value("Several segmentations + relations").run()
    assert not at.exception
    next(b for b in at.button if b.label == "Plan all segmentations").click()
    at.run()
    assert not at.exception
    plans = [c for c in ssh.commands if "--plan" in c]
    assert len(plans) == 2
    rows = next(
        d.value for d in at.dataframe if "segmentation" in d.value.columns
    )
    assert list(rows["segmentation"]) == ["nuclei_labels", "cyto_labels"]
