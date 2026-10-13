"""Tests for the Streamlit launcher's logic (workflow/launcher)."""

import json
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


def test_stop_unlock_and_script_commands():
    import shlex

    slurm = {"slurm_job": "42", "config": "/w/c.yaml", "kind": "single"}
    assert core.stop_command(slurm) == "scancel --full --signal=INT 42"
    assert core.stop_command(slurm, force=True) == "scancel 42"
    login = {"pid": "99", "config": "/w/m.yaml", "kind": "multi"}
    assert core.stop_command(login) == "kill -INT -- -99"
    assert shlex.split(core.unlock_command(login, "/w")) == [
        "python",
        "scripts/run_multi.py",
        "--config",
        "/w/m.yaml",
        "--unlock",
    ]
    assert core.unlock_command(slurm, "/w").endswith("c.yaml --unlock")
    convert = {"config": "/w/c.yaml", "kind": "convert", "work_dir": "/s/r"}
    assert "--directory /s/r/.snakemake_convert" in core.unlock_command(
        convert, "/w"
    )
    assert core.unlock_command({"kind": "check", "config": "x"}, "/w") is None
    assert core.check_command("/w/m.yaml", sample=3).split()[-2:] == [
        "--sample",
        "3",
    ]
    assert core.parse_json_output('log line\n{"a": 1}\nmore\n') == {"a": 1}
    assert core.parse_json_output("Traceback") is None


def test_profile_overrides_and_account_limits():
    import yaml

    text = (ROOT / "workflow" / "profile" / "slurm" / "config.yaml").read_text()
    res = core.profile_resources(text)
    assert res["segment"]["gres"] == "gpu:1"
    assert core.first_attempt(res["segment"]["mem_mb"]) == 128000
    assert (
        res["convert"]["slurm_partition"] == res["default"]["slurm_partition"]
    )
    new = yaml.safe_load(
        core.profile_with_overrides(
            text, {"segment": {"mem_mb": 96000, "qos": "", "runtime": 300}}
        )
    )
    seg = new["set-resources"]["segment"]
    assert seg["mem_mb"] == "attempt * 96000" and "qos" not in seg
    assert seg["runtime"] == 300
    # prepare sizes "auto" tiles from its own memory: it follows segment.
    assert new["set-resources"]["prepare"]["mem_mb"] == "attempt * 96000"
    assert new["executor"] == "slurm"  # the rest of the profile is kept

    limits = core.parse_account_limits(
        "#assoc\nscicore|6hours,1day\nrtx4090|rtx4090-6hours\n"
        "#qos\n6hours|06:00:00\n1day|1-00:00:00\nrtx4090-6hours|06:00:00\n"
    )
    assert limits["maxwall"]["1day"] == 1440
    ok = {
        "segment": {
            "slurm_partition": "rtx4090",
            "qos": "rtx4090-6hours",
            "runtime": 300,
        }
    }
    assert core.resource_problems(ok, limits) == []
    too_long = {"segment": {**ok["segment"], "runtime": 1400}}
    assert "over QOS" in core.resource_problems(too_long, limits)[0]
    wrong_qos = {"segment": {**ok["segment"], "qos": "1day"}}
    assert (
        "not one your account may use on"
        in core.resource_problems(wrong_qos, limits)[0]
    )
    no_part = {"merge": {"slurm_partition": "a100", "runtime": 10}}
    assert "no association" in core.resource_problems(no_part, limits)[0]
    assert core.slurm_minutes("2-03:04:05") == 2 * 1440 + 3 * 60 + 5
    assert core.slurm_minutes("UNLIMITED") is None


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
    plan = {
        "shape": [10, 2048, 2048],
        "tile_shape": [10, 1024, 1024],
        "grid": [1, 2, 2],
        "tiles": 4,
        "tiles_with_signal": None,
        "gpu_jobs": None,
        "tile_read_bytes": 1073741824,
        "labels_bytes_uncompressed": 0,
        "seconds_per_tile": None,
    }
    row = core.plan_row("cells", plan)
    assert row["tiles"] == 4 and row["grid"] == "1 × 2 × 2"
    assert row["with signal"] == "?" and "GPU hours" not in row
    assert row["read per tile (GiB)"] == 1.0
    timed = core.plan_row(
        "cells",
        {
            **plan,
            "seconds_per_tile": 12.0,
            "hours_per_job": 0.5,
            "gpu_hours": 3.2,
        },
    )
    assert timed["s / tile"] == 12.0 and timed["GPU hours"] == 3.2
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
    def __init__(self, name, is_dir, size=0):
        import stat

        self.filename = name
        self.st_mode = stat.S_IFDIR if is_dir else stat.S_IFREG
        self.st_size = size


class _FakeSFTP:
    """Just enough of paramiko's SFTPClient over a dict of folders, and of
    files (path -> bytes) it can read and write."""

    tree: ClassVar[dict[str, list[str]]] = {
        "/home/u": ["patchworks", "data"],
        "/home/u/patchworks": ["workflow"],
        "/home/u/patchworks/workflow": ["Snakefile", "config"],
        "/home/u/data": ["scan.czi", "old.zarr"],
    }

    def __init__(self, files=None):
        self.files = {} if files is None else files

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
        if path in self.files:
            return _FakeAttr(name, False, len(self.files[path]))
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
        import io

        if "a" in mode:
            files = self.files

            class _Append(io.BytesIO):
                def write(self, text):
                    data = text.encode() if isinstance(text, str) else text
                    files[path] = files.get(path, b"") + data

            return _Append()
        if path not in self.files:
            raise FileNotFoundError(path)
        return io.BytesIO(self.files[path])

    def putfo(self, fh, path):
        self.files[path] = fh.read()


class _FakeSSH:
    """An SSH client: canned answers by command, files kept in memory."""

    def __init__(self, answers=None, files=None):
        self.commands = []
        self.active = True
        self.answers = answers or {}
        self.files = {} if files is None else files

    def get_transport(self):
        from types import SimpleNamespace

        return SimpleNamespace(is_active=lambda: self.active)

    def open_sftp(self):
        return _FakeSFTP(self.files)

    def exec_command(self, command, timeout=None):
        import io as _io
        from types import SimpleNamespace

        self.commands.append(command)
        out = next(
            (v for k, v in self.answers.items() if k in command), ""
        ).encode()
        chan = SimpleNamespace(recv_exit_status=lambda: 0)
        stdout = SimpleNamespace(read=_io.BytesIO(out).read, channel=chan)
        return None, stdout, SimpleNamespace(read=lambda: b"")

    def close(self):
        pass


WF = "/home/u/patchworks/workflow"


def _app(ssh, **state):
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(
        str(ROOT / "workflow/launcher/app.py"), default_timeout=30
    )
    at.session_state["ssh_client"] = ssh
    at.session_state["ssh_target"] = "u@host:22"
    at.session_state["wf_dir"] = WF
    for key, value in state.items():
        at.session_state[key] = value
    return at


def _report(labels, converted, plans=True):
    plan = {
        "shape": [10, 2048, 2048],
        "tile_shape": [10, 1024, 1024],
        "grid": [1, 2, 2],
        "tiles": 4,
        "tiles_with_signal": 3,
        "gpu_jobs": 1,
        "tile_read_bytes": 1,
        "labels_bytes_uncompressed": 1,
        "seconds_per_tile": None,
    }
    return json.dumps(
        {
            "shared_problems": [],
            "converted": converted,
            "valid": True,
            "segmentations": [
                {
                    "label_name": label,
                    "config": f"{label}.yaml",
                    "problems": [],
                    "plan": plan if converted and plans else None,
                }
                for label in labels
            ],
        }
    )


def test_app_connected_flow_validates_and_checks_on_the_cluster():
    """With a (fake) connection: required fields, then check.py's report."""
    pytest.importorskip("paramiko")
    ssh = _FakeSSH({"scripts/check.py": _report(["labels"], converted=False)})
    at = _app(ssh)
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

    next(b for b in at.button if b.label == "Check and plan").click().run()
    assert not at.exception
    assert any("Not converted yet" in i.value for i in at.info)
    # The configs went up first, then check.py ran on them, there.
    uploaded = [p for p in ssh.files if p.startswith(f"{WF}/config/launcher_")]
    assert uploaded and any(
        "scripts/check.py" in c and uploaded[0] in c for c in ssh.commands
    )


def test_app_plans_every_segmentation_of_a_multi_run():
    pytest.importorskip("paramiko")
    ssh = _FakeSSH(
        {
            "scripts/check.py": _report(
                ["nuclei_labels", "cyto_labels"], converted=True
            )
        }
    )
    at = _app(ssh, input="/home/u/data/scan.czi", work_dir="/scratch/run")
    at.run()
    at.button_group[0].set_value("Several segmentations + relations").run()
    assert not at.exception
    next(b for b in at.button if b.label == "Check and plan").click().run()
    assert not at.exception
    rows = next(
        d.value for d in at.dataframe if "segmentation" in d.value.columns
    )
    assert list(rows["segmentation"]) == ["nuclei_labels", "cyto_labels"]
    # check.py ran once, on the multi config.
    checks = [c for c in ssh.commands if "scripts/check.py" in c]
    assert len(checks) == 1 and "multi.yaml" in checks[0]


def test_app_resources_change_for_one_run():
    """A changed resource goes into this run's own copy of the profile."""
    pytest.importorskip("paramiko")
    profile = (ROOT / "workflow/profile/slurm/config.yaml").read_bytes()
    ssh = _FakeSSH(
        {"sbatch": "123\n", "sacctmgr": "#assoc\nrtx4090|other\n#qos\n"},
        files={f"{WF}/profile/slurm/config.yaml": profile},
    )
    at = _app(ssh, input="/home/u/data/scan.czi", work_dir="/scratch/run")
    at.run()
    assert not at.exception
    next(
        b for b in at.button if b.label == "Check against my account"
    ).click().run()
    assert any("rtx4090-1day" in e.value for e in at.error)
    at.radio[0].set_value(core.SLURM_JOB).run()
    next(b for b in at.button if b.label == "Launch").click().run()
    assert not at.exception
    assert any("sbatch" in c for c in ssh.commands)
    record = json.loads(ssh.files[f"{WF}/{core.REGISTRY}"].decode())
    assert record["slurm_job"] == "123" and record["kind"] == "single"
    assert record["inner"] and record["setup"] is not None


def _job(**extra):
    return {
        "name": "cells_20261013-120000",
        "log": f"{WF}/{core.LOG_DIR}/cells.log",
        "started": "20261013-120000",
        "mode": core.SLURM_JOB,
        "slurm_job": "42",
        "config": f"{WF}/config/launcher_cells.yaml",
        "kind": "single",
        "setup": 'eval "$(pixi shell-hook)"',
        "inner": "cd /w && snakemake ...",
        "slurm": {"time": "1:00:00"},
        **extra,
    }


def test_jobs_tab_progress_results_and_actions():
    pytest.importorskip("paramiko")
    status = {
        "work_dir": "/scratch/run",
        "segmentations": [
            {
                "label_name": "nuclei",
                "stage": "done",
                "batches": 4,
                "batches_done": 4,
                "objects": 1234,
                "seams": {"x": {"seam_rate": 0.1, "interior_rate": 0.05}},
            },
            {
                "label_name": "cells",
                "stage": "segmenting",
                "batches": 10,
                "batches_done": 3,
                "objects": None,
                "seams": None,
            },
        ],
        "relations": [
            {"a": "nuclei", "b": "cells", "files": ["/scratch/run/n_c.xlsx"]}
        ],
        "bundle": None,
    }
    ssh = _FakeSSH(
        {
            "scripts/status.py": json.dumps(status),
            "squeue": "DONE\n",
            "sbatch": "77\n",
            "tail -n": "the log\n",
        },
        files={
            f"{WF}/{core.REGISTRY}": core.registry_line(_job()).encode(),
            "/scratch/run/n_c.xlsx": b"xlsx bytes",
        },
    )
    at = _app(ssh, input="/x.czi", work_dir="/scratch/run")
    at.run()
    assert not at.exception
    texts = [p.proto.text for p in at.get("progress")]
    assert any("cells** · 3/10 GPU jobs done" in t for t in texts)
    assert any("nuclei** · merged" in t for t in texts)
    rows = next(d.value for d in at.dataframe if "objects" in d.value.columns)
    assert rows["objects"].tolist() == [1234]
    labels = [b.label for b in at.button]
    assert "Unlock" in labels and "Relaunch" in labels
    # Stop is offered only while the run is alive.
    assert "Stop it" not in labels

    next(b for b in at.button if b.label == "Unlock").click().run()
    assert any("--unlock" in c for c in ssh.commands)
    next(b for b in at.button if b.label == "Relaunch").click().run()
    lines = ssh.files[f"{WF}/{core.REGISTRY}"].decode().splitlines()
    assert len(lines) == 2 and json.loads(lines[1])["inner"] == _job()["inner"]

    fetch = next(b for b in at.button if "n_c.xlsx" in b.label)
    fetch.click().run()
    assert not at.exception
    assert at.session_state["dl_/scratch/run/n_c.xlsx"] == b"xlsx bytes"


def test_jobs_tab_stop_while_running():
    pytest.importorskip("paramiko")
    ssh = _FakeSSH(
        {"squeue": "RUNNING\n", "scripts/status.py": "{}"},
        files={f"{WF}/{core.REGISTRY}": core.registry_line(_job()).encode()},
    )
    at = _app(ssh, input="/x.czi", work_dir="/scratch/run")
    at.run()
    assert not at.exception
    labels = [b.label for b in at.button]
    assert "Unlock" not in labels and "Relaunch" not in labels
    next(b for b in at.button if b.label == "Stop it").click().run()
    assert any("scancel --full --signal=INT 42" in c for c in ssh.commands)


def test_a_folder_of_tiffs_becomes_a_glob_and_a_pattern():
    tifffile = pytest.importorskip("tifffile")
    listing = [
        *(
            (f"s_T0_Z{z:03d}_C{c}_V0.tif", False)
            for z in range(3)
            for c in (0, 1)
        ),
        ("notes.txt", False),
        (".hidden.tif", False),
        ("sub.tif", True),
    ]
    names = core.tiff_names(listing)
    assert len(names) == 6
    assert core.tiff_glob("/data/scan", names) == "/data/scan/*.tif"
    assert core.tiff_glob("/d", ["a.TIFF", "b.TIFF"]) == "/d/*.TIFF"
    pattern = core.guess_sequence_pattern(names)
    # What the converter hands it to: the constant T0 is no axis.
    axes, shape = tifffile.tifffile.parse_filenames(names, pattern)[:2]
    assert (axes, shape) == (("Z", "C"), (3, 2))
    numbered = [f"plane{z:04d}.tif" for z in range(5)]
    pattern = core.guess_sequence_pattern(numbered)
    assert tifffile.tifffile.parse_filenames(numbered, pattern)[:2] == (
        ("Z",),
        (5,),
    )
    assert core.guess_sequence_pattern(["one.tif"]) is None

    cfg = core.build_config(
        {**BASE, "input": "/d/*.tif", "method": "threshold"}
    )
    assert any("sequence_pattern" in p for p in core.config_problems(cfg))
    cfg["sequence_pattern"] = pattern
    assert not core.config_problems(cfg)


def test_stitch_is_per_segmentation(tmp_path):
    """touch for the nuclei, iou for the cells, in one multi run."""
    import run_multi
    from _pw import validate_config

    shared = {"input": "/data/scan.czi", "work_dir": str(tmp_path / "run")}
    segs = [
        {"label_name": "nuclei", "method": "threshold", "stitch": "touch"},
        {
            "label_name": "cells",
            "method": "custom",
            "custom_module": "patchworks.plugins.watershed",
            "stitch": "iou",
            "iou_threshold": 0.3,
        },
    ]
    files = core.build_multi(shared, segs, [], directory=str(tmp_path / "c"))
    paths = [p for p in files if not p.endswith("multi.yaml")]
    cfgs = [files[p] for p in paths]
    assert [c["stitch"] for c in cfgs] == ["touch", "iou"]
    assert cfgs[1]["iou_threshold"] == 0.3
    for cfg in cfgs:
        validate_config(cfg)
        assert not core.config_problems(cfg)
    assert core.multi_problems(files) == []
    from pathlib import Path

    assert run_multi._validate_configs([Path(p) for p in paths], cfgs)

    # A method that fills space, stitched by touch: refused, as the
    # workflow would.
    touch = core.build_config({**shared, **segs[1], "stitch": "touch"})
    assert core.config_problems(touch)
    with pytest.raises(ValueError, match="iou"):
        validate_config(touch)


def test_fills_space_lists_the_plugins_that_say_so():
    import importlib

    for module in core.FILLS_SPACE:
        try:
            plugin = importlib.import_module(module)
        except ImportError:
            continue
        assert getattr(plugin.segment, "patchworks_stitch", None) == "iou"
    assert core.fills_space(
        {"method": "custom", "custom_module": "patchworks.plugins.plantseg"}
    )
    assert not core.fills_space({"method": "cellpose"})


def test_a_dropped_connection_asks_to_log_in_again():
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
    ssh.active = False
    at.run()
    assert not at.exception
    assert any(
        "connection to the cluster was lost" in w.value for w in at.warning
    )
    assert any("Connect to a cluster" in i.value for i in at.info)
