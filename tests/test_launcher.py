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

    info = core.parse_cluster_info(CLUSTER)
    assert info["default_partition"] == "scicore"
    assert info["accounts"] == ["biehlmai"]
    assert info["maxwall"]["1day"] == 1440
    gpu = info["partitions"]["rtx4090"]
    # Two sinfo lines (node groups) of one partition add up.
    assert gpu["gres"] == ["gpu:1", "gpu:rtx4090:1"] and gpu["cpus"] == 192
    assert core.usable_qos(info, "rtx4090") == [
        "rtx4090-1day",
        "rtx4090-6hours",
    ]
    assert core.usable_qos(info, "a100") == []
    ok = {
        "segment": {
            "slurm_partition": "rtx4090",
            "qos": "rtx4090-6hours",
            "gres": "gpu:1",
            "runtime": 300,
        }
    }
    assert core.resource_problems(ok, info) == []
    too_long = {"segment": {**ok["segment"], "runtime": 1400}}
    assert (
        "23:20:00 is longer than QOS 'rtx4090-6hours' allows (06:00:00)"
        in core.resource_problems(too_long, info)[0]
    )
    wrong_qos = {"segment": {**ok["segment"], "qos": "1day"}}
    assert (
        "not one your account may use on"
        in core.resource_problems(wrong_qos, info)[0]
    )
    down = core.resource_problems(
        {"merge": {"slurm_partition": "a100", "runtime": 10}}, info
    )
    assert "is down" in down[0] and "no association" in down[1]
    unknown = {"merge": {"slurm_partition": "h100"}}
    assert "no partition 'h100'" in core.resource_problems(unknown, info)[0]
    # No partition named: the cluster's default, and its time limit.
    long_merge = {"merge": {"runtime": 2000}}
    assert (
        "than partition 'scicore' allows (1-00:00:00)"
        in core.resource_problems(long_merge, info)[0]
    )
    no_gpu = {"segment": {"slurm_partition": "scicore", "gres": "gpu:1"}}
    assert "has no GPUs" in core.resource_problems(no_gpu, info)[0]
    rows = core.partition_rows(info)
    assert rows[-1]["partition"] == "scicore (default)"  # GPU ones first
    assert rows[1]["idle CPUs"] == "82 / 192"
    assert core.parse_cluster_info("")["partitions"] == {}
    # QOS left blank: the default QOS's MaxWall holds (sciCORE: 6 hours).
    assert core.default_qos(info) == "6hours"
    assert core.default_qos(info, "rtx4090") == "rtx4090-6hours"
    controller = {"controller": {"runtime": 1000}}
    (problem,) = core.resource_problems(controller, info)
    assert "default QOS '6hours'" in problem and "QOS '1day'" in problem
    assert core.qos_for(info, None, 3 * 1440) == "1week"
    assert core.qos_for(info, None, 300) == ""  # the default allows it
    assert core.qos_for(info, "scicore", 1000) == "1day"  # the shortest
    assert core.qos_for(info, None, 30 * 1440) == ""  # none allows it
    # A QOS chosen that is too short: the one to switch to.
    assert core.better_qos(info, None, "1day", 3 * 1440) == "1week"
    assert core.better_qos(info, None, "1week", 3 * 1440) == ""
    assert core.better_qos(info, None, "", 300) == ""  # the default fits
    # Default QOS not known (inherited from the account): name one.
    unknown = {**info, "default_qos": {}}
    assert core.qos_for(unknown, None, 3 * 1440) == "1week"
    # Older output without the defaultqos column still parses.
    old = core.parse_cluster_info("#assoc\nbiehlmai|scicore|1day")
    assert old["assoc"] == {"scicore": {"1day"}} and old["default_qos"] == {}
    why = core.explain_sbatch_error(
        "sbatch: error: QOSMaxWallDurationPerJobLimit",
        info,
        {"time": "3-00:00:00", "partition": "", "qos": ""},
    )
    assert "Choose QOS '1week'" in why and "1week (7-00:00:00)" in why
    assert core.explain_sbatch_error("sbatch: error: other", info, {}) is None
    assert core.slurm_minutes("2-03:04:05") == 2 * 1440 + 3 * 60 + 5
    assert core.slurm_minutes("UNLIMITED") is None
    assert core.slurm_minutes("3 days") is None


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
        "/home/u/patchworks/workflow": ["Snakefile", "config", "scripts"],
        "/home/u/patchworks/workflow/scripts": [
            s.split("/")[-1] for s in core.REQUIRED_SCRIPTS
        ],
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
        # Files written in this test show in their folders too.
        written = {}
        for f in self.files:
            if f.startswith(path.rstrip("/") + "/"):
                rest = f[len(path.rstrip("/")) + 1 :]
                written[rest.split("/")[0]] = "/" in rest
        if path not in self.tree and not written:
            raise FileNotFoundError(path)
        return [
            _FakeAttr(n, self._is_dir(f"{path}/{n}"))
            for n in self.tree.get(path, [])
        ] + [_FakeAttr(n, d) for n, d in written.items()]

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

CLUSTER = """#partitions
scicore*|up|1-00:00:00|(null)|1200/400/0/1600
rtx4090|up|7-00:00:00|gpu:rtx4090:8(S:0-1)|100/60/0/160
rtx4090|up|7-00:00:00|gpu:rtx4090:4|10/22/0/32
a100|down|2-00:00:00|gpu:a100:4|0/0/64/64
#assoc
biehlmai|scicore|1day,1week,6hours|6hours
biehlmai|rtx4090|rtx4090-1day,rtx4090-6hours|rtx4090-6hours
#qos
1day|1-00:00:00
1week|7-00:00:00
6hours|06:00:00
rtx4090-1day|1-00:00:00
rtx4090-6hours|06:00:00
"""


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
    cluster = CLUSTER.replace("rtx4090-1day,", "")  # not the profile's QOS
    ssh = _FakeSSH(
        {"sbatch": "123\n", "#partitions": cluster},
        files={f"{WF}/profile/slurm/config.yaml": profile},
    )
    at = _app(ssh, input="/home/u/data/scan.czi", work_dir="/scratch/run")
    at.run()
    assert not at.exception
    # Checked as soon as shown: the profile's segment QOS is not the account's.
    assert any("'rtx4090-1day' is not one" in e.value for e in at.error)
    partitions = next(
        d.value for d in at.dataframe if "partition" in d.value.columns
    )
    assert "rtx4090" in partitions["partition"].tolist()
    next(r for r in at.radio if r.label == "run mode").set_value(
        core.SLURM_JOB
    ).run()
    # The controller's partition, QOS and account are dropdowns of what
    # the cluster has; anything else may still be typed.
    assert "rtx4090" in at.selectbox(key="ctrl_partition").options
    assert at.selectbox(key="ctrl_account").options == ["(default)", "biehlmai"]
    at.selectbox(key="ctrl_partition").set_value("scicore").run()
    assert "1day" in at.selectbox(key="ctrl_qos").options
    assert "rtx4090-6hours" not in at.selectbox(key="ctrl_qos").options
    # The default QOS (6 hours) is too short for the controller's 3 days:
    # one that allows them is chosen.
    assert at.selectbox(key="ctrl_qos").value == "1week"
    # A QOS picked that is too short: flagged, with a button to fix it.
    at.selectbox(key="ctrl_qos").set_value("1day").run()
    assert any("longer than QOS '1day'" in e.value for e in at.error)
    next(b for b in at.button if b.label == "Use QOS 1week").click().run()
    assert at.selectbox(key="ctrl_qos").value == "1week"
    assert not any("QOS '1day'" in e.value for e in at.error)
    # A shorter time: the QOS chosen for the user follows it.
    next(t for t in at.text_input if t.label == "time").set_value(
        "04:00:00"
    ).run()
    assert at.selectbox(key="ctrl_qos").value == ""
    next(t for t in at.text_input if t.label == "time").set_value(
        "3-00:00:00"
    ).run()
    next(b for b in at.button if b.label == "Launch").click().run()
    assert not at.exception
    fetched = next(
        i for i, c in enumerate(ssh.commands) if "fetch_models.py" in c
    )
    submitted = next(i for i, c in enumerate(ssh.commands) if "sbatch" in c)
    assert fetched < submitted  # the models first, on the login node
    record = json.loads(ssh.files[f"{WF}/{core.REGISTRY}"].decode())
    assert record["slurm_job"] == "123" and record["kind"] == "single"
    # Cellpose's "auto" tiles: sized for the segment partition's RTX 4090,
    # not the 8 GB fallback.
    import yaml

    uploaded = yaml.safe_load(ssh.files[record["config"]])
    assert uploaded["gpu_memory_gb"] == 24
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
            "nuclei_channel": 1,  # the watershed grows from nuclei
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
        # The launcher wants a path on the (Linux) cluster: tmp_path is a
        # C:\ path on Windows.
        assert not core.config_problems({**cfg, "work_dir": "/scratch/run"})
    assert core.multi_problems(files) == []
    from pathlib import Path

    assert run_multi._validate_configs([Path(p) for p in paths], cfgs)

    # A method that fills space, stitched by touch: refused, as the
    # workflow would.
    touch = core.build_config({**shared, **segs[1], "stitch": "touch"})
    (problem,) = core.config_problems({**touch, "work_dir": "/scratch/run"})
    assert 'use stitch "iou"' in problem
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


def _roundtrip_files(directory):
    shared = {
        "input": "/data/planes/*.tif",
        "sequence_pattern": "_Z(?P<Z>\\d+)",
        "work_dir": "/scratch/run",
        "tile_shape": "[16, 512, 512]",
        "compression": "zstd:3",
        "level": 1,
        "tiles_per_job": 2,
        "gpu_memory_gb": 24,
        "skip_empty": False,
        "empty_threshold": 120.0,
        "pyramid_levels": 4,
        "table_channels": [0, 2],
        "notify_email": "me@example.org",
        "notify_events": ["error"],
        "merge_workers": 8,
    }
    segs = [
        {
            "label_name": "nuclei",
            "method": "cellpose",
            "channel": 1,
            "overlap": "[4, 30, 30]",
            "cp_model": "nuclei",
            "cp_diameter": 17.0,
            "cp_do_3d": True,
            "cp_extra_kwargs": "flow_threshold: 0.6",
            "fill_holes": "per_plane",
            "min_volume": 5.0,
        },
        {
            "label_name": "cells",
            "method": "plantseg",
            "channel": 0,
            # One cell grown from each of the nuclei segmented above.
            "seed_labels": "nuclei",
            "ps_model": "generic_light_sheet_3D_unet",
            "ps_segmentation": "nuclei_watershed",
            "ps_beta": 0.5,
            "max_radius_um": 12.0,
            "extra_kwargs": "patch: [64, 128, 128]",
            "stitch": "iou",
            "iou_threshold": 0.3,
            "dilate": 2,
        },
        {
            "label_name": "cilia",
            "method": "dog",
            "channel": 2,
            "dog_low_sigma": 0.5,
            "dog_high_sigma": 2.0,
            "dog_threshold": 0.05,
            "dog_sigma_units": "um",
            "dog_use_gpu": True,
            "dog_psf": "/psf.tif",
            "dog_wavelength": 488,
            "dog_na": 1.1,
            "dog_nimm": 1.33,
            "dog_dup_rev_z": "on",
            "open_radius": 1,
        },
    ]
    return core.build_multi(
        shared,
        segs,
        [
            {"a": "nuclei", "b": "cells", "output": "n_c.xlsx"},
            {
                "a": "cilia",
                "b": "cells",
                "output": "c_c.xlsx",
                "max_distance_um": 1.5,
            },
        ],
        directory=directory,
        relate={"partition": "gpu", "mem": "64G", "cpus": 8, "time": 90},
        bundle={"format": "iso"},
        review={
            "expect": {"cells": {"nuclei": 1, "cilia": [0, 2]}},
            "min_overlap": 0.7,
            "position": {"cilia": {"parent": "cells", "apical": "nuclei"}},
        },
    )


def test_a_past_run_loads_back_into_the_form_unchanged():
    """Every setting the form writes comes back from the config files."""
    pytest.importorskip("paramiko")
    import yaml

    directory = f"{WF}/config/launcher_multi_1"
    files = _roundtrip_files(directory)
    multi_path = f"{directory}/multi.yaml"
    remote = {p: yaml.safe_dump(c).encode() for p, c in files.items()}
    remote[f"{WF}/{core.REGISTRY}"] = core.registry_line(
        _job(config=multi_path, kind="multi")
    ).encode()
    at = _app(_FakeSSH({"squeue": "DONE"}, files=remote))
    at.run()
    assert not at.exception
    past = next(s for s in at.selectbox if s.label == "a past launch")
    past.set_value(multi_path).run()
    next(b for b in at.button if b.key == "load_past").click().run()
    assert not at.exception
    note = next(s.value for s in at.success if "loaded" in s.value)
    assert "left out" not in note
    rebuilt = at.session_state["files"]
    by_label = {
        c["label_name"]: c
        for p, c in rebuilt.items()
        if not p.endswith("multi.yaml")
    }
    for path, cfg in files.items():
        if path.endswith("multi.yaml"):
            continue
        assert by_label[cfg["label_name"]] == cfg, cfg["label_name"]
    new_multi = next(c for p, c in rebuilt.items() if p.endswith("multi.yaml"))
    old_multi = files[multi_path]
    for key in ("relations", "relate", "bundle", "review"):
        assert new_multi.get(key) == old_multi.get(key), key


def test_form_state_says_what_the_form_cannot_hold():
    cfg = core.build_config({**BASE, "method": "threshold"})
    cfg["seed_labels"] = "nuclei"
    cfg["denoise"] = {"model": "n2v"}
    state, ignored = core.form_state([cfg])
    assert ignored == ["denoise"]  # seed_labels has a field now
    assert (
        state["run_kind"] == core.RUN_KINDS[0]
        and state["method"] == "threshold"
    )


def test_a_decon_without_optics_keeps_the_plugins_defaults():
    """The form's defaults (an oil objective) are not what a config that
    leaves them out meant: the plugin's own are."""
    cfg = core.build_config(
        {**BASE, "method": "dog", "dog_psf": "/p.tif", "dog_wavelength": 488}
    )
    del cfg["custom"]["kwargs"]["decon_kwargs"]["wavelength"]
    state, _ = core.form_state([cfg])
    assert (state["wl"], state["na"], state["ni"]) == (520, 1.25, 1.3)


def test_a_config_file_loads_into_the_one_segmentation_form():
    pytest.importorskip("paramiko")
    import yaml

    cfg = core.build_config(
        {
            **BASE,
            "method": "watershed",
            "nuclei_channel": 1,
            "ws_nuclei_min_size": 200,
            "foreground": "none",
            "dilate": 1,
            "dilate_gpu": True,
            "stitch": "iou",
            "overlap": "[2, 20, 20]",
            "max_volume": 900.0,
            "shard": True,
        }
    )
    path = "/home/u/my_run.yaml"
    at = _app(_FakeSSH(files={path: yaml.safe_dump(cfg).encode()}))
    at.run()
    at.text_input(key="load_path").set_value(path).run()
    next(b for b in at.button if b.key == "load_file").click().run()
    assert not at.exception
    (rebuilt,) = at.session_state["files"].values()
    assert rebuilt == cfg


@pytest.mark.parametrize(
    "extra",
    [
        {"method": "watershed", "nuclei_channel": 1, "max_radius_um": "15"},
        {"method": "watershed", "seed_labels": "nuclei", "foreground": "none"},
        {
            "method": "plantseg",
            "ps_segmentation": "nuclei_watershed",
            "seed_labels": "nuclei",
        },
        {"method": "plantseg", "ps_model": "my_own_model", "ps_beta": 0.3},
    ],
)
def test_seeded_methods_write_what_the_workflow_accepts(extra):
    """PlantSeg and the watershed from the form: valid to the workflow,
    but for packages this test environment may lack."""
    from _pw import environment_problems, validate_config

    cfg = core.build_config({**BASE, "stitch": "iou", **extra})
    assert cfg["method"] == "custom"
    assert cfg["custom"]["module"] == core.SEEDED[extra["method"]]
    assert core.config_problems(cfg) == []
    missing = environment_problems(cfg)
    try:
        validate_config(cfg)
    except ValueError as exc:
        others = [
            line
            for line in str(exc).splitlines()[1:]
            if not any(m in line for m in missing)
        ]
        assert others == [], others


def test_seed_rules():
    def problems(**extra):
        return core.config_problems(
            core.build_config({**BASE, "stitch": "iou", **extra})
        )

    assert "nuclei" in problems(method="watershed")[0]
    assert (
        "not both"
        in problems(method="watershed", seed_labels="nuclei", nuclei_channel=1)[
            0
        ]
    )
    assert "from itself" in problems(method="watershed", seed_labels="cells")[0]
    assert (
        "would ignore it"
        in problems(method="plantseg", seed_labels="nuclei")[0]
    )
    assert (
        "uses nuclei"
        in problems(method="plantseg", ps_segmentation="lifted_multicut")[0]
    )


def test_gpu_memory_from_the_segment_gpu():
    info = core.parse_cluster_info(CLUSTER)
    rtx = {"slurm_partition": "rtx4090", "gres": "gpu:1"}
    assert core.gpu_memory_gb(rtx, info) == (24, "rtx4090")
    assert core.gpu_memory_gb({"gres": "gpu:a100:1"}, info) == (40, "a100")
    assert core.gpu_memory_gb({"gres": "gpu:mystery:1"}, info)[0] is None
    assert core.gpu_memory_gb({"slurm_partition": "scicore"}, info)[0] is None

    cellpose = core.build_config({**BASE, "method": "cellpose"})
    threshold = core.build_config({**BASE, "method": "threshold"})
    fixed = core.build_config(
        {**BASE, "method": "cellpose", "tile_shape": "[16, 512, 512]"}
    )
    assert core.needs_gpu_memory(cellpose)
    assert not core.needs_gpu_memory(threshold)  # sized to host memory
    assert not core.needs_gpu_memory(fixed)
    files = core.with_gpu_memory({"a": cellpose, "b": threshold, "m": {}}, 24)
    assert files["a"]["gpu_memory_gb"] == 24
    assert files["b"]["gpu_memory_gb"] is None and files["m"] == {}


def test_launcher_and_workflow_agree_on_the_8gb_fallback():
    from _pw import sized_for_fallback_gpu, with_defaults

    for extra in (
        {"method": "cellpose"},
        {"method": "cellpose", "cp_gpu": False},
        {"method": "threshold"},
        {"method": "cellpose", "gpu_memory_gb": 24},
        {"method": "cellpose", "tile_shape": "[8, 256, 256]"},
    ):
        cfg = with_defaults(core.build_config({**BASE, **extra}))
        assert core.needs_gpu_memory(cfg) == sized_for_fallback_gpu(cfg), extra


def test_jobs_tab_shows_a_tried_tile():
    pytest.importorskip("paramiko")
    report = json.loads(_report(["cells"], converted=True))
    report["segmentations"][0]["plan"].update(
        {
            "seconds_per_tile": 4.2,
            "preview": "/scratch/run/previews/1/cells.png",
            "preview_tile": 3,
            "preview_objects": 57,
            "preview_plane": 8,
        }
    )
    import io

    from PIL import Image

    png = io.BytesIO()
    Image.new("RGB", (8, 4)).save(png, format="PNG")
    job = _job(kind="check", previews="/scratch/run/previews/1")
    ssh = _FakeSSH(
        {"squeue": "DONE\n", "tail -n": "[check] log\n" + json.dumps(report)},
        files={
            f"{WF}/{core.REGISTRY}": core.registry_line(job).encode(),
            "/scratch/run/previews/1/cells.png": png.getvalue(),
        },
    )
    at = _app(ssh, input="/x.czi", work_dir="/scratch/run")
    at.run()
    assert not at.exception
    (image,) = [e for e in at.main if type(e).__name__ == "Image"]
    assert "tile 3, plane 8 of the tile · 57 objects · 4.2 s per tile" in str(
        image.proto
    )


def test_configure_shows_the_channels():
    pytest.importorskip("paramiko")
    import io

    from PIL import Image

    png = io.BytesIO()
    Image.new("L", (4, 4)).save(png, format="PNG")
    info = {
        "store": "/scratch/run/image.zarr",
        "axes": "czyx",
        "shape": [3, 40, 2048, 2048],
        "channels": 3,
        "voxel_um": {"z": 0.5, "y": 0.1, "x": 0.1},
        "level": 1,
        "pictures": [
            f"/scratch/run/previews/channels/channel_{c}.png" for c in range(3)
        ],
    }
    ssh = _FakeSSH(
        {"scripts/channels.py": json.dumps(info)},
        files={p: png.getvalue() for p in info["pictures"]},
    )
    at = _app(ssh, input="/x.czi", work_dir="/scratch/run")
    at.run()
    at.button(key="show_channels").click().run()
    assert not at.exception
    images = [e for e in at.main if type(e).__name__ == "Image"]
    assert len(images) == 3
    assert any(
        "3 × 40 × 2048 × 2048 · voxel z 0.5 µm" in c.value for c in at.caption
    )


QUEUE = """101|pw-cells|RUNNING|None|1:02:03|3-00:00:00|scicore
102|pw-cells_3f2a|PENDING|Priority|0:00|1-00:00:00|rtx4090
103|pw-cells_3f2a|PENDING|QOSMaxWallDurationPerJobLimit|0:00|2-00:00:00|rtx4090
104|my_other_job|PENDING|Priority|0:00|1:00:00|scicore
"""


def test_queue_rows_say_why_jobs_wait():
    rows = core.parse_queue(QUEUE)
    assert [r["job"] for r in rows] == ["101", "102", "103"]  # ours only
    assert rows[0]["running for"] == "1:02:03" and rows[0]["why it waits"] == ""
    assert rows[1]["why it waits"].startswith("waiting its turn")
    assert rows[2]["why it waits"].startswith("NEVER STARTS")


def test_jobs_tab_shows_failures_and_the_queue():
    pytest.importorskip("paramiko")
    status = {
        "work_dir": "/scratch/run",
        "segmentations": [
            {
                "label_name": "cells",
                "stage": "segmenting",
                "batches": 10,
                "batches_done": 3,
                "objects": None,
                "seams": None,
                "failures": [
                    {
                        "step": "segment job 4",
                        "log": "/scratch/run/cells/logs/segment/4.log",
                        "error": "torch.OutOfMemoryError: CUDA out of memory",
                        "hint": "the GPU ran out of memory: smaller tiles",
                    }
                ],
            }
        ],
        "failures": [],
    }
    ssh = _FakeSSH(
        {
            "squeue --me": QUEUE,
            "squeue": "RUNNING\n",
            "scripts/status.py": json.dumps(status),
            "tail -n 80": "CUDA out of memory\n",
        },
        files={f"{WF}/{core.REGISTRY}": core.registry_line(_job()).encode()},
    )
    at = _app(ssh, input="/x.czi", work_dir="/scratch/run")
    at.run()
    assert not at.exception
    assert any("segment job 4" in e.value for e in at.error)
    assert any("GPU ran out of memory" in m.value for m in at.markdown)
    assert any("NEVER STARTS" in e.value and "103" in e.value for e in at.error)
    queue = next(
        d.value for d in at.dataframe if "why it waits" in d.value.columns
    )
    assert len(queue) == 3


CONTROLLER_LOG = """[Mon Oct 19 09:44:01 2026]
rule convert:
    output: /scratch/run/image.zarr/zarr.json
    log: /scratch/run/logs/convert.log
    jobid: 0
    reason: Missing output files: /scratch/run/image.zarr/zarr.json
Job 0 has been submitted with SLURM jobid 201 (log: /s/slurm_logs/rule_convert/201.log).
[Mon Oct 19 09:48:01 2026]
Error in rule convert:
    message: SLURM-job '201' failed
    jobid: 0
    log: /scratch/run/logs/convert.log (check log file(s) for error details)
Trying to restart job 0.
[Mon Oct 19 09:56:32 2026]
rule convert:
    output: /scratch/run/image.zarr/zarr.json
    log: /scratch/run/logs/convert.log
    jobid: 0
Job 0 has been submitted with SLURM jobid 203 (log: /s/slurm_logs/rule_convert/203.log).
"""


def test_steps_from_the_controllers_log():
    (step,) = core.parse_steps(CONTROLLER_LOG)
    assert step["rule"] == "convert" and step["attempt"] == 2
    assert step["slurm_job"] == "203" and step["state"] == "submitted"
    assert step["log"] == "/scratch/run/logs/convert.log"
    assert step["slurm_log"] == "/s/slurm_logs/rule_convert/203.log"
    done = core.parse_steps(
        CONTROLLER_LOG + "Finished jobid: 0 (Rule: convert)\n"
    )
    assert done[0]["state"] == "done"
    progress = core.last_progress(
        "12:00 INFO patchworks - image.zarr/0: 1,234/5,000 (24%) after 3m, "
        "~9m left\nother line\n"
    )
    assert progress["label"] == "image.zarr/0" and progress["left"] == "9m"
    assert round(progress["fraction"], 2) == 0.25
    assert core.last_progress("nothing yet") is None


def test_jobs_tab_convert_only_shows_the_convert_step():
    """A convert-only run: its progress is the conversion's, from the step's
    own log, not the segmentations'; each step's log is one click away."""
    pytest.importorskip("paramiko")
    status = {
        "work_dir": "/scratch/run",
        "segmentations": [
            {
                "label_name": "nuclei",
                "stage": "not started",
                "batches": None,
                "batches_done": 0,
                "objects": None,
                "seams": None,
            }
        ],
        "failures": [],
        "converted": False,
        "occupancy": False,
    }
    ssh = _FakeSSH(
        {
            "squeue --me": "203|patchworks_95|RUNNING|None|12:00|6:00:00|scicore\n",
            "squeue": "RUNNING\n",
            "scripts/status.py": json.dumps(status),
            "grep -E": CONTROLLER_LOG,
            "/scratch/run/logs/convert.log": "image.zarr/0: 1,234/5,000 "
            "(24%) after 3m, ~9m left\n",
            "tail -n": "the log\n",
        },
        files={
            f"{WF}/{core.REGISTRY}": core.registry_line(
                _job(kind="convert", slurm_job="42")
            ).encode()
        },
    )
    at = _app(ssh, input="/x.czi", work_dir="/scratch/run")
    at.run()
    assert not at.exception
    texts = [p.proto.text for p in at.get("progress")]
    assert any(
        "convert** · image.zarr/0: 25%, ~9m left (attempt 2)" in t
        for t in texts
    ), texts
    assert any("occupancy** · waiting to start" in t for t in texts)
    assert not any("nuclei" in t for t in texts)
    steps = next(d.value for d in at.dataframe if "step" in d.value.columns)
    assert steps["state"].tolist() == ["running"]
    assert any("convert · running" in e.label for e in at.expander)
    assert any(
        "updated" in c.value and "every 15 s" in c.value for c in at.caption
    )


def test_cellpose_model_dropdown_lists_the_clusters_models():
    pytest.importorskip("paramiko")
    ssh = _FakeSSH({"available_models": 'MODELS:["cpsam"]\n'})
    at = _app(ssh, input="/x.czi", work_dir="/scratch/run")
    at.run()
    assert not at.exception
    model = at.selectbox(key="cpm")
    # Cellpose 4 only: its model, and the form's default is not offered.
    assert model.options == ["cpsam"] and model.value == "cpsam"
    (cfg,) = at.session_state["files"].values()
    assert cfg["cellpose"]["model"] == "cpsam"
    # Asked once, not on every rerun.
    at.run()
    assert sum("available_models" in c for c in ssh.commands) == 1
    assert core.parse_models("Traceback") == []


def test_template_files():
    single = {
        "/w/config/launcher_a.yaml": core.build_config(
            {**BASE, "method": "threshold"}
        )
    }
    ((path, cfg),) = core.template_files(single, "my run", "/t").items()
    assert path == "/t/my-run.yaml"
    assert "input" not in cfg and "work_dir" not in cfg
    kept = core.template_files(single, "x", "/t", with_image=True)
    assert kept["/t/x.yaml"]["input"] == BASE["input"]

    files = _roundtrip_files("/w/config/launcher_multi_1")
    out = core.template_files(files, "lab", "/t")
    multi = out["/t/lab/multi.yaml"]
    assert multi["segmentations"] == [
        "seg_nuclei.yaml",
        "seg_cells.yaml",
        "seg_cilia.yaml",
    ]
    assert (
        multi["relations"]
        == files["/w/config/launcher_multi_1/multi.yaml"]["relations"]
    )
    assert "input" not in out["/t/lab/seg_cells.yaml"]
    assert core.template_choices(
        [("lab", True), ("my-run.yaml", False), ("notes.txt", False)], "/t"
    ) == {"lab": "/t/lab/multi.yaml", "my-run": "/t/my-run.yaml"}


def test_save_a_template_and_start_the_next_run_from_it():
    """Settings saved without the image; loaded, the form keeps its own."""
    pytest.importorskip("paramiko")
    ssh = _FakeSSH()
    at = _app(ssh, input="/data/first.czi", work_dir="/scratch/first")
    at.run()
    at.selectbox(key="cpm").set_value("nuclei")
    at.text_input(key="ov").set_value("[4, 20, 20]").run()
    at.text_input(key="template_name").set_value("nuclei 3D").run()
    next(b for b in at.button if b.label == "Save").click().run()
    assert not at.exception
    saved = f"{WF}/config/templates/nuclei-3D.yaml"
    import yaml

    template = yaml.safe_load(ssh.files[saved])
    assert template["cellpose"]["model"] == "nuclei" and "input" not in template

    # The next image: another form, its own image, the template's settings.
    at = _app(ssh, input="/data/second.czi", work_dir="/scratch/second")
    at.run()
    next(s for s in at.selectbox if s.label == "a template").set_value(
        "nuclei-3D"
    ).run()
    at.button(key="load_template").click().run()
    assert not at.exception
    (cfg,) = at.session_state["files"].values()
    assert cfg["input"] == "/data/second.czi"
    assert cfg["work_dir"] == "/scratch/second"
    assert cfg["cellpose"]["model"] == "nuclei" and cfg["overlap"] == [
        4,
        20,
        20,
    ]


INPUTS = (
    "TextInput",
    "NumberInput",
    "Selectbox",
    "Checkbox",
    "TextArea",
    "Radio",
    "Multiselect",
    "ButtonGroup",
)


@pytest.mark.parametrize(
    "method",
    ["cellpose", "plantseg", "watershed", "dog", "threshold", "custom"],
)
def test_every_setting_explains_itself(method):
    """Each field has its question mark: what it does, and an example."""
    pytest.importorskip("paramiko")
    profile = (ROOT / "workflow/profile/slurm/config.yaml").read_bytes()
    ssh = _FakeSSH(
        {"#partitions": CLUSTER},
        files={f"{WF}/profile/slurm/config.yaml": profile},
    )
    at = _app(
        ssh,
        input="/data/planes/*.tif",
        sequence_pattern="_Z(?P<Z>\\d+)",
        work_dir="/scratch/run",
    )
    at.run()
    at.button_group[0].set_value(core.RUN_KINDS[1]).run()
    next(b for b in at.button_group if b.proto.label == "method").set_value(
        method
    ).run()
    # The fields shown only once others are set, and what a method needs
    # before the launch settings show: nuclei for the watershed, a module.
    for key, value in (
        ("s0_psf", "/psf.tif"),
        ("s0_nuc", "1" if method == "watershed" else ""),
        ("s0_cm", "my_seg"),
    ):
        try:
            at.text_input(key=key).set_value(value).run()
        except KeyError:
            pass  # not this method's field
    at.number_input(key="s0_dil").set_value(1).run()
    next(r for r in at.radio if r.label == "run mode").set_value(
        core.SLURM_JOB
    ).run()
    assert not at.exception
    silent = sorted(
        {
            e.proto.label
            for e in list(at.main) + list(at.sidebar)
            if type(e).__name__ in INPUTS and not e.proto.help
        }
    )
    assert silent == []


def test_help_keys_point_at_texts():
    """Every short key of a segmentation's fields names an explanation."""
    import ast

    from help_text import HELP

    tree = ast.parse((ROOT / "workflow/launcher/app.py").read_text())
    short = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and getattr(node.targets[0], "id", "") == "SHORT_KEYS"
    )
    targets = [ast.literal_eval(v) for v in short.values]
    assert [t for t in targets if t not in HELP] == []
    assert all(len(text) > 20 for text in HELP.values())


def test_an_older_workflow_on_the_cluster_is_said_and_updated():
    """The launcher pulled on a computer, the cluster's clone not: its new
    scripts are missing there. Said in the sidebar, with Update it."""
    pytest.importorskip("paramiko")
    scripts = f"{WF}/scripts"
    old = [n for n in _FakeSFTP.tree[scripts] if n != "channels.py"]

    class Old(_FakeSSH):
        updated = False

        def open_sftp(self):
            ssh = self

            class Live(_FakeSFTP):
                # What the clone holds now: one SFTP channel is reused.
                @property
                def tree(self):
                    return {
                        **_FakeSFTP.tree,
                        scripts: _FakeSFTP.tree[scripts]
                        if ssh.updated
                        else old,
                    }

            return Live(self.files)

        def exec_command(self, command, timeout=None):
            if "git pull" in command:
                self.updated = True
            return super().exec_command(command, timeout)

    ssh = Old({"git pull": "Fast-forward\n 5 files changed\n"})
    at = _app(ssh, input="/x.czi", work_dir="/scratch/run")
    at.run()
    assert any(
        "older than this launcher" in w.value for w in at.sidebar.warning
    )
    at.button(key="update_workflow").click().run()
    assert any("git pull --ff-only" in c for c in ssh.commands)
    assert any(
        "updated: 5 files changed" in m.value for m in at.sidebar.success
    )
    assert not any(
        "older than this launcher" in w.value for w in at.sidebar.warning
    )


def test_a_missing_script_says_what_to_do():
    error = (
        "python: can't open file '/scicore/home/lab/me/patchworks/workflow/"
        "scripts/channels.py': [Errno 2] No such file or directory"
    )
    said = core.explain_remote_error(error)
    assert "no scripts/channels.py" in said and "git pull" in said
    assert core.explain_remote_error("x\nValueError: y") == "ValueError: y"
