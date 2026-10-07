"""Tests for the multi-config driver's SLURM-facing behaviour."""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "workflow" / "scripts")
)

import numpy as np  # noqa: E402
import openpyxl  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402

from run_multi import (  # noqa: E402
    _CONVERT_KEYS,
    _relate_cmd,
    _snakemake_cmd,
    _validate_configs,
    slurm_jobname_prefix,
)

# The SLURM executor's own rule (snakemake_executor_plugin_slurm): it raises a
# WorkflowError and aborts the whole run if the prefix does not match.
_EXECUTOR_RULE = re.compile(r"^[A-Za-z0-9_-]{1,50}$")


def test_jobname_prefix_satisfies_the_executor():
    """Whatever a label_name contains, the prefix must stay submittable.

    The executor names jobs after a UUID and refuses a --job-name override, so
    this prefix is the only thing that makes squeue readable -- and an invalid
    one fails the run rather than degrading.
    """
    for label in ("nuclei_labels", "cyto_labels", "convert", "a"):
        assert _EXECUTOR_RULE.match(slurm_jobname_prefix(label))

    # Characters a label might plausibly pick up are sanitised, not passed on.
    assert _EXECUTOR_RULE.match(slurm_jobname_prefix("cilia/v2 (test)"))
    assert _EXECUTOR_RULE.match(slurm_jobname_prefix("run 1: nuclei"))
    # And an over-long label is truncated to the executor's 50-char limit.
    assert _EXECUTOR_RULE.match(slurm_jobname_prefix("x" * 200))


def test_jobname_prefix_keeps_the_label_readable():
    """The label must lead, since that is what a queue listing truncates to."""
    assert slurm_jobname_prefix("nuclei_labels") == "pw-nuclei_labels"
    assert slurm_jobname_prefix("convert") == "pw-convert"


def test_common_configfile_is_merged_under_the_per_config_one():
    """Snakemake merges --configfile values in order, later winning.

    That ordering is the whole mechanism: shared settings come from common.yaml
    and the per-config file overrides only what differs. Swap the two and every
    config would silently get the shared defaults instead of its own channel.
    """
    cmd = _snakemake_cmd(
        Path("config/config_nuclei.yaml"),
        workflow_dir=Path("workflow"),
        profile=None,
        cores=8,
        dry_run=False,
        common=Path("config/common.yaml"),
    )
    i = cmd.index("--configfile")
    assert cmd[i + 1].endswith("common.yaml")
    assert cmd[i + 2].endswith("config_nuclei.yaml")

    # Without a common file the invocation is unchanged: one configfile, so
    # a self-contained config keeps working exactly as before.
    plain = _snakemake_cmd(
        Path("config/config_nuclei.yaml"),
        workflow_dir=Path("workflow"),
        profile=None,
        cores=8,
        dry_run=False,
    )
    j = plain.index("--configfile")
    assert plain[j + 1].endswith("config_nuclei.yaml")
    assert not plain[j + 2].endswith(".yaml")


def test_convert_keys_must_agree_across_configs(tmp_path):
    """`convert` runs once from the first config, so a later one is ignored.

    Setting shard on the second config and watching a million files appear
    anyway is invisible without this check -- there is no log line saying the
    value was dropped, because nothing ever read it.
    """
    paths = [Path("a.yaml"), Path("b.yaml")]
    # A creatable work_dir: "/w" only passed where the tests ran as root.
    w = str(tmp_path / "w")
    base = {"work_dir": w, "tile_shape": [16, 512, 512], "level": 0}
    good = [
        {**base, "label_name": "a", "shard": True},
        {**base, "label_name": "b", "shard": True},
    ]
    assert _validate_configs(paths, good) == w

    bad = [
        {**base, "label_name": "a", "shard": True},
        {**base, "label_name": "b", "shard": False},
    ]
    # It reports every problem and exits, rather than raising, so that a
    # mistake costs one readable message instead of a traceback.
    with pytest.raises(SystemExit):
        _validate_configs(paths, bad)


@pytest.mark.parametrize("multi_file", ["multi.yaml", "multi_plantseg.yaml"])
def test_shipped_multi_configs_are_consistent(
    tmp_path, multi_file, monkeypatch
):
    """The shipped example must satisfy its own validator.

    It is the thing users copy, so a config set that run_multi would refuse to
    start is worse than no example at all.

    The templates' paths are placeholders, which the validator now rejects on
    purpose (running them unedited used to die deep in pathlib instead). What
    is being checked here is the *structure* they teach -- one shared
    work_dir, one tile_shape, convert's keys in the shared file -- so point
    them at a real directory first and check that.
    """
    cfg_dir = Path(__file__).resolve().parents[1] / "workflow" / "config"
    multi = yaml.safe_load((cfg_dir / multi_file).read_text())
    common = yaml.safe_load((cfg_dir.parent / multi["common"]).read_text())
    paths = [cfg_dir.parent / p for p in multi["segmentations"]]
    cfgs = [{**common, **yaml.safe_load(p.read_text())} for p in paths]

    work_dir = str(tmp_path / "results")
    for cfg in cfgs:
        cfg["work_dir"] = work_dir
        cfg["input"] = str(tmp_path / "scan.ims")

    assert _validate_configs(paths, cfgs) == work_dir
    # Every key convert reads comes from the shared file, not a per-config one.
    for path in paths:
        own = yaml.safe_load(path.read_text())
        assert not set(own) & set(_CONVERT_KEYS), path.name
    # Each config passes prepare's own checks too (custom kwargs included),
    # and the relations and review rules name label images that get made.
    import _pw
    from _pw import validate_config
    from run_multi import _review_rules

    # PlantSeg is not installed here; its own environment has it.
    monkeypatch.setattr(_pw, "has_module", lambda name: True)
    for cfg in cfgs:
        validate_config(cfg)
    names = [c["label_name"] for c in cfgs]
    for rel in multi.get("relations") or ():
        assert {rel["a"], rel["b"]} <= set(names)
    if multi.get("review"):
        _review_rules(multi, names)


def _workflow_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "workflow"


def test_occupancy_is_a_submitted_rule_not_a_localrule():
    """The occupancy build must never run on the submit host.

    It streams the entire image. Doing that in the run_multi driver ran it on
    a login node, where a multi-terabyte read is killed with no traceback --
    the run just returned to the prompt. Only fetch_model may be local (it
    needs network); everything else has to get a real allocation.
    """
    wf = _workflow_dir()
    snakefile = (wf / "Snakefile").read_text()
    local_block = snakefile.split("localrules:")[1].split("rule ")[0]
    local = {
        line.strip().rstrip(",")
        for line in local_block.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    assert local == {"fetch_model"}, local

    rules = (wf / "rules" / "convert.smk").read_text()
    assert "rule occupancy:" in rules


def test_occupancy_is_not_rebuilt_by_the_driver():
    """run_multi must ask Snakemake for the map, not build it in-process.

    An in-process build bypasses the scheduler entirely, which is how it ended
    up on the login node.
    """
    src = (_workflow_dir() / "scripts" / "run_multi.py").read_text()
    assert "build_occupancy_map(" not in src
    assert "occupancy.zarr" in src


def test_relate_is_submitted_via_slurm_under_profile():
    """The relate step must never run in-process on the submit host.

    Same failure mode as the occupancy map: label_relations() streams every
    chunk of two full-resolution label volumes. A prior fix moved the map
    build off the login node; the relate step made the identical mistake and
    hung there for the same reason until this fix.
    """
    src = (_workflow_dir() / "scripts" / "run_multi.py").read_text()
    assert "from relate import run_relations" in src
    assert '"srun"' in src
    assert "label_relations(" not in src


def _relate_kwargs(**overrides):
    kwargs = dict(
        work_dir="/w",
        image_store="/w/image.zarr",
        workflow_dir=Path("/workflow"),
        relate_partition="scicore",
        relate_mem="32G",
        relate_cpus=8,
        relate_time=180,
        relate_qos=None,
    )
    kwargs.update(overrides)
    return kwargs


def test_relate_cmd_is_one_job_per_relation_pair():
    """A killed shared job used to lose every relation still queued behind

    the one that was running -- one srun per pair means a slow or failing
    pair can no longer starve, or take down, its siblings' time budget.
    """
    relations = [
        {"a": "nuclei_labels", "b": "cyto_labels", "output": "n2c.xlsx"},
        {"a": "cilia_labels", "b": "cyto_labels", "output": "c2c.xlsx"},
    ]
    cmds = [_relate_cmd(rel, **_relate_kwargs()) for rel in relations]

    assert len(cmds) == 2
    for cmd, rel in zip(cmds, relations):
        assert cmd[0] == "srun"
        assert "--relations" in cmd
        # Each job's payload is *only* its own pair, not the whole list.
        payload = json.loads(cmd[cmd.index("--relations") + 1])
        assert payload == [rel]


def test_relate_cmd_job_name_identifies_the_pair():
    cmd = _relate_cmd(
        {"a": "cilia_labels", "b": "cyto_labels"}, **_relate_kwargs()
    )
    name = cmd[cmd.index("--job-name") + 1]
    assert _EXECUTOR_RULE.match(name)
    assert "cilia_labels" in name
    assert "cyto_labels" in name


def test_relate_cmd_gives_each_pair_its_own_log():
    """Concurrent per-pair jobs sharing one relate.log would interleave --

    each pair's --log must be a distinct file, or the whole point of
    splitting the log the way segment/<batch>.log already does is lost.
    """
    cmds = [
        _relate_cmd(rel, **_relate_kwargs())
        for rel in (
            {"a": "nuclei_labels", "b": "cyto_labels"},
            {"a": "cilia_labels", "b": "cyto_labels"},
        )
    ]
    logs = [cmd[cmd.index("--log") + 1] for cmd in cmds]
    assert len(set(logs)) == 2
    assert all(log.startswith("/w/logs/relate/") for log in logs)


def test_relate_cmd_omits_qos_by_default():
    cmd = _relate_cmd({"a": "a", "b": "b"}, **_relate_kwargs())
    assert "--qos" not in cmd


def test_relate_cmd_passes_qos_when_set():
    """A default QOS whose MaxWall is shorter than --relate-time is exactly

    what killed a real run (QOSMaxWallDurationPerJobLimit) -- --relate-qos
    lets a longer one be requested explicitly instead of guessed at.
    """
    cmd = _relate_cmd({"a": "a", "b": "b"}, **_relate_kwargs(relate_qos="1day"))
    assert cmd[cmd.index("--qos") + 1] == "1day"


def test_relate_script_has_the_real_bookkeeping():
    """relate.py must be the actual implementation, not a stub.

    Submitting the wrong (or a trimmed-down) script would silently produce a
    workbook missing the unmatched-label rows the docstring promises.
    """
    src = (_workflow_dir() / "scripts" / "relate.py").read_text()
    assert "def run_relations(" in src
    assert "label_relations" in src
    # The workbook is written from the object tables (with every object,
    # unmatched ones included, and the review decisions applied).
    assert "relation_workbook" in src


def test_view_script_loads_every_label_by_default():
    """view.py must not override labels=, or auto-load stops working.

    view_in_napari's labels=None default is what auto-loads every label
    group under <image>/labels/<name>/ as its own layer -- passing an
    explicit labels= here would silently drop that and show only one.
    """
    src = (_workflow_dir() / "scripts" / "view.py").read_text()
    assert "from patchworks.plugins.napari import view_in_napari" in src
    assert "labels=" not in src


def test_viewer_is_a_separate_opt_in_pixi_environment():
    """napari's Qt/GUI deps must stay out of the default headless env.

    Adding them to the default `[pypi-dependencies]` would pull heavy GUI
    dependencies into every SLURM job's environment for a feature only used
    interactively.
    """
    import tomllib

    src = (_workflow_dir() / "pixi.toml").read_text()
    default_deps = src.split("[pypi-dependencies]")[1].split("[feature")[0]
    assert "napari" not in default_deps

    # Asserted on the parsed manifest rather than an exact line: the literal
    # `viewer = { features = ["viewer"] }` broke the moment the environment
    # gained no-default-feature, which is not what this test is about.
    pixi = tomllib.loads(src)
    assert pixi["environments"]["viewer"]["features"] == ["viewer"]
    assert (
        "napari"
        in (
            pixi["feature"]["viewer"]["pypi-dependencies"]["patchworks"][
                "extras"
            ]
        )
    )


def test_relate_writes_its_own_log():
    """relate.py runs via srun, not a Snakemake rule -- nothing else wires up

    its logging (see prepare/segment/merge's `log:` directives), so main()
    has to call start_log() itself or its output only ever streams to
    whatever invoked srun and is gone once that terminal scrolls past it.
    """
    src = (_workflow_dir() / "scripts" / "relate.py").read_text()
    assert "from _pw import start_log" in src
    assert "start_log(" in src
    assert '"logs" / "relate.log"' in src


def test_relate_rechunks_mismatched_label_arrays(tmp_path):
    """A chunk-layout mismatch must be rechunked away, not require a re-run.

    Two configs are free to have segmented at different tile_shape (one
    published before the other's config changed, or a cheaper method sized
    its own tile differently) -- label_relations() itself refuses mismatched
    chunks by design, but that only means the caller has to rechunk one side
    first, not that the whole segmentation needs redoing.
    """
    import zarr

    from relate import run_relations

    image_store = str(tmp_path / "image.zarr")

    # a: labels 1 and 2, split at x=5. b: a single label 10 covering all of
    # a's label 1 and none of label 2 -- built with a *different* chunking.
    a_data = np.zeros((1, 10), dtype=np.int32)
    a_data[0, :5] = 1
    a_data[0, 5:] = 2
    b_data = np.zeros((1, 10), dtype=np.int32)
    b_data[0, :5] = 10

    root = zarr.open_group(image_store, mode="w")
    labels = root.require_group("labels")
    a_grp = labels.require_group("nuclei_labels")
    a_arr = a_grp.create_array(
        name="0", shape=a_data.shape, chunks=(1, 2), dtype=np.int32
    )
    a_arr[:] = a_data
    a_grp.attrs["sequential_labels"] = True
    a_grp.attrs["n_objects"] = 2

    b_grp = labels.require_group("cyto_labels")
    b_arr = b_grp.create_array(
        name="0", shape=b_data.shape, chunks=(1, 5), dtype=np.int32
    )
    b_arr[:] = b_data
    b_grp.attrs["sequential_labels"] = True
    b_grp.attrs["n_objects"] = 1

    out_dir = tmp_path / "work"
    out_dir.mkdir()
    run_relations(
        str(out_dir),
        image_store,
        [{"a": "nuclei_labels", "b": "cyto_labels", "output": "rel.xlsx"}],
    )

    wb = openpyxl.load_workbook(out_dir / "rel.xlsx")
    rows = {
        row[0]: (row[1], row[2], row[3])
        for row in wb["nuclei_labels"].iter_rows(min_row=2, values_only=True)
    }
    assert rows[1] == (10, 5, 1.0)  # label 1 fully inside b's label 10
    assert rows[2] == (None, 0, 0)  # label 2 touches nothing in b


def test_relation_up_to_date_missing_output_is_false(tmp_path):
    from relate import _relation_up_to_date

    assert not _relation_up_to_date(
        str(tmp_path), "a", "b", tmp_path / "nope.xlsx"
    )


def test_relation_up_to_date_missing_marker_is_false(tmp_path):
    """No labels.done for a label means its state can't be judged -- treat

    that as "recompute", not as "trust the existing workbook".
    """
    from relate import _relation_up_to_date

    out = tmp_path / "rel.xlsx"
    out.write_text("x")
    assert not _relation_up_to_date(str(tmp_path), "a", "b", out)


def test_relation_up_to_date_true_only_when_newer_than_both_markers(
    tmp_path,
):
    from relate import _relation_up_to_date

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "labels.done").touch()
    (tmp_path / "b" / "labels.done").touch()
    out = tmp_path / "rel.xlsx"
    out.write_text("x")

    now = time.time()
    os.utime(tmp_path / "a" / "labels.done", (now, now))
    os.utime(tmp_path / "b" / "labels.done", (now, now))

    # older than both markers -> stale
    os.utime(out, (now - 10, now - 10))
    assert not _relation_up_to_date(str(tmp_path), "a", "b", out)

    # newer than both markers -> up to date
    os.utime(out, (now + 10, now + 10))
    assert _relation_up_to_date(str(tmp_path), "a", "b", out)

    # b re-merged after the workbook was written -> stale again
    os.utime(tmp_path / "b" / "labels.done", (now + 20, now + 20))
    assert not _relation_up_to_date(str(tmp_path), "a", "b", out)


def test_relate_skips_a_relation_whose_workbook_is_up_to_date(tmp_path):
    """A retry must not recompute what already finished -- only what a

    shared, now-split-per-pair job left missing after a partial failure.
    Proven by planting sentinel content no real relation would produce: if
    run_relations() recomputed anyway, the sentinel would be gone.
    """
    from relate import run_relations

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    for name in ("a_labels", "b_labels"):
        (work_dir / name).mkdir()
        (work_dir / name / "labels.done").touch()

    out_path = work_dir / "rel.xlsx"
    sentinel = openpyxl.Workbook()
    sentinel.active.append(["sentinel"])
    sentinel.save(out_path)
    os.utime(out_path, (time.time() + 60, time.time() + 60))

    # No image_store/labels at all -- if this weren't skipped, run_relations
    # would raise trying to open them, not just produce the wrong content.
    run_relations(
        str(work_dir),
        str(tmp_path / "image.zarr"),
        [{"a": "a_labels", "b": "b_labels", "output": "rel.xlsx"}],
    )

    wb = openpyxl.load_workbook(out_path)
    assert wb.active["A1"].value == "sentinel"


def test_mixed_nuclei_channel_auto_passes_validation(tmp_path):
    """A channel-count mismatch under `tile_shape: "auto"` is no longer

    refused at validation time -- it's resolved automatically instead (see
    `_resolve_shared_tile_shape`), which needs the converted image's real
    shape/dtype and so can only run after phase A, not from
    `_validate_configs()`. This used to `sys.exit` here; asserting that
    would now be testing the wrong layer.
    """
    paths = [Path("a.yaml"), Path("b.yaml")]
    w = str(tmp_path / "w")  # creatable, without root
    base = {"work_dir": w, "tile_shape": "auto", "level": 0}

    mixed = [
        {**base, "label_name": "a", "channel": 0, "nuclei_channel": 1},
        {**base, "label_name": "b", "channel": 2},
    ]
    assert _validate_configs(paths, mixed) == w

    # Same pair with one explicit shape is fine: both get that tile.
    pinned = [{**c, "tile_shape": [16, 512, 512]} for c in mixed]
    assert _validate_configs(paths, pinned) == w

    # And "auto" is fine when every config carries the same channel count.
    both = [{**mixed[0]}, {**mixed[1], "nuclei_channel": 3}]
    assert _validate_configs(paths, both) == w


def test_resolve_shared_tile_shape_pins_the_smallest_candidate(
    tmp_path, monkeypatch
):
    """The shared tile must be the tightest of every config's own budget.

    A larger tile than some config's own sizer output would ask that config
    for more memory than its settings were judged to need -- only the
    smallest candidate is safe for every config at once.
    """
    import run_multi

    class _FakeImage:
        shape = (10, 100, 100)
        dtype = "uint16"

    calls = []

    def _fake_load_ome_zarr(store, *, channel, level):
        calls.append((store, channel, level))
        return _FakeImage()

    # One tile per config, matched up by call order (channel 0 then 1).
    fake_tiles = [(8, 64, 64), (4, 32, 32)]

    def _fake_sizer_cellpose(shape, dtype, **kwargs):
        return fake_tiles[len(calls) - 1]

    monkeypatch.setattr(
        "patchworks.load_ome_zarr", _fake_load_ome_zarr, raising=False
    )
    monkeypatch.setattr(
        "patchworks.auto_tile_shape_cellpose",
        _fake_sizer_cellpose,
        raising=False,
    )

    seg_cfgs = [
        {
            "channel": 0,
            "level": 0,
            "method": "cellpose",
            "cellpose": {"do_3D": True, "gpu": True},
        },
        {
            "channel": 1,
            "level": 0,
            "nuclei_channel": 2,
            "method": "cellpose",
            "cellpose": {"do_3D": True, "gpu": True},
        },
    ]
    out = run_multi._resolve_shared_tile_shape(
        seg_cfgs, "/w/image.zarr", str(tmp_path)
    )
    assert out == tmp_path / ".multi_tile_shape.generated.yaml"
    written = yaml.safe_load(out.read_text())
    assert written == {"tile_shape": [4, 32, 32]}  # the smaller candidate
    assert len(calls) == 2


def test_merge_reshards_level_zero_when_asked():
    """`shard_labels` has to reshard level 0 *after* everything wrote to it.

    Ordering is the whole correctness argument: the merge and the volume
    filter both rewrite level 0 a chunk at a time, so a shard created before
    either of them would be read-modify-written by several writers and lose
    chunks. It also has to land before `register_labels`, so the pyramid is
    built from the level that will actually be on disk.
    """
    src = (_workflow_dir() / "scripts" / "merge.py").read_text()
    assert 'cfg.get("shard_labels", False)' in src
    assert "reshard_level(" in src

    reshard = src.index("reshard_level(label_group")
    assert src.index("merge_tile_labels(") < reshard
    assert src.index("filter_labels_by_size(") < reshard
    assert reshard < src.index("register_labels(\n")


def test_merge_shards_the_label_pyramid():
    """`shard:` has to reach the label pyramid, not just the conversion.

    Only the pyramid levels can take it: level 0 is written one chunk at a
    time by concurrent segment jobs (or the merge's own pool), and a shard
    must be written whole by a single writer -- see _write_pyramid's note.
    Levels 1..N go through one dask pass, so they can be sharded.
    """
    src = (_workflow_dir() / "scripts" / "merge.py").read_text()
    # The call's own arguments contain ")", so end it on the closing line.
    register = src.split("register_labels(")[1].split("\n)")[0]
    assert 'shard=cfg.get("shard", False)' in register


def test_ngff_version_reaches_convert_and_merge():
    """Both writers need it, or a store ends up half one version.

    convert makes the image; merge adds the label pyramid to it, in a
    separate job hours later. If only one of them read the key the store
    would mix zarr v2 arrays with v3 ones.
    """
    convert = (_workflow_dir() / "scripts" / "convert.py").read_text()
    merge = (_workflow_dir() / "scripts" / "merge.py").read_text()
    for name, src in (("convert.py", convert), ("merge.py", merge)):
        assert 'ngff_version=cfg.get("ngff_version", "auto")' in src, name


def test_ngff_version_must_agree_across_configs():
    """It decides the store's format, and convert runs once, from config #1."""
    from run_multi import _CONVERT_KEYS

    assert "ngff_version" in _CONVERT_KEYS


def test_store_marker_file_follows_the_zarr_format():
    """The rules watch a file whose *name* is the zarr format's.

    zarr v3 writes "zarr.json", v2 writes ".zgroup". NGFF 0.4 means a v2
    store, so a hardcoded zarr.json would leave `convert` waiting forever on
    a file that is never written.
    """
    smk = (_workflow_dir() / "rules" / "common.smk").read_text()
    assert "ZARR_ROOT_FILE" in smk
    assert '".zgroup"' in smk
    assert '"zarr.json"' in smk
    assert 'IMAGE_OK = f"{IMAGE}/{ZARR_ROOT_FILE}"' in smk
    # The occupancy map is a private zarr v3 array whatever ngff_version
    # says: under 0.4 a .zgroup marker for it was never written.
    assert 'OCCUPANCY_OK = f"{OCCUPANCY}/zarr.json"' in smk

    # run_multi asks Snakemake for the same markers: it asked for zarr.json
    # under 0.4 and every multi run writing 0.4 failed to convert.
    import run_multi

    assert run_multi.zarr_root_file({"ngff_version": "0.4"}) == ".zgroup"
    assert run_multi.zarr_root_file({}) == "zarr.json"
    assert run_multi.zarr_root_file({"ngff_version": "0.5"}) == "zarr.json"

    # Both scripts that turn the marker back into a store path must strip
    # whichever name is in use.
    for script in ("convert.py", "build_occupancy.py"):
        src = (_workflow_dir() / "scripts" / script).read_text()
        assert '.removesuffix("/zarr.json")' in src, script
        assert '.removesuffix("/.zgroup")' in src, script


class _NoRelateFlags:
    relate_partition = None
    relate_mem = None
    relate_cpus = None
    relate_time = None
    relate_qos = None


def test_relate_settings_fall_back_to_the_documented_defaults():
    from run_multi import RELATE_DEFAULTS, _relate_settings

    assert _relate_settings({}, _NoRelateFlags()) == RELATE_DEFAULTS


def test_relate_settings_read_the_multi_config():
    """`pixi run multi-slurm` is a fixed command, so the QOS must be config.

    A cluster whose default QOS caps the wall time below `time` otherwise has
    to abandon the shipped task for a hand-written command line.
    """
    from run_multi import _relate_settings

    out = _relate_settings(
        {"relate": {"qos": "1day", "time": 720}}, _NoRelateFlags()
    )
    assert out["qos"] == "1day"
    assert out["time"] == 720
    # untouched keys keep their defaults
    assert out["partition"] == "scicore"
    assert out["mem"] == "32G"


def test_relate_flag_overrides_the_config():
    from run_multi import _relate_settings

    class _Flags(_NoRelateFlags):
        relate_time = 999

    out = _relate_settings({"relate": {"qos": "1day", "time": 720}}, _Flags())
    assert out["time"] == 999
    assert out["qos"] == "1day"  # not overridden, so the config still wins


def test_relate_block_typos_are_rejected():
    """A silently ignored typo would run with the default the user replaced."""
    import pytest
    from run_multi import _relate_settings

    with pytest.raises(ValueError, match="unknown key"):
        _relate_settings({"relate": {"qs": "1day"}}, _NoRelateFlags())
    with pytest.raises(ValueError, match="must be a mapping"):
        _relate_settings({"relate": ["1day"]}, _NoRelateFlags())


def test_shipped_multi_config_relate_block_is_valid():
    """The template's commented example must match what the parser accepts."""
    import re

    import yaml
    from run_multi import RELATE_DEFAULTS

    text = (_workflow_dir() / "config" / "multi.yaml").read_text()
    block = re.search(r"^# relate:\n((?:^#   .*\n)+)", text, re.M)
    assert block, "multi.yaml no longer documents the relate: block"
    uncommented = "relate:\n" + re.sub(r"^# ", "", block.group(1), flags=re.M)
    parsed = yaml.safe_load(uncommented)["relate"]
    assert set(parsed) <= set(RELATE_DEFAULTS), set(parsed) - set(
        RELATE_DEFAULTS
    )


def test_template_placeholders_are_refused_with_a_useful_message(capsys):
    """Running the shipped templates unedited must say so, not crash.

    `pixi run multi-slurm` points at workflow/config/, so someone who has
    their own configs elsewhere gets the templates by default. That used to
    reach state-directory creation and die four pathlib frames deep on
    `PermissionError: '/path'`, naming neither the setting nor the file.
    """
    cfgs = [
        {
            "work_dir": "/path/to/results",
            "input": "/path/to/scan.ims",
            "tile_shape": [16, 1024, 1024],
            "level": 0,
        }
    ]
    with pytest.raises(SystemExit):
        _validate_configs([Path("config_nuclei.yaml")], cfgs)
    # Problems are printed and exited on, so the message is on stderr.
    message = capsys.readouterr().err
    assert "placeholder" in message
    assert "config_nuclei.yaml" in message
    assert "work_dir" in message and "input" in message


def test_uncreatable_work_dir_is_refused(monkeypatch, tmp_path, capsys):
    """A work_dir you cannot write fails before the first job is launched."""
    from run_multi import os as run_multi_os

    base = tmp_path / "readonly"
    base.mkdir()
    cfgs = [
        {
            "work_dir": str(base / "results"),
            "input": str(tmp_path / "scan.ims"),
            "tile_shape": [16, 1024, 1024],
            "level": 0,
        }
    ]
    # Writable: accepted, and the work_dir need not exist yet.
    assert _validate_configs([Path("c.yaml")], cfgs) == str(base / "results")

    # Not writable: refused. Patched rather than chmod'ed because the test
    # suite may run as root, for whom every directory is writable.
    real_access = run_multi_os.access
    monkeypatch.setattr(
        run_multi_os,
        "access",
        lambda p, mode: False if str(p) == str(base) else real_access(p, mode),
    )
    with pytest.raises(SystemExit):
        _validate_configs([Path("c.yaml")], cfgs)
    assert "not creatable" in capsys.readouterr().err


def test_iso_export_command_preserves_a_zarr_tree(monkeypatch):
    """The .iso has to be readable on Windows, not just on Linux.

    A zarr store nests deeper than ISO-9660's 8 levels, so without `-D` the
    tree is relocated into RR_MOVED -- which Rock Ridge readers undo
    transparently and Windows does not, leaving Explorer a scrambled tree
    that still *looks* like it copied fine. Joliet carries the long names
    Windows reads; level 3 stores a >4 GB shard as multiple extents.
    """
    import importlib.util
    from pathlib import Path as _Path

    spec = importlib.util.spec_from_file_location(
        "export_iso", _workflow_dir() / "scripts" / "export_iso.py"
    )
    export_iso = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export_iso)
    # The flags are what is under test, not whether this machine happens to
    # have an ISO builder installed.
    monkeypatch.setattr(
        export_iso,
        "find_builder",
        lambda: ("/usr/bin/xorriso", ["-as", "mkisofs"]),
    )

    cmd = export_iso.build_command(
        _Path("/data/image.zarr"), _Path("/out/image.zarr.iso")
    )
    for flag in ("-R", "-J", "-joliet-long", "-D", "-graft-points"):
        assert flag in cmd, flag
    assert cmd[cmd.index("-iso-level") + 1] == "3"
    # The store must land as its own directory inside the image, not as a
    # bare 0/ 1/ labels/ at the root.
    assert cmd[-1] == "/image.zarr=/data/image.zarr"


def test_iso_volume_id_is_always_acceptable():
    """xorriso rejects a volid over 32 chars or with odd characters."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "export_iso", _workflow_dir() / "scripts" / "export_iso.py"
    )
    export_iso = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export_iso)

    for name in ("image", "Elena out-multi", "x" * 80, "...", ""):
        volid = export_iso.volume_id(name)
        assert 1 <= len(volid) <= 32, (name, volid)
        assert all(c.isalnum() or c == "_" for c in volid), (name, volid)


def test_merge_reports_the_label_chunk_cost():
    """The file count must be visible in the log, not discovered by `ls`.

    Label chunks are inherited from tile_shape, so an auto-sized tile that
    is not a multiple of the cap (729 against 1024) multiplies the file
    count ~2.25x with nothing in the config hinting at it.
    """
    src = (_workflow_dir() / "scripts" / "merge.py").read_text()
    assert "LABEL_CHUNK_CAP = (16, 1024, 1024)" in src
    assert "-> {n_chunks:,} chunks" in src
    # The note has to name the remedy, and only fire when it is not taken.
    # Both keys, not just shard_labels: each pyramid level holds about as
    # many chunks as level 0, so level 0 is under a third of the group.
    assert 'if (not shard_labels or not cfg.get("shard"))' in src
    assert "`shard` covers levels 1..N" in src
    # And echo the values it read: a key set in the wrong config file is
    # otherwise indistinguishable from the feature not working.
    assert "[patchworks] sharding: shard=" in src
    assert "shard_labels={shard_labels!r}" in src


def test_reshard_task_is_wired_up():
    """The retrofit script has to ship, not live in a chat message.

    A store written before `shard_labels` was turned on needs repacking, and
    telling someone to save a file by hand loses it -- `git pull` should
    deliver it like every other script here.
    """
    wf = _workflow_dir()
    assert (wf / "scripts" / "reshard_store.py").is_file()
    pixi = (wf / "pixi.toml").read_text()
    assert 'reshard = "python scripts/reshard_store.py"' in pixi
    # The QOS trap bit us on the merge; the task's own comment must warn.
    assert "--qos=1day" in pixi


def test_reshard_script_skips_already_sharded_arrays():
    """Re-running it must be a no-op, not a second full rewrite."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "reshard_store", _workflow_dir() / "scripts" / "reshard_store.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    src = (_workflow_dir() / "scripts" / "reshard_store.py").read_text()
    assert "already sharded" in src
    assert "--labels-only" in src
    assert "--dry-run" in src
    # It must never silently eat a merge-state attr.
    assert "patchworks_merge_state" in src


def test_iso_tool_is_not_a_conda_dependency():
    """Declaring xorriso broke `pixi install` for every environment.

    conda-forge carries no ISO-building C tool, so `xorriso = "*"` is
    unsolvable and takes the whole environment down with it -- a worse
    failure than the missing tool, because it blocks segmentation too.
    """
    import tomllib

    pixi = tomllib.loads((_workflow_dir() / "pixi.toml").read_text())
    for tool in ("xorriso", "genisoimage", "mkisofs", "libisoburn"):
        assert tool not in pixi["dependencies"], tool
    assert pixi["tasks"]["iso"] == "python scripts/export_iso.py"


def test_iso_script_accepts_any_mkisofs_compatible_builder():
    """xorriso, genisoimage and mkisofs take the same options.

    Only xorriso needs `-as mkisofs` to emulate them, so whichever the
    cluster happens to provide should work.
    """
    import importlib.util
    from pathlib import Path as _Path

    spec = importlib.util.spec_from_file_location(
        "export_iso", _workflow_dir() / "scripts" / "export_iso.py"
    )
    export_iso = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export_iso)

    assert [n for n, _ in export_iso._BUILDERS] == [
        "xorriso",
        "genisoimage",
        "mkisofs",
    ]

    export_iso.shutil.which = lambda n: (
        f"/usr/bin/{n}" if n == "genisoimage" else None
    )
    cmd = export_iso.build_command(
        _Path("/data/image.zarr"), _Path("/out/i.iso")
    )
    assert cmd[0].endswith("genisoimage")
    assert "-as" not in cmd  # only xorriso needs the emulation prefix
    for flag in ("-R", "-J", "-D", "-graft-points"):
        assert flag in cmd, flag

    export_iso.shutil.which = lambda n: None
    with pytest.raises(SystemExit) as excinfo:
        export_iso.find_builder()
    message = str(excinfo.value)
    assert "conda-forge does not package any of them" in message
    assert "pure-Python ISO builder" in message


def test_iso_script_says_to_submit_it():
    """A login node kills a whole-store pass with no message at all."""
    src = (_workflow_dir() / "scripts" / "export_iso.py").read_text()
    assert "do not run it on a login node" in src
    assert "sbatch" in src and "--qos=1day" in src


def test_dog_extra_has_its_native_library():
    """patchworks[dog] pulls pycudadecon, which binds to the cudadecon lib.

    Without it the deconvolution step in config_cilia.yaml imports fine and
    fails at first use, on a GPU node, hours into a run.
    """
    import tomllib

    pixi = tomllib.loads((_workflow_dir() / "pixi.toml").read_text())
    assert "cudadecon" in pixi["dependencies"]
    # cupy is deliberately absent: cupy-cuda12x/13x is chosen per cluster.
    assert not any(k.startswith("cupy") for k in pixi["dependencies"])


def test_zip_export_needs_no_external_tool(tmp_path):
    """Where no ISO builder exists, the zip format must still work.

    scicore has none of xorriso/genisoimage/mkisofs and conda-forge cannot
    supply one, so a store there can only be bundled this way.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "export_iso", _workflow_dir() / "scripts" / "export_iso.py"
    )
    export_iso = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export_iso)

    store = tmp_path / "image.zarr"
    (store / "labels" / "cells" / "0" / "c").mkdir(parents=True)
    (store / "zarr.json").write_text("{}")
    (store / "labels" / "cells" / "0" / "c" / "0").write_bytes(b"chunk")

    output = tmp_path / "image.zarr.zip"
    # No builder available must not matter for this path.
    export_iso.shutil.which = lambda n: None
    export_iso.write_zip(store, output, progress=False)

    import zipfile

    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
    # Paths are relative to the store's parent, so it unpacks to a usable
    # store rather than a bare 0/ labels/.
    assert "image.zarr/zarr.json" in names
    assert "image.zarr/labels/cells/0/c/0" in names
    # Stored, not deflated: the chunks are already compressed.
    with zipfile.ZipFile(output) as archive:
        assert all(
            i.compress_type == zipfile.ZIP_STORED for i in archive.infolist()
        )


def test_zip_task_is_wired_up():
    import tomllib

    pixi = tomllib.loads((_workflow_dir() / "pixi.toml").read_text())
    assert pixi["tasks"]["zip"] == ("python scripts/export_iso.py --format zip")


def test_cupy_is_available_without_editing_pixi_toml():
    """A cluster needing cupy must not have to patch this file.

    cupy's wheel is CUDA-version-specific, so it cannot be a plain
    dependency -- but carrying it as a local commit meant a rebase conflict
    on every release that touched pixi.toml. Ship both variants as opt-in
    environments instead.
    """
    import tomllib

    pixi = tomllib.loads((_workflow_dir() / "pixi.toml").read_text())

    assert pixi["feature"]["cuda12"]["pypi-dependencies"] == {
        "cupy-cuda12x": "*"
    }
    assert pixi["feature"]["cuda13"]["pypi-dependencies"] == {
        "cupy-cuda13x": "*"
    }
    for env in ("cuda12", "cuda13"):
        assert env in pixi["environments"], env
    # The default env must stay free of cupy: it is GPU- and
    # CUDA-version-specific, and most steps do not need it.
    assert not any(k.startswith("cupy") for k in pixi["dependencies"])
    assert not any(
        k.startswith("cupy") for k in pixi.get("pypi-dependencies", {})
    )


def test_gpu_environments_combine_with_the_cellpose_pins():
    """A cluster can need both a pinned Cellpose and cupy."""
    import tomllib

    pixi = tomllib.loads((_workflow_dir() / "pixi.toml").read_text())
    for env, feats in (
        ("cellpose4-cuda12", {"cp4", "cuda12"}),
        ("cellpose4-cuda13", {"cp4", "cuda13"}),
    ):
        assert set(pixi["environments"][env]["features"]) == feats, env


class _NoBundleFlags:
    bundle_format = None
    bundle_output = None
    bundle_partition = None
    bundle_mem = None
    bundle_cpus = None
    bundle_time = None
    bundle_qos = None


def test_bundling_is_off_unless_asked():
    """It is a full read of everything the run produced."""
    from run_multi import _bundle_settings

    assert _bundle_settings({}, _NoBundleFlags())["format"] is None


def test_bundle_settings_read_the_multi_config():
    """`pixi run multi-slurm` is fixed, so this has to work from config."""
    from run_multi import _bundle_settings

    out = _bundle_settings(
        {"bundle": {"format": "zip", "qos": "1day"}}, _NoBundleFlags()
    )
    assert out["format"] == "zip"
    assert out["qos"] == "1day"
    assert out["mem"] == "8G"  # untouched keys keep their defaults


def test_bundle_rejects_a_bad_format_or_typo():
    """Silently not bundling after a run of hours is the wrong failure."""
    import pytest
    from run_multi import _bundle_settings

    with pytest.raises(ValueError, match='must be "zip" or "iso"'):
        _bundle_settings({"bundle": {"format": "tar"}}, _NoBundleFlags())
    with pytest.raises(ValueError, match="unknown key"):
        _bundle_settings({"bundle": {"frmat": "zip"}}, _NoBundleFlags())
    with pytest.raises(ValueError, match="must be a mapping"):
        _bundle_settings({"bundle": ["zip"]}, _NoBundleFlags())


def test_bundle_is_submitted_under_a_profile():
    """It reads every file the run produced -- not on the login node."""
    from pathlib import Path as _Path

    from run_multi import _bundle_cmd

    bundle = {
        "format": "zip",
        "output": None,
        "partition": "scicore",
        "mem": "8G",
        "cpus": 2,
        "time": 720,
        "qos": "1day",
    }
    cmd = _bundle_cmd("/w/image.zarr", _Path("/workflow"), bundle, True)
    assert cmd[0] == "srun"
    assert "--qos" in cmd and "1day" in cmd
    assert cmd[cmd.index("--time") + 1] == "720"
    assert cmd[-1] == "--overwrite"
    assert "--format" in cmd and "zip" in cmd
    assert str(_Path("/workflow/scripts/export_iso.py")) in cmd

    # Without a profile it runs in-process, no srun.
    local = _bundle_cmd("/w/image.zarr", _Path("/workflow"), bundle, False)
    assert "srun" not in local


def test_bundle_runs_after_the_relations_not_before():
    """A bundle of a run whose relations failed would look complete."""
    src = (_workflow_dir() / "scripts" / "run_multi.py").read_text()
    # The failure branch exits without bundling...
    failed = src.index("relation(s) failed")
    exit_one = src.index("sys.exit(1)", failed)
    bundle_after = src.index("_finish()", exit_one)
    assert exit_one < bundle_after
    # ...as does a run with relations skipped; the success path bundles.
    finish = src[src.index("def _finish():") :].split("\n\n")[0]
    assert finish.index("if skipped:") < finish.index("_run_bundle(")
    assert src.count("_finish()") == 4


def test_bundle_store_path_is_made_absolute():
    """The command runs with cwd=workflow_dir, not the driver's cwd.

    A relative work_dir would otherwise be resolved against the workflow
    directory, and the step would fail with "not a directory" after the
    whole run had already succeeded.
    """
    from pathlib import Path as _Path

    from run_multi import _bundle_cmd

    bundle = {
        "format": "zip",
        "output": "out/bundle.zip",
        "partition": "scicore",
        "mem": "8G",
        "cpus": 2,
        "time": 720,
        "qos": None,
    }
    cmd = _bundle_cmd("results/image.zarr", _Path("/workflow"), bundle, False)
    store = cmd[cmd.index("--store") + 1]
    output = cmd[cmd.index("--output") + 1]
    assert _Path(store).is_absolute(), store
    assert _Path(output).is_absolute(), output
    assert store.endswith("results/image.zarr")


def test_manifest_declares_exactly_one_platform():
    """A platform declared anywhere makes *every* environment solve for it.

    Not the intersection per environment, as the per-feature `platforms`
    key suggests: putting win-64/osx-arm64 on the viewer feature alone made
    `cellpose4` fail on osx-arm64, where cudadecon has no build. So the GPU
    stack pins the whole manifest to linux-64, and viewing a result
    elsewhere is a plain venv, not a pixi environment.
    """
    import tomllib

    pixi = tomllib.loads((_workflow_dir() / "pixi.toml").read_text())
    assert pixi["workspace"]["platforms"] == ["linux-64"]
    for name, feature in pixi["feature"].items():
        assert "platforms" not in feature, name


def test_viewer_environment_is_lean():
    """It carries what a viewer needs, not the GPU segmentation stack."""
    import tomllib

    pixi = tomllib.loads((_workflow_dir() / "pixi.toml").read_text())
    viewer = pixi["feature"]["viewer"]
    assert pixi["environments"]["viewer"]["no-default-feature"] is True
    # ...and therefore has to carry its own python.
    assert "python" in viewer["dependencies"]
    assert "napari" in viewer["tasks"]


def test_viewer_requires_a_patchworks_that_reads_bundles():
    """An older patchworks fails to open a .zip, with an error about bioio
    or groups, not about its age: 2.8.0 still sent a bundle to bioio
    ("reading .zip requires bioio"). Opening one needs >= 3.0.0."""
    import tomllib

    pixi = tomllib.loads((_workflow_dir() / "pixi.toml").read_text())
    spec = pixi["feature"]["viewer"]["pypi-dependencies"]["patchworks"]
    assert _patchworks_is_recent(spec, (3, 0, 0)), spec
    assert "napari" in spec["extras"]


def test_viewer_workspace_solves_everywhere():
    """Viewing a result must work on a laptop, and only a second manifest can.

    Platforms are not scoped per feature: one declared anywhere goes into
    the set pixi solves every environment in that manifest for, so putting
    win-64 on the workflow manifest breaks the GPU environments, which need
    linux-64-only cudadecon. A separate workspace has none of them.
    """
    import tomllib

    manifest = _workflow_dir() / "viewer" / "pixi.toml"
    assert manifest.is_file()
    pixi = tomllib.loads(manifest.read_text())

    for platform in ("linux-64", "win-64", "osx-arm64"):
        assert platform in pixi["workspace"]["platforms"], platform
    # Nothing linux-only may leak in, or it stops solving elsewhere.
    assert "cudadecon" not in pixi["dependencies"]
    assert set(pixi["pypi-dependencies"]) == {"patchworks"}
    spec = pixi["pypi-dependencies"]["patchworks"]
    assert spec["extras"] == ["napari"]
    floor = tuple(int(x) for x in spec["version"].removeprefix(">=").split("."))
    assert spec["version"].startswith(">=") and floor >= (3, 0, 0), spec[
        "version"
    ]

    # It reuses the workflow's viewer script rather than duplicating it.
    task = pixi["tasks"]["napari"]
    assert task == "python ../scripts/view.py"
    assert (manifest.parent / "../scripts/view.py").resolve().is_file()


def test_review_block_is_checked_before_anything_runs(capsys):
    from run_multi import _review_rules

    names = ["nuclei_labels", "cyto_labels"]
    ok = {"review": {"expect": {"cyto_labels": {"nuclei_labels": [1, 2]}}}}
    assert _review_rules(ok, names) == ok["review"]
    assert _review_rules({}, names) == {}
    for bad in (
        {"review": {"expect": {"cells": {"nuclei_labels": 1}}}},
        {"review": {"expect": {"cyto_labels": {"nuclei_labels": [2, 1]}}}},
        {"review": {"expct": {}}},
    ):
        with pytest.raises(SystemExit):
            _review_rules(bad, names)
    assert "not a segmentation's label_name" in capsys.readouterr().err


def test_relate_writes_workbooks_from_the_reviewed_tables(tmp_path):
    """The workbook reflects review decisions: re-running relate after a
    review gives the corrected numbers, not the raw ones."""
    pytest.importorskip("pandas")
    from relate import run_relations
    from test_tables import make_scene

    from patchworks import Review

    store = make_scene(tmp_path / "image.zarr")
    Review(store).decide("cilia_labels", 4, "wrong")
    rel = [{"a": "cilia_labels", "b": "cyto_labels", "output": "c.xlsx"}]
    run_relations(str(tmp_path), store, rel)
    wb = openpyxl.load_workbook(tmp_path / "c.xlsx")
    ids = [
        r[0] for r in wb["cilia_labels"].iter_rows(min_row=2, values_only=True)
    ]
    assert ids == [1, 2, 3, 5]  # the rejected cilium is gone
    counts = {
        r[0]: r[1]
        for r in wb["cyto_labels"].iter_rows(min_row=2, values_only=True)
    }
    assert counts == {1: 1, 2: 2, 3: 1, 4: 0}


def test_relate_rewrites_a_workbook_after_review_without_rereading(
    tmp_path, monkeypatch
):
    """Decisions made after the workbook make it stale -- and only the
    workbook is rewritten: the overlaps are already in the tables."""
    pytest.importorskip("pandas")
    import relate
    from test_tables import make_scene

    from patchworks import Review

    store = make_scene(tmp_path / "image.zarr")
    for name in ("cilia_labels", "cyto_labels"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "labels.done").touch()
    rel = [{"a": "cilia_labels", "b": "cyto_labels", "output": "c.xlsx"}]
    relate.run_relations(str(tmp_path), store, rel)
    first = openpyxl.load_workbook(tmp_path / "c.xlsx")["cilia_labels"].max_row

    time.sleep(1.1)  # decision strictly newer than the workbook
    Review(store).decide("cilia_labels", 4, "wrong")
    monkeypatch.setattr(
        "patchworks.label_relations",
        lambda *a, **k: pytest.fail("labels re-read for a review change"),
    )
    relate.run_relations(str(tmp_path), store, rel)
    wb = openpyxl.load_workbook(tmp_path / "c.xlsx")
    assert wb["cilia_labels"].max_row == first - 1
    os.utime(tmp_path / "c.xlsx")
    relate.run_relations(str(tmp_path), store, rel)  # now simply skipped


def test_position_rules_and_distances_are_checked(capsys):
    from run_multi import _review_rules

    names = ["nuclei_labels", "cyto_labels", "cilia_labels"]
    rel = [
        {"a": "nuclei_labels", "b": "cyto_labels"},
        {"a": "cilia_labels", "b": "cyto_labels", "max_distance_um": 1.5},
    ]
    good = {
        "relations": rel,
        "review": {
            "position": {
                "cilia_labels": {
                    "parent": "cyto_labels",
                    "apical": "nuclei_labels",
                }
            }
        },
    }
    assert _review_rules(good, names) == good["review"]
    for review, relations in (
        (
            {
                "position": {
                    "cilia_labels": {"parent": "cyto_labels", "apical": "up"}
                }
            },
            rel,
        ),
        (
            {
                "position": {
                    "cilia_labels": {"parent": "nuclei_labels", "apical": "+z"}
                }
            },
            rel,
        ),
        (
            {
                "position": {
                    "cilia_labels": {
                        "parent": "cyto_labels",
                        "apical": "+z",
                        "x": 1,
                    }
                }
            },
            rel,
        ),
        (
            {},
            [{"a": "cilia_labels", "b": "cyto_labels", "max_distance_um": -1}],
        ),
    ):
        with pytest.raises(SystemExit):
            _review_rules({"relations": relations, "review": review}, names)
    err = capsys.readouterr().err
    assert "needs the relation cilia_labels -> nuclei_labels" in err
    assert "max_distance_um must be a positive number" in err


def test_pixi_manifests_require_the_patchworks_the_scripts_use():
    """pixi keeps a locked patchworks for as long as it satisfies the
    manifest, so a floor below what the repository's scripts use leaves old
    environments broken: 2.8.0 in the viewer failed to open a .zip bundle,
    and the workflow scripts write object tables (3.1.0)."""
    import tomllib

    wf = _workflow_dir()
    main = tomllib.loads((wf / "pixi.toml").read_text())
    viewer = tomllib.loads((wf / "viewer" / "pixi.toml").read_text())
    specs = [
        main["pypi-dependencies"]["patchworks"],
        main["feature"]["viewer"]["pypi-dependencies"]["patchworks"],
        viewer["pypi-dependencies"]["patchworks"],
    ]
    for spec in specs:
        assert _patchworks_is_recent(spec, (3, 1, 0)), spec
    # The workflow's own environments run the scripts: same revision.
    assert "git" in main["pypi-dependencies"]["patchworks"]
    assert "review" in viewer["tasks"]


def _patchworks_is_recent(spec, floor):
    """The manifest takes patchworks from GitHub main, or a release >= floor."""
    if "git" in spec:
        return (
            spec["git"].rstrip("/").removesuffix(".git")
            == "https://github.com/imcf/patchworks"
            and spec.get("branch") == "main"
        )
    pin = spec["version"]
    if not pin.startswith(">="):
        return False
    return tuple(int(x) for x in pin[2:].split(".")) >= floor


class _FakeProc:
    def __init__(self, log, name, rc, steps=2):
        self.log, self.name, self.rc, self.steps = log, name, rc, steps
        self.returncode = None
        log.append(("start", name))

    def poll(self):
        self.steps -= 1
        if self.steps <= 0 and self.returncode is None:
            self.returncode = self.rc
            self.log.append(("end", self.name))
        return self.returncode


def test_seeded_config_waits_for_the_labels_it_grows_from():
    from run_multi import run_after, seed_dependencies

    cfgs = [
        {"label_name": "cyto_labels", "seed_labels": "nuclei_labels"},
        {"label_name": "nuclei_labels"},
        {"label_name": "cilia_labels"},
        {"label_name": "old_cells", "seed_labels": "from_an_earlier_run"},
    ]
    deps = seed_dependencies(cfgs)
    assert deps == {0: 1}
    log = []
    names = ["cyto", "nuclei", "cilia", "old"]
    status = run_after(
        names, deps, lambda i: _FakeProc(log, names[i], 0), poll_seconds=0
    )
    assert status == [(n, "ok") for n in names]
    # Everything independent starts together; cyto only after nuclei ended
    assert log[:3] == [
        ("start", "nuclei"),
        ("start", "cilia"),
        ("start", "old"),
    ]
    assert log.index(("start", "cyto")) > log.index(("end", "nuclei"))


def test_a_failed_seed_config_skips_its_dependent_only():
    from run_multi import run_after

    log = []
    names = ["cyto", "nuclei", "cilia"]
    rcs = {"nuclei": 1}
    status = run_after(
        names,
        {0: 1},
        lambda i: _FakeProc(log, names[i], rcs.get(names[i], 0)),
        poll_seconds=0,
    )
    assert status == [
        ("cyto", "skipped (nuclei failed)"),
        ("nuclei", "FAILED"),
        ("cilia", "ok"),
    ]
    assert ("start", "cyto") not in log


def test_configs_seeding_each_other_are_refused(tmp_path, capsys):
    base = {
        "work_dir": str(tmp_path / "w"),
        "tile_shape": [16, 64, 64],
        "level": 0,
    }
    cfgs = [
        {**base, "label_name": "a", "seed_labels": "b"},
        {**base, "label_name": "b", "seed_labels": "a"},
    ]
    with pytest.raises(SystemExit):
        _validate_configs([Path("a.yaml"), Path("b.yaml")], cfgs)
    assert "from each other" in capsys.readouterr().err
    del cfgs[1]["seed_labels"]
    assert _validate_configs([Path("a.yaml"), Path("b.yaml")], cfgs)


def test_main_runs_with_an_explicit_tile_shape_seeds_first(
    tmp_path, monkeypatch
):
    """main() past the conversion, with the shipped style of config: an
    explicit tile_shape list (which used to crash it, unhashable in a set)
    and a config seeded by another, listed first but started second."""
    import run_multi

    common = tmp_path / "common.yaml"
    common.write_text(
        yaml.safe_dump(
            {
                "input": str(tmp_path / "scan.zarr"),
                "work_dir": str(tmp_path / "results"),
                "tile_shape": [16, 512, 512],
                "level": 0,
            }
        )
    )
    cells = tmp_path / "cells.yaml"
    cells.write_text(
        yaml.safe_dump({"label_name": "cyto_labels", "seed_labels": "nuclei"})
    )
    nuclei = tmp_path / "nuclei.yaml"
    nuclei.write_text(yaml.safe_dump({"label_name": "nuclei"}))
    multi = tmp_path / "multi.yaml"
    multi.write_text(
        yaml.safe_dump(
            {"common": str(common), "segmentations": [str(cells), str(nuclei)]}
        )
    )
    log = []
    monkeypatch.setattr(run_multi, "_run", lambda cmd, wd: 0)  # convert

    def popen(cmd, cwd=None):
        name = Path(cmd[cmd.index("--configfile") + 2]).stem
        return _FakeProc(log, name, 0)

    monkeypatch.setattr(run_multi.subprocess, "Popen", popen)
    monkeypatch.setattr(
        sys, "argv", ["run_multi.py", "--config", str(multi), "--cores", "1"]
    )
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(SystemExit) as done:  # no relations: exits after
        run_multi.main()
    assert done.value.code == 0
    assert log.index(("start", "cells")) > log.index(("end", "nuclei"))


def test_config_option_resolves_from_where_pixi_runs(tmp_path, monkeypatch):
    """`pixi run multi-slurm --config my_multi.yaml` from a project folder:
    pixi runs the task in workflow/ but says where it was called from
    (INIT_CWD); the configs the multi lists are found next to it."""
    import run_multi

    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    common = {
        "input": str(tmp_path / "scan.zarr"),
        "work_dir": str(tmp_path / "results"),
        "tile_shape": [16, 512, 512],
        "level": 0,
    }
    (project / "configs" / "common.yaml").write_text(yaml.safe_dump(common))
    (project / "configs" / "a.yaml").write_text(
        yaml.safe_dump({"label_name": "a"})
    )
    (project / "my_multi.yaml").write_text(
        yaml.safe_dump(
            {
                "common": "configs/common.yaml",
                "segmentations": ["configs/a.yaml"],
            }
        )
    )
    started = []
    monkeypatch.setattr(run_multi, "_run", lambda cmd, wd: 0)
    monkeypatch.setattr(
        run_multi.subprocess,
        "Popen",
        lambda cmd, cwd=None: started.append(cmd) or _FakeProc([], "a", 0, 1),
    )
    monkeypatch.setenv("INIT_CWD", str(project))
    monkeypatch.chdir(_workflow_dir())  # where pixi runs every task
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_multi.py", "--config", "my_multi.yaml", "--cores", "1"],
    )
    with pytest.raises(SystemExit) as done:
        run_multi.main()
    assert done.value.code == 0
    cmd = started[0]
    i = cmd.index("--configfile")
    assert cmd[i + 1 : i + 3] == [
        str(project / "configs" / "common.yaml"),
        str(project / "configs" / "a.yaml"),
    ]


def test_config_defaults_to_the_shipped_multi_and_names_a_missing_one(
    tmp_path, monkeypatch, capsys
):
    import run_multi

    wf = _workflow_dir()
    monkeypatch.delenv("INIT_CWD", raising=False)
    monkeypatch.chdir(tmp_path)
    assert run_multi._resolve(
        "config/multi.yaml", *run_multi._config_bases(wf)
    ) == (wf / "config" / "multi.yaml")
    monkeypatch.setattr(sys, "argv", ["run_multi.py", "--config", "nope.yaml"])
    with pytest.raises(SystemExit) as err:
        run_multi.main()
    assert "nope.yaml not found" in str(err.value.code)
    # The pixi tasks leave --config to the caller, defaulting to multi.yaml
    tasks = (wf / "pixi.toml").read_text()
    assert (
        'multi-slurm = "python scripts/run_multi.py --profile profile/slurm"'
        in tasks
    )


def test_plantseg_and_careamics_environment_cannot_break_the_others():
    """pixi solves every environment, even to run a task in the default one,
    so one unsolvable optional environment stops everybody. PlantSeg's conda
    stack and CAREamics (via microssim) disagree on scipy/tqdm: shared with
    the plantseg solve group, `pixi run multi-slurm` failed on the cluster
    with "failed to solve the pypi requirements of environment 'plantseg'"."""
    import tomllib

    pixi = tomllib.loads((_workflow_dir() / "pixi.toml").read_text())
    envs = pixi["environments"]
    both = envs["plantseg-careamics"]
    assert both["solve-group"] != envs["plantseg"]["solve-group"]
    assert "plantseg-careamics" in both["features"]
    held = pixi["feature"]["plantseg-careamics"]["dependencies"]
    assert held["scipy"] == "<=1.17.1" and held["tqdm"] == "<=4.67.3"
    # The SLURM profile's slurm-jobname-prefix needs executor plugin >=2.2;
    # PlantSeg's conda pins pushed the solver down to 1.4.0, and every
    # snakemake call failed on "unrecognized arguments". pandas<3 lets
    # 2.2-2.6 fit next to conda's numpy 2.3.
    plantseg = pixi["feature"]["plantseg"]
    assert (
        plantseg["pypi-dependencies"]["snakemake-executor-plugin-slurm"]
        == ">=2.2"
    )
    assert plantseg["dependencies"]["pandas"] == ">=2.2.3,<3"
    # and the plantseg U-Net gets a CUDA build, not conda-forge's CPU one
    assert (
        pixi["feature"]["plantseg"]["dependencies"]["pytorch"]["build"]
        == "cuda*"
    )


def test_a_missing_package_stops_the_run_before_converting(
    tmp_path, monkeypatch, capsys
):
    import _pw
    import run_multi

    common = tmp_path / "common.yaml"
    common.write_text(
        yaml.safe_dump(
            {
                "input": str(tmp_path / "scan.zarr"),
                "work_dir": str(tmp_path / "results"),
                "tile_shape": [16, 512, 512],
                "level": 0,
            }
        )
    )
    cells = tmp_path / "cells.yaml"
    cells.write_text(
        yaml.safe_dump(
            {
                "label_name": "cyto_labels_plantseg",
                "method": "custom",
                "custom": {"module": "patchworks.plugins.plantseg"},
            }
        )
    )
    multi = tmp_path / "multi.yaml"
    multi.write_text(
        yaml.safe_dump({"common": str(common), "segmentations": [str(cells)]})
    )
    monkeypatch.setattr(_pw, "has_module", lambda name: name != "plantseg")
    ran = []
    monkeypatch.setattr(run_multi, "_run", lambda *a: ran.append(a) or 0)
    monkeypatch.setattr(sys, "argv", ["run_multi.py", "--config", str(multi)])
    with pytest.raises(SystemExit) as done:
        run_multi.main()
    assert done.value.code == 1 and not ran  # nothing converted
    assert "pixi run -e plantseg" in capsys.readouterr().err


def test_seed_plan_warns_when_the_seeds_come_from_no_config(tmp_path):
    """A seed_labels name that no config makes (nuclei_labels for
    nuclei_labels_cpsam) used to start the cells at once, from whatever old
    labels the store held under that name -- silently."""
    from run_multi import seed_plan

    store = tmp_path / "image.zarr"
    (store / "labels" / "nuclei_labels").mkdir(parents=True)
    cfgs = [
        {"label_name": "nuclei_labels_cpsam"},
        {"label_name": "cyto_labels_plantseg", "seed_labels": "nuclei_labels"},
        {"label_name": "cells_b", "seed_labels": "nuclei_labels_cpsam"},
        {"label_name": "cells_c", "seed_labels": "nope"},
    ]
    lines = seed_plan(cfgs, str(store))
    assert lines[0].startswith("WARNING: cyto_labels_plantseg grows from")
    assert "already in the store, NOT re-made" in lines[0]
    assert "nuclei_labels_cpsam" in lines[0]
    assert lines[1] == (
        "cells_b grows from nuclei_labels_cpsam: starts once "
        "nuclei_labels_cpsam is done"
    )
    assert "not in the store either" in lines[2]


def test_a_config_failing_on_a_stale_lock_says_how_to_unlock(
    tmp_path, monkeypatch, capsys
):
    """A run interrupted mid-way leaves Snakemake's lock behind; the next
    run of that config fails at once with a LockException, which run_multi
    only reported as FAILED."""
    import run_multi

    work = tmp_path / "results"
    locks = work / "cells" / ".snakemake" / ".snakemake" / "locks"
    locks.mkdir(parents=True)
    (locks / "0.input.lock").write_text("x")
    assert run_multi.held_locks(work / "cells" / ".snakemake")
    assert run_multi.held_locks(work / "other" / ".snakemake") == []

    common = tmp_path / "common.yaml"
    common.write_text(
        yaml.safe_dump(
            {
                "input": str(tmp_path / "scan.zarr"),
                "work_dir": str(work),
                "tile_shape": [16, 512, 512],
                "level": 0,
            }
        )
    )
    cells = tmp_path / "cells.yaml"
    cells.write_text(yaml.safe_dump({"label_name": "cells"}))
    multi = tmp_path / "multi.yaml"
    multi.write_text(
        yaml.safe_dump({"common": str(common), "segmentations": [str(cells)]})
    )
    monkeypatch.setattr(run_multi, "_run", lambda cmd, wd: 0)
    monkeypatch.setattr(
        run_multi.subprocess,
        "Popen",
        lambda cmd, cwd=None: _FakeProc([], "cells", 1, 1),
    )
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setattr(sys, "argv", ["run_multi.py", "--config", str(multi)])
    with pytest.raises(SystemExit):
        run_multi.main()
    err = capsys.readouterr().err
    assert "cells.yaml: its Snakemake directory is locked" in err
    assert f"--config {multi} --unlock" in err


def test_one_run_multi_per_work_dir(tmp_path, monkeypatch):
    """A run_multi left running from an earlier attempt drove the same
    work_dir as the new one: two Snakemakes per config, a lock, and two
    merges heading for one label group."""
    import json
    import socket

    import run_multi

    work = tmp_path / "results"
    marker = run_multi.claim_driver(work, Path("multi.yaml"))
    mine = json.loads(marker.read_text())
    assert mine["pid"] == os.getpid() and mine["host"] == socket.gethostname()

    # Another live driver on this machine: refused, saying how to stop it
    marker.write_text(json.dumps({**mine, "pid": 999999, "config": "old.yaml"}))
    monkeypatch.setattr(run_multi, "_driver_alive", lambda pid: True)
    with pytest.raises(SystemExit) as err:
        run_multi.claim_driver(work, Path("multi.yaml"))
    assert "already driving" in str(err.value) and "kill 999999" in str(
        err.value
    )

    # Its marker left behind by a driver that died: taken over
    monkeypatch.setattr(run_multi, "_driver_alive", lambda pid: False)
    run_multi.claim_driver(work, Path("multi.yaml"))
    assert json.loads(marker.read_text())["pid"] == os.getpid()

    # A driver on another login node cannot be checked from here: refused
    marker.write_text(json.dumps({**mine, "pid": 4242, "host": "login99"}))
    with pytest.raises(SystemExit) as err:
        run_multi.claim_driver(work, Path("multi.yaml"))
    assert "login99" in str(err.value) and str(marker) in str(err.value)


def test_driver_alive_rejects_a_recycled_pid():
    import run_multi

    assert (
        run_multi._driver_alive(os.getpid()) is False
    )  # pytest, not run_multi
    assert run_multi._driver_alive(2**22 + 12345) is False  # no such process


@pytest.mark.skipif(os.name == "nt", reason="no liveness check on Windows")
def test_driver_alive_finds_a_running_driver(tmp_path):
    import run_multi

    script = tmp_path / "run_multi.py"
    script.write_text("import time\ntime.sleep(60)\n")
    proc = subprocess.Popen([sys.executable, str(script)])
    try:
        time.sleep(0.5)
        assert run_multi._driver_alive(proc.pid) is True
    finally:
        proc.kill()
        proc.wait()


def _label_store(tmp_path, names, ngff_version="0.5"):
    import zarr

    from patchworks.plugins.ome_zarr import register_labels

    store = tmp_path / "results" / "image.zarr"
    zarr_format = 2 if ngff_version == "0.4" else 3
    labels = zarr.open_group(
        str(store), mode="w", zarr_format=zarr_format
    ).require_group("labels")
    for name in names:
        labels.require_group(name).create_array(
            "0", shape=(4, 8, 8), dtype="int32"
        )
        register_labels(
            str(store),
            name,
            axes="zyx",
            n_levels=1,
            progress=False,
            ngff_version=ngff_version,
        )
    return store


@pytest.mark.parametrize("ngff_version", ["0.4", "0.5"])
def test_label_ready_needs_a_registered_label_image(tmp_path, ngff_version):
    import zarr

    import run_multi

    store = _label_store(tmp_path, ["nuclei"], ngff_version)
    assert run_multi.label_ready(store, "nuclei")
    assert not run_multi.label_ready(store, "cilia")
    # Tiles written in place, merge not finished: a bare group
    zarr.open_group(str(store / "labels")).require_group("cells").create_array(
        "0", shape=(4, 8, 8), dtype="int32"
    )
    assert not run_multi.label_ready(store, "cells")


def test_relations_must_name_labels_this_run_makes_or_has(tmp_path):
    """A relation on a label image nobody makes failed as a zarr
    ArrayNotFoundError in its relate job, after every segmentation."""
    import run_multi

    store = _label_store(tmp_path, ["old_nuclei"])
    cfgs = [{"label_name": "cyto"}, {"label_name": "cilia_dog"}]
    rels = [
        {"a": "old_nuclei", "b": "cyto"},  # in the store: fine
        {"a": "cilia_labels", "b": "cyto"},  # nobody makes it
    ]
    problems = run_multi.relation_problems(rels, cfgs, store)
    assert len(problems) == 1
    assert "'cilia_labels'" in problems[0] and "cilia_dog" in problems[0]


def test_a_done_run_whose_labels_are_gone_is_reported(tmp_path):
    """labels.done in the run directory, labels/<name> missing from the
    store: Snakemake reported the config ok and nothing re-made it."""
    import run_multi

    store = _label_store(tmp_path, ["nuclei"])
    work = tmp_path / "results"
    cfgs = [
        {"work_dir": str(work), "label_name": n}
        for n in ("nuclei", "cilia_labels", "cyto")
    ]
    for name in ("nuclei", "cilia_labels"):
        (work / name).mkdir()
        (work / name / "labels.done").touch()
    problems = run_multi.stale_runs(cfgs, store)
    assert len(problems) == 1
    assert problems[0].startswith("cilia_labels:")
    assert f"Remove {work / 'cilia_labels'}" in problems[0]


def test_relations_on_labels_a_run_left_missing_are_skipped(
    tmp_path, monkeypatch
):
    """Every config reported ok but one left no label image: its relations
    are skipped with a message, the others still run, and the exit says
    something is missing."""
    import types

    import run_multi

    work = tmp_path / "results"
    common = tmp_path / "common.yaml"
    common.write_text(
        yaml.safe_dump(
            {
                "input": str(tmp_path / "scan.zarr"),
                "work_dir": str(work),
                "tile_shape": [16, 512, 512],
                "level": 0,
            }
        )
    )
    paths = []
    for name in ("nuclei", "cyto", "cilia"):
        paths.append(tmp_path / f"{name}.yaml")
        paths[-1].write_text(yaml.safe_dump({"label_name": name}))
    multi = tmp_path / "multi.yaml"
    multi.write_text(
        yaml.safe_dump(
            {
                "common": str(common),
                "segmentations": [str(p) for p in paths],
                "relations": [
                    {"a": "nuclei", "b": "cyto", "output": "n.xlsx"},
                    {"a": "cilia", "b": "cyto", "output": "c.xlsx"},
                ],
            }
        )
    )
    log = []
    monkeypatch.setattr(run_multi, "_run", lambda cmd, wd: 0)  # convert

    def popen(cmd, cwd=None):
        name = Path(cmd[cmd.index("--configfile") + 2]).stem
        if name != "cilia":  # cilia "succeeds" without its labels
            _label_store(tmp_path, ["nuclei", "cyto"])
        return _FakeProc(log, name, 0)

    ran = []
    monkeypatch.setitem(
        sys.modules,
        "relate",
        types.SimpleNamespace(
            run_relations=lambda w, s, rels: ran.extend(rels)
        ),
    )
    monkeypatch.setattr(run_multi.subprocess, "Popen", popen)
    monkeypatch.setattr(
        sys, "argv", ["run_multi.py", "--config", str(multi), "--cores", "1"]
    )
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(SystemExit) as done:
        run_multi.main()
    assert done.value.code == 1
    assert [r["a"] for r in ran] == ["nuclei"]


def test_a_position_rule_orders_its_relations():
    """cilia -> cells is classified by nuclei -> cells: run side by side,
    the cilia job finished first and found no nuclei related."""
    from run_multi import relation_dependencies, run_after

    rels = [
        {"a": "cilia", "b": "cells_ps"},
        {"a": "nuclei", "b": "cells_cp"},
        {"a": "nuclei", "b": "cells_ps"},
        {"a": "cilia", "b": "cells_cp"},
    ]
    rules = {
        "position": {
            "cilia": {"parent": "cells_ps", "apical": "towards:nuclei"}
        }
    }
    deps = relation_dependencies(rels, rules)
    assert deps == {0: 2}  # cilia -> cells_cp has no rule: free
    assert relation_dependencies(rels, None) == {}

    log = []
    names = [f"{r['a']}>{r['b']}" for r in rels]
    status = run_after(
        names,
        deps,
        lambda i: _FakeProc(log, names[i], 1 if i == 2 else 0),
        poll_seconds=0,
        skip_after_failure=False,
    )
    # Started after its dependency, and still run although it failed
    assert log.index(("start", names[0])) > log.index(("end", names[2]))
    assert dict(status) == {
        names[0]: "ok",
        names[1]: "ok",
        names[2]: "FAILED",
        names[3]: "ok",
    }
