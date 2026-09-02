"""Tests for the multi-config driver's SLURM-facing behaviour."""

import re
import sys
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


def test_convert_keys_must_agree_across_configs():
    """`convert` runs once from the first config, so a later one is ignored.

    Setting shard on the second config and watching a million files appear
    anyway is invisible without this check -- there is no log line saying the
    value was dropped, because nothing ever read it.
    """
    paths = [Path("a.yaml"), Path("b.yaml")]
    base = {"work_dir": "/w", "tile_shape": [16, 512, 512], "level": 0}
    good = [
        {**base, "label_name": "a", "shard": True},
        {**base, "label_name": "b", "shard": True},
    ]
    assert _validate_configs(paths, good) == "/w"

    bad = [
        {**base, "label_name": "a", "shard": True},
        {**base, "label_name": "b", "shard": False},
    ]
    # It reports every problem and exits, rather than raising, so that a
    # mistake costs one readable message instead of a traceback.
    with pytest.raises(SystemExit):
        _validate_configs(paths, bad)


def test_shipped_multi_configs_are_consistent():
    """The shipped example must satisfy its own validator.

    It is the thing users copy, so a config set that run_multi would refuse to
    start is worse than no example at all.
    """
    cfg_dir = Path(__file__).resolve().parents[1] / "workflow" / "config"
    multi = yaml.safe_load((cfg_dir / "multi.yaml").read_text())
    common = yaml.safe_load((cfg_dir.parent / multi["common"]).read_text())
    paths = [cfg_dir.parent / p for p in multi["segmentations"]]
    cfgs = [{**common, **yaml.safe_load(p.read_text())} for p in paths]

    assert _validate_configs(paths, cfgs) == common["work_dir"]
    # Every key convert reads comes from the shared file, not a per-config one.
    for path in paths:
        own = yaml.safe_load(path.read_text())
        assert not set(own) & set(_CONVERT_KEYS), path.name


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


def test_relate_script_has_the_real_bookkeeping():
    """relate.py must be the actual implementation, not a stub.

    Submitting the wrong (or a trimmed-down) script would silently produce a
    workbook missing the unmatched-label rows the docstring promises.
    """
    src = (_workflow_dir() / "scripts" / "relate.py").read_text()
    assert "def run_relations(" in src
    assert "label_relations" in src
    assert "openpyxl" in src


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
    src = (_workflow_dir() / "pixi.toml").read_text()
    default_deps = src.split("[pypi-dependencies]")[1].split("[feature")[0]
    assert "napari" not in default_deps
    assert 'viewer = { features = ["viewer"] }' in src
    assert 'extras = ["napari"]' in src


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


def test_mixed_nuclei_channel_auto_passes_validation():
    """A channel-count mismatch under `tile_shape: "auto"` is no longer

    refused at validation time -- it's resolved automatically instead (see
    `_resolve_shared_tile_shape`), which needs the converted image's real
    shape/dtype and so can only run after phase A, not from
    `_validate_configs()`. This used to `sys.exit` here; asserting that
    would now be testing the wrong layer.
    """
    paths = [Path("a.yaml"), Path("b.yaml")]
    base = {"work_dir": "/w", "tile_shape": "auto", "level": 0}

    mixed = [
        {**base, "label_name": "a", "channel": 0, "nuclei_channel": 1},
        {**base, "label_name": "b", "channel": 2},
    ]
    assert _validate_configs(paths, mixed) == "/w"

    # Same pair with one explicit shape is fine: both get that tile.
    pinned = [{**c, "tile_shape": [16, 512, 512]} for c in mixed]
    assert _validate_configs(paths, pinned) == "/w"

    # And "auto" is fine when every config carries the same channel count.
    both = [{**mixed[0]}, {**mixed[1], "nuclei_channel": 3}]
    assert _validate_configs(paths, both) == "/w"


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
