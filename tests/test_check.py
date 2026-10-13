"""Tests for workflow/scripts/check.py and _pw.plan_run (before a run)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workflow" / "scripts"))

import check  # noqa: E402


def _converted(work, with_occupancy=True):
    from patchworks import build_occupancy_map
    from patchworks.plugins.ome_zarr import to_ome_zarr

    img = np.zeros((2, 8, 64, 64), dtype="uint16")
    img[0, 2:6, 4:20, 4:20] = 1000  # signal in the first tile only
    store = str(work / "image.zarr")
    to_ome_zarr(img, store, axes="czyx", n_levels=1, progress=False)
    if with_occupancy:
        build_occupancy_map(store, level=0, block=(8, 16, 16))
    return store


def _config(tmp_path, **extra):
    cfg = {
        "input": "/data/scan.czi",
        "work_dir": str(tmp_path / "w"),
        "label_name": "cells",
        "method": "threshold",
        "tile_shape": [8, 32, 32],
        "overlap": [0, 4, 4],
        "tiles_per_job": 2,
        **extra,
    }
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_plan_matches_what_prepare_would_do(tmp_path):
    (tmp_path / "w").mkdir()
    _converted(tmp_path / "w")
    report = check.check(_config(tmp_path), sample=2)
    assert report["valid"] and report["converted"]
    plan = report["segmentations"][0]["plan"]
    assert plan["tile_shape"] == [8, 32, 32] and plan["grid"] == [1, 2, 2]
    assert plan["tiles"] == 4
    assert plan["tiles_with_signal"] == 1 and plan["gpu_jobs"] == 1
    # Sampled among the tiles with signal: only one has any.
    assert plan["seconds_per_tile"] is not None and plan["sampled_tiles"] == 1
    text = check.format_report(report)
    assert "4 tiles, 1 with signal -> 1 GPU jobs of up to 2" in text


def test_without_the_occupancy_map_signal_is_unknown(tmp_path):
    (tmp_path / "w").mkdir()
    _converted(tmp_path / "w", with_occupancy=False)
    plan = check.check(_config(tmp_path))["segmentations"][0]["plan"]
    assert plan["tiles_with_signal"] is None and plan["gpu_jobs"] is None


def test_problems_and_a_failing_plan_make_it_invalid(
    tmp_path, capsys, monkeypatch
):
    report = check.check(_config(tmp_path, stitch="tuoch"))
    assert not report["valid"] and not report["converted"]
    assert any("tuoch" in p for p in report["segmentations"][0]["problems"])

    (tmp_path / "w").mkdir()
    _converted(tmp_path / "w")
    path = _config(tmp_path, overlap=[8, 4, 4])
    report = check.check(path)
    assert not report["valid"]
    assert "overlap[0]=8" in report["segmentations"][0]["plan_error"]

    monkeypatch.setattr(sys, "argv", ["check.py", str(path), "--json"])
    with pytest.raises(SystemExit) as stop:
        check.main()
    assert stop.value.code == 1
    assert json.loads(capsys.readouterr().out)["valid"] is False


def test_a_multi_config_is_checked_as_a_set(tmp_path):
    common = tmp_path / "common.yaml"
    common.write_text(
        yaml.safe_dump({"input": "/x.czi", "method": "threshold"})
    )
    paths = []
    for name, work in (("a", "w1"), ("b", "w2")):
        paths.append(tmp_path / f"{name}.yaml")
        paths[-1].write_text(
            yaml.safe_dump(
                {"label_name": name, "work_dir": str(tmp_path / work)}
            )
        )
    multi = tmp_path / "multi.yaml"
    multi.write_text(
        yaml.safe_dump(
            {"common": str(common), "segmentations": [str(p) for p in paths]}
        )
    )
    report = check.check(multi)
    assert not report["valid"]
    assert any("work_dir" in p for p in report["shared_problems"])
