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


def test_fetch_models_names_what_it_fetches(monkeypatch):
    import _pw

    calls = []
    import patchworks.plugins.cellpose as cp

    monkeypatch.setattr(cp, "_get_model", lambda spec: calls.append(spec))
    assert "Cellpose model nuclei" in _pw.fetch_models(
        {"method": "cellpose", "cellpose": {"model": "nuclei"}}
    )
    assert calls == [{"model": "nuclei", "gpu": False}]
    try:
        import patchworks.plugins.plantseg as ps
    except ImportError:
        pytest.skip("no PlantSeg plugin import here")
    monkeypatch.setattr(ps, "fetch_model", lambda m: calls.append(m))
    spec = {
        "method": "custom",
        "custom": {
            "module": "patchworks.plugins.plantseg",
            "kwargs": {"model": "generic_light_sheet_3D_unet"},
        },
    }
    assert "PlantSeg model generic_light_sheet_3D_unet" in _pw.fetch_models(
        spec
    )
    assert calls[-1] == "generic_light_sheet_3D_unet"
    assert "no model" in _pw.fetch_models({"method": "threshold"})


def test_fetch_script_reports_each_segmentation(tmp_path, monkeypatch, capsys):
    import _pw
    import fetch_models

    monkeypatch.setattr(
        _pw, "fetch_models", lambda cfg: f"ok {cfg['label_name']}"
    )
    path = _config(tmp_path)
    monkeypatch.setattr(sys, "argv", ["fetch_models.py", str(path)])
    with pytest.raises(SystemExit) as stop:
        fetch_models.main()
    assert stop.value.code == 0
    assert "[fetch] cells: ok cells" in capsys.readouterr().out


def test_preview_draws_a_segmented_tile(tmp_path):
    from PIL import Image

    (tmp_path / "w").mkdir()
    _converted(tmp_path / "w")
    report = check.check(
        _config(tmp_path), sample=2, preview=tmp_path / "previews"
    )
    plan = report["segmentations"][0]["plan"]
    assert plan["preview"] == str(tmp_path / "previews" / "cells.png")
    assert plan["preview_objects"] == 1  # the one bright square
    assert plan["preview_plane"] == 4  # the middle of 8 planes
    picture = Image.open(plan["preview"])
    # The tile (32 x 32), a gap, the tile again with the outlines.
    assert picture.size == (32 + 8 + 32, 32)
    rgb = picture.convert("RGB")
    left, right = rgb.crop((0, 0, 32, 32)), rgb.crop((40, 0, 72, 32))
    assert left.tobytes() != right.tobytes()  # outlined on the right
    assert "1 objects, drawn in" in check.format_report(report)


def test_channel_pictures(tmp_path, capsys, monkeypatch):
    import channels
    from PIL import Image

    (tmp_path / "w").mkdir()
    _converted(tmp_path / "w")
    path = _config(tmp_path)
    monkeypatch.setattr(sys, "argv", ["channels.py", str(path), "--json"])
    channels.main()
    info = json.loads(capsys.readouterr().out)
    assert info["channels"] == 2 and info["shape"] == [2, 8, 64, 64]
    assert info["axes"] == "czyx" and len(info["pictures"]) == 2
    first = Image.open(info["pictures"][0])
    assert first.size == (64, 64)
    # Channel 0 has the bright square, channel 1 nothing.
    assert first.getextrema()[1] == 255


def test_channels_needs_a_converted_image(tmp_path, monkeypatch):
    import channels

    monkeypatch.setattr(sys, "argv", ["channels.py", str(_config(tmp_path))])
    with pytest.raises(SystemExit, match="not converted yet"):
        channels.main()


def test_overlap_auto_from_tried_tiles_then_an_earlier_runs_table(tmp_path):
    """overlap "auto": from the objects themselves -- the tiles tried with
    --sample, and once there is one, the segmentation's own table."""
    import json

    import _pw

    # Objects measured: whole ones only, per axis.
    labels = np.zeros((10, 40, 40), "uint16")
    labels[2:6, 5:25, 5:15] = 1  # 4 x 20 x 10
    labels[0:3, 30:39, 30:39] = 2  # touches the edge in z: left out
    assert _pw.object_extents(labels).tolist() == [[4, 20, 10]]
    assert _pw.overlap_from_extents([[4, 20, 10]]) == [5, 22, 11]

    _converted(tmp_path / "w")
    cfg = {
        "work_dir": str(tmp_path / "w"),
        "label_name": "cells",
        "overlap": "auto",
    }
    spec, source = _pw.resolve_overlap(cfg, 3)
    assert spec == _pw.OVERLAP_FALLBACK and "no measurement" in source

    # Tiles tried: their objects size it, and are kept for the run.
    report = check.check(_config(tmp_path, overlap="auto"), sample=2)
    (seg,) = report["segmentations"]
    assert seg["plan"]["suggested_overlap"] == [5, 18, 18]  # the 4x16x16 blob
    saved = json.loads(
        (tmp_path / "w" / "cells" / _pw.EXTENTS_FILE).read_text()
    )
    assert saved["extents"] == [[4, 16, 16]]
    spec, source = _pw.resolve_overlap(cfg, 3, (16, 64, 64))
    assert spec == [5, 18, 18] and "tried tile" in source
    # Tiles smaller than the objects: half the tile, and said.
    spec, source = _pw.resolve_overlap(cfg, 3, (8, 32, 32))
    assert spec == [4, 16, 16] and "capped at half the tile" in source

    # A number set by hand is kept as it is.
    assert _pw.resolve_overlap({**cfg, "overlap": [2, 8, 8]}, 3)[0] == [2, 8, 8]


def test_overlap_auto_prefers_an_earlier_runs_object_table(tmp_path):
    import _pw
    import zarr

    from patchworks._tables import compute_table
    from patchworks.plugins.ome_zarr import register_labels

    store = _converted(tmp_path / "w")
    labels = np.zeros((8, 64, 64), "uint32")
    labels[1:7, 10:40, 10:30] = 1  # 6 x 30 x 20
    labels[2:5, 45:55, 45:60] = 2  # 3 x 10 x 15
    group = zarr.open_group(store, mode="a").require_group("labels")
    group.require_group("cells").create_array(
        "0", data=labels, chunks=(8, 32, 32)
    )
    register_labels(store, "cells", n_levels=1, progress=False)
    compute_table(store, "cells")
    cfg = {
        "work_dir": str(tmp_path / "w"),
        "label_name": "cells",
        "overlap": "",
    }
    spec, source = _pw.resolve_overlap(cfg, 3, (32, 256, 256))
    assert "earlier run's table" in source
    # 90th percentile of each axis, plus 10%.
    assert spec == [7, 31, 22]  # p90 [5.7, 28, 19.5] * 1.1, rounded up
