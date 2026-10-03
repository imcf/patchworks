"""Tests for workflow/scripts/_pw.py's config-to-segmentation-function wiring."""

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "workflow" / "scripts")
)

import numpy as np  # noqa: E402


def test_with_voxel_size_fills_in_cellposes_calibration(tmp_path):
    """The same auto-fill a custom function's voxel_size gets (see

    _with_voxel_size's docstring) must also reach cellpose_fn, or do_3D
    silently assumes isotropic voxels -- fragmenting objects across z for
    any real (anisotropic) calibration. This checks the actual calibration
    read + injection against a real store, not just that the parameter
    exists (see test_cellpose.py for that).
    """
    from _pw import _with_voxel_size
    from patchworks.plugins.cellpose import cellpose_fn
    from patchworks.plugins.ome_zarr import to_ome_zarr

    arr = np.zeros((4, 8, 8), dtype="uint16")
    to_ome_zarr(
        arr,
        str(tmp_path / "image.zarr"),
        axes="zyx",
        pixel_size={"z": 0.24, "y": 0.10833, "x": 0.10833},
        n_levels=1,
        progress=False,
    )
    cfg = {"work_dir": str(tmp_path)}

    kwargs = _with_voxel_size(cellpose_fn, {}, cfg)

    assert kwargs["voxel_size"]["z"] == 0.24
    assert kwargs["voxel_size"]["x"] == 0.10833


def test_with_voxel_size_never_overrides_an_explicit_value(tmp_path):
    """An explicit voxel_size must win, and win *cheaply*: no image.zarr

    exists in work_dir at all here, so if this read past the early return it
    would raise, not just return the wrong value.
    """
    from _pw import _with_voxel_size
    from patchworks.plugins.cellpose import cellpose_fn

    cfg = {"work_dir": str(tmp_path)}
    explicit = {"z": 1.0, "y": 1.0, "x": 1.0}

    kwargs = _with_voxel_size(cellpose_fn, {"voxel_size": explicit}, cfg)

    assert kwargs["voxel_size"] == explicit


def test_validate_config_accepts_a_positive_min_volume():
    from _pw import validate_config

    validate_config({"method": "threshold", "min_volume": 5.0})


def test_validate_config_accepts_no_min_volume():
    from _pw import validate_config

    validate_config({"method": "threshold"})
    validate_config({"method": "threshold", "min_volume": None})


def test_validate_config_rejects_a_non_positive_min_volume():
    import pytest
    from _pw import validate_config

    with pytest.raises(ValueError, match="min_volume"):
        validate_config({"method": "threshold", "min_volume": 0})
    with pytest.raises(ValueError, match="min_volume"):
        validate_config({"method": "threshold", "min_volume": -1.0})
    with pytest.raises(ValueError, match="min_volume"):
        validate_config({"method": "threshold", "min_volume": "5"})


def test_validate_config_accepts_a_positive_max_volume():
    from _pw import validate_config

    validate_config({"method": "threshold", "max_volume": 500.0})
    validate_config({"method": "threshold", "max_volume": None})


def test_validate_config_rejects_a_non_positive_max_volume():
    import pytest
    from _pw import validate_config

    with pytest.raises(ValueError, match="max_volume"):
        validate_config({"method": "threshold", "max_volume": 0})
    with pytest.raises(ValueError, match="max_volume"):
        validate_config({"method": "threshold", "max_volume": -1.0})
    with pytest.raises(ValueError, match="max_volume"):
        validate_config({"method": "threshold", "max_volume": "5"})


def test_validate_config_accepts_max_volume_above_min_volume():
    from _pw import validate_config

    validate_config(
        {"method": "threshold", "min_volume": 5.0, "max_volume": 500.0}
    )


def test_validate_config_rejects_max_volume_at_or_below_min_volume():
    import pytest
    from _pw import validate_config

    with pytest.raises(ValueError, match="max_volume"):
        validate_config(
            {"method": "threshold", "min_volume": 5.0, "max_volume": 5.0}
        )
    with pytest.raises(ValueError, match="max_volume"):
        validate_config(
            {"method": "threshold", "min_volume": 500.0, "max_volume": 5.0}
        )


def test_validate_config_rejects_a_model_the_install_does_not_have(
    monkeypatch,
):
    """Cellpose does not raise on an unknown model name -- it logs and loads

    its default. A v3 name against a v4 install therefore segments every
    tile with a model nobody chose, silently. That has to fail in prepare.
    """
    import pytest
    from _pw import validate_config

    import patchworks.plugins.cellpose as cp

    monkeypatch.setattr(cp, "available_models", lambda: ["cpsam", "cpsam_v2"])
    with pytest.raises(ValueError, match="cyto3"):
        validate_config({"method": "cellpose", "cellpose": {"model": "cyto3"}})


def test_validate_config_accepts_a_model_the_install_has(monkeypatch):
    from _pw import validate_config

    import patchworks.plugins.cellpose as cp

    monkeypatch.setattr(cp, "available_models", lambda: ["cpsam", "cpsam_v2"])
    validate_config({"method": "cellpose", "cellpose": {"model": "cpsam"}})


def test_validate_config_leaves_a_custom_model_path_alone(
    monkeypatch, tmp_path
):
    """A path is a custom-trained model; no name list can vouch for it."""
    from _pw import validate_config

    import patchworks.plugins.cellpose as cp

    monkeypatch.setattr(cp, "available_models", lambda: ["cpsam"])
    custom = tmp_path / "my_model.pth"
    custom.write_text("")
    validate_config({"method": "cellpose", "cellpose": {"model": str(custom)}})


def test_validate_config_accepts_shard_and_shard_labels():
    """Both take true/false or an explicit shard shape."""
    from _pw import validate_config

    for key in ("shard", "shard_labels"):
        validate_config({"method": "threshold", key: True})
        validate_config({"method": "threshold", key: False})
        validate_config({"method": "threshold", key: None})
        validate_config({"method": "threshold", key: [16, 512, 512]})


def test_validate_config_rejects_a_malformed_shard_shape():
    """A bad shape would otherwise surface hours later, inside the merge."""
    import pytest
    from _pw import validate_config

    for key in ("shard", "shard_labels"):
        with pytest.raises(ValueError, match=key):
            validate_config({"method": "threshold", key: "auto"})
        with pytest.raises(ValueError, match=key):
            validate_config({"method": "threshold", key: [16, 0, 512]})
        with pytest.raises(ValueError, match=key):
            validate_config({"method": "threshold", key: []})


def test_validate_config_accepts_the_supported_ngff_versions():
    from _pw import validate_config

    for value in ("auto", "0.4", "0.5", None):
        validate_config({"method": "threshold", "ngff_version": value})


def test_validate_config_rejects_an_unwritable_ngff_version():
    """0.6 is released; naming it must explain why it is still refused."""
    import pytest
    from _pw import validate_config

    with pytest.raises(ValueError, match="coordinateSystems"):
        validate_config({"method": "threshold", "ngff_version": "0.6"})
    with pytest.raises(ValueError, match="ngff_version"):
        validate_config({"method": "threshold", "ngff_version": "latest"})


def test_validate_config_rejects_sharding_on_ngff_04():
    """Zarr v2 has no sharding codec, so the pair is a contradiction.

    Silently ignoring `shard` here would be the worst outcome: the whole
    point of turning it on is a filesystem that cannot take the file count.
    """
    import pytest
    from _pw import validate_config

    with pytest.raises(ValueError, match="no sharding codec"):
        validate_config(
            {"method": "threshold", "ngff_version": "0.4", "shard": True}
        )
    # ...but 0.4 on its own is fine.
    validate_config({"method": "threshold", "ngff_version": "0.4"})


def test_segment_progress_resumes_only_the_same_batch(tmp_path):
    """A retried batch skips the tiles an earlier attempt already staged."""
    import zarr

    from _pw import (
        load_segment_progress,
        save_segment_progress,
        segment_progress_path,
    )

    from patchworks import create_stage

    stage = create_stage(tmp_path / "stage.zarr", (4, 8, 8), (1, 8, 8))
    path = segment_progress_path(stage, 3)
    assert load_segment_progress(path, [0, 1, 2], (1, 8, 8)) == {}

    save_segment_progress(path, [0, 1, 2], (1, 8, 8), {0: 5, 1: 0})
    assert load_segment_progress(path, [0, 1, 2], (1, 8, 8)) == {0: 5, 1: 0}
    # A different batch layout or tile shape is stale, not resumable.
    assert load_segment_progress(path, [0, 1], (1, 8, 8)) == {}
    assert load_segment_progress(path, [0, 1, 2], (2, 8, 8)) == {}
    # The checkpoint sits inside the store without confusing zarr ...
    assert zarr.open_group(stage, mode="r")["staged"].shape == (4, 8, 8)
    # ... and dies with it when prepare recreates the stage.
    create_stage(stage, (4, 8, 8), (1, 8, 8))
    assert load_segment_progress(path, [0, 1, 2], (1, 8, 8)) == {}


def test_validate_config_checks_stitch_and_iou_threshold():
    import pytest
    from _pw import validate_config

    validate_config({"method": "threshold", "stitch": "iou"})
    validate_config({"method": "threshold", "iou_threshold": 0.3})
    with pytest.raises(ValueError, match="stitch"):
        validate_config({"method": "threshold", "stitch": "glue"})
    with pytest.raises(ValueError, match="iou_threshold"):
        validate_config({"method": "threshold", "iou_threshold": 0})


def test_build_fn_applies_fill_holes_and_opening():
    import numpy as np
    from _pw import build_fn

    tile = np.zeros((1, 12, 12), "uint16")
    tile[0, 2:9, 2:9] = 1000
    tile[0, 5, 5] = 0  # a hole the threshold leaves
    fn = build_fn(
        {"method": "threshold", "fill_holes": "per_plane", "open_radius": 1}
    )
    out = fn(tile)
    assert out[0, 5, 5] == out[0, 3, 3] != 0


def test_validate_config_checks_the_denoise_block(tmp_path):
    import pytest
    from _pw import validate_config

    model = tmp_path / "n2v.ckpt"
    model.write_bytes(b"")
    validate_config({"method": "threshold", "denoise": {"model": str(model)}})
    with pytest.raises(ValueError) as err:
        validate_config(
            {
                "method": "threshold",
                "denoise": {
                    "model": str(tmp_path / "missing.ckpt"),
                    "tile": [16, 256, 256],
                    "tile_size": "big",
                },
            }
        )
    msg = str(err.value)
    assert "missing.ckpt" in msg and "unknown denoise keys ['tile']" in msg
    assert "denoise.tile_size" in msg
    with pytest.raises(ValueError, match="needs a model"):
        validate_config({"method": "threshold", "denoise": {}})


def test_custom_plugins_validate_through_their_factories():
    import pytest
    from _pw import validate_config

    for module in (
        "patchworks.plugins.watershed",
        "patchworks.plugins.plantseg",
    ):
        with pytest.raises(ValueError, match="unknown custom.kwargs"):
            validate_config(
                {
                    "method": "custom",
                    "custom": {"module": module, "kwargs": {"bogus": 1}},
                }
            )


def _seeded_store(tmp_path):
    """image.zarr with a membrane-like channel 0 and a nuclei label image."""
    from patchworks.plugins.ome_zarr import to_ome_zarr, write_labels

    img = np.zeros((2, 4, 32, 32), "uint16")
    img[0, :, 15:17, :] = 500  # a wall across y = 16
    store = to_ome_zarr(img, tmp_path / "image.zarr", axes="czyx", n_levels=2)
    nuclei = np.zeros((4, 32, 32), "uint32")
    nuclei[1:3, 5:9, 10:14] = 70_001
    nuclei[1:3, 22:26, 10:14] = 70_002
    write_labels(store, nuclei, name="nuclei_labels", progress=False)
    return nuclei


def test_open_image_stacks_the_seed_labels(tmp_path):
    from _pw import check_seed_labels, open_image

    nuclei = _seeded_store(tmp_path)
    arr = open_image(tmp_path, 0, 0, seed_labels="nuclei_labels")
    assert arr.shape == (2, 4, 32, 32)
    np.testing.assert_array_equal(np.asarray(arr[1]), nuclei)  # ids exact
    assert np.asarray(arr[0])[0, 16, 0] == 500
    check_seed_labels(
        tmp_path, {"seed_labels": "nuclei_labels", "channel": 0, "level": 0}
    )

    import pytest

    with pytest.raises(ValueError, match="does not exist"):
        check_seed_labels(
            tmp_path, {"seed_labels": "cells", "channel": 0, "level": 0}
        )
    with pytest.raises(ValueError, match="same level"):
        open_image(tmp_path, 0, 1, seed_labels="nuclei_labels")
    with pytest.raises(ValueError, match="set one"):
        open_image(
            tmp_path, 0, 0, nuclei_channel=1, seed_labels="nuclei_labels"
        )


def test_seed_labels_grow_one_cell_per_given_nucleus(tmp_path):
    from _pw import build_fn, open_image

    nuclei = _seeded_store(tmp_path)
    cfg = {
        "method": "custom",
        "work_dir": str(tmp_path),
        "seed_labels": "nuclei_labels",
        "custom": {
            "module": "patchworks.plugins.watershed",
            "kwargs": {"foreground": None},
        },
    }
    fn = build_fn(cfg)
    assert fn.keywords["seeds"] == "labels"
    cells = fn(
        np.asarray(open_image(tmp_path, 0, 0, seed_labels="nuclei_labels"))
    )
    # The wall at y = 16 parts the two cells, one per nucleus
    assert len(np.unique(cells[cells > 0])) == 2
    assert cells[2, 7, 12] != cells[2, 24, 12]
    assert (cells[:, :15] == cells[2, 7, 12]).all()
    assert nuclei.max() > cells.max()  # renumbered per tile, merged later


def test_validate_config_checks_seed_labels():
    import pytest
    from _pw import validate_config

    ok = {
        "method": "custom",
        "label_name": "cyto_labels",
        "seed_labels": "nuclei_labels",
        "custom": {"module": "patchworks.plugins.watershed"},
    }
    validate_config(ok)
    cases = {
        "own label_name": {**ok, "seed_labels": "cyto_labels"},
        "set one": {**ok, "nuclei_channel": 1},
        'needs method: "custom"': {
            **ok,
            "method": "cellpose",
            "cellpose": {"model": "cyto3"},
        },
        "takes no `seeds`": {
            **ok,
            "custom": {"module": "patchworks.plugins.dog"},
        },
        "custom.kwargs.seeds": {
            **ok,
            "custom": {
                "module": "patchworks.plugins.watershed",
                "kwargs": {"seeds": "channel"},
            },
        },
        "name of a label image": {**ok, "seed_labels": 3},
    }
    for message, cfg in cases.items():
        with pytest.raises(ValueError, match=message):
            validate_config(cfg)
