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
