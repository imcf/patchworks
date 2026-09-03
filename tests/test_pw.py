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
