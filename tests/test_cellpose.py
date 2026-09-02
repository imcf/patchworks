"""Tests for the cellpose plugin's anisotropy handling.

cellpose itself is not a test dependency (heavy, GPU-oriented), and
cellpose_fn() calls _require_cellpose() as its very first line, so it can't
be exercised end-to-end here. cellpose_anisotropy() is a pure function with
no cellpose import at all, and is where the actual math lives -- that's
what's covered directly. The wiring that fills cellpose_fn's voxel_size
parameter in from the image's calibration (workflow/scripts/_pw.py's
_with_voxel_size) is covered in tests/test_pw.py.
"""

import inspect


def test_cellpose_anisotropy_from_calibration():
    from patchworks.plugins.cellpose import cellpose_anisotropy

    calibration = {"z": 0.24, "y": 0.10833, "x": 0.10833}
    assert cellpose_anisotropy(calibration) == 0.24 / 0.10833


def test_cellpose_anisotropy_falls_back_to_y_when_x_is_missing():
    from patchworks.plugins.cellpose import cellpose_anisotropy

    assert cellpose_anisotropy({"z": 0.2, "y": 0.1}) == 2.0


def test_cellpose_anisotropy_missing_calibration_returns_none():
    from patchworks.plugins.cellpose import cellpose_anisotropy

    assert cellpose_anisotropy({}) is None
    assert cellpose_anisotropy({"z": 0.2}) is None  # no lateral size at all
    assert cellpose_anisotropy({"x": 0.1}) is None  # no z size


def test_cellpose_fn_declares_a_voxel_size_parameter():
    """workflow/scripts/_pw.py's _with_voxel_size() finds this by signature

    inspection to decide whether to fill it in -- if this parameter were
    ever renamed, that wiring would silently stop working rather than error.
    """
    from patchworks.plugins.cellpose import cellpose_fn

    params = inspect.signature(cellpose_fn).parameters
    assert "voxel_size" in params
    # anisotropy itself goes through **cellpose_kwargs (it's a cellpose
    # model.eval() argument, not a patchworks-specific one) -- only the raw
    # calibration is a named parameter.
    assert "anisotropy" not in params
