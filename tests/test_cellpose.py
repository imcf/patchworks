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

import numpy as np


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


def test_available_models_is_a_list():
    """Empty when Cellpose isn't installed -- then no name can be rejected."""
    from patchworks.plugins.cellpose import available_models

    assert isinstance(available_models(), list)


def test_v4_gets_the_model_name_as_pretrained_model():
    """v4 accepts `model_type=` and then ignores it ("not used in v4.0.1+"),

    leaving pretrained_model at its default -- so passing the configured name
    there segmented every config with the same default model whatever
    `model:` said. The name has to reach `pretrained_model=` on v4.
    """
    import inspect

    from patchworks.plugins import cellpose as cp

    src = inspect.getsource(cp._get_model)
    v4_branch = src.split("if _CELLPOSE_V4:")[1].split("else:")[0]
    assert "pretrained_model=model_type" in v4_branch
    assert "model_type=model_type" not in v4_branch


class _FakeModel:
    def __init__(self):
        self.seen = None

    def eval(self, img, **kwargs):
        self.seen = (np.array(img), kwargs)
        spatial = (
            img.shape[1:] if kwargs.get("channel_axis") == 0 else img.shape
        )
        return (np.ones(spatial, dtype="int32"),)


def test_intensity_range_scales_every_tile_alike(monkeypatch):
    """With an image-wide range, two tiles of different brightness reach
    Cellpose scaled by the same numbers, and Cellpose's own per-input
    normalisation is off -- else each tile gets its own contrast."""
    from patchworks.plugins import cellpose as cp

    fake = _FakeModel()
    monkeypatch.setattr(cp, "_require_cellpose", lambda: None)
    monkeypatch.setattr(cp, "_get_model", lambda _cfg: fake)
    fn = cp.cellpose_fn("cpsam", do_3D=True, intensity_range=(100, 300))
    for level in (100, 200):
        fn(np.full((2, 4, 4), level, dtype="uint16"))
        img, kwargs = fake.seen
        assert kwargs["normalize"] is False
        np.testing.assert_allclose(img, (level - 100) / 200)


def test_intensity_range_per_channel(monkeypatch):
    from patchworks.plugins import cellpose as cp

    fake = _FakeModel()
    monkeypatch.setattr(cp, "_require_cellpose", lambda: None)
    monkeypatch.setattr(cp, "_get_model", lambda _cfg: fake)
    fn = cp.cellpose_fn(
        "cpsam",
        do_3D=True,
        channel_axis=0,
        intensity_range=[(0, 10), (100, 200)],
    )
    tile = np.stack([np.full((2, 3, 3), 5), np.full((2, 3, 3), 150)])
    assert fn(tile).shape == (2, 3, 3)
    img, _ = fake.seen
    np.testing.assert_allclose(img[0], 0.5)
    np.testing.assert_allclose(img[1], 0.5)


def test_intensity_range_and_normalize_are_exclusive(monkeypatch):
    import pytest

    from patchworks.plugins import cellpose as cp

    monkeypatch.setattr(cp, "_require_cellpose", lambda: None)
    with pytest.raises(ValueError, match="not both"):
        cp.cellpose_fn("cpsam", intensity_range=(0, 1), normalize=True)


def test_image_intensity_range_samples_full_resolution(tmp_path):
    """Sampled at full resolution, in the given regions, ignoring the
    exact-zero padding of unacquired regions."""
    from patchworks import intensity_range
    from patchworks.plugins.ome_zarr import to_ome_zarr

    rng = np.random.default_rng(0)
    img = np.zeros((2, 4, 64, 64), dtype="uint16")
    img[0, :, :, :32] = rng.integers(100, 201, (4, 64, 32))
    img[0, :, :, 32:] = 5000  # outside the regions sampled
    img[1] = 1000
    store = str(tmp_path / "s.zarr")
    to_ome_zarr(img, store, axes="czyx", n_levels=3, progress=False)
    left = [
        (slice(0, 4), slice(y, y + 16), slice(0, 32)) for y in range(0, 64, 16)
    ]
    (lo0, hi0), (lo1, hi1) = intensity_range(
        store, [0, 1], regions=left, sample_shape=(4, 16, 16)
    )
    assert 100 <= lo0 < 105 and 195 < hi0 <= 200
    assert lo1 == 1000 and hi1 == 1001  # flat channel: a unit-wide range
    # Without regions, random crops; zeros alone give the identity range.
    blank = str(tmp_path / "b.zarr")
    to_ome_zarr(
        np.zeros((4, 32, 32), "uint16"), blank, axes="zyx", progress=False
    )
    assert intensity_range(blank, None) == [(0.0, 1.0)]


def test_3d_cellpose_folds_fragments_and_smooths_flows(monkeypatch):
    """3-D Cellpose fragments cells; by default the slivers are folded back
    into their cells, and the flows smoothed where Cellpose supports it."""
    from patchworks.plugins import cellpose as cp

    class Model:
        def eval(self, img, flow3D_smooth=0, **kwargs):
            self.smooth = flow3D_smooth
            masks = np.zeros(img.shape, "int32")
            masks[1:9, 2:30, 2:30] = 1
            masks[1:9, 28:30, 2:30] = 2  # a sliver of cell 1
            return (masks,)

    model = Model()
    monkeypatch.setattr(cp, "_require_cellpose", lambda: None)
    monkeypatch.setattr(cp, "_get_model", lambda _cfg: model)
    monkeypatch.setattr(
        cp, "_cellpose_models", type("M", (), {"CellposeModel": Model})
    )
    out = cp.cellpose_fn("cpsam", do_3D=True)(np.zeros((10, 32, 32), "uint16"))
    assert set(np.unique(out)) == {0, 1}
    assert model.smooth == 1
    # Off on request, and off by default in 2-D.
    out = cp.cellpose_fn("cpsam", do_3D=True, fragments=None, flow3D_smooth=0)(
        np.zeros((10, 32, 32), "uint16")
    )
    assert set(np.unique(out)) == {0, 1, 2} and model.smooth == 0


def test_applied_defaults_are_what_cellpose_fn_fills_in(monkeypatch):
    """The workflow records these in the labels' provenance, so they must be
    exactly what cellpose_fn applies -- explicit settings win."""
    from patchworks.plugins import cellpose as cp

    class Model:
        def eval(self, img, flow3D_smooth=0): ...

    monkeypatch.setattr(
        cp, "_cellpose_models", type("M", (), {"CellposeModel": Model})
    )
    assert cp.applied_defaults(True, {}) == {
        "fragments": 0.1,
        "flow3D_smooth": 1,
    }
    assert cp.applied_defaults(False, {}) == {"fragments": None}
    assert cp.applied_defaults(
        True, {"fragments": None, "flow3D_smooth": 0}
    ) == {"fragments": None}
