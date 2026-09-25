"""Self-contained tests for the dog plugin. No frameworks, no fixtures."""

import numpy as np
import pytest


def _make_blob_image(shape=(1, 64, 64)):
    img = np.zeros(shape, dtype="float32")
    img[0, 28:36, 28:36] = 1.0
    return img


def test_dog_label_fn_cpu_finds_blob():
    from patchworks.plugins.dog import dog_label_fn

    fn = dog_label_fn(low_sigma=1.0, high_sigma=4.0, threshold=0.01)
    labels = fn(_make_blob_image())

    assert labels.shape == (1, 64, 64)
    assert labels.dtype == np.int32
    assert labels.max() >= 1  # the blob was detected
    assert labels[0, 0, 0] == 0  # background stays unlabeled


def test_segment_adapter_matches_factory():
    # method: "custom" calls segment(tile, **kwargs) directly — must match
    # dog_label_fn(**kwargs)(tile) exactly.
    from patchworks.plugins.dog import dog_label_fn, segment

    kwargs = dict(low_sigma=1.0, high_sigma=4.0, threshold=0.01)
    img = _make_blob_image()

    via_adapter = segment(img, **kwargs)
    via_factory = dog_label_fn(**kwargs)(img)

    np.testing.assert_array_equal(via_adapter, via_factory)


def test_dog_label_fn_with_tile_process():
    import dask.array as da

    from patchworks import tile_process
    from patchworks.plugins.dog import dog_label_fn

    arr = da.from_array(_make_blob_image((1, 64, 64)), chunks=(1, 64, 64))
    fn = dog_label_fn(low_sigma=1.0, high_sigma=4.0, threshold=0.01)
    result = tile_process(arr, fn, overlap=4).compute()

    assert result.shape == (1, 64, 64)
    assert result.max() >= 1


def test_decon_voxel_kwargs_from_calibration():
    """Voxel sizes come from the image, not from retyped config values.

    A deconvolution told the wrong voxel size does not fail -- it returns a
    subtly wrong result -- so deriving them is the point.
    """
    from patchworks.plugins.dog import decon_voxel_kwargs

    assert decon_voxel_kwargs({"z": 0.2, "y": 0.1, "x": 0.1}) == {
        "dxdata": 0.1,
        "dzdata": 0.2,
        "dxpsf": 0.1,
        "dzpsf": 0.2,
    }
    # A PSF sampled differently from the data keeps its own sizes.
    assert decon_voxel_kwargs(
        {"z": 0.2, "y": 0.1, "x": 0.1}, {"z": 0.1, "y": 0.05, "x": 0.05}
    ) == {"dxdata": 0.1, "dzdata": 0.2, "dxpsf": 0.05, "dzpsf": 0.1}
    # An uncalibrated axis is simply omitted rather than guessed.
    assert decon_voxel_kwargs({"y": 0.1, "x": 0.1}) == {
        "dxdata": 0.1,
        "dxpsf": 0.1,
    }
    assert decon_voxel_kwargs({}) == {}


class _FakeDecon:
    """pycudadecon's step API, recording every call."""

    def __init__(self):
        self.calls = []

    def make_otf(self, psf, outpath, **kw):
        self.calls.append(("make_otf", psf, kw))
        return outpath

    def rl_init(self, shape, otfpath, **kw):
        self.calls.append(("rl_init", tuple(shape), kw))

    def rl_decon(self, im, **kw):
        self.calls.append(("rl_decon", im.shape, kw))
        return im.astype("float32")

    def rl_cleanup(self):
        self.calls.append(("rl_cleanup",))

    def decon(self, images, **kw):
        self.calls.append(("decon", images.shape, kw))
        return images.astype("float32")

    def named(self, name):
        return [c for c in self.calls if c[0] == name]


@pytest.fixture
def fake_decon(monkeypatch):
    import sys

    from patchworks.plugins import dog

    fake = _FakeDecon()
    monkeypatch.setitem(sys.modules, "pycudadecon", fake)
    monkeypatch.setattr(dog, "_require_pycudadecon", lambda: None)
    dog._drop_decon_state()
    yield fake
    dog._decon_state.clear()


def test_explicit_decon_kwargs_win_over_the_calibration(fake_decon):
    """Anything set by hand must survive; only gaps are filled."""
    from patchworks.plugins import dog

    fn = dog.dog_label_fn(
        low_sigma=1.0,
        high_sigma=3.0,
        threshold=0.5,
        # dxpsf set by hand: the PSF was sampled finer than the data
        decon_kwargs={"psf": "psf.tif", "dxpsf": 0.05},
        voxel_size={"z": 0.2, "y": 0.1, "x": 0.1},
    )
    fn(np.zeros((4, 8, 8), "uint16"))

    ((_, _, otf_kw),) = fake_decon.named("make_otf")
    ((_, _, init_kw),) = fake_decon.named("rl_init")
    assert otf_kw["dxpsf"] == 0.05, "an explicit value must not be replaced"
    assert init_kw["dxdata"] == 0.1  # filled from the calibration
    assert init_kw["dzdata"] == 0.2
    assert otf_kw["dzpsf"] == init_kw["dzpsf"] == 0.2


def test_decon_setup_is_built_once_per_worker(fake_decon):
    """The OTF once; cudaDecon once per tile shape -- not once per tile."""
    from patchworks.plugins import dog

    fn = dog.dog_label_fn(
        1.0, 3.0, 0.5, decon_kwargs={"psf": "psf.tif", "n_iters": 5}
    )
    for shape in [(4, 8, 8)] * 3 + [(4, 8, 6)] + [(4, 8, 8)]:
        fn(np.zeros(shape, "uint16"))
    assert len(fake_decon.named("make_otf")) == 1
    assert [c[1] for c in fake_decon.named("rl_init")] == [
        (4, 8, 8),
        (4, 8, 6),
        (4, 8, 8),
    ]
    assert len(fake_decon.named("rl_decon")) == 5
    assert all(c[2]["n_iters"] == 5 for c in fake_decon.named("rl_decon"))
    assert not fake_decon.named("decon")


def test_decon_falls_back_for_what_the_cache_does_not_model(fake_decon):
    from patchworks.plugins import dog

    psf = np.ones((3, 3, 3), "float32")
    dog.dog_label_fn(1.0, 3.0, 0.5, decon_kwargs={"psf": psf})(
        np.zeros((4, 8, 8), "uint16")
    )
    assert len(fake_decon.named("decon")) == 1


def test_dup_rev_z_auto_follows_tile_depth_vs_psf(fake_decon):
    """Shallow tiles (vs the PSF's axial extent) get z-mirroring."""
    from patchworks.plugins import dog

    kw = {
        "psf": "psf.tif",
        "dup_rev_z": "auto",
        "wavelength": 525,
        "na": 1.4,
        "nimm": 1.515,
        "dzdata": 0.2,
    }
    fn = dog.dog_label_fn(1.0, 3.0, 0.5, decon_kwargs=kw)
    fn(np.zeros((4, 8, 8), "uint16"))  # 0.8 um deep, FWHM ~0.5 um
    fn(np.zeros((40, 8, 8), "uint16"))  # 8 um deep
    flags = [c[2]["dup_rev_z"] for c in fake_decon.named("rl_decon")]
    assert flags == [True, False]


def test_restore_shape_anchors_a_cropped_decon_at_the_origin():
    """cudaDecon can hand back a smaller volume than it was given.

    Observed on a real tile: (32, 1084, 1084) in, (32, 1080, 1080) out --
    each axis rounded down to an FFT-efficient length, with the excess taken
    off the high end. Restoring it *centred* (what this used to do) moved
    every voxel by excess // 2, measured as a 2 px y/x shift on real data:
    invisible on a cell, glaring on a cilium a few voxels across.
    """
    from patchworks.plugins.dog import _restore_shape

    arr = np.arange(13 * 1020 * 1020, dtype="float32").reshape(13, 1020, 1020)
    out = _restore_shape(arr, (14, 1024, 1024))
    assert out.shape == (14, 1024, 1024)
    # Content keeps its original indices -- voxel 0 stays voxel 0.
    assert np.array_equal(out[0:13, 0:1020, 0:1020], arr)


def test_restore_shape_crops_from_the_high_end():
    """The mirror case: an axis that came back too long keeps its low corner."""
    from patchworks.plugins.dog import _restore_shape

    arr = np.arange(6 * 12, dtype="float32").reshape(6, 12)
    out = _restore_shape(arr, (4, 8))
    assert out.shape == (4, 8)
    assert np.array_equal(out, arr[0:4, 0:8])


def test_restore_shape_handles_growth_and_exact_fit():
    """It must be a no-op when shapes already match, and crop when larger."""
    from patchworks.plugins.dog import _restore_shape

    same = np.ones((4, 8, 8), dtype="float32")
    assert _restore_shape(same, (4, 8, 8)).shape == (4, 8, 8)
    bigger = np.ones((6, 12, 12), dtype="float32")
    assert _restore_shape(bigger, (4, 8, 8)).shape == (4, 8, 8)
    # And a mix: one axis short, one long.
    mixed = np.ones((2, 12), dtype="float32")
    assert _restore_shape(mixed, (4, 8)).shape == (4, 8)


def test_stage_tile_rejects_a_shape_changing_function(tmp_path):
    """A wrong-shaped return must name the culprit, not blow up inside zarr.

    This used to surface as "could not broadcast input array from shape
    (13,1020,1020) into shape (14,1024,1024)" six frames deep in zarr's codec
    pipeline, which says nothing about which function misbehaved.
    """
    import dask.array as da
    import pytest

    from patchworks import create_stage, stage_tile

    image = da.zeros((8, 32, 32), chunks=(4, 16, 16), dtype="uint16")
    stage = str(tmp_path / "stage.zarr")
    create_stage(stage, image.shape, (4, 16, 16))

    def crops(block):
        """Stand-in for a deconvolution backend that trims its output."""
        return np.zeros(tuple(s - 1 for s in block.shape), dtype="int32")

    with pytest.raises(ValueError, match="one label per input voxel"):
        stage_tile(image, crops, stage, 0, tile_shape=(4, 16, 16), overlap=2)


def test_physical_sigmas_follow_the_calibration():
    """sigma_units="um" blurs every axis by the same distance."""
    from patchworks.plugins.dog import _fit_sigma, _physical_sigma

    vox = {"z": 0.5, "y": 0.1, "x": 0.1}
    assert _physical_sigma(0.3, vox) == pytest.approx((0.6, 3.0, 3.0))
    assert _physical_sigma((0.2, 0.2), vox) == pytest.approx((2.0, 2.0))
    assert _fit_sigma((0.6, 3.0, 3.0), 2) == (3.0, 3.0)


def test_sigma_units_um_segments_like_the_equivalent_pixels():
    from patchworks.plugins.dog import dog_label_fn

    rng = np.random.default_rng(0)
    img = rng.random((6, 40, 40)).astype("float32") * 0.01
    img[2:4, 15:25, 15:25] += 1.0
    vox = {"z": 0.5, "y": 0.1, "x": 0.1}
    um = dog_label_fn(0.1, 0.3, 0.02, voxel_size=vox, sigma_units="um")(img)
    px = dog_label_fn((0.2, 1.0, 1.0), (0.6, 3.0, 3.0), 0.02)(img)
    np.testing.assert_array_equal(um, px)
    with pytest.raises(ValueError, match="voxel_size"):
        dog_label_fn(0.1, 0.3, 0.02, sigma_units="um")
