"""Self-contained tests for the dog plugin. No frameworks, no fixtures."""

import numpy as np


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


def test_explicit_decon_kwargs_win_over_the_calibration(monkeypatch):
    """Anything set by hand must survive; only gaps are filled."""
    import sys
    import types

    from patchworks.plugins import dog

    captured = {}

    def _decon(images, **kwargs):
        captured.update(kwargs)
        return images

    monkeypatch.setitem(
        sys.modules, "pycudadecon", types.SimpleNamespace(decon=_decon)
    )
    monkeypatch.setattr(dog, "_require_pycudadecon", lambda: None)

    fn = dog.dog_label_fn(
        low_sigma=1.0,
        high_sigma=3.0,
        threshold=0.5,
        # dxpsf set by hand: the PSF was sampled finer than the data
        decon_kwargs={"psf": "psf.tif", "dxpsf": 0.05},
        voxel_size={"z": 0.2, "y": 0.1, "x": 0.1},
    )
    fn(np.zeros((4, 8, 8), "uint16"))

    assert captured["dxpsf"] == 0.05, "an explicit value must not be replaced"
    assert captured["dxdata"] == 0.1  # filled from the calibration
    assert captured["dzdata"] == 0.2
    assert captured["dzpsf"] == 0.2
