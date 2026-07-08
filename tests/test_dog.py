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
