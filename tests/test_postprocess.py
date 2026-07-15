"""Self-contained tests for the dilate_labels post-processing wrapper."""

import pickle

import numpy as np


def _make_blob_labels(shape=(1, 64, 64)):
    labels = np.zeros(shape, dtype="int32")
    labels[0, 28:36, 28:36] = 1
    return labels


def test_dilate_labels_grows_mask():
    from patchworks import dilate_labels

    fn = lambda tile: _make_blob_labels(tile.shape)  # noqa: E731
    plain = fn(np.zeros((1, 64, 64)))
    dilated = dilate_labels(fn, iterations=2)(np.zeros((1, 64, 64)))

    assert (dilated > 0).sum() > (plain > 0).sum()


def test_dilate_labels_zero_iterations_is_noop():
    from patchworks import dilate_labels

    fn = lambda tile: _make_blob_labels(tile.shape)  # noqa: E731

    assert dilate_labels(fn, iterations=0) is fn


def test_dilate_labels_picklable():
    from patchworks.plugins.dog import dog_label_fn

    from patchworks import dilate_labels

    fn = dilate_labels(
        dog_label_fn(low_sigma=1.0, high_sigma=4.0, threshold=0.01),
        iterations=2,
    )
    pickle.loads(pickle.dumps(fn))
