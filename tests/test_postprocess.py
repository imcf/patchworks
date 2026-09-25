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


def test_dilate_labels_does_not_eat_a_touching_neighbour():
    """Growth goes into background only; existing objects keep their voxels.

    A bare max filter let the higher id overwrite its lower neighbour along
    their shared edge, shrinking the neighbour instead of growing anything.
    """
    from patchworks import dilate_labels

    labels = np.zeros((1, 8, 16), dtype="int32")
    labels[0, 2:6, 2:8] = 3
    labels[0, 2:6, 8:14] = 9  # touches 3 along x=7|8

    dilated = dilate_labels(lambda t: labels, iterations=1)(labels)

    assert ((labels == 3) <= (dilated == 3)).all()
    assert ((labels == 9) <= (dilated == 9)).all()
    assert (dilated > 0).sum() > (labels > 0).sum()


def test_fill_holes_fills_enclosed_background_only():
    from patchworks import fill_holes

    labels = np.zeros((1, 12, 12), "int32")
    labels[0, 2:7, 2:7] = 4
    labels[0, 4, 4] = 0  # a hole: enclosed by object 4
    labels[0, 8:12, 8:12] = 6
    labels[0, 10, 11] = 0  # not a hole: touches the tile border
    out = fill_holes(lambda t: labels.copy(), per_plane=True)(labels)
    assert out[0, 4, 4] == 4
    assert out[0, 10, 11] == 0
    assert (out[labels > 0] == labels[labels > 0]).all()


def test_fill_holes_3d_vs_per_plane():
    """A ring through a thin stack is a hole only plane by plane."""
    from patchworks import fill_holes

    ring = np.zeros((3, 9, 9), "int32")
    ring[:, 2:7, 2:7] = 1
    ring[:, 4, 4] = 0  # a tube: open at the stack's top and bottom
    assert fill_holes(lambda t: ring.copy())(ring)[1, 4, 4] == 0
    assert fill_holes(lambda t: ring.copy(), per_plane=True)(ring)[1, 4, 4] == 1


def test_open_labels_cuts_spurs_and_keeps_neighbours_apart():
    from patchworks import open_labels

    labels = np.zeros((1, 16, 20), "int32")
    labels[0, 3:12, 2:9] = 1  # a blob ...
    labels[0, 7, 9:15] = 1  # ... with a one-voxel spur
    labels[0, 3:12, 15:19] = 2  # a touching neighbour
    out = open_labels(lambda t: labels.copy(), radius=1)(labels)
    assert (out[0, 7, 10:14] == 0).all()  # spur gone
    assert (out[0, 5:10, 4:7] == 1).all()  # body kept
    assert (out[labels == 2] == 2).all()
    assert open_labels(lambda t: t, radius=0)(labels) is labels
