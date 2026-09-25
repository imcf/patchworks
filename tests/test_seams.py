"""Tests for the seam quality report."""

import numpy as np
import zarr

from patchworks import seam_report


def _blobs(shape=(1, 128, 128), n=150, r=3, seed=0):
    from skimage.draw import disk
    from skimage.measure import label

    rng = np.random.default_rng(seed)
    mask = np.zeros(shape[1:], bool)
    for cy, cx in rng.integers(r, shape[1] - r, (n, 2)):
        mask[disk((cy, cx), r, shape=mask.shape)] = True
    return label(mask).astype("int32")[None]


def _store(tmp_path, arr, name):
    z = zarr.open_group(str(tmp_path / name), mode="w").create_array(
        "0", shape=arr.shape, chunks=arr.shape, dtype=arr.dtype
    )
    z[:] = arr
    return str(tmp_path / name)


def test_clean_stitching_shows_no_seams(tmp_path):
    report = seam_report(_store(tmp_path, _blobs(), "clean.zarr"), (1, 32, 32))
    x = report["axes"][2]
    assert x["seam_labels"] > 0
    assert x["seam_rate"] <= 2 * x["interior_rate"] + 0.05
    assert 0 not in report["axes"]  # a one-plane axis has no seams


def test_one_plane_tiles_have_no_interior_to_compare(tmp_path):
    stack = np.concatenate([_blobs(seed=s) for s in range(4)])
    report = seam_report(_store(tmp_path, stack, "z.zarr"), (1, 64, 64))
    assert report["axes"][0]["interior_rate"] is None
    assert report["axes"][0]["ratio"] is None


def test_objects_chopped_at_seams_are_reported(tmp_path, caplog):
    labels = _blobs()
    # Every object crossing an x seam loses the part beyond it, as when the
    # neighbouring tile did not see it.
    for pos in range(32, 128, 32):
        crossing = np.intersect1d(
            labels[0, :, pos - 1][labels[0, :, pos - 1] > 0],
            labels[0, :, pos][labels[0, :, pos] > 0],
        )
        right = labels[0, :, pos : pos + 32]
        right[np.isin(right, crossing)] = 0
    report = seam_report(_store(tmp_path, labels, "chopped.zarr"), (1, 32, 32))
    x = report["axes"][2]
    assert x["seam_rate"] > 0.5
    assert x["ratio"] > 2
    assert any(f["axis"] == 2 for f in report["worst_seams"][:5])
    assert "the tiling shows" in caplog.text
