"""Tests for the global, post-merge label volume filter."""

from pathlib import Path

import numpy as np
import zarr


def test_voxel_volume_multiplies_given_axes():
    from patchworks import voxel_volume

    assert voxel_volume({"z": 0.24, "y": 0.10833, "x": 0.10833}) == (
        0.24 * 0.10833 * 0.10833
    )


def test_voxel_volume_missing_axis_treated_as_one():
    from patchworks import voxel_volume

    assert voxel_volume({"y": 0.5, "x": 0.5}) == 0.25


def test_min_voxels_for_volume_rounds_up():
    from patchworks import min_voxels_for_volume

    calibration = {"z": 0.24, "y": 0.10833, "x": 0.10833}
    assert min_voxels_for_volume(5.0, calibration) == 1776


def test_max_voxels_for_volume_rounds_down():
    from patchworks import max_voxels_for_volume

    calibration = {"z": 0.24, "y": 0.10833, "x": 0.10833}
    assert max_voxels_for_volume(5.0, calibration) == 1775


def _write_labels(path, array, chunks):
    root = zarr.open_group(path, mode="w")
    arr = root.create_array(
        "labels", shape=array.shape, chunks=chunks, dtype="int32"
    )
    arr[:] = array
    return root


def test_filter_labels_by_size_drops_small_objects(tmp_path):
    from patchworks import filter_labels_by_size

    array = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 2],
            [0, 0, 0, 0],
        ]
    )
    path = str(tmp_path / "labels.zarr")
    root = _write_labels(path, array, chunks=(4, 4))

    n_kept, n_removed = filter_labels_by_size(path, "labels", min_voxels=2)

    assert (n_kept, n_removed) == (1, 1)
    assert np.array_equal(
        np.asarray(root["labels"]),
        [
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
    )


def test_filter_labels_by_size_relabels_surviving_ids_sequentially(tmp_path):
    """Dropping id 1 must not leave id 2 with a gap where 1 used to be."""
    from patchworks import filter_labels_by_size

    array = np.array(
        [
            [1, 0, 2, 2],
            [0, 0, 2, 2],
        ]
    )
    path = str(tmp_path / "labels.zarr")
    root = _write_labels(path, array, chunks=(2, 4))

    filter_labels_by_size(path, "labels", min_voxels=2)

    assert np.array_equal(
        np.asarray(root["labels"]),
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
    )


def test_filter_labels_by_size_relabel_false_keeps_original_ids(tmp_path):
    from patchworks import filter_labels_by_size

    array = np.array(
        [
            [1, 0, 5, 5],
            [0, 0, 5, 5],
        ]
    )
    path = str(tmp_path / "labels.zarr")
    root = _write_labels(path, array, chunks=(2, 4))

    filter_labels_by_size(path, "labels", min_voxels=2, relabel=False)

    assert np.array_equal(
        np.asarray(root["labels"]),
        [
            [0, 0, 5, 5],
            [0, 0, 5, 5],
        ],
    )


def test_filter_labels_by_size_works_across_multiple_chunks(tmp_path):
    """A removed object's id can exceed every surviving id and still must

    stay in bounds of the LUT built from the largest id *seen*, not just
    the largest surviving one -- this is the case a single-chunk array
    can't exercise, since chunking is what makes the scan streaming at all.
    """
    from patchworks import filter_labels_by_size

    array = np.zeros((4, 4), dtype="int32")
    array[0:2, 0:2] = 1  # 4 voxels, kept
    array[2:4, 2:4] = 2  # 4 voxels, kept
    array[0, 3] = (
        3  # 1 voxel, dropped -- id 3 exceeds nothing survives above it
    )
    path = str(tmp_path / "labels.zarr")
    root = _write_labels(path, array, chunks=(2, 2))

    n_kept, n_removed = filter_labels_by_size(path, "labels", min_voxels=2)

    assert (n_kept, n_removed) == (2, 1)
    out = np.asarray(root["labels"])
    assert set(np.unique(out).tolist()) == {0, 1, 2}
    assert out[0, 3] == 0


def test_filter_labels_by_size_drops_large_objects(tmp_path):
    from patchworks import filter_labels_by_size

    array = np.array(
        [
            [1, 0, 2, 2],
            [0, 0, 2, 2],
        ]
    )
    path = str(tmp_path / "labels.zarr")
    root = _write_labels(path, array, chunks=(2, 4))

    n_kept, n_removed = filter_labels_by_size(path, "labels", max_voxels=2)

    assert (n_kept, n_removed) == (1, 1)
    assert np.array_equal(
        np.asarray(root["labels"]),
        [
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ],
    )


def test_filter_labels_by_size_min_and_max_together_keeps_the_middle(tmp_path):
    from patchworks import filter_labels_by_size

    array = np.array([1, 0, 2, 2, 0, 3, 3, 3])  # sizes: 1, 2, 3
    path = str(tmp_path / "labels.zarr")
    root = _write_labels(path, array, chunks=(8,))

    n_kept, n_removed = filter_labels_by_size(
        path, "labels", min_voxels=2, max_voxels=2, relabel=False
    )

    assert (n_kept, n_removed) == (1, 2)
    assert np.array_equal(np.asarray(root["labels"]), [0, 0, 2, 2, 0, 0, 0, 0])


def test_filter_labels_by_size_requires_at_least_one_bound(tmp_path):
    import pytest
    from patchworks import filter_labels_by_size

    array = np.array([1, 1, 0, 2])
    path = str(tmp_path / "labels.zarr")
    _write_labels(path, array, chunks=(4,))

    with pytest.raises(ValueError, match="min_voxels.*max_voxels"):
        filter_labels_by_size(path, "labels")


def test_size_filter_skips_background_chunks_and_matches_serial(tmp_path):
    """Threads give the serial result, objects spanning chunks are counted
    whole, and a background chunk is neither rewritten nor materialised."""
    from patchworks import filter_labels_by_size

    rng = np.random.default_rng(0)
    array = np.zeros((32, 32), "uint32")
    array[2:6, 2:14] = 7  # 48 voxels across two chunks: kept
    array[20:22, 20:22] = 3  # 4 voxels: dropped
    array[9:12, 25:28] = rng.integers(10, 12, (3, 3))  # small specks
    results = []
    for workers in (1, 4):
        path = str(tmp_path / f"labels{workers}.zarr")
        root = zarr.open_group(path, mode="w")
        arr = root.create_array(
            "labels", shape=array.shape, chunks=(8, 8), dtype="uint32"
        )
        arr[:] = array
        chunks = Path(path, "labels", "c")
        before = sorted(p for p in chunks.rglob("*") if p.is_file())
        assert 0 < len(before) < 16  # zarr stores only non-empty chunks
        out = filter_labels_by_size(
            path, "labels", min_voxels=10, n_workers=workers
        )
        after = sorted(p for p in chunks.rglob("*") if p.is_file())
        assert set(after) <= set(before)  # no background chunk written
        results.append((out, np.asarray(root["labels"])))
    (out1, lab1), (out4, lab4) = results
    assert out1 == out4 and np.array_equal(lab1, lab4)
    assert set(np.unique(lab1).tolist()) == {0, 1}
    assert (lab1[2:6, 2:14] == 1).all()
