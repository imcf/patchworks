import dask.array as da
import numpy as np
import pytest

from patchworks import label_relations


def test_label_relations_majority_overlap():
    # nucleus 1 sits fully inside cell 10; nucleus 2 straddles cells 20/21
    # with more voxels in 20 -> should match 20, not 21.
    a = np.zeros((4, 10), dtype=np.int32)
    a[0:2, 0:2] = 1
    a[0:2, 4:8] = 2
    b = np.zeros((4, 10), dtype=np.int32)
    b[0:3, 0:3] = 10
    b[0:2, 4:6] = 20  # 4 voxels overlap with label 2
    b[0:2, 6:8] = 21  # 4 voxels overlap with label 2 -> tie, but let's skew it
    b[0:2, 6:7] = 20  # tip the tie: label 20 now has 6 voxels vs 21's 2

    table = label_relations(da.from_array(a, chunks=(2, 5)), da.from_array(b, chunks=(2, 5)))

    assert table[1]["match"] == 10
    assert table[1]["overlap_voxels"] == 4
    assert table[1]["overlap_fraction"] == 1.0

    assert table[2]["match"] == 20
    assert table[2]["overlap_voxels"] == 6


def test_label_relations_no_overlap_omitted():
    a = np.zeros((2, 2), dtype=np.int32)
    a[0, 0] = 1
    b = np.zeros((2, 2), dtype=np.int32)  # all background, no overlap anywhere

    table = label_relations(da.from_array(a, chunks=(2, 2)), da.from_array(b, chunks=(2, 2)))
    assert table == {}


def test_label_relations_chunk_mismatch_raises():
    a = da.zeros((4, 4), chunks=(2, 4), dtype=np.int32)
    b = da.zeros((4, 4), chunks=(4, 4), dtype=np.int32)
    with pytest.raises(ValueError, match="chunk layout"):
        label_relations(a, b)
