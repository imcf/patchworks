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

    table = label_relations(
        da.from_array(a, chunks=(2, 5)), da.from_array(b, chunks=(2, 5))
    )

    assert table[1]["match"] == 10
    assert table[1]["overlap_voxels"] == 4
    assert table[1]["overlap_fraction"] == 1.0

    assert table[2]["match"] == 20
    assert table[2]["overlap_voxels"] == 6


def test_label_relations_no_overlap_omitted():
    a = np.zeros((2, 2), dtype=np.int32)
    a[0, 0] = 1
    b = np.zeros((2, 2), dtype=np.int32)  # all background, no overlap anywhere

    table = label_relations(
        da.from_array(a, chunks=(2, 2)), da.from_array(b, chunks=(2, 2))
    )
    assert table == {}


def test_label_relations_chunk_mismatch_raises():
    a = da.zeros((4, 4), chunks=(2, 4), dtype=np.int32)
    b = da.zeros((4, 4), chunks=(4, 4), dtype=np.int32)
    with pytest.raises(ValueError, match="chunk layout"):
        label_relations(a, b)


def test_label_relations_reports_progress(caplog, monkeypatch):
    """A multi-hour scan must say it is alive, not just print at the end.

    `ex.map` returns results in submission order, so one slow early chunk
    withheld every later result and the log stayed empty however many had
    actually finished -- a real run showed two lines after 1.5 hours,
    indistinguishable from a hang.
    """
    import dask.array as da
    import numpy as np

    from patchworks import _relations

    # Report on every chunk rather than once a minute, so the mechanism is
    # observable in a test instead of only in an hours-long job.
    monkeypatch.setattr(_relations, "_PROGRESS_INTERVAL_S", 0.0)

    rng = np.random.default_rng(0)
    a = da.from_array(
        rng.integers(0, 5, (8, 64, 64), dtype="uint32"), chunks=(4, 32, 32)
    )
    b = da.from_array(
        rng.integers(0, 5, (8, 64, 64), dtype="uint32"), chunks=(4, 32, 32)
    )

    with caplog.at_level("INFO"):
        _relations.label_relations(a, b)

    text = caplog.text
    assert "scanning 8 chunk(s)" in text
    # Periodic lines while it runs, including a final 100%.
    assert "label_relations: 1/8" in text
    assert "8/8 (100%)" in text


def test_label_relations_is_independent_of_completion_order():
    """Switching to as_completed must not change a single row.

    Chunks now land in whatever order they finish rather than in submission
    order, so the merge downstream has to be order-insensitive.
    """
    import dask.array as da
    import numpy as np

    from patchworks import label_relations

    rng = np.random.default_rng(3)
    a_arr = rng.integers(0, 12, (12, 96, 96), dtype="uint32")
    b_arr = rng.integers(0, 12, (12, 96, 96), dtype="uint32")
    a = da.from_array(a_arr, chunks=(4, 32, 32))
    b = da.from_array(b_arr, chunks=(4, 32, 32))

    assert label_relations(a, b, n_workers=1) == label_relations(
        a, b, n_workers=8
    )
