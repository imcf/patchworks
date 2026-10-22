"""Merge internals: stitching across tile boundaries."""

import numpy as np


def test_pair_stats_by_key_matches_unique_over_rows():
    """The int64-key pair count is the row-unique one, order included."""
    from patchworks._merge import _pair_stats

    rng = np.random.default_rng(3)
    a = rng.integers(0, 50, (30, 40)) * 1_000_003
    b = rng.integers(0, 60, (30, 40)) * 7
    pairs, inter, a_area, b_area = _pair_stats(a, b)
    av, bv = a.ravel().astype(np.int64), b.ravel().astype(np.int64)
    both = (av > 0) & (bv > 0)
    want, want_n = np.unique(
        np.stack([av[both], bv[both]], 1), axis=0, return_counts=True
    )
    np.testing.assert_array_equal(pairs, want)
    np.testing.assert_array_equal(inter, want_n)
    assert sum(a_area.values()) == int((av > 0).sum())
