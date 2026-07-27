"""Tests for exact tile occupancy via the max-pooled summary map."""

import numpy as np
import pytest
import zarr

from patchworks import (
    block_for_tile,
    build_occupancy_map,
    estimate_empty_tiles,
    occupancy_path,
    tile_occupancy,
)


def _write_store(tmp_path, data):
    """Write a minimal single-level OME-ZARR holding ``(c, z, y, x)`` data."""
    store = str(tmp_path / "image.zarr")
    root = zarr.open_group(store, mode="w")
    root.create_array(
        name="0", shape=data.shape, chunks=(1, 2, 32, 32), dtype=data.dtype
    )
    root["0"][:] = data
    root.attrs["multiscales"] = [
        {
            "datasets": [{"path": "0"}],
            "axes": [{"name": a} for a in "czyx"],
        }
    ]
    return store


def test_corner_signal_is_found(tmp_path):
    """A voxel in a tile's extreme corner must mark that tile occupied.

    The centred-window sampler reads only the middle of each tile, so this is
    exactly the signal it cannot see -- and in the workflow an unseen tile is
    never segmented.
    """
    data = np.zeros((1, 4, 64, 64), "uint16")
    data[0, 0, 63, 63] = 1000  # last tile, extreme corner
    store = _write_store(tmp_path, data)
    tile = (4, 32, 32)

    build_occupancy_map(store, block=(1, 8, 8))
    info = tile_occupancy(store, tile, channel=0, threshold=100)

    assert info["occupancy"].shape == (1, 2, 2)
    assert info["occupancy"][0, 1, 1], "corner signal missed"
    assert info["n_occupied"] == 1, "only the corner tile holds signal"

    # The sampler this replaces cannot see it: its window covers the tile
    # centre only, so the corner voxel never enters the comparison.
    sampled = estimate_empty_tiles(
        data[0], tile, threshold=100, sample_window=(4, 8, 8)
    )
    assert not sampled["occupancy"][0, 1, 1]


def test_edge_tile_verdict_uses_only_its_own_extent(tmp_path):
    """A partial edge tile must not inherit its neighbour's signal.

    The old clamp (``min(start, s - w)``) slid the last tile's sampling window
    backwards into the previous tile whenever the last tile was narrower than
    the centring offset.
    """
    # x = 40 with tile 32 -> tiles [0,32) and [32,40): the last one is 8 wide.
    data = np.zeros((1, 2, 8, 40), "uint16")
    data[0, :, :, :32] = 1000  # signal lives entirely in the FIRST tile
    tile = (2, 8, 32)

    sampled = estimate_empty_tiles(
        data[0], tile, threshold=100, sample_window=(2, 8, 16)
    )
    assert sampled["occupancy"][0, 0, 0], "first tile genuinely has signal"
    assert not sampled["occupancy"][0, 0, 1], (
        "edge tile is empty; its verdict must not come from its neighbour"
    )


def test_occupancy_matches_a_full_scan(tmp_path):
    """Pooled maxima give the same verdicts as testing every voxel.

    ``block_max > t`` holds exactly when some voxel in the block exceeds
    ``t``, so this is an equality, not an approximation.
    """
    rng = np.random.default_rng(0)
    data = (rng.random((2, 4, 64, 64)) * 50).astype("uint16")
    data[0, 1, 5, 5] = 900
    data[0, 2, 40, 60] = 900
    data[1, 3, 33, 3] = 900
    store = _write_store(tmp_path, data)
    tile = (4, 32, 32)
    threshold = 500

    build_occupancy_map(store, block=(1, 8, 8))
    for channel in (0, 1):
        info = tile_occupancy(store, tile, channel=channel, threshold=threshold)
        expected = np.zeros(info["occupancy"].shape, dtype=bool)
        for idx in np.ndindex(*expected.shape):
            sl = tuple(
                slice(i * t, min((i + 1) * t, s))
                for i, t, s in zip(idx, tile, data.shape[1:])
            )
            expected[idx] = data[(channel, *sl)].max() > threshold
        assert np.array_equal(info["occupancy"], expected), (
            f"channel {channel} disagrees with a full scan"
        )


def test_build_is_idempotent_and_covers_ragged_edges(tmp_path):
    """A second build reuses the map; blocks past the edge don't invent signal."""
    # 20 is not a multiple of the block size 8 -> the last block is padded.
    data = np.zeros((1, 1, 20, 20), "uint16")
    data[0, 0, 19, 19] = 700
    store = _write_store(tmp_path, data)

    path = build_occupancy_map(store, block=(1, 8, 8))
    assert path == occupancy_path(store, 0)
    pooled = np.asarray(zarr.open_array(path, mode="r")[0])
    assert pooled.shape == (1, 3, 3), "ceil(20/8) = 3 blocks per axis"
    assert pooled[0, 2, 2] == 700, "ragged edge block keeps its maximum"
    assert pooled.max() == 700, "padding must not invent a larger value"

    assert build_occupancy_map(store, block=(1, 8, 8)) == path

    # ...but a *different* block must rebuild rather than serve a stale map,
    # or changing tile_shape would silently keep the old resolution forever.
    build_occupancy_map(store, block=(1, 4, 4))
    assert tuple(zarr.open_array(path, mode="r").attrs["block"]) == (1, 4, 4)

    # And an explicit overwrite must actually take effect: the guard against
    # a concurrent build used to discard the freshly built map.
    build_occupancy_map(store, block=(1, 8, 8), overwrite=True)
    assert tuple(zarr.open_array(path, mode="r").attrs["block"]) == (1, 8, 8)


def test_map_lives_outside_the_image_store(tmp_path):
    """The map must not make the image store an invalid zarr hierarchy.

    It is not an NGFF array, so zarr refuses to walk a hierarchy containing
    it -- putting it inside image.zarr made every members()/arrays() call on
    the user's own image raise a ZarrUserWarning.
    """
    import warnings

    data = np.zeros((1, 2, 16, 16), "uint16")
    store = _write_store(tmp_path, data)
    path = build_occupancy_map(store, block=(1, 8, 8))

    assert not path.startswith(store + "/"), "map must be a sibling, not a node"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        members = sorted(
            k for k, _ in zarr.open_group(store, mode="r").members()
        )
    assert members == ["0"], (
        f"image store should hold only its levels: {members}"
    )


def test_all_channels_built_in_one_traversal(tmp_path):
    """Every channel is filled, and the source is read once, not once each.

    Looping channels on the outside traversed the whole image per channel --
    three full reads for a three-channel stack.
    """
    data = np.zeros((3, 2, 16, 16), "uint16")
    for c in range(3):
        data[c, 0, 0, 0] = 100 * (c + 1)  # a distinct value per channel
    store = _write_store(tmp_path, data)

    reads = {"n": 0}
    real_getitem = zarr.Array.__getitem__

    def _counting(self, key):
        if self.basename == "0":  # the image level, not the map being written
            reads["n"] += 1
        return real_getitem(self, key)

    zarr.Array.__getitem__ = _counting
    try:
        build_occupancy_map(store, block=(1, 8, 8))
    finally:
        zarr.Array.__getitem__ = real_getitem

    pooled = np.asarray(zarr.open_array(occupancy_path(store, 0), mode="r"))
    assert pooled.shape[0] == 3
    for c in range(3):
        assert pooled[c].max() == 100 * (c + 1), f"channel {c} not built"

    # 2x2 blocks per plane x 2 planes = 4 output regions... but the region
    # step is sized to ~128 MB, so this tiny image is a single region: one
    # read per channel, and no more.
    assert reads["n"] == 3, (
        f"expected one read per channel, got {reads['n']} -- the image is "
        "being traversed more than once"
    )


def test_block_must_be_finer_than_the_tile(tmp_path):
    """A block as coarse as the tile makes every tile test occupied.

    The map can only resolve whole blocks, so one block spanning the tile is
    over-covered by every tile -- correct, but useless as a skip list. This is
    what silently disabled skip_empty on smaller images.
    """
    assert block_for_tile((16, 1024, 1024)) == (1, 128, 128)
    assert block_for_tile((8, 32, 32)) == (1, 8, 8)
    assert block_for_tile((1, 4, 4)) == (1, 1, 1)  # never below 1

    # A tile-sized block marks everything occupied...
    data = np.zeros((1, 8, 128, 128), "uint16")
    data[0, 2:5, 4:20, 4:20] = 900  # one blob in one corner
    store = _write_store(tmp_path, data)
    tile = (8, 32, 32)

    build_occupancy_map(store, block=(1, 128, 128))
    coarse = tile_occupancy(store, tile, channel=0, threshold=100)
    assert coarse["n_occupied"] == coarse["n_tiles"], "coarse blocks over-cover"

    # ...while a tile-derived block finds only the tile that has signal.
    build_occupancy_map(store, block=block_for_tile(tile), overwrite=True)
    fine = tile_occupancy(store, tile, channel=0, threshold=100)
    assert fine["n_occupied"] == 1, f"expected 1 occupied, got {fine}"


def test_tile_shape_dimensionality_is_checked(tmp_path):
    data = np.zeros((1, 2, 16, 16), "uint16")
    store = _write_store(tmp_path, data)
    build_occupancy_map(store, block=(1, 8, 8))
    with pytest.raises(ValueError, match="occupancy map is"):
        tile_occupancy(store, (16, 16), channel=0, threshold=1)
