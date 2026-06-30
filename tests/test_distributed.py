"""Tests for the per-tile distributed building blocks."""

import numpy as np

from patchworks import (
    create_stage,
    merge_tile_labels,
    spatial_tiles,
    stage_tile,
)


def _fn(tile):
    from skimage.measure import label

    return label(tile > 0).astype("int32")


def test_spatial_tiles_cover():
    """Tiles tile the array exactly, in row-major order."""
    tiles = spatial_tiles((4, 5), (2, 2))
    assert len(tiles) == 2 * 3  # ceil(4/2) * ceil(5/2)
    assert tiles[0] == (slice(0, 2), slice(0, 2))
    assert tiles[-1] == (slice(2, 4), slice(4, 5))  # clipped last tile


def test_stage_then_merge_stitches_boundary(tmp_path):
    """Per-tile staging + merge reproduces a cross-boundary single object."""
    img = np.zeros((16, 32), "uint16")
    img[4:12, 8:24] = 500  # block straddling the x=16 tile boundary

    stage = str(tmp_path / "stage.zarr")
    tile = (16, 16)
    create_stage(stage, img.shape, tile)
    for i in range(len(spatial_tiles(img.shape, tile))):
        stage_tile(img, _fn, stage, i, tile_shape=tile, overlap=4)

    merged = merge_tile_labels(
        stage,
        write_to=str(tmp_path / "out.zarr"),
        input_component="staged",
        sequential_labels=True,
    ).compute()
    ids = np.unique(merged[merged > 0])
    assert ids.size == 1, f"object split into {ids.size} labels"


def test_separate_objects_keep_distinct_labels(tmp_path):
    """Different objects in different tiles must NOT collapse to one label.

    Each tile produces local labels (1..N); without the merge's global-uniqueness
    pass every tile's "1" would fuse into a single object.
    """
    img = np.zeros((16, 32), "uint16")
    # four separate objects, one per (16x16) tile, none touching a boundary
    img[3:6, 3:6] = 500
    img[3:6, 19:22] = 500
    img[10:13, 3:6] = 500
    img[10:13, 19:22] = 500

    stage = str(tmp_path / "stage.zarr")
    tile = (16, 16)
    create_stage(stage, img.shape, tile)
    for i in range(len(spatial_tiles(img.shape, tile))):
        stage_tile(img, _fn, stage, i, tile_shape=tile, overlap=4)

    merged = merge_tile_labels(
        stage,
        write_to=str(tmp_path / "out.zarr"),
        input_component="staged",
        sequential_labels=True,
    ).compute()
    ids = np.unique(merged[merged > 0])
    assert ids.size == 4, f"expected 4 distinct objects, got {ids.size}"
    assert set(ids.tolist()) == {1, 2, 3, 4}  # contiguous after relabel
