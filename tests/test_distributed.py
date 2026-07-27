"""Tests for the per-tile distributed building blocks."""

import numpy as np
import pytest

from patchworks import (
    auto_overlap,
    create_stage,
    merge_tile_labels,
    normalize_overlap,
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


def test_normalize_overlap_scalar_and_per_axis():
    """A scalar spreads to every axis; a sequence is taken as-is."""
    assert normalize_overlap(4, 3) == (4, 4, 4)
    assert normalize_overlap([1, 30, 30], 3) == (1, 30, 30)
    with pytest.raises(ValueError, match="entries but the tile"):
        normalize_overlap([1, 30], 3)
    with pytest.raises(ValueError, match="non-negative"):
        normalize_overlap([-1, 0, 0], 3)


def test_auto_overlap_anisotropy_shrinks_z():
    """A 20x coarser z step needs 20x fewer planes for the same distance."""
    assert auto_overlap(30) == 30  # unchanged without voxel_size
    z, y, x = auto_overlap(30, voxel_size=(2.0, 0.1, 0.1))
    assert (y, x) == (30, 30)
    assert z == 2, f"expected 2 z-planes for 3 um at a 2 um step, got {z}"


def test_per_axis_overlap_reads_less_but_still_stitches(tmp_path):
    """A thin z-halo keeps the boundary merge correct on an anisotropic tile.

    The x-halo still has to cover the object, but z carries no boundary here,
    so a per-axis overlap trims the read without changing the result.
    """
    img = np.zeros((4, 16, 32), "uint16")
    img[1:3, 4:12, 8:24] = 500  # straddles the x=16 tile boundary

    tile = (4, 16, 16)
    results = {}
    for name, overlap in (("scalar", 6), ("per_axis", [0, 6, 6])):
        stage = str(tmp_path / f"stage_{name}.zarr")
        create_stage(stage, img.shape, tile)
        for i in range(len(spatial_tiles(img.shape, tile))):
            stage_tile(img, _fn, stage, i, tile_shape=tile, overlap=overlap)
        results[name] = merge_tile_labels(
            stage,
            write_to=str(tmp_path / f"out_{name}.zarr"),
            input_component="staged",
            sequential_labels=True,
        ).compute()

    for name, merged in results.items():
        ids = np.unique(merged[merged > 0])
        assert ids.size == 1, f"{name}: object split into {ids.size} labels"
    assert np.array_equal(results["scalar"], results["per_axis"])


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
