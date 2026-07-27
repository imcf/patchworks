"""Tests for the per-tile distributed building blocks."""

from pathlib import Path

import numpy as np
import pytest
import zarr

from patchworks import (
    auto_overlap,
    capped_output_chunks,
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


def test_capped_output_chunks_never_shreds_an_axis():
    """A tile axis with no usable divisor keeps its size, not chunk-size 1.

    `auto` tile sizing takes a square root, so an axis can be any integer. A
    prime one above the cap has no divisor but 1 -- which would mean one chunk
    per voxel along that axis.
    """
    assert capped_output_chunks((16, 2048, 2048), (16, 1024, 1024)) == (
        16,
        1024,
        1024,
    )
    assert capped_output_chunks((16, 1024, 1024), (16, 1024, 1024)) == (
        16,
        1024,
        1024,
    )
    assert capped_output_chunks((1500,), (1024,)) == (750,)  # a real divisor
    for prime in (1031, 1093):
        assert capped_output_chunks((prime,), (1024,)) == (prime,)

    # Whatever it returns must still divide the tile, or concurrent workers
    # would read-modify-write a chunk they share.
    for tile in (1024, 2048, 1500, 1031, 697, 3000):
        (chunk,) = capped_output_chunks((tile,), (1024,))
        assert tile % chunk == 0, f"{chunk} does not divide {tile}"


def _relabel_canonical(arr):
    """Canonical form so two labellings can be compared up to renumbering."""
    out = np.zeros_like(arr, dtype=np.int64)
    for new, old in enumerate(
        sorted(np.unique(arr[arr > 0]).tolist()), start=1
    ):
        out[arr == old] = new
    return out


def test_offsets_match_the_renumber_pass(tmp_path):
    """Per-tile counts must produce the same labelling as the scan pass.

    The counts path replaces _make_globally_unique -- a full read+write of the
    store -- with a cumulative sum. This is the guard that deleting that pass
    changed nothing observable.
    """
    img = np.zeros((8, 48), "uint16")
    img[1:4, 4:12] = 500  # inside tile 0
    img[2:6, 14:26] = 500  # straddles the x=16 boundary
    img[5:7, 30:44] = 500  # straddles the x=32 boundary
    tile = (8, 16)

    merged = {}
    for name, use_counts in (("scan", False), ("counts", True)):
        stage = str(tmp_path / f"stage_{name}.zarr")
        create_stage(stage, img.shape, tile)
        counts = {}
        for i in range(len(spatial_tiles(img.shape, tile))):
            counts[i] = stage_tile(
                img, _fn, stage, i, tile_shape=tile, overlap=2
            )
        merged[name] = merge_tile_labels(
            stage,
            write_to=str(tmp_path / f"out_{name}.zarr"),
            input_component="staged",
            sequential_labels=True,
            label_counts=counts if use_counts else None,
        ).compute()

    assert np.array_equal(
        _relabel_canonical(merged["scan"]), _relabel_canonical(merged["counts"])
    ), "offset path disagrees with the renumber pass"
    ids = np.unique(merged["counts"][merged["counts"] > 0])
    assert set(ids.tolist()) == {1, 2, 3}, f"expected 3 objects, got {ids}"


def test_empty_chunks_are_skipped_not_just_short_circuited(tmp_path):
    """Chunks with no labels must cost neither I/O nor disk.

    A count of 0 already tells the merge the chunk is background, so it need
    not be read at all -- and leaving the output chunk unwritten means zarr
    never materialises it.
    """
    img = np.zeros((8, 64), "uint16")
    img[1:4, 1:7] = 500  # wholly inside tile 0; the other seven are empty
    tile = (8, 8)

    stage = str(tmp_path / "stage.zarr")
    create_stage(stage, img.shape, tile)
    counts = {
        i: stage_tile(img, _fn, stage, i, tile_shape=tile, overlap=0)
        for i in range(len(spatial_tiles(img.shape, tile)))
    }
    assert sum(1 for n in counts.values() if n == 0) == 7, "7 empty tiles"

    out = str(tmp_path / "out.zarr")
    merged = merge_tile_labels(
        stage,
        write_to=out,
        input_component="staged",
        sequential_labels=True,
        label_counts=counts,
    ).compute()

    # Correctness first: the one real object survives, everything else is 0.
    assert set(np.unique(merged).tolist()) == {0, 1}
    assert merged[1:4, 1:7].min() == 1

    # And the empty chunks were never written: zarr stores only the chunks it
    # was given, so the skipped ones leave no file behind.
    chunk_dir = Path(out) / "labels" / "c" / "0"
    written = sorted(p.name for p in chunk_dir.iterdir())
    assert written == ["0"], f"only the occupied chunk should exist: {written}"


def test_in_place_merge_matches_the_two_store_merge(tmp_path):
    """Relabelling back into the source must give the same labelling.

    It is safe because the boundary scan finishes before any chunk is
    rewritten, so nothing still needs the original ids -- and it saves writing
    the whole volume a second time.
    """
    img = np.zeros((8, 48), "uint16")
    img[1:4, 4:12] = 500  # inside tile 0
    img[2:6, 14:26] = 500  # straddles the x=16 boundary
    tile = (8, 16)

    results = {}
    for name in ("two_store", "in_place"):
        stage = str(tmp_path / f"stage_{name}.zarr")
        create_stage(stage, img.shape, tile)
        counts = {
            i: stage_tile(img, _fn, stage, i, tile_shape=tile, overlap=2)
            for i in range(len(spatial_tiles(img.shape, tile)))
        }
        # in place = the merge's output *is* its input
        out = (
            stage if name == "in_place" else str(tmp_path / f"out_{name}.zarr")
        )
        component = "staged" if name == "in_place" else "labels"
        results[name] = merge_tile_labels(
            stage,
            write_to=out,
            input_component="staged",
            output_component=component,
            sequential_labels=True,
            label_counts=counts,
        ).compute()

    assert np.array_equal(results["two_store"], results["in_place"]), (
        "in-place relabelling changed the result"
    )
    assert set(np.unique(results["in_place"]).tolist()) == {0, 1, 2}

    # And it must refuse a rechunk it cannot honour, rather than silently
    # ignoring it: in place the array is rewritten, never recreated.
    stage = str(tmp_path / "stage_guard.zarr")
    create_stage(stage, img.shape, tile)
    with pytest.raises(ValueError, match="cannot differ"):
        merge_tile_labels(
            stage,
            write_to=stage,
            input_component="staged",
            output_component="staged",
            output_chunks=(8, 8),
        )


def test_stage_tile_returns_dense_label_count(tmp_path):
    """stage_tile reports how many labels it wrote, and writes exactly 1..n."""
    img = np.zeros((8, 16), "uint16")
    img[1:3, 1:3] = 500
    img[5:7, 10:13] = 500
    stage = str(tmp_path / "stage.zarr")
    create_stage(stage, img.shape, (8, 16))

    n = stage_tile(img, _fn, stage, 0, tile_shape=(8, 16), overlap=0)
    assert n == 2

    written = np.asarray(zarr.open_group(stage, mode="r")["staged"])
    assert set(np.unique(written).tolist()) == {0, 1, 2}, "ids must be dense"


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
