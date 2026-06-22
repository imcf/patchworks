"""Self-contained tests for patchworks. No frameworks, no fixtures."""
import numpy as np
import pytest


def _make_image(shape=(4, 64, 64), dtype="uint16"):
    rng = np.random.default_rng(42)
    return rng.integers(0, 1000, shape, dtype=dtype)


def _label_fn(tile: np.ndarray) -> np.ndarray:
    """Simple threshold segmentation for testing."""
    from skimage.measure import label as sk_label
    binary = tile > tile.mean()
    return sk_label(binary).astype("int32")


def test_tile_process_numpy_array():
    import dask.array as da
    from patchworks import tile_process

    arr = da.from_array(_make_image((4, 64, 64)), chunks=(1, 64, 64))
    result = tile_process(arr, _label_fn).compute()
    assert result.shape == (4, 64, 64)
    assert result.dtype in (np.int32, np.int64, np.uint16, np.uint32)
    assert result.max() > 0


def test_tile_process_with_overlap():
    import dask.array as da
    from patchworks import tile_process

    arr = da.from_array(_make_image((2, 64, 64)), chunks=(1, 64, 64))
    result = tile_process(arr, _label_fn, overlap=8).compute()
    assert result.shape == (2, 64, 64)


def test_tile_process_overlap_multitile_shape():
    # Multiple tiles along y and x: exercises the halo trim across real
    # interior boundaries. Output must keep the original shape (halo trimmed).
    import dask.array as da
    from patchworks import tile_process

    arr = da.from_array(_make_image((1, 96, 96)), chunks=(1, 48, 48))
    result = tile_process(arr, _label_fn, overlap=8).compute()
    assert result.shape == (1, 96, 96)


def test_tile_process_merges_object_across_boundary():
    # An object spanning a tile boundary must end up with a single label.
    import dask.array as da
    from patchworks import tile_process

    data = np.zeros((1, 16, 32), dtype="uint16")
    data[0, 4:12, 8:24] = 500  # one solid block straddling the x=16 boundary
    arr = da.from_array(data, chunks=(1, 16, 16))

    def fn(tile):
        from skimage.measure import label
        return label(tile > 0).astype("int32")

    result = tile_process(arr, fn).compute()
    ids = np.unique(result[result > 0])
    assert ids.size == 1, f"object split into {ids.size} labels, expected 1"


def test_tile_process_write_to(tmp_path):
    import dask.array as da
    import zarr
    from patchworks import tile_process

    arr = da.from_array(_make_image((2, 32, 32)), chunks=(1, 32, 32))
    out = str(tmp_path / "labels.zarr")
    tile_process(arr, _label_fn, write_to=out, output_component="labels")

    root = zarr.open_group(out, mode="r")
    assert "labels" in root
    assert root["labels"].shape == (2, 32, 32)


def test_tile_process_skip_empty():
    import dask.array as da
    from patchworks import tile_process

    # First two tiles are zeros (empty), last two have signal
    arr_data = _make_image((4, 32, 32))
    arr_data[:2] = 0
    arr = da.from_array(arr_data, chunks=(1, 32, 32))

    call_count = [0]
    def counting_fn(tile):
        call_count[0] += 1
        return _label_fn(tile)

    tile_process(arr, counting_fn, skip_empty=True, empty_threshold=0)
    # With staging, fn is called once per non-empty tile
    assert call_count[0] == 2, f"Expected 2 fn calls, got {call_count[0]}"


def test_tile_process_sequential_labels():
    import dask.array as da
    from patchworks import tile_process

    arr = da.from_array(_make_image((2, 32, 32)), chunks=(1, 32, 32))
    result = tile_process(arr, _label_fn, sequential_labels=True).compute()
    labels = np.unique(result)
    labels = labels[labels > 0]
    # Sequential: no gaps
    assert np.all(labels == np.arange(1, len(labels) + 1))


def test_merge_tile_labels_standalone(tmp_path):
    # Standalone merge of a dask array of per-tile labels: an object straddling
    # a tile boundary must collapse to a single label.
    import dask.array as da
    from patchworks import merge_tile_labels

    data = np.zeros((1, 16, 32), dtype="uint16")
    data[0, 4:12, 8:24] = 1  # block crossing the x=16 boundary
    image = da.from_array(data, chunks=(1, 16, 16))

    def fn(tile):
        from skimage.measure import label
        return label(tile > 0).astype("int32")

    labeled = image.map_blocks(
        fn, dtype="int32", meta=np.empty((0,) * image.ndim, dtype="int32")
    )
    out = str(tmp_path / "merged.zarr")
    merged = merge_tile_labels(labeled, write_to=out, sequential_labels=True)
    arr = merged.compute()
    ids = np.unique(arr[arr > 0])
    assert ids.size == 1, f"object split into {ids.size} labels, expected 1"


def test_merge_transitive_three_tiles(tmp_path):
    # A cell that spans 3 tiles (A→B→C) must be merged into one label even
    # though A and C never directly touch. Transitivity via connected_components.
    import dask.array as da
    from patchworks._merge import zarr_native_merge
    import zarr

    sp = str(tmp_path / "stage.zarr")
    root = zarr.open_group(sp, mode="w")
    a = root.zeros("staged", shape=(3, 4, 4), chunks=(1, 4, 4), dtype=np.int32)
    a[0] = np.full((4, 4), 10)   # label 10 in tile 0
    a[1] = np.full((4, 4), 20)   # label 20 in tile 1 (touches 10 and 30)
    a[2] = np.full((4, 4), 30)   # label 30 in tile 2

    out = str(tmp_path / "out.zarr")
    zarr_native_merge(sp, "staged", out, "labels", n_workers=1)
    r = np.asarray(zarr.open_group(out)["labels"])

    assert r[0, 0, 0] == r[1, 0, 0] == r[2, 0, 0], (
        f"transitive merge failed: tile0={r[0,0,0]} tile1={r[1,0,0]} tile2={r[2,0,0]}"
    )


def test_merge_isolated_labels_not_merged(tmp_path):
    # Two cells that never touch across any boundary must stay separate.
    import dask.array as da
    from patchworks._merge import zarr_native_merge
    import zarr

    sp = str(tmp_path / "stage.zarr")
    root = zarr.open_group(sp, mode="w")
    a = root.zeros("staged", shape=(2, 4, 8), chunks=(1, 4, 4), dtype=np.int32)
    # tile (z=0, x-left): label 1 only in left half, no boundary voxel
    # tile (z=0, x-right): label 2 only in right half
    # They share the x=4 boundary but fill opposite ends → no touching voxel
    a[0, :, :3] = 1   # left side of tile 0
    a[0, :, 5:] = 0
    a[1, :, :3] = 0
    a[1, :, 5:] = 2   # right side of tile 1

    out = str(tmp_path / "out.zarr")
    zarr_native_merge(sp, "staged", out, "labels", n_workers=1)
    r = np.asarray(zarr.open_group(out)["labels"])
    unique = set(np.unique(r[r > 0]).tolist())
    assert len(unique) == 2, f"isolated labels were incorrectly merged: {unique}"


def test_auto_tile_shape():
    from patchworks import auto_tile_shape

    shape = (128, 2048, 2048)
    tile = auto_tile_shape(shape, "uint16", target_bytes=64 * 1024**2)
    assert len(tile) == 3
    assert all(t <= s for t, s in zip(tile, shape))
    nbytes = np.prod(tile) * np.dtype("uint16").itemsize
    assert nbytes <= 200 * 1024**2  # reasonable upper bound


def test_auto_tile_shape_cellpose():
    from patchworks import auto_tile_shape_cellpose

    tile = auto_tile_shape_cellpose((128, 2048, 2048), "uint16", diameter=30)
    assert tile[0] == 1  # z=1 for 2-D cellpose


def test_relabel_sequential_array():
    from patchworks import relabel_sequential_array

    labels = np.array([0, 500, 500, 7, 7, 7, 0, 1000], dtype=np.int32)
    out = relabel_sequential_array(labels)
    assert out[0] == 0
    assert out[1] == out[2]  # 500 → same id
    assert out[3] == out[4] == out[5]  # 7 → same id
    assert out[6] == 0
    # Should be contiguous
    uniq = np.unique(out)
    uniq = uniq[uniq > 0]
    assert np.all(uniq == np.arange(1, len(uniq) + 1))


def test_estimate_empty_tiles():
    import dask.array as da
    from patchworks import estimate_empty_tiles

    arr_data = np.zeros((4, 32, 32), dtype="uint16")
    arr_data[2:] = 1000  # tiles 2 and 3 have signal
    arr = da.from_array(arr_data, chunks=(1, 32, 32))

    info = estimate_empty_tiles(arr, tile_shape=(1, 32, 32))
    assert info["n_tiles"] == 4
    assert info["n_occupied"] == 2
    assert info["empty_fraction"] == 0.5
