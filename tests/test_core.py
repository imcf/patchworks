"""Self-contained tests for patchworks. No frameworks, no fixtures."""

import numpy as np


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

    # overlap=0 so empty tiles are not fed signal from a neighbour's halo
    tile_process(
        arr, counting_fn, overlap=0, skip_empty=True, empty_threshold=0
    )
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


def test_merge_tile_labels_return_count(tmp_path):
    # sequential_labels=True already computes the exact object count while
    # renumbering to 1..N — return_count=True surfaces it instead of
    # discarding it, so a caller can persist it (e.g. write_labels'
    # n_objects=) for a downstream consumer to skip re-deriving the id set.
    import dask.array as da

    from patchworks import merge_tile_labels

    data = np.zeros((1, 16, 32), dtype="uint16")
    data[0, 2:6, 2:6] = 1
    data[0, 2:6, 10:14] = 1  # same tile-local label, different object
    image = da.from_array(data, chunks=(1, 16, 16))

    def fn(tile):
        from skimage.measure import label

        return label(tile > 0).astype("int32")

    labeled = image.map_blocks(
        fn, dtype="int32", meta=np.empty((0,) * image.ndim, dtype="int32")
    )
    out = str(tmp_path / "merged.zarr")
    merged, n_objects = merge_tile_labels(
        labeled, write_to=out, sequential_labels=True, return_count=True
    )
    arr = merged.compute()
    ids = np.unique(arr[arr > 0])
    assert n_objects == ids.size
    assert n_objects == 2


def test_merge_tile_labels_return_count_none_without_sequential(tmp_path):
    import dask.array as da

    from patchworks import merge_tile_labels

    data = np.zeros((1, 16, 32), dtype="uint16")
    data[0, 4:12, 8:24] = 1
    image = da.from_array(data, chunks=(1, 16, 16))

    def fn(tile):
        from skimage.measure import label

        return label(tile > 0).astype("int32")

    labeled = image.map_blocks(
        fn, dtype="int32", meta=np.empty((0,) * image.ndim, dtype="int32")
    )
    out = str(tmp_path / "merged.zarr")
    merged, n_objects = merge_tile_labels(
        labeled, write_to=out, sequential_labels=False, return_count=True
    )
    assert n_objects is None


def test_merge_transitive_three_tiles(tmp_path):
    # A cell that spans 3 tiles (A→B→C) must be merged into one label even
    # though A and C never directly touch. Transitivity via connected_components.
    import zarr

    from patchworks._merge import zarr_native_merge

    sp = str(tmp_path / "stage.zarr")
    root = zarr.open_group(sp, mode="w")
    a = root.zeros(
        name="staged", shape=(3, 4, 4), chunks=(1, 4, 4), dtype=np.int32
    )
    a[0] = np.full((4, 4), 10)  # label 10 in tile 0
    a[1] = np.full((4, 4), 20)  # label 20 in tile 1 (touches 10 and 30)
    a[2] = np.full((4, 4), 30)  # label 30 in tile 2

    out = str(tmp_path / "out.zarr")
    zarr_native_merge(sp, "staged", out, "labels", n_workers=1)
    r = np.asarray(zarr.open_group(out)["labels"])

    assert r[0, 0, 0] == r[1, 0, 0] == r[2, 0, 0], (
        f"transitive merge failed: tile0={r[0, 0, 0]} tile1={r[1, 0, 0]} tile2={r[2, 0, 0]}"
    )


def test_merge_isolated_labels_not_merged(tmp_path):
    # Two cells that never touch across any boundary must stay separate.
    import zarr

    from patchworks._merge import zarr_native_merge

    sp = str(tmp_path / "stage.zarr")
    root = zarr.open_group(sp, mode="w")
    a = root.zeros(
        name="staged", shape=(2, 4, 8), chunks=(1, 4, 4), dtype=np.int32
    )
    # tile (z=0, x-left): label 1 only in left half, no boundary voxel
    # tile (z=0, x-right): label 2 only in right half
    # They share the x=4 boundary but fill opposite ends → no touching voxel
    a[0, :, :3] = 1  # left side of tile 0
    a[0, :, 5:] = 0
    a[1, :, :3] = 0
    a[1, :, 5:] = 2  # right side of tile 1

    out = str(tmp_path / "out.zarr")
    zarr_native_merge(sp, "staged", out, "labels", n_workers=1)
    r = np.asarray(zarr.open_group(out)["labels"])
    unique = set(np.unique(r[r > 0]).tolist())
    assert len(unique) == 2, (
        f"isolated labels were incorrectly merged: {unique}"
    )


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


def test_relabel_sequential_zarr(tmp_path):
    """The standalone zarr relabeller stays correct on its own.

    The merge folds this into its LUT now, so nothing internal exercises it —
    but it is public API for label stores written by other pipelines.
    """
    import zarr

    from patchworks import relabel_sequential_zarr

    store = str(tmp_path / "labels.zarr")
    root = zarr.open_group(store, mode="w")
    root.create_array(name="labels", shape=(4, 8), chunks=(4, 4), dtype="i4")
    data = np.zeros((4, 8), dtype="int32")
    data[0, :3] = 500  # gappy ids spanning both chunks
    data[1, 5:] = 7
    data[3, 2] = 100000
    root["labels"][:] = data

    n = relabel_sequential_zarr(store, "labels")
    out = np.asarray(zarr.open_group(store, mode="r")["labels"])

    assert n == 3, f"three distinct objects, got {n}"
    assert set(np.unique(out).tolist()) == {0, 1, 2, 3}, "must be contiguous"
    # Same voxels keep the same id; background stays background.
    assert len(set(out[0, :3].tolist())) == 1
    assert len(set(out[1, 5:].tolist())) == 1
    assert out[2, 0] == 0


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


def test_safe_worker_count_bounds():
    import os

    from patchworks._chunks import safe_worker_count

    # GPU → always serial (no VRAM contention)
    assert safe_worker_count(10**6, use_gpu=True) == 1
    # Absurdly large tile → memory-bound to 1
    assert safe_worker_count(10**15) == 1
    # Tiny tile → CPU-bound, leaves a core free, always >= 1
    n = safe_worker_count(1024)
    assert 1 <= n <= max(1, (os.cpu_count() or 1) - 1)


def test_tile_process_max_workers():
    import dask.array as da

    from patchworks import tile_process

    arr = da.from_array(_make_image((2, 32, 32)), chunks=(1, 32, 32))
    result = tile_process(arr, _label_fn, max_workers=1).compute()
    assert result.shape == (2, 32, 32)


def test_channel_selection_respects_the_stores_axes(tmp_path):
    """`channel: 0` on a store with no channel axis must not slice away z.

    arr[channel] was applied unconditionally, so a single-channel image
    written as plain zyx silently lost its z axis. The array stayed valid,
    just one dimension short, and only surfaced later as a tile count that
    disagreed with the occupancy grid.
    """
    import dask.array as da
    import pytest

    from patchworks import load_ome_zarr
    from patchworks.plugins.ome_zarr import to_ome_zarr

    vol = da.zeros((8, 64, 64), chunks=(4, 32, 32), dtype="uint16")
    store = str(tmp_path / "zyx.zarr")
    to_ome_zarr(vol, store, axes="zyx", n_levels=1, progress=False)

    # channel 0 on a zyx store: keep the whole volume, do not index axis 0.
    assert load_ome_zarr(store, channel=0, level=0).shape == (8, 64, 64)
    assert load_ome_zarr(store, channel=None, level=0).shape == (8, 64, 64)
    # A non-zero channel really is a mistake here, and is named as one.
    with pytest.raises(ValueError, match="no channel axis"):
        load_ome_zarr(store, channel=2, level=0)

    # A czyx store still selects the channel as before.
    vol4 = da.zeros((3, 8, 64, 64), chunks=(1, 4, 32, 32), dtype="uint16")
    store4 = str(tmp_path / "czyx.zarr")
    to_ome_zarr(vol4, store4, axes="czyx", n_levels=1, progress=False)
    assert load_ome_zarr(store4, channel=1, level=0).shape == (8, 64, 64)


def test_auto_tile_shape_charges_for_extra_channels():
    """A 2-channel tile must fit the same byte budget, not twice it.

    `nuclei_channel` doubles what a tile holds while the tile geometry stays
    single-channel, so a sizer blind to it hands the GPU a tile needing twice
    the VRAM it budgeted for.
    """
    import pytest

    from patchworks import auto_tile_shape, auto_tile_shape_cellpose

    shape, dtype = (128, 2048, 2048), "uint16"

    one = auto_tile_shape(shape, dtype)
    two = auto_tile_shape(shape, dtype, n_channels=2)
    # Same bytes overall: 2 channels of roughly half the area each.
    assert np.prod(two) * 2 <= np.prod(one)
    assert np.prod(two) * 2 >= np.prod(one) * 0.9

    kw = dict(
        diameter=30,
        do_3D=True,
        use_gpu=True,
        gpu_memory=24 * 1024**3,
        # Generous on purpose: this test is about GPU-vs-channel scaling,
        # not the host-RAM ceiling, so host RAM must stay non-binding here.
        available_memory=64 * 1024**3,
    )
    cp_one = auto_tile_shape_cellpose(shape, dtype, **kw)
    cp_two = auto_tile_shape_cellpose(shape, dtype, n_channels=2, **kw)
    assert np.prod(cp_two) * 2 <= np.prod(cp_one)
    assert cp_two[0] == cp_one[0]  # do_3D still pins z to the full extent

    with pytest.raises(ValueError, match="n_channels"):
        auto_tile_shape(shape, dtype, n_channels=0)
