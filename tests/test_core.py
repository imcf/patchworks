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


def test_stage_stores_never_collide(tmp_path):
    """Two runs sharing a directory must each get their own stage store.

    A fixed ``_pws_stage.zarr`` name let concurrent runs writing side by side
    (``nuclei.zarr`` and ``cyto.zarr``) overwrite each other's tiles.
    """
    from patchworks._merge import _scratch_store

    a, _ = _scratch_store(tmp_path, "stage")
    b, _ = _scratch_store(tmp_path, "stage")
    assert a != b
    assert all(p.startswith(str(tmp_path)) for p in (a, b))


def test_tile_process_leaves_no_scratch_behind(tmp_path):
    """A successful run removes its stage store from the output directory."""
    import dask.array as da

    from patchworks import tile_process

    arr = da.from_array(_make_image((2, 64, 64)), chunks=(1, 64, 64))
    tile_process(arr, _label_fn, write_to=tmp_path / "out.zarr")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.zarr"]


def test_tile_process_cleans_up_when_fn_fails(tmp_path):
    """A failing fn must not leave a stage store the size of the image."""
    import dask.array as da
    import pytest

    from patchworks import tile_process

    def boom(tile):
        raise RuntimeError("segmentation failed")

    arr = da.from_array(_make_image((2, 64, 64)), chunks=(1, 64, 64))
    with pytest.raises(RuntimeError, match="segmentation failed"):
        tile_process(arr, boom, write_to=tmp_path / "out.zarr", progress=False)
    assert not [p for p in tmp_path.iterdir() if p.name.startswith("_pws_")]


def test_tile_process_keep_stage_survives(tmp_path):
    """keep_stage=True keeps the stage store (under a unique name)."""
    import dask.array as da

    from patchworks import tile_process

    arr = da.from_array(_make_image((2, 64, 64)), chunks=(1, 64, 64))
    tile_process(
        arr,
        _label_fn,
        write_to=tmp_path / "out.zarr",
        stage_dir=tmp_path / "stage",
        keep_stage=True,
        progress=False,
    )
    kept = list((tmp_path / "stage").iterdir())
    assert len(kept) == 1 and kept[0].name.startswith("_pws_stage_")


def test_tile_process_writes_no_log_file_by_default(tmp_path):
    """A library call must not drop files next to the user's data."""
    import dask.array as da

    from patchworks import tile_process

    arr = da.from_array(_make_image((2, 64, 64)), chunks=(1, 64, 64))
    tile_process(arr, _label_fn, write_to=tmp_path / "out.zarr")
    assert not (tmp_path / "patchworks.log").exists()


def test_zarr_path_detection_tolerates_a_trailing_slash():
    """``image.zarr/`` is still a store the labels go back into."""
    from patchworks._core import _is_zarr_path

    assert _is_zarr_path("/data/image.zarr")
    assert _is_zarr_path("/data/image.zarr/")
    assert not _is_zarr_path("/data/image.tif")


def test_unknown_tile_shape_string_is_rejected():
    import dask.array as da
    import pytest

    from patchworks import tile_process

    arr = da.from_array(_make_image((2, 64, 64)), chunks=(1, 64, 64))
    with pytest.raises(ValueError, match="Unknown tile_shape"):
        tile_process(arr, _label_fn, tile_shape="big")


def test_otsu_matches_scikit_image():
    """The built-in Otsu is a faithful port, so skip_empty needs no skimage."""
    import pytest

    threshold_otsu = pytest.importorskip("skimage.filters").threshold_otsu
    from patchworks._io import _otsu_threshold

    rng = np.random.default_rng(0)
    samples = [
        rng.integers(0, 1000, 5000).astype("uint16"),
        rng.integers(-500, 500, 3000).astype("int16"),
        np.concatenate(
            [rng.normal(10, 3, 3000), rng.normal(80, 10, 2000)]
        ).astype("float32"),
        np.concatenate(
            [np.zeros(4000, "uint8"), rng.integers(50, 255, 1000, "uint8")]
        ),
        np.full(10, 7, "uint16"),
    ]
    for s in samples:
        assert np.isclose(_otsu_threshold(s), threshold_otsu(s), atol=1e-3)
    assert _otsu_threshold(np.array([], "uint16")) == 0.0


@__import__("pytest").mark.skipif(
    not __import__("sys").platform.startswith("linux"),
    reason="fork is only used (and safe) on Linux; macOS and Windows spawn, "
    "which needs a __main__ guard by design",
)
def test_merge_pool_does_not_rerun_an_unguarded_script(tmp_path):
    """A script without a __main__ guard must still merge with >1 worker.

    Python 3.14 made forkserver the Linux default, which re-imports the
    caller's main module in every worker: the pipeline's Snakemake merge.py
    and the docs' examples then re-run their whole body per worker and die.
    """
    import subprocess
    import sys
    import textwrap

    script = tmp_path / "unguarded.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import numpy as np, zarr
            from patchworks._merge import zarr_native_merge
            print("TOP-LEVEL", flush=True)
            g = zarr.open_group({str(tmp_path / "s.zarr")!r}, mode="w")
            a = g.create_array(
                "staged", shape=(4, 8, 8), chunks=(1, 8, 8), dtype="int32"
            )
            a[:] = 1
            zarr_native_merge(
                {str(tmp_path / "s.zarr")!r}, "staged",
                {str(tmp_path / "o.zarr")!r}, "labels", n_workers=2,
            )
            print("MERGED", flush=True)
            """
        )
    )
    run = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.split() == ["TOP-LEVEL", "MERGED"]


def test_single_worker_merge_releases_the_lut(tmp_path):
    """An in-process relabel must not keep the (deleted) LUT mapped."""
    import zarr

    from patchworks import _merge
    from patchworks._merge import zarr_native_merge

    sp = str(tmp_path / "stage.zarr")
    a = zarr.open_group(sp, mode="w").create_array(
        "staged", shape=(2, 4, 4), chunks=(1, 4, 4), dtype="int32"
    )
    a[:] = 1
    zarr_native_merge(sp, "staged", str(tmp_path / "o.zarr"), "l", n_workers=1)
    assert _merge._merge_lut is None
    assert _merge._merge_src is None and _merge._merge_dst is None


def test_merge_does_not_wrap_narrow_label_dtypes(tmp_path):
    """400 objects in uint8 tiles must stay 400 objects, not wrap to 255.

    Per-tile masks are often narrow (Cellpose returns uint16); the global
    renumbering sums every tile, so the merge must widen, not overflow.
    """
    import dask.array as da

    from patchworks import merge_tile_labels

    tile = np.zeros((16, 16), "uint8")
    ys, xs = np.divmod(np.arange(0, 256, 2)[:100], 16)
    tile[ys, xs] = np.arange(1, 101)
    lab = da.from_array(np.block([[tile, tile], [tile, tile]]), chunks=16)

    out = merge_tile_labels(lab, write_to=tmp_path / "o.zarr").compute()
    assert len(np.unique(out)) - 1 == 400


def test_merge_widens_the_output_for_counted_narrow_stores(tmp_path):
    """With label_counts the store is not rewritten; the output widens."""
    import zarr

    from patchworks._merge import zarr_native_merge

    sp = str(tmp_path / "s.zarr")
    a = zarr.open_group(sp, mode="w").create_array(
        "staged", shape=(2, 16, 16), chunks=(1, 16, 16), dtype="uint8"
    )
    # 128 single-pixel objects per tile on disjoint pixels (even vs odd), so
    # nothing touches across the boundary: 256 objects, one more than uint8.
    flat = np.zeros((2, 256), "uint8")
    flat[0, 0::2] = np.arange(1, 129)
    flat[1, 1::2] = np.arange(1, 129)
    a[:] = flat.reshape(2, 16, 16)
    zarr_native_merge(
        sp,
        "staged",
        str(tmp_path / "o.zarr"),
        "l",
        n_workers=1,
        label_counts=[128, 128],
    )
    out = zarr.open_group(str(tmp_path / "o.zarr"), mode="r")["l"]
    assert np.dtype(out.dtype).itemsize >= 4
    assert len(np.unique(np.asarray(out[:]))) - 1 == 256


def test_in_place_merge_refuses_to_overflow(tmp_path):
    import pytest
    import zarr

    from patchworks._merge import zarr_native_merge

    sp = str(tmp_path / "s.zarr")
    a = zarr.open_group(sp, mode="w").create_array(
        "staged", shape=(2, 16, 16), chunks=(1, 16, 16), dtype="uint8"
    )
    tile = (np.arange(256).reshape(16, 16) % 200).astype("uint8")
    a[:] = np.stack([tile, tile])
    with pytest.raises(OverflowError):
        zarr_native_merge(
            sp, "staged", sp, "staged", n_workers=1, label_counts=[199, 199]
        )


def test_relabel_sequential_keeps_every_object_without_background(tmp_path):
    """No 0 in the data must not turn the smallest object into background."""
    import zarr

    from patchworks import relabel_sequential_array, relabel_sequential_zarr

    np.testing.assert_array_equal(
        relabel_sequential_array(np.array([5, 5, 7])), [1, 1, 2]
    )
    assert relabel_sequential_array(np.array([], "int32")).size == 0

    sp = str(tmp_path / "l.zarr")
    z = zarr.open_group(sp, mode="w").create_array(
        "labels", shape=(2, 2), chunks=(1, 2), dtype="int32"
    )
    z[:] = [[5, 5], [9, 7]]
    assert relabel_sequential_zarr(sp) == 3
    np.testing.assert_array_equal(z[:], [[1, 1], [3, 2]])


def test_tile_process_names_a_function_returning_the_wrong_shape(tmp_path):
    import dask.array as da
    import pytest

    from patchworks import tile_process

    def cropping_fn(tile):
        return np.zeros(tuple(s - 1 for s in tile.shape), "int32")

    arr = da.from_array(_make_image((2, 64, 64)), chunks=(1, 64, 64))
    with pytest.raises(ValueError, match="cropping_fn.*returned shape"):
        tile_process(arr, cropping_fn, write_to=tmp_path / "o.zarr")


def test_auto_empty_threshold_samples_full_windows(monkeypatch):
    """Every sample window is full-size, even on an axis just over 64."""
    import dask.array as da

    from patchworks import _io

    seen = []
    monkeypatch.setattr(
        _io, "_otsu_threshold", lambda s: seen.append(s.size) or 0.0
    )
    _io.auto_empty_threshold(da.zeros((4, 70, 300), dtype="uint16"), 0, 0)
    assert seen == [3 * 4 * 64 * 64]


def test_zip_bundles_are_recognised_by_path_component_only(tmp_path):
    """A directory merely containing ".zip" in its name is not a bundle."""
    from patchworks import load_ome_zarr
    from patchworks._io import split_zip_path
    from patchworks.plugins.ome_zarr import to_ome_zarr

    assert split_zip_path("/d/my.zipfiles/a.zarr") is None
    assert split_zip_path("/d/a.zarr.zip") == ("/d/a.zarr.zip", "")
    assert split_zip_path("/d/a.zarr.zip/labels/c") == (
        "/d/a.zarr.zip",
        "labels/c",
    )
    assert split_zip_path("C:\\d\\a.ZIP\\labels\\c") == (
        "C:\\d\\a.ZIP",
        "labels/c",
    )

    store = tmp_path / "my.zipfiles" / "a.zarr"
    to_ome_zarr(np.zeros((2, 16, 16), "uint16"), store, axes="zyx", n_levels=1)
    assert load_ome_zarr(store).shape == (2, 16, 16)


def _two_touching_cells():
    img = np.zeros((1, 16, 64), "uint16")
    img[0, 4:12, 20:32] = 1  # cell A, ends at the x=32 seam
    img[0, 4:12, 32:44] = 2  # cell B, touching it across the seam
    img[0, 2:6, 50:60] = 1  # a third cell, far away
    return img


def _label_by_value(tile):
    from skimage.measure import label

    return label(tile, background=0, connectivity=1).astype("int32")


def test_tile_process_iou_stitching_keeps_touching_cells_apart(tmp_path):
    import dask.array as da

    from patchworks import tile_process

    arr = da.from_array(_two_touching_cells(), chunks=(1, 16, 32))
    touch = tile_process(
        arr, _label_by_value, overlap=6, write_to=tmp_path / "t.zarr"
    ).compute()
    iou = tile_process(
        arr,
        _label_by_value,
        overlap=6,
        stitch="iou",
        write_to=tmp_path / "i.zarr",
    ).compute()
    assert len(np.unique(touch)) - 1 == 2  # A and B fused at the seam
    assert len(np.unique(iou)) - 1 == 3
    assert not [p for p in tmp_path.iterdir() if p.name.startswith("_pws_")]


def test_tile_process_resumes_where_it_stopped(tmp_path):
    """A failed resumable run keeps its tiles; the rerun only does the rest."""
    import dask.array as da
    import pytest

    from patchworks import tile_process

    img = _make_image((4, 32, 32))
    arr = da.from_array(img, chunks=(1, 32, 32))
    calls = {"n": 0, "fail_at": 3}

    def flaky(tile):
        calls["n"] += 1
        if calls["n"] == calls["fail_at"]:
            raise RuntimeError("job killed")
        return _label_fn(tile)

    kw = dict(
        overlap=0,
        write_to=tmp_path / "o.zarr",
        resume=True,
        max_workers=1,
        progress=False,
    )
    with pytest.raises(RuntimeError, match="job killed"):
        tile_process(arr, flaky, **kw)
    kept = [p for p in tmp_path.iterdir() if p.name.startswith("_pws_resume")]
    assert len(kept) == 1  # kept to resume from

    calls.update(n=0, fail_at=-1)
    resumed = tile_process(arr, flaky, **kw).compute()
    assert calls["n"] == 2  # only the two tiles the first run never staged
    assert not [p for p in tmp_path.iterdir() if p.name.startswith("_pws_")]

    fresh = tile_process(
        arr, _label_fn, overlap=0, write_to=tmp_path / "f.zarr"
    ).compute()
    assert len(np.unique(resumed)) == len(np.unique(fresh))
    assert ((resumed > 0) == (fresh > 0)).all()


def test_tile_process_rejects_an_unknown_stitch():
    import dask.array as da
    import pytest

    from patchworks import tile_process

    arr = da.from_array(_make_image((1, 16, 16)), chunks=(1, 16, 16))
    with pytest.raises(ValueError, match="stitch"):
        tile_process(arr, _label_fn, stitch="glue")


def test_compression_applies_to_everything_written(tmp_path):
    import dask.array as da
    import pytest
    import zarr

    from patchworks import compression, tile_process
    from patchworks.plugins.ome_zarr import to_ome_zarr

    arr = da.from_array(_make_image((2, 32, 32)), chunks=(1, 32, 32))
    with compression("blosc:lz4"):
        tile_process(arr, _label_fn, write_to=tmp_path / "o.zarr")
    codec = zarr.open_group(str(tmp_path / "o.zarr"), mode="r")[
        "labels"
    ].compressors[0]
    cname = getattr(codec.cname, "value", codec.cname)  # enum or str by version
    assert type(codec).__name__ == "BloscCodec" and cname == "lz4"

    out = to_ome_zarr(
        _make_image((2, 32, 32)),
        tmp_path / "img.zarr",
        axes="zyx",
        n_levels=2,
        compression="zstd:3",
    )
    for level in ("0", "1"):
        c = zarr.open_group(str(out), mode="r")[level].compressors[0]
        assert type(c).__name__ == "ZstdCodec" and c.level == 3
    # ... and the setting does not leak out of the call.
    tile_process(arr, _label_fn, write_to=tmp_path / "d.zarr")
    c = zarr.open_group(str(tmp_path / "d.zarr"), mode="r")["labels"]
    assert c.compressors[0].level == 1

    with pytest.raises(ValueError, match="unknown compression"):
        compression("gzip").__enter__()


def test_compression_none_and_zarr_v2(tmp_path):
    import zarr

    from patchworks.plugins.ome_zarr import to_ome_zarr

    out = to_ome_zarr(
        _make_image((2, 16, 16)),
        tmp_path / "n.zarr",
        axes="zyx",
        n_levels=1,
        compression="none",
    )
    assert zarr.open_group(str(out), mode="r")["0"].compressors == ()
    v2 = to_ome_zarr(
        _make_image((2, 16, 16)),
        tmp_path / "v2.zarr",
        axes="zyx",
        n_levels=1,
        compression="blosc",
        ngff_version="0.4",
    )
    c = zarr.open_group(str(v2), mode="r")["0"].compressors[0]
    assert type(c).__name__ == "Blosc"


def test_labels_record_how_they_were_made(tmp_path):
    import dask.array as da

    from patchworks import read_provenance, tile_process
    from patchworks.plugins.ome_zarr import to_ome_zarr

    store = to_ome_zarr(
        _make_image((2, 32, 32)), tmp_path / "a.zarr", axes="zyx", n_levels=1
    )
    tile_process(
        store,
        _label_fn,
        tile_shape=(1, 32, 32),
        overlap=2,
        stitch="iou",
        progress=False,
    )
    rec = read_provenance(f"{store}/labels/labels")
    assert rec["settings"]["stitch"] == "iou"
    assert rec["settings"]["tile_shape"] == [1, 32, 32]
    assert rec["settings"]["fn"].endswith("_label_fn")
    assert rec["settings"]["input"] == store
    assert "patchworks" in rec["versions"] and rec["created"]

    arr = da.from_array(_make_image((2, 32, 32)), chunks=(1, 32, 32))
    tile_process(arr, _label_fn, write_to=tmp_path / "o.zarr", progress=False)
    rec = read_provenance(tmp_path / "o.zarr", component="labels")
    assert rec["settings"]["stitch"] == "touch"


@__import__("pytest").mark.skipif(
    not __import__("sys").platform.startswith("linux"),
    reason="multi-GPU workers are forked (Linux only)",
)
def test_gpus_pin_one_worker_process_per_device(tmp_path, monkeypatch):
    """Each GPU worker sees exactly its own device, and all are used."""
    import os

    import dask.array as da

    from patchworks import tile_process

    seen = tmp_path / "seen"
    seen.mkdir()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5,7")

    def recording(tile):
        import time

        dev = os.environ["CUDA_VISIBLE_DEVICES"]
        (seen / f"{os.getpid()}").write_text(dev)
        time.sleep(0.05)  # long enough that both workers get tiles
        return _label_fn(tile)

    arr = da.from_array(_make_image((8, 32, 32)), chunks=(1, 32, 32))
    multi = tile_process(
        arr,
        recording,
        overlap=0,
        gpus=2,
        use_gpu=True,
        write_to=tmp_path / "m.zarr",
        progress=False,
    ).compute()
    devices = {p.read_text() for p in seen.iterdir()}
    assert devices == {"3", "5"}
    assert all("," not in d for d in devices)  # one device per process
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3,5,7"  # parent untouched

    single = tile_process(
        arr,
        _label_fn,
        overlap=0,
        write_to=tmp_path / "s.zarr",
        progress=False,
    ).compute()
    assert ((multi > 0) == (single > 0)).all()
    assert len(np.unique(multi)) == len(np.unique(single))


def test_resolve_gpus(monkeypatch):
    import pytest

    from patchworks._core import _resolve_gpus

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert _resolve_gpus(None) is None
    assert _resolve_gpus(2) == ["0", "1"]
    assert _resolve_gpus([1, "GPU-abc"]) == ["1", "GPU-abc"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,6")
    assert _resolve_gpus(2) == ["4", "6"]
    with pytest.raises(ValueError, match="visible"):
        _resolve_gpus(3)


def test_dry_run_plans_without_writing(tmp_path):
    import dask.array as da

    from patchworks import tile_process

    img = np.zeros((4, 64, 64), "uint16")
    img[:, 5:20, 5:20] = 1000  # signal in one corner tile only
    arr = da.from_array(img, chunks=(4, 32, 32))
    calls = []

    def fn(tile):
        calls.append(tile.shape)
        return _label_fn(tile)

    plan = tile_process(
        arr,
        fn,
        overlap=4,
        skip_empty=True,
        empty_threshold=10,
        write_to=tmp_path / "o.zarr",
        dry_run=True,
        plan_sample=2,
    )
    assert plan["tiles"] == 4 and plan["grid"] == [1, 2, 2]
    assert plan["tiles_with_signal"] == 1
    assert plan["labels_bytes_uncompressed"] == 4 * 64 * 64 * 4
    assert plan["estimated_seconds"] is not None
    assert calls == [(4, 36, 36)]  # the one tile with signal, halo included
    assert list(tmp_path.iterdir()) == []  # nothing written


def test_cli_plan(tmp_path, capsys):
    import json

    from patchworks.cli import main
    from patchworks.plugins.ome_zarr import to_ome_zarr

    store = to_ome_zarr(
        _make_image((2, 64, 64)), tmp_path / "a.zarr", axes="zyx", n_levels=1
    )
    assert main(["segment", store, "--tile-shape", "2,32,32", "--plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["tiles"] == 4
    assert not (tmp_path / "a.zarr" / "labels").exists()
