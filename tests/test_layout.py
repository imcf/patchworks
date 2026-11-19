"""The viewer layout: planes of small chunks inside shards."""

import numpy as np
import pytest
import zarr

from patchworks._chunks import dask_from_zarr, io_chunks
from patchworks._relabel import relabel_sequential_zarr
from patchworks._volume_filter import filter_labels_by_size
from patchworks.plugins.ome_zarr import (
    is_viewer_layout,
    label_layout,
    reshard_level,
    to_ome_zarr,
    viewer_layout,
    write_labels,
)


def _blobs(shape=(20, 96, 96), n=60, seed=0):
    """Labels: *n* boxes of different sizes, ids with gaps."""
    rng = np.random.default_rng(seed)
    lab = np.zeros(shape, "int32")
    for i in range(n):
        z, y, x = (rng.integers(0, s - 6) for s in shape)
        d = rng.integers(1, 6)
        lab[z : z + d, y : y + 3 * d, x : x + 3 * d] = 3 * i + 7
    return lab


def _sharded(tmp_path, lab):
    """*lab* as a sharded array of small chunks: shards (8, 32, 32)."""
    arr = zarr.open_group(str(tmp_path / "s.zarr"), mode="w").create_array(
        "0",
        shape=lab.shape,
        chunks=(1, 16, 16),
        shards=(8, 32, 32),
        dtype=lab.dtype,
    )
    arr[...] = lab
    return str(tmp_path / "s.zarr")


def test_passes_over_a_sharded_array_go_shard_by_shard(tmp_path):
    path = _sharded(tmp_path, _blobs())
    arr = zarr.open_array(path, path="0", mode="r")
    assert io_chunks(arr) == (8, 32, 32)
    blocks = dask_from_zarr(path, component="0")
    assert blocks.chunksize == (8, 32, 32)
    assert blocks.npartitions == 3 * 3 * 3  # not 20 * 6 * 6 chunks


@pytest.mark.parametrize("workers", [1, 8])
def test_rewriting_in_place_loses_no_chunk_of_a_shard(tmp_path, workers):
    """Threads writing chunks of one shard lost each other's: a pass now
    writes whole shards."""
    lab = _blobs()
    path = _sharded(tmp_path, lab)
    ids, counts = np.unique(lab[lab > 0], return_counts=True)
    small = set(ids[counts < 20].tolist())
    kept, dropped = filter_labels_by_size(
        path, "0", min_voxels=20, relabel=False, n_workers=workers
    )
    assert dropped == len(small)
    expected = np.where(np.isin(lab, list(small)), 0, lab)
    assert np.array_equal(zarr.open_array(path, path="0")[...], expected)

    n = relabel_sequential_zarr(path, "0")
    out = zarr.open_array(path, path="0")[...]
    assert n == kept and set(np.unique(out)) == set(range(n + 1))
    assert np.array_equal(out > 0, expected > 0)


def test_rewriting_into_the_viewer_layout_keeps_data_and_attrs(tmp_path):
    lab = _blobs()
    group = tmp_path / "g.zarr"
    arr = zarr.open_group(str(group), mode="w").create_array(
        "0", data=lab, chunks=(7, 40, 40)
    )
    arr.attrs["patchworks_merge_state"] = {"done": True}
    chunks, shards = viewer_layout(lab.shape, "zyx")
    reshard_level(str(group), "0", shard=shards, chunks=chunks, progress=False)
    out = zarr.open_array(str(group), path="0", mode="r")
    assert out.chunks == (1, 96, 96) and out.shards == (16, 96, 96)
    assert is_viewer_layout(out)
    assert np.array_equal(out[...], lab)
    assert dict(out.attrs)["patchworks_merge_state"] == {"done": True}


def test_a_store_keeps_one_layout_for_its_labels(tmp_path):
    img = np.zeros((20, 96, 96), "uint16")
    store = to_ome_zarr(
        img, tmp_path / "a.zarr", axes="zyx", n_levels=2, progress=False
    )
    assert label_layout(store) == "viewer"  # no label image yet
    write_labels(store, _blobs(), name="cells", progress=False)
    cells = zarr.open_group(f"{store}/labels/cells", mode="r")
    assert is_viewer_layout(cells["0"]) and is_viewer_layout(cells["1"])
    assert label_layout(store) == "viewer"

    old = to_ome_zarr(
        img, tmp_path / "b.zarr", axes="zyx", n_levels=2, progress=False
    )
    write_labels(old, _blobs(), name="cells", layout="tiles", progress=False)
    assert label_layout(old) == "tiles"
    # The label image written again does not decide for itself
    assert label_layout(old, exclude="cells") == "viewer"
    write_labels(old, _blobs(seed=1), name="nuclei", progress=False)
    nuclei = zarr.open_group(f"{old}/labels/nuclei", mode="r")
    assert not is_viewer_layout(nuclei["0"])

    v2 = to_ome_zarr(
        img,
        tmp_path / "c.zarr",
        axes="zyx",
        n_levels=1,
        ngff_version="0.4",
        progress=False,
    )
    assert label_layout(v2) == "tiles"


def test_relating_two_viewer_label_images(tmp_path):
    """Measured and related shard by shard, with the same results."""
    from patchworks._tables import compute_table, read_columns, relate_tables

    img = np.zeros((20, 96, 96), "uint16")
    store = to_ome_zarr(
        img, tmp_path / "a.zarr", axes="zyx", n_levels=1, progress=False
    )
    cells = np.zeros((20, 96, 96), "int32")
    cells[:, :48], cells[:, 48:] = 1, 2
    nuclei = _blobs(n=10)
    write_labels(store, cells, name="cells", progress=False)
    write_labels(store, nuclei, name="nuclei", progress=False)
    for name in ("cells", "nuclei"):
        compute_table(store, name)
    relate_tables(store, "nuclei", "cells")
    cols = read_columns(f"{store}/labels/nuclei")
    for lab, parent in zip(cols["label"], cols["cells_id"]):
        zs, ys, xs = np.nonzero(nuclei == lab)
        inside = cells[zs, ys, xs]
        assert parent == np.bincount(inside).argmax()


def test_a_converted_image_in_the_viewer_layout(tmp_path):
    img = np.random.default_rng(0).integers(0, 999, (2, 6, 80, 80), "uint16")
    store = to_ome_zarr(
        img,
        tmp_path / "v.zarr",
        axes="czyx",
        n_levels=2,
        layout="viewer",
        progress=False,
    )
    root = zarr.open_group(store, mode="r")
    assert root["0"].chunks == (1, 1, 80, 80)
    assert root["0"].shards == (1, 1, 80, 80)
    assert np.array_equal(root["0"][...], img)
    assert root["1"].shape == (2, 6, 40, 40) and is_viewer_layout(root["1"])


def _reshard_script():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parent.parent
        / "workflow"
        / "scripts"
        / "reshard_store.py"
    )
    spec = importlib.util.spec_from_file_location("reshard_store", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_an_old_store_rewritten_for_viewers(tmp_path, monkeypatch, capsys):
    """Every array into the viewer layout, in one job: the data, the tables
    (still current), the objects' relations and validity all kept."""
    import sys

    from patchworks._tables import (
        compute_table,
        is_stale,
        read_columns,
        relate_tables,
    )
    from patchworks._validate import patchworks_findings

    img = np.random.default_rng(0).integers(0, 999, (2, 20, 96, 96), "uint16")
    store = to_ome_zarr(
        img, tmp_path / "old.zarr", axes="czyx", n_levels=2, progress=False
    )
    cells = np.zeros((20, 96, 96), "int32")
    cells[:, :48], cells[:, 48:] = 1, 2
    for name, lab in (("cells", cells), ("nuclei", _blobs(n=10))):
        write_labels(store, lab, name=name, layout="tiles", progress=False)
        compute_table(store, name)
    relate_tables(store, "nuclei", "cells")
    before = read_columns(f"{store}/labels/nuclei")

    script = _reshard_script()
    argv = ["reshard_store.py", "--store", store, "--layout", "viewer"]
    monkeypatch.setattr(sys, "argv", [*argv, "--dry-run"])
    script.main()
    assert "6 array(s) need rewriting, 0 already in the viewer layout" in (
        capsys.readouterr().out
    )
    monkeypatch.setattr(sys, "argv", argv)
    script.main()
    root = zarr.open_group(store, mode="r")
    assert root["0"].chunks == (1, 1, 96, 96) and root["0"].shards == (
        1,
        1,
        96,
        96,
    )
    assert np.array_equal(root["0"][...], img)
    for name, lab in (("cells", cells), ("nuclei", _blobs(n=10))):
        group = root["labels"][name]
        assert all(is_viewer_layout(group[p]) for p in ("0", "1"))
        assert np.array_equal(group["0"][...], lab)
        assert not is_stale(f"{store}/labels/{name}")
    assert label_layout(store) == "viewer"
    relate_tables(store, "nuclei", "cells")
    after = read_columns(f"{store}/labels/nuclei")
    assert np.array_equal(after["cells_id"], before["cells_id"])
    findings, _ = patchworks_findings(store)
    assert all(not f.error for f in findings)

    script.main()  # again: nothing left to do
    assert "0 array(s) need rewriting" in capsys.readouterr().out
