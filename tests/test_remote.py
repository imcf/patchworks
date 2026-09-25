"""Remote (fsspec URL) stores, with fsspec's in-memory filesystem as bucket."""

import os
import uuid

import numpy as np
import pytest
import zarr

fsspec = pytest.importorskip("fsspec")

from patchworks import load_ome_zarr, tile_process  # noqa: E402
from patchworks._io import is_remote  # noqa: E402
from patchworks.plugins.ome_zarr import (  # noqa: E402
    read_pixel_size,
    reshard_level,
    to_ome_zarr,
)


@pytest.fixture
def bucket(tmp_path, monkeypatch):
    """An OME-ZARR uploaded to memory://, and a clean working directory."""
    img = np.zeros((2, 64, 64), "uint16")
    img[:, 10:30, 20:40] = 1000
    local = to_ome_zarr(
        img,
        tmp_path / "a.zarr",
        axes="zyx",
        n_levels=2,
        pixel_size={"z": 0.5, "y": 0.2, "x": 0.2},
    )
    key = f"/b-{uuid.uuid4().hex[:8]}/a.zarr"
    fsspec.filesystem("memory").put(local, key, recursive=True)
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    return f"memory:/{key}", work


def _seg(tile):
    return (tile > 500).astype("int32")


def test_is_remote():
    assert is_remote("s3://b/x.zarr") and is_remote("https://h/x.zarr")
    assert not is_remote("/data/x.zarr") and not is_remote("x.zarr")
    assert not is_remote("file:///data/x.zarr") and not is_remote(None)


def test_segment_a_remote_store_into_itself(bucket):
    url, work = bucket
    assert load_ome_zarr(url).shape == (2, 64, 64)
    assert read_pixel_size(url) == {"z": 0.5, "y": 0.2, "x": 0.2}
    tile_process(url, _seg, tile_shape=(2, 32, 32), overlap=4, progress=False)
    labels = zarr.open_group(f"{url}/labels/labels", mode="r")["0"][:]
    assert len(np.unique(labels)) == 2
    # Scratch stayed local and tidy: nothing named after the URL scheme.
    assert os.listdir(work) == []


def test_remote_input_local_output(bucket, tmp_path):
    url, work = bucket
    out = tile_process(
        url,
        _seg,
        tile_shape=(2, 32, 32),
        write_to=tmp_path / "o.zarr",
        stitch="iou",
        progress=False,
    )
    assert len(np.unique(np.asarray(out))) == 2
    assert os.listdir(work) == []


def test_reshard_refuses_a_remote_store(bucket):
    url, _ = bucket
    with pytest.raises(ValueError, match="remote"):
        reshard_level(url, "0", shard=True)
