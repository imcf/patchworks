"""Tests for the patchworks command line."""

import json

import numpy as np
import pytest
import zarr

from patchworks.cli import main
from patchworks.plugins.ome_zarr import read_pixel_size, to_ome_zarr


@pytest.fixture
def blobs():
    img = np.zeros((2, 64, 64), "uint16")
    img[:, 10:20, 10:20] = 1000
    img[:, 30:50, 28:40] = 900  # crosses the 32-px tile seams
    return img


def test_segment_info_seams(tmp_path, blobs, capsys):
    store = to_ome_zarr(blobs, tmp_path / "a.zarr", axes="zyx", n_levels=2)
    args = ["segment", store, "--method", "threshold"]
    args += ["--tile-shape", "2,32,32", "--overlap", "4", "--stitch", "iou"]
    assert main(args) == 0
    labels = zarr.open_group(f"{store}/labels/labels", mode="r")["0"][:]
    assert len(np.unique(labels)) - 1 == 2  # the seam-crossing blob is one

    capsys.readouterr()
    assert main(["info", store]) == 0
    out = capsys.readouterr().out
    assert "labels/labels" in out and "level 1" in out
    assert "ZstdCodec" in out

    seams = ["seams", f"{store}/labels/labels", "--tile-shape", "2,32,32"]
    assert main(seams) == 0
    report = json.loads(capsys.readouterr().out)
    assert set(report) == {"axes", "worst_seams"}


def test_convert_with_pixel_size_and_codec(tmp_path, blobs):
    src = to_ome_zarr(blobs, tmp_path / "src.zarr", axes="zyx", n_levels=1)
    out = str(tmp_path / "c.zarr")
    args = ["convert", src, out, "--axes", "zyx", "--levels", "2"]
    args += ["--pixel-size", "0.5,0.2,0.2", "--compression", "blosc"]
    assert main(args) == 0
    assert read_pixel_size(out) == {"z": 0.5, "y": 0.2, "x": 0.2}
    codec = zarr.open_group(out, mode="r")["0"].compressors[0]
    assert type(codec).__name__ == "BloscCodec"


def test_segment_rejects_incomplete_methods(tmp_path, blobs):
    store = to_ome_zarr(blobs, tmp_path / "a.zarr", axes="zyx", n_levels=1)
    with pytest.raises(SystemExit, match="--fn"):
        main(["segment", store, "--method", "custom"])
    with pytest.raises(SystemExit):
        main(["segment", store, "--channel", "red"])
