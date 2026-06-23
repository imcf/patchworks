"""Tests for the OME-ZARR conversion plugin."""

import dask.array as da
import numpy as np
import pytest

import zarr

from patchworks import load_ome_zarr
from patchworks.plugins.ome_zarr import (
    add_pyramid,
    to_ome_zarr,
    write_labels,
)


def _level_scale(store, level):
    root = zarr.open_group(str(store), mode="r")
    ds = root.attrs["multiscales"][0]["datasets"][level]
    return ds["coordinateTransformations"][0]["scale"]


def test_pixel_size_written_and_scaled(tmp_path):
    """Physical voxel size lands in NGFF scale; X/Y scale, Z stays."""
    out = tmp_path / "cal.zarr"
    to_ome_zarr(
        da.zeros((8, 8, 8), "uint16"),
        out,
        axes="zyx",
        pixel_size={"z": 2.0, "y": 0.5, "x": 0.5},
        n_levels=2,
    )
    # level 0 = physical size; level 1 doubles X/Y, keeps Z.
    assert _level_scale(out, 0) == [2.0, 0.5, 0.5]
    assert _level_scale(out, 1) == [2.0, 1.0, 1.0]
    root = zarr.open_group(str(out), mode="r")
    units = [a.get("unit") for a in root.attrs["multiscales"][0]["axes"]]
    assert units == ["micrometer", "micrometer", "micrometer"]


def test_imaris_without_reader(tmp_path):
    """A .ims path without the reader raises an actionable ImportError."""
    try:
        import imaris_ims_file_reader  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="imaris"):
            to_ome_zarr(str(tmp_path / "scan.ims"), tmp_path / "o.zarr")
    else:
        pytest.skip("imaris reader installed; ImportError path not exercised")


def test_pyramid_roundtrip(tmp_path):
    """Levels are written, downsampled by striding, and read back intact."""
    a = np.arange(8 * 8 * 8, dtype="int32").reshape(8, 8, 8)
    out = tmp_path / "vol.zarr"

    to_ome_zarr(a, out, axes="zyx", n_levels=3, downscale=2)

    l0 = load_ome_zarr(out, channel=None, level=0)
    l1 = load_ome_zarr(out, channel=None, level=1)
    l2 = load_ome_zarr(out, channel=None, level=2)

    # Z is kept at full resolution; only X/Y are downsampled.
    assert l0.shape == (8, 8, 8)
    assert l1.shape == (8, 4, 4)
    assert l2.shape == (8, 2, 2)
    # Full resolution is byte-identical; downsampling is nearest (label-safe).
    assert np.array_equal(np.asarray(l0), a)
    assert np.array_equal(np.asarray(l1), a[:, ::2, ::2])


def test_non_spatial_axis_not_downsampled(tmp_path):
    """A channel axis keeps its size across pyramid levels."""
    a = da.zeros((3, 16, 16), dtype="uint16")
    out = tmp_path / "cyx.zarr"

    to_ome_zarr(a, out, axes="cyx", n_levels=2, downscale=2)

    assert load_ome_zarr(out, channel=None, level=1).shape == (3, 8, 8)


def test_axes_length_mismatch(tmp_path):
    with pytest.raises(ValueError):
        to_ome_zarr(
            np.zeros((4, 4), "uint8"), tmp_path / "bad.zarr", axes="zyx"
        )


def test_unreadable_format_without_bioio(tmp_path):
    """A non-zarr file with bioio absent raises an actionable ImportError."""
    try:
        import bioio  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="bioio"):
            to_ome_zarr(str(tmp_path / "scan.czi"), tmp_path / "out.zarr")
    else:
        pytest.skip("bioio installed; ImportError path not exercised")


def test_add_pyramid_to_flat_store(tmp_path):
    """add_pyramid turns a single-array store into a multi-scale one."""
    base = np.arange(8 * 8 * 8, dtype="int32").reshape(8, 8, 8)
    store = str(tmp_path / "flat.zarr")
    da.to_zarr(da.from_array(base, chunks=(8, 8, 8)), store, component="0")

    add_pyramid(store, base="0", axes="zyx", n_levels=3, downscale=2)

    assert load_ome_zarr(store, channel=None, level=0).shape == (8, 8, 8)
    l1 = load_ome_zarr(store, channel=None, level=1)
    assert l1.shape == (8, 4, 4)  # Z preserved
    assert np.array_equal(np.asarray(l1), base[:, ::2, ::2])


def test_write_labels_into_store(tmp_path):
    """Labels land under labels/<name>/ as a registered NGFF pyramid."""
    store = to_ome_zarr(
        np.zeros((8, 8, 8), "uint16"), tmp_path / "img.zarr", n_levels=2
    )
    labels = np.ones((8, 8, 8), dtype="int32")

    group = write_labels(store, labels, name="cells", n_levels=2)

    # registered in the parent labels group
    labels_grp = zarr.open_group(f"{store}/labels", mode="r")
    assert "cells" in labels_grp.attrs["labels"]
    # readable as a multi-scale label image with image-label metadata
    assert load_ome_zarr(group, channel=None, level=0).shape == (8, 8, 8)
    assert load_ome_zarr(group, channel=None, level=1).shape == (8, 4, 4)
    lg = zarr.open_group(group, mode="r")
    assert lg.attrs["image-label"]["version"]


def test_reuse_pyramid_ignored_for_arrays(tmp_path):
    """reuse_pyramid only affects .ims inputs; arrays still rebuild."""
    out = to_ome_zarr(
        np.zeros((8, 8, 8), "uint16"),
        tmp_path / "arr.zarr",
        n_levels=2,
        reuse_pyramid=True,
    )
    assert load_ome_zarr(out, channel=None, level=1).shape == (8, 4, 4)
