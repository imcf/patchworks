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


def test_pyramid_roundtrip(tmp_path):
    """Levels are written, downsampled by striding, and read back intact."""
    a = np.arange(8 * 8 * 8, dtype="int32").reshape(8, 8, 8)
    out = tmp_path / "vol.zarr"

    to_ome_zarr(a, out, axes="zyx", n_levels=3, downscale=2)

    l0 = load_ome_zarr(out, channel=None, level=0)
    l1 = load_ome_zarr(out, channel=None, level=1)
    l2 = load_ome_zarr(out, channel=None, level=2)

    assert l0.shape == (8, 8, 8)
    assert l1.shape == (4, 4, 4)
    assert l2.shape == (2, 2, 2)
    # Full resolution is byte-identical; downsampling is nearest (label-safe).
    assert np.array_equal(np.asarray(l0), a)
    assert np.array_equal(np.asarray(l1), a[::2, ::2, ::2])


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
    assert l1.shape == (4, 4, 4)
    assert np.array_equal(np.asarray(l1), base[::2, ::2, ::2])


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
    assert load_ome_zarr(group, channel=None, level=1).shape == (4, 4, 4)
    lg = zarr.open_group(group, mode="r")
    assert lg.attrs["image-label"]["version"]
