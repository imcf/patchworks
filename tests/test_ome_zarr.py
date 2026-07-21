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


def test_tiff_sequence_conversion(tmp_path):
    """A folder of single-plane TIFFs is wrapped lazily and converted.

    The filename pattern lists Z before C, but the output must still come
    out channel-first (patchworks' tczyx convention), since load_ome_zarr /
    tile_process hard-assume axis 0 is the channel axis.
    """
    tifffile = pytest.importorskip("tifffile")
    n_z, n_c, size = 3, 2, 8
    for z in range(n_z):
        for c in range(n_c):
            img = np.full((size, size), z * 10 + c, dtype="uint16")
            tifffile.imwrite(
                tmp_path / f"sample_Z{z:03d}_C{c}_V0.tif",
                img,
                resolution=(20000.0, 20000.0),
                resolutionunit="CENTIMETER",
            )

    out = tmp_path / "out.zarr"
    to_ome_zarr(
        str(tmp_path / "*.tif"),
        out,
        sequence_pattern=r"_Z(?P<Z>\d+)_C(?P<C>\d+)_V\d+",
        n_levels=1,
    )

    result = np.asarray(load_ome_zarr(out, channel=None))
    assert result.shape == (n_c, n_z, size, size)  # channel-first, not z-first
    # each plane's constant value encodes its (z, c) position.
    assert (
        result[:, :, 0, 0]
        == [[z * 10 + c for z in range(n_z)] for c in range(n_c)]
    ).all()
    assert _level_scale(out, 0) == pytest.approx([1.0, 1.0, 0.5, 0.5])

    # per-channel selection picks the right plane regardless of pattern order.
    ch1 = np.asarray(load_ome_zarr(out, channel=1))
    assert ch1.shape == (n_z, size, size)
    assert (ch1[:, 0, 0] == [z * 10 + 1 for z in range(n_z)]).all()


def test_tiff_sequence_drops_singleton_time_axis(tmp_path):
    """A constant T in the pattern is dropped, keeping channel at axis 0."""
    tifffile = pytest.importorskip("tifffile")
    n_z, n_c, size = 2, 3, 8
    for z in range(n_z):
        for c in range(n_c):
            img = np.full((size, size), z * 10 + c, dtype="uint16")
            tifffile.imwrite(tmp_path / f"sample_T0_Z{z:03d}_C{c}_V0.tif", img)

    out = tmp_path / "out.zarr"
    to_ome_zarr(
        str(tmp_path / "*.tif"),
        out,
        sequence_pattern=r"_T(?P<T>\d+)_Z(?P<Z>\d+)_C(?P<C>\d+)_V\d+",
        n_levels=1,
    )

    result = np.asarray(load_ome_zarr(out, channel=None))
    assert result.shape == (n_c, n_z, size, size)  # no leftover T axis
    ch2 = np.asarray(load_ome_zarr(out, channel=2))
    assert ch2.shape == (n_z, size, size)
    assert (ch2[:, 0, 0] == [z * 10 + 2 for z in range(n_z)]).all()


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


def test_write_labels_n_objects_persisted(tmp_path):
    """n_objects lands in the label group's attrs for a downstream reader."""
    store = to_ome_zarr(
        np.zeros((8, 8, 8), "uint16"), tmp_path / "img.zarr", n_levels=2
    )
    labels = np.ones((8, 8, 8), dtype="int32")

    group = write_labels(store, labels, name="cells", n_levels=2, n_objects=42)

    lg = zarr.open_group(group, mode="r")
    assert lg.attrs["n_objects"] == 42
    assert lg.attrs["sequential_labels"] is True


def test_write_labels_no_n_objects_by_default(tmp_path):
    """Without n_objects=, no misleading count is written."""
    store = to_ome_zarr(
        np.zeros((8, 8, 8), "uint16"), tmp_path / "img.zarr", n_levels=2
    )
    labels = np.ones((8, 8, 8), dtype="int32")

    group = write_labels(store, labels, name="cells", n_levels=2)

    lg = zarr.open_group(group, mode="r")
    assert "n_objects" not in lg.attrs
    assert "sequential_labels" not in lg.attrs


def test_reuse_pyramid_ignored_for_arrays(tmp_path):
    """reuse_pyramid only affects .ims inputs; arrays still rebuild."""
    out = to_ome_zarr(
        np.zeros((8, 8, 8), "uint16"),
        tmp_path / "arr.zarr",
        n_levels=2,
        reuse_pyramid=True,
    )
    assert load_ome_zarr(out, channel=None, level=1).shape == (8, 4, 4)


def test_sharding(tmp_path):
    """shard=True/tuple writes zarr-v3 shards; data round-trips intact."""
    import zarr as _zarr

    a = np.arange(4 * 64 * 64, dtype="uint16").reshape(4, 64, 64)

    out = to_ome_zarr(
        a,
        tmp_path / "s.zarr",
        axes="zyx",
        n_levels=2,
        chunks=(2, 16, 16),
        shard=True,
    )
    z0 = _zarr.open_array(f"{out}/0", mode="r")
    assert z0.chunks == (2, 16, 16)
    assert z0.shards is not None and z0.shards != z0.chunks
    assert np.array_equal(
        np.asarray(load_ome_zarr(out, channel=None, level=0)), a
    )

    out2 = to_ome_zarr(
        a,
        tmp_path / "e.zarr",
        axes="zyx",
        n_levels=1,
        chunks=(2, 16, 16),
        shard=(2, 32, 32),
    )
    assert _zarr.open_array(f"{out2}/0", mode="r").shards == (2, 32, 32)

    out3 = to_ome_zarr(
        a, tmp_path / "n.zarr", axes="zyx", n_levels=1, chunks=(2, 16, 16)
    )
    assert (
        getattr(_zarr.open_array(f"{out3}/0", mode="r"), "shards", None) is None
    )
