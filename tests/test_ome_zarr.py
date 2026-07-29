"""Tests for the OME-ZARR conversion plugin."""

import dask.array as da
import numpy as np
import pytest

import zarr

from patchworks import load_ome_zarr
from patchworks.plugins.ome_zarr import (
    add_pyramid,
    read_ngff_attr,
    to_ome_zarr,
    write_labels,
)


def test_streaming_pyramid_matches_dask(tmp_path):
    """The streaming levels must be byte-identical to the dask-built ones.

    The streaming path exists to bound memory (one source chunk per task); it
    must not change a single voxel of the result.
    """
    rng = np.random.default_rng(0)
    data = rng.integers(0, 500, size=(9, 100, 100), dtype="uint16")

    streamed = tmp_path / "streamed.zarr"
    to_ome_zarr(data, streamed, axes="zyx", n_levels=4, progress=False)
    # chunks= forces the dask route through the same public entry point.
    dasked = tmp_path / "dasked.zarr"
    to_ome_zarr(
        data,
        dasked,
        axes="zyx",
        n_levels=4,
        chunks=(16, 1024, 1024),
        progress=False,
    )

    a_root = zarr.open_group(str(streamed), mode="r")
    b_root = zarr.open_group(str(dasked), mode="r")
    a_levels = [
        d["path"]
        for d in read_ngff_attr(a_root.attrs, "multiscales")[0]["datasets"]
    ]
    b_levels = [
        d["path"]
        for d in read_ngff_attr(b_root.attrs, "multiscales")[0]["datasets"]
    ]
    assert a_levels == b_levels, "both routes must produce the same levels"
    for lvl in a_levels:
        assert np.array_equal(
            np.asarray(a_root[lvl]), np.asarray(b_root[lvl])
        ), f"level {lvl} differs between the streaming and dask routes"


def test_streaming_level_reads_one_source_chunk_per_task(tmp_path):
    """Each output chunk must map onto exactly one source chunk.

    That alignment is what makes peak memory ``n_workers x one chunk`` and so
    is the property the OOM fix rests on -- not an incidental detail.
    """
    from patchworks.plugins.ome_zarr import _level_chunks

    # A (16, 1024, 1024) source at stride (1, 2, 2) -> (16, 512, 512).
    out = _level_chunks((16, 1024, 1024), (1, 2, 2), "zyx", (16, 4096, 4096))
    assert out == (16, 512, 512)
    # Deep levels get floored rather than fragmenting into tiny chunks.
    floored = _level_chunks((16, 128, 128), (1, 2, 2), "zyx", (16, 256, 256))
    assert floored == (16, 128, 128)
    # Never larger than the level itself.
    clamped = _level_chunks((16, 1024, 1024), (1, 2, 2), "zyx", (4, 100, 100))
    assert clamped == (4, 100, 100)


def test_ngff_layout_matches_the_zarr_version(tmp_path):
    """A zarr v3 store must carry 0.5-style nested metadata, not 0.4's.

    NGFF 0.4 is defined over zarr v2 with the keys at the top level; 0.5 is
    the v3 revision and nests them under "ome". Writing v3 data with 0.4's
    layout matches neither, and a strict 0.5 reader finds nothing.
    """
    import json

    from patchworks.plugins.ome_zarr import _ZARR_V3

    out = tmp_path / "layout.zarr"
    to_ome_zarr(np.zeros((4, 8, 8), "uint16"), out, axes="zyx", n_levels=2)
    write_labels(out, np.ones((4, 8, 8), "int32"), name="cells", n_objects=1)

    root = json.load(open(out / "zarr.json"))["attributes"]
    if _ZARR_V3:
        assert "ome" in root, "v3 must nest NGFF keys under 'ome'"
        assert root["ome"]["version"] == "0.5"
        assert "multiscales" in root["ome"]
        assert "multiscales" not in root, "0.4 layout must not linger"
    else:
        assert root["multiscales"][0]["version"] == "0.4"

    # Both label keys land in the same place, written at different times.
    lg = zarr.open_group(f"{out}/labels/cells", mode="r")
    assert read_ngff_attr(lg.attrs, "multiscales") is not None
    assert read_ngff_attr(lg.attrs, "image-label") is not None
    # patchworks' own hints stay top-level so consumers need not know NGFF.
    assert lg.attrs["n_objects"] == 1

    # And everything still round-trips through our own readers.
    assert load_ome_zarr(out, channel=None, level=1).shape == (4, 4, 4)


def test_reader_accepts_the_legacy_04_layout(tmp_path):
    """Stores written by older patchworks (0.4 top-level) must still load."""
    store = tmp_path / "legacy.zarr"
    root = zarr.open_group(str(store), mode="w")
    root.create_array(name="0", shape=(2, 4, 4), chunks=(2, 4, 4), dtype="u2")
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "legacy",
            "axes": [{"name": a, "type": "space"} for a in "zyx"],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.0, 1.0]}
                    ],
                }
            ],
        }
    ]
    assert load_ome_zarr(store, channel=None, level=0).shape == (2, 4, 4)
    assert read_ngff_attr(root.attrs, "multiscales")[0]["name"] == "legacy"


def _level_scale(store, level):
    root = zarr.open_group(str(store), mode="r")
    ds = read_ngff_attr(root.attrs, "multiscales")[0]["datasets"][level]
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
    units = [
        a.get("unit")
        for a in read_ngff_attr(root.attrs, "multiscales")[0]["axes"]
    ]
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
    """A constant T in the pattern is dropped, keeping channel at axis 0.

    Each file's own array has leading singleton dims (1, 1, y, x) — what
    tifffile.imread reports even for a plain single-page TIFF in practice —
    to make sure those get stripped instead of being mistaken for real
    sequence axes.
    """
    tifffile = pytest.importorskip("tifffile")
    n_z, n_c, size = 2, 3, 8
    for z in range(n_z):
        for c in range(n_c):
            img = np.full((1, 1, size, size), z * 10 + c, dtype="uint16")
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
    assert "cells" in read_ngff_attr(labels_grp.attrs, "labels")
    # readable as a multi-scale label image with image-label metadata
    assert load_ome_zarr(group, channel=None, level=0).shape == (8, 8, 8)
    assert load_ome_zarr(group, channel=None, level=1).shape == (8, 4, 4)
    lg = zarr.open_group(group, mode="r")
    assert read_ngff_attr(lg.attrs, "image-label")["version"]


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


def test_glob_without_sequence_pattern_says_so(tmp_path):
    """A glob input must name the missing setting, not blame the format.

    Without sequence_pattern the glob fell through to bioio, which reads one
    file and could only report "does not support the image" -- pointing at the
    path instead of at the config key that was missing.
    """
    with pytest.raises(ValueError, match="sequence_pattern is not set"):
        to_ome_zarr(str(tmp_path / "*.tif"), tmp_path / "out.zarr")

    # A real single file must still reach the normal readers.
    with pytest.raises(Exception) as exc:
        to_ome_zarr(str(tmp_path / "scan.ims"), tmp_path / "o2.zarr")
    assert "sequence_pattern" not in str(exc.value)


def test_base_chunks_never_group_source_chunks(tmp_path):
    """One output chunk must not require several source chunks.

    A folder of stitched TIFFs gives one dask chunk per file, so a plane can
    be gigabytes. The z cap of 16 would group sixteen of them into one output
    chunk -- ~58 GB held to write 32 MB, against a 64 GB job. Splitting a
    source chunk is fine; combining several is not.
    """
    from patchworks.plugins.ome_zarr import _default_chunks

    shape = (4, 126, 45961, 42072)
    # one chunk per file: (c=1, z=1, whole plane)
    assert _default_chunks(
        shape, "czyx", source_chunks=(1, 1, 45961, 42072)
    ) == (
        1,
        1,
        1024,
        1024,
    )
    # a normally-chunked source is unaffected by the cap
    assert _default_chunks(shape, "czyx", source_chunks=shape) == (
        1,
        16,
        1024,
        1024,
    )
    # and omitting it keeps the previous behaviour
    assert _default_chunks(shape, "czyx") == (1, 16, 1024, 1024)


def test_tiff_sequence_keeps_one_plane_per_chunk(tmp_path):
    """End to end: a per-file source must not be z-grouped on write."""
    tifffile = pytest.importorskip("tifffile")
    n_z, n_c, size = 4, 2, 8
    for z in range(n_z):
        for c in range(n_c):
            tifffile.imwrite(
                tmp_path / f"s_Z{z:03d}_C{c}_V0.tif",
                np.full((size, size), z * 10 + c, "uint16"),
            )

    out = tmp_path / "seq.zarr"
    to_ome_zarr(
        str(tmp_path / "*.tif"),
        out,
        sequence_pattern=r"_Z(?P<Z>\d+)_C(?P<C>\d+)_V\d+",
        n_levels=1,
        progress=False,
    )
    level0 = zarr.open_array(str(out) + "/0", mode="r")
    assert level0.chunks[1] == 1, (
        f"z was grouped into {level0.chunks[1]} planes per chunk; each source "
        "chunk is a whole file, so that multiplies the read"
    )
    # and the data still round-trips
    result = np.asarray(load_ome_zarr(out, channel=None))
    assert result.shape == (n_c, n_z, size, size)
    assert (
        result[:, :, 0, 0]
        == [[z * 10 + c for z in range(n_z)] for c in range(n_c)]
    ).all()


def _ome_xml(x, y, z, unit="um"):
    return (
        '<?xml version="1.0"?>'
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        f'<Image><Pixels PhysicalSizeX="{x}" PhysicalSizeXUnit="{unit}"'
        f' PhysicalSizeY="{y}" PhysicalSizeYUnit="{unit}"'
        f' PhysicalSizeZ="{z}" PhysicalSizeZUnit="{unit}"/></Image></OME>'
    )


def test_pixel_size_from_ome_xml(tmp_path):
    """OME-XML PhysicalSize* must be read, not just the resolution tags.

    Stitched output commonly writes OME-XML into ImageDescription and leaves
    the TIFF tags at their defaults, so a tag-only reader calls a perfectly
    calibrated image uncalibrated -- and it is the only source that can give
    z for a sequence of single planes.
    """
    tifffile = pytest.importorskip("tifffile")
    from patchworks.plugins.ome_zarr import _tiff_pixel_size

    path = tmp_path / "ome.tif"
    tifffile.imwrite(
        path,
        np.zeros((8, 8), "uint16"),
        description=_ome_xml(0.325, 0.325, 1.5),
    )
    assert _tiff_pixel_size(str(path)) == pytest.approx(
        {"z": 1.5, "y": 0.325, "x": 0.325}
    )

    # mm is converted, not taken at face value
    mm = tmp_path / "mm.tif"
    tifffile.imwrite(
        mm,
        np.zeros((8, 8), "uint16"),
        description=_ome_xml(0.001, 0.001, 0.002, "mm"),
    )
    assert _tiff_pixel_size(str(mm)) == pytest.approx(
        {"z": 2.0, "y": 1.0, "x": 1.0}
    )


def test_ome_xml_wins_over_the_resolution_tags(tmp_path):
    """The explicit source wins per axis when a file carries both."""
    tifffile = pytest.importorskip("tifffile")
    from patchworks.plugins.ome_zarr import _tiff_pixel_size

    path = tmp_path / "both.tif"
    tifffile.imwrite(
        path,
        np.zeros((8, 8), "uint16"),
        description=_ome_xml(0.325, 0.325, 1.5),
        resolution=(20000.0, 20000.0),  # would say 0.5 µm
        resolutionunit="CENTIMETER",
    )
    got = _tiff_pixel_size(str(path))
    assert got["x"] == pytest.approx(0.325), "OME-XML must win over the tag"
    assert got["z"] == pytest.approx(1.5)


def test_uncalibrated_tiff_reports_nothing(tmp_path):
    """No metadata gives an empty dict rather than an invented default."""
    tifffile = pytest.importorskip("tifffile")
    from patchworks.plugins.ome_zarr import _tiff_pixel_size

    path = tmp_path / "bare.tif"
    tifffile.imwrite(path, np.zeros((8, 8), "uint16"))
    assert _tiff_pixel_size(str(path)) == {}
