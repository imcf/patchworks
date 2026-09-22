"""Tests for the napari plugin's lazy data resolvers (no display needed)."""

import dask.array as da
import numpy as np
import pytest

from patchworks.plugins import napari as nplugin
from patchworks.plugins.ome_zarr import to_ome_zarr


def test_resolve_image_multiscale(tmp_path):
    """An OME-ZARR pyramid resolves to a multi-scale list of dask arrays."""
    to_ome_zarr(
        np.zeros((16, 16, 16), "uint16"), tmp_path / "img.zarr", n_levels=3
    )
    out = nplugin._resolve_image(tmp_path / "img.zarr", channel=None)
    assert isinstance(out, list)
    assert len(out) == 3
    assert all(isinstance(lvl, da.Array) for lvl in out)
    assert out[1].shape == (16, 8, 8)  # Z preserved, only X/Y downsampled


def test_resolve_labels_plain_zarr(tmp_path):
    """A plain tile_process label store resolves via its component, as int32."""
    labels = da.from_array(np.ones((4, 8, 8), "int64"), chunks=(4, 8, 8))
    da.to_zarr(labels, str(tmp_path / "labels.zarr"), component="labels")
    out = nplugin._resolve_labels(tmp_path / "labels.zarr", component="labels")
    assert isinstance(out, da.Array)
    assert out.dtype == np.int32
    assert out.shape == (4, 8, 8)


def test_require_napari_message(monkeypatch):
    """Missing napari yields an actionable ImportError; otherwise it imports."""
    try:
        import napari  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="patchworks\\[napari\\]"):
            nplugin._require_napari()
    else:
        assert nplugin._require_napari() is napari


def test_inner_label_discovery(tmp_path):
    """Labels written into a store are discoverable for auto-overlay."""
    import numpy as np

    from patchworks.plugins.ome_zarr import to_ome_zarr, write_labels

    store = to_ome_zarr(
        np.zeros((8, 8, 8), "uint16"), tmp_path / "scan.zarr", n_levels=2
    )
    write_labels(store, np.ones((8, 8, 8), "int32"), name="cells", n_levels=2)

    assert nplugin._inner_label_names(store) == ["cells"]
    levels = nplugin._multiscale_levels(f"{store}/labels/cells", None)
    assert len(levels) == 2
    assert levels[1].shape == (8, 4, 4)  # Z preserved, XY downsampled


def test_inner_label_discovery_none(tmp_path):
    """A store without labels yields an empty list (image-only view)."""
    import numpy as np

    from patchworks.plugins.ome_zarr import to_ome_zarr

    store = to_ome_zarr(
        np.zeros((8, 8, 8), "uint16"), tmp_path / "img.zarr", n_levels=1
    )
    assert nplugin._inner_label_names(store) == []


def test_label_hint_present_when_n_objects_written(tmp_path):
    """write_labels(..., n_objects=...) is readable back via _label_hint."""
    from patchworks.plugins.ome_zarr import to_ome_zarr, write_labels

    store = to_ome_zarr(
        np.zeros((8, 8, 8), "uint16"), tmp_path / "scan.zarr", n_levels=1
    )
    write_labels(
        store,
        np.ones((8, 8, 8), "int32"),
        name="cells",
        n_levels=1,
        n_objects=17,
    )

    hint = nplugin._label_hint(f"{store}/labels/cells")
    assert hint == {"n_objects": 17, "sequential_labels": True}


def test_label_hint_empty_without_n_objects(tmp_path):
    """No n_objects= at write time -> no hint, not a misleading default."""
    from patchworks.plugins.ome_zarr import to_ome_zarr, write_labels

    store = to_ome_zarr(
        np.zeros((8, 8, 8), "uint16"), tmp_path / "scan.zarr", n_levels=1
    )
    write_labels(store, np.ones((8, 8, 8), "int32"), name="cells", n_levels=1)

    assert nplugin._label_hint(f"{store}/labels/cells") == {}


def test_label_hint_missing_store_returns_empty():
    """A path that doesn't exist (or isn't a label group) just yields {}."""
    assert nplugin._label_hint("/no/such/store.zarr") == {}


def test_a_zip_bundle_opens_exactly_like_the_directory(tmp_path):
    """`pixi run napari` must work on a bundle, not only on a directory.

    Every reader here builds group paths by string-joining
    (f"{store}/labels/{name}"), which a bundle breaks: the archive is a
    file. Without this, packing a store made it unviewable.
    """
    import zipfile
    from pathlib import Path

    import zarr

    from patchworks.plugins import napari as napari_plugin
    from patchworks.plugins.ome_zarr import (
        read_pixel_size,
        register_labels,
        to_ome_zarr,
    )

    image = np.arange(4 * 64 * 64, dtype="uint16").reshape(4, 64, 64)
    store = to_ome_zarr(
        image,
        tmp_path / "image.zarr",
        axes="zyx",
        n_levels=3,
        chunks=(2, 32, 32),
        pixel_size={"z": 0.24, "y": 0.108, "x": 0.108},
        progress=False,
    )
    labels = np.zeros((4, 64, 64), dtype="uint32")
    labels[1:3, 10:30, 10:30] = 7
    group = zarr.open_group(f"{store}/labels/cilia", mode="a")
    base = group.create_array(
        "0", shape=labels.shape, chunks=(2, 32, 32), dtype="uint32"
    )
    base[:] = labels
    register_labels(store, "cilia", n_levels=3, progress=False, n_objects=1)

    bundle = tmp_path / "image.zarr.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_STORED) as archive:
        for item in sorted(Path(store).rglob("*")):
            if item.is_file():
                archive.write(item, item.relative_to(Path(store).parent))

    for source in (str(store), str(bundle)):
        label_group = f"{source}/labels/cilia"
        assert napari_plugin._inner_label_names(source) == ["cilia"], source
        assert napari_plugin._has_multiscales(source), source
        assert [
            tuple(level.shape)
            for level in napari_plugin._multiscale_levels(source, None)
        ] == [(4, 64, 64), (4, 32, 32), (4, 16, 16)], source
        assert [
            tuple(level.shape)
            for level in napari_plugin._multiscale_levels(label_group, None)
        ] == [(4, 64, 64), (4, 32, 32), (4, 16, 16)], source
        assert napari_plugin._label_hint(label_group)["n_objects"] == 1
        assert napari_plugin._pyramid_calibration(label_group, 3) == (
            [0.24, 0.108, 0.108],
            ["micrometer"] * 3,
        ), source
        assert read_pixel_size(source) == {
            "z": 0.24,
            "y": 0.108,
            "x": 0.108,
        }, source

    # And the pixel data itself round-trips.
    from patchworks import load_ome_zarr

    assert np.array_equal(
        np.asarray(load_ome_zarr(str(bundle), channel=None, level=0)), image
    )
