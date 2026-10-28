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
    assert hint == {
        "patchworks_labels": f"{store}/labels/cells",
        "n_objects": 17,
        "sequential_labels": True,
    }


def test_label_hint_empty_without_n_objects(tmp_path):
    """No n_objects= at write time -> no hint, not a misleading default."""
    from patchworks.plugins.ome_zarr import to_ome_zarr, write_labels

    store = to_ome_zarr(
        np.zeros((8, 8, 8), "uint16"), tmp_path / "scan.zarr", n_levels=1
    )
    write_labels(store, np.ones((8, 8, 8), "int32"), name="cells", n_levels=1)

    # Where the labels are (for their object table), but no object count.
    assert nplugin._label_hint(f"{store}/labels/cells") == {
        "patchworks_labels": f"{store}/labels/cells"
    }


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

    # The entry points view_in_napari actually calls. Testing only the
    # helpers above missed that _is_zarr gated on a ".zarr" suffix, so a
    # bundle was handed to bioio and failed on a missing optional
    # dependency -- every reader below it worked fine.
    for source in (str(store), str(bundle)):
        assert napari_plugin._is_zarr(source), source
        resolved = napari_plugin._resolve_image(source, None)
        assert [tuple(level.shape) for level in resolved] == [
            (4, 64, 64),
            (4, 32, 32),
            (4, 16, 16),
        ], source
    # Not asserted here: _resolve_labels() with an explicit label-group path
    # ("<store>/labels/<name>"). It gates multiscale detection on _is_zarr,
    # which that path fails for a directory store, so it tries to open a
    # group as an array. Pre-existing, and not the path view_in_napari takes
    # for auto-loaded labels -- that one goes through _multiscale_levels,
    # asserted above for both sources.

    # A non-zarr path must still go to bioio, not be mistaken for a store.
    assert not napari_plugin._is_zarr("scan.ims")
    assert not napari_plugin._is_zarr(42)


@pytest.fixture
def stub_napari(monkeypatch):
    """A recording stand-in for napari, so view_in_napari can be driven here.

    The plugin's resolvers were covered individually while the function that
    calls them was not, which is how a bundle reached bioio: every helper
    worked, the dispatcher above them did not. This drives the real
    entry point and hands back the layers it would have added.
    """
    import sys
    import types

    added = []

    class _Viewer:
        def add_image(self, data, **kwargs):
            added.append(("image", kwargs.get("name"), data, kwargs))

        def add_labels(self, data, **kwargs):
            added.append(("labels", kwargs.get("name"), data, kwargs))

    napari = types.ModuleType("napari")
    napari.Viewer = _Viewer
    utils = types.ModuleType("napari.utils")
    colormaps = types.ModuleType("napari.utils.colormaps")
    colormaps.CyclicLabelColormap = type("CyclicLabelColormap", (), {})
    napari.utils = utils
    utils.colormaps = colormaps
    for name, module in (
        ("napari", napari),
        ("napari.utils", utils),
        ("napari.utils.colormaps", colormaps),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return added


def _multi_store(tmp_path):
    """An image with three label groups, as a real run leaves it."""
    import zarr

    from patchworks.plugins.ome_zarr import register_labels

    image = np.arange(2 * 4 * 32 * 32, dtype="uint16").reshape(2, 4, 32, 32)
    store = to_ome_zarr(
        image,
        tmp_path / "image.zarr",
        axes="czyx",
        n_levels=2,
        chunks=(1, 2, 16, 16),
        pixel_size={"z": 0.24, "y": 0.10833, "x": 0.10833},
        progress=False,
    )
    for index, name in enumerate(
        ("nuclei_labels", "cyto_labels", "cilia_labels"), start=1
    ):
        labels = np.zeros((4, 32, 32), dtype="uint32")
        labels[1:3, 4:12, 4:12] = index
        group = zarr.open_group(f"{store}/labels/{name}", mode="a")
        base = group.create_array(
            "0", shape=labels.shape, chunks=(2, 16, 16), dtype="uint32"
        )
        base[:] = labels
        register_labels(
            store, name, n_levels=2, progress=False, n_objects=index
        )
    return store


def _bundle(store, tmp_path, name="image.zarr.zip"):
    import zipfile
    from pathlib import Path as _Path

    out = tmp_path / name
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as archive:
        for item in sorted(_Path(store).rglob("*")):
            if item.is_file():
                archive.write(item, item.relative_to(_Path(store).parent))
    return out


def test_view_in_napari_loads_a_bundle_exactly_like_a_directory(
    tmp_path, stub_napari
):
    """Every layer, name, scale and hint must match between the two."""
    store = _multi_store(tmp_path)
    bundle = _bundle(store, tmp_path)

    def layers_for(source):
        stub_napari.clear()
        nplugin.view_in_napari(str(source), show=False, glasbey=False)
        return [
            (
                kind,
                name,
                [tuple(level.shape) for level in data],
                tuple(kwargs.get("scale") or ()),
                (kwargs.get("metadata") or {}).get("n_objects"),
            )
            for kind, name, data, kwargs in stub_napari
        ]

    from_directory = layers_for(store)
    from_bundle = layers_for(bundle)

    # The image plus all three label groups, auto-loaded.
    assert [(kind, name) for kind, name, *_ in from_directory] == [
        ("image", ["image [0]", "image [1]"]),
        ("labels", "nuclei_labels"),
        ("labels", "cyto_labels"),
        ("labels", "cilia_labels"),
    ]
    assert from_bundle == from_directory


def test_view_in_napari_bundle_keeps_channel_selection(tmp_path, stub_napari):
    """--channel has to work on a bundle too, not just the whole stack."""
    store = _multi_store(tmp_path)
    bundle = _bundle(store, tmp_path)

    for source in (store, bundle):
        stub_napari.clear()
        nplugin.view_in_napari(
            str(source), channel=1, show=False, glasbey=False
        )
        kind, name, data, _ = stub_napari[0]
        assert kind == "image"
        # Channel axis dropped, so the leading axis is z.
        assert data[0].shape == (4, 32, 32), source


def test_bundle_name_need_not_match_the_store(tmp_path, stub_napari):
    """`--output whatever.zip` must still open.

    The inner store name is read from the archive listing, not guessed
    from the filename.
    """
    store = _multi_store(tmp_path)
    bundle = _bundle(store, tmp_path, name="results-for-elena.zip")

    stub_napari.clear()
    nplugin.view_in_napari(str(bundle), show=False, glasbey=False)
    assert [name for _, name, _, _ in stub_napari] == [
        ["image [0]", "image [1]"],
        "nuclei_labels",
        "cyto_labels",
        "cilia_labels",
    ]


def test_resolve_image_reads_other_formats_and_picks_the_channel(monkeypatch):
    """A non-zarr file goes through bioio; that path crashed on unpacking."""
    import dask.array as da
    import numpy as np

    from patchworks.plugins import napari as napari_plugin
    from patchworks.plugins import ome_zarr

    cyx = da.from_array(np.arange(2 * 4 * 4).reshape(2, 4, 4))
    monkeypatch.setattr(
        ome_zarr, "_open_bioio", lambda path, scene: (cyx, "cyx", {})
    )
    every = napari_plugin._resolve_image("scan.czi", None)
    assert every.shape == (2, 4, 4)
    second = napari_plugin._resolve_image("scan.czi", 1)
    np.testing.assert_array_equal(second, cyx[1])


def test_numba_cache_kept_short_unless_set(monkeypatch, tmp_path):
    """numba's default cache sits next to napari's code; on a network share
    that path passes Windows' 260-character limit and labels fail to draw.
    The viewer moves it to a short local directory -- unless one is set."""
    import os

    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    nplugin.short_numba_cache()
    assert os.environ["NUMBA_CACHE_DIR"].endswith("patchworks-numba-cache")
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path))
    nplugin.short_numba_cache()
    assert os.environ["NUMBA_CACHE_DIR"] == str(tmp_path)


def test_view_script_sets_the_numba_cache_before_importing_napari():
    """view.py runs from the checkout, so its own setting reaches users
    before a patchworks release does -- and must come before the import."""
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1] / "workflow" / "scripts" / "view.py"
    ).read_text()
    assert src.index("NUMBA_CACHE_DIR") < src.index(
        "from patchworks.plugins.napari import"
    )
    # napari reads its settings when imported: asynchronous loading too.
    assert src.index("NAPARI_ASYNC") < src.index(
        "from patchworks.plugins.napari import"
    )


def test_axes_come_from_the_store_not_a_guess(tmp_path, caplog):
    """Every channel shown is a 4-D image: its axes are read from the
    store, without the "guessing 'czyx'" warning."""
    import logging

    import numpy as np

    from patchworks.plugins.napari import _pyramid_calibration
    from patchworks.plugins.ome_zarr import to_ome_zarr

    store = str(tmp_path / "s.zarr")
    to_ome_zarr(
        np.zeros((2, 4, 8, 8), "uint16"),
        store,
        axes="czyx",
        pixel_size={"z": 2.0, "y": 0.5, "x": 0.5},
        n_levels=1,
        progress=False,
    )
    with caplog.at_level(logging.WARNING):
        scale, _ = _pyramid_calibration(store, 4)
    assert scale[1:] == [2.0, 0.5, 0.5]
    assert "guessing" not in caplog.text


def test_channels_on_a_slider_or_split(tmp_path, stub_napari):
    """The image goes to napari lazily (dask): napari cuts the region on
    screen out of every axis before picking the plane, which read every
    channel and plane of it from a zarr array -- 8 minutes for a whole
    slide. Each channel is a layer, so several can be seen at once; only
    the first is visible (napari reads a hidden layer only once shown)."""
    import dask.array as da

    store = _multi_store(tmp_path)
    image = np.arange(2 * 4 * 32 * 32, dtype="uint16").reshape(2, 4, 32, 32)
    nplugin.view_in_napari(str(store), show=False, glasbey=False)
    (kind, name, data, kwargs) = stub_napari[0]
    assert name == ["image [0]", "image [1]"] and kwargs["channel_axis"] == 0
    assert kwargs["visible"] == [True, False]
    assert len(kwargs["scale"]) == len(kwargs["units"]) == 3
    assert kwargs["scale"][0] == 0.24
    assert all(isinstance(level, da.Array) for level in data)
    assert np.array_equal(np.asarray(data[0][1, 2]), image[1, 2])
    assert all(
        isinstance(level, da.Array)
        for _, _, data, _ in stub_napari[1:]
        for level in data
    )
    stub_napari.clear()
    nplugin.view_in_napari(str(store), show=False, glasbey=False, shown=[0, 1])
    assert stub_napari[0][3]["visible"] == [True, True]
    stub_napari.clear()
    nplugin.view_in_napari(
        str(store), show=False, glasbey=False, split_channels=False
    )
    (kind, name, data, kwargs) = stub_napari[0]
    assert name == "image" and "channel_axis" not in kwargs
    assert len(kwargs["scale"]) == 4


def test_sliders_are_named_after_the_axes(tmp_path):
    """napari labels its sliders -3, -4: they are named from the store."""
    import types

    pytest.importorskip("napari")
    from napari.components import ViewerModel

    store = _multi_store(tmp_path)
    viewer = ViewerModel()
    for data, name, kwargs in nplugin.image_layers(str(store), None):
        viewer.add_image(data, name=name, **kwargs)
    nplugin._name_axes(viewer, str(store))
    assert tuple(viewer.dims.axis_labels) == ("z", "y", "x")
    viewer = ViewerModel()
    for data, name, kwargs in nplugin.image_layers(
        str(store), None, split_channels=False
    ):
        viewer.add_image(data, name=name, **kwargs)
    nplugin._name_axes(viewer, str(store))
    assert tuple(viewer.dims.axis_labels) == ("channel", "z", "y", "x")
    # The test stand-in has no dims: nothing to name, no error.
    nplugin._name_axes(types.SimpleNamespace(layers=[]), str(store))


def test_label_colours_need_no_glasbey_package():
    """The palette is stored: importing glasbey took ~15 s and building the
    palette ~3 s, at every opening of the viewer."""
    from patchworks.plugins.napari import GLASBEY_256

    assert len(GLASBEY_256) == len(set(GLASBEY_256)) == 256
    assert GLASBEY_256[0] == "db202d"
    assert all(len(h) == 6 and int(h, 16) >= 0 for h in GLASBEY_256)


def test_object_tables_are_read_after_the_window_opens(monkeypatch):
    """A whole slide's tables took minutes and the window waited for them:
    they are read in a background thread and reach the layers afterwards."""
    import time
    import types

    pytest.importorskip("napari")
    qt = pytest.importorskip("qtpy.QtWidgets")
    app = qt.QApplication.instance() or qt.QApplication([])

    def slow(store):
        time.sleep(0.3)
        return {"cells": "features"}

    monkeypatch.setattr(nplugin, "_hover_tables", slow)
    layers = {"cells": types.SimpleNamespace(features=None)}
    viewer = types.SimpleNamespace(layers=layers)
    nplugin._show_relations_later(viewer, "store")
    assert layers["cells"].features is None  # not waited for
    end = time.time() + 10
    while layers["cells"].features is None and time.time() < end:
        app.processEvents()
        time.sleep(0.02)
    assert layers["cells"].features == "features"


def test_channel_layers_are_named_after_the_channels(tmp_path, stub_napari):
    """Each channel's layer takes its name from the store (NGFF omero
    metadata), as converting from a CZI, LIF, ND2 or IMS file now writes."""
    from patchworks.plugins.ome_zarr import channel_names, write_channel_names

    store = _multi_store(tmp_path)
    assert channel_names(str(store)) is None
    with pytest.raises(ValueError, match="3 channel name"):
        write_channel_names(str(store), ["a", "b", "c"])
    write_channel_names(str(store), ["DAPI", "Phalloidin"])
    assert channel_names(str(store)) == ["DAPI", "Phalloidin"]
    nplugin.view_in_napari(str(store), show=False, glasbey=False)
    assert stub_napari[0][1] == ["DAPI", "Phalloidin"]
    # The names travel in a bundle too.
    stub_napari.clear()
    nplugin.view_in_napari(
        str(_bundle(store, tmp_path)), show=False, glasbey=False
    )
    assert stub_napari[0][1] == ["DAPI", "Phalloidin"]


def test_fix_metadata_names_the_channels(tmp_path, capsys):
    from patchworks.cli import main
    from patchworks.plugins.ome_zarr import channel_names

    store = _multi_store(tmp_path)
    assert main(["fix-metadata", str(store), "--channel-names", "A, B"]) == 0
    assert channel_names(str(store)) == ["A", "B"]
    assert "channels named: A, B" in capsys.readouterr().out


def test_conversion_keeps_the_source_channel_names(tmp_path, monkeypatch):
    """bioio reports a file's channel names: the new store carries them."""
    import sys
    import types

    from patchworks.plugins import ome_zarr
    from patchworks.plugins.ome_zarr import channel_names, source_channel_names

    class BioImage:
        scenes = ["s0"]
        channel_names = ["DAPI", "Cilia"]

        def __init__(self, path):
            pass

        def set_scene(self, scene):
            pass

    monkeypatch.setitem(
        sys.modules, "bioio", types.SimpleNamespace(BioImage=BioImage)
    )
    assert source_channel_names("scan.czi") == ["DAPI", "Cilia"]
    # Converting (here from a store that has names) writes them.
    named = _multi_store(tmp_path / "a")
    ome_zarr.write_channel_names(str(named), ["DAPI", "Cilia"])
    out = ome_zarr.to_ome_zarr(
        str(named), tmp_path / "b.zarr", n_levels=1, progress=False
    )
    assert channel_names(out) == ["DAPI", "Cilia"]


def _tiff_folder(folder, how):
    import tifffile

    folder.mkdir()
    for z in range(2):
        for c, name in enumerate(("DAPI", "Membrane", "Cilia")):
            plane = np.full((8, 8), c, "uint16")
            path = folder / f"s_Z{z:03d}_C{c}.tif"
            if how == "imagej":
                tifffile.imwrite(
                    path, plane, imagej=True, metadata={"Labels": [name]}
                )
            elif how == "ome":
                tifffile.imwrite(
                    path, plane, ome=True, metadata={"Channel": {"Name": name}}
                )
            else:
                tifffile.imwrite(path, plane)
    return str(folder / "*.tif")


def test_a_folder_of_tiffs_names_its_channels(tmp_path, monkeypatch):
    """Single-plane TIFFs carry their channel's name too: ImageJ's Labels,
    OME-XML, or a companion OME-XML beside them."""
    from patchworks.cli import main
    from patchworks.plugins.ome_zarr import (
        channel_names,
        tiff_sequence_channel_names,
        to_ome_zarr,
    )

    pattern = r"_Z(?P<Z>\d+)_C(?P<C>\d+)"
    want = ["DAPI", "Membrane", "Cilia"]
    for how in ("imagej", "ome"):
        glob = _tiff_folder(tmp_path / how, how)
        assert tiff_sequence_channel_names(glob, pattern) == want, how
    plain = _tiff_folder(tmp_path / "plain", "none")
    assert tiff_sequence_channel_names(plain, pattern) is None
    (tmp_path / "plain" / "dataset.companion.ome").write_text(
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        '<Image ID="Image:0"><Pixels ID="Pixels:0">'
        + "".join(
            f'<Channel ID="Channel:0:{i}" Name="{n}"/>'
            for i, n in enumerate(want)
        )
        + "</Pixels></Image></OME>"
    )
    assert tiff_sequence_channel_names(plain, pattern) == want
    # Converting the folder names the store's channels.
    out = to_ome_zarr(
        _tiff_folder(tmp_path / "conv", "imagej"),
        tmp_path / "conv.zarr",
        sequence_pattern=pattern,
        n_levels=1,
        progress=False,
    )
    assert channel_names(out) == want
    # And an older store, from the folder.
    store = to_ome_zarr(
        _tiff_folder(tmp_path / "old", "none"),
        tmp_path / "old.zarr",
        sequence_pattern=pattern,
        n_levels=1,
        progress=False,
    )
    assert channel_names(store) is None
    args = ["fix-metadata", store, "--channels-from"]
    assert main([*args, glob, "--sequence-pattern", pattern]) == 0
    assert channel_names(store) == want
