"""PlantSeg plugin, against a stand-in for PlantSeg's functional API.

PlantSeg is conda-forge only (vigra, nifty, elf), so these tests replace it
with a module exposing the same functions (plant-seg 2.0) that records how it
is called; the real thing runs where the workflow's ``plantseg`` environment
is installed.
"""

import sys
import types

import numpy as np
import pytest

pytest.importorskip("skimage")

from test_watershed import epithelium  # noqa: E402

RES = (0.235, 0.15, 0.15)


@pytest.fixture
def fake_plantseg(monkeypatch):
    calls = {}

    def unet_prediction(raw, input_layout, model_name, model_id, **kw):
        calls["predict"] = dict(
            shape=raw.shape, layout=input_layout, model=model_name, **kw
        )
        # The "U-Net": a normalized membrane image, as (C, Z, Y, X)
        lo, hi = np.percentile(raw, [1, 99.8])
        return np.clip((raw - lo) / (hi - lo), 0, 1)[None]

    def dt_watershed(pmaps, **kw):
        calls["dt_watershed"] = kw
        from skimage.measure import label

        return label(pmaps < 0.5).astype("uint32")

    def gasp(pmaps, superpixels, **kw):
        calls["gasp"] = kw
        return superpixels

    def lifted(pmaps, nuclei_seg, superpixels, **kw):
        calls["lifted"] = dict(n_nuclei=int(nuclei_seg.max()), **kw)
        return superpixels

    zoo = types.SimpleNamespace(
        get_model_resolution=lambda name: RES,
        get_model_names=lambda: ["generic_confocal_3D_unet"],
        get_model_by_name=lambda name, **kw: calls.setdefault("fetched", name),
    )
    mods = {
        "plantseg": types.ModuleType("plantseg"),
        "plantseg.functionals": types.ModuleType("plantseg.functionals"),
        "plantseg.functionals.prediction": types.ModuleType("p"),
        "plantseg.functionals.segmentation": types.ModuleType("s"),
        "plantseg.core": types.ModuleType("plantseg.core"),
        "plantseg.core.zoo": types.ModuleType("plantseg.core.zoo"),
    }
    mods["plantseg.functionals.prediction"].unet_prediction = unet_prediction
    seg = mods["plantseg.functionals.segmentation"]
    seg.dt_watershed = dt_watershed
    seg.gasp = gasp
    seg.mutex_ws = gasp
    seg.multicut = gasp
    seg.lifted_multicut_from_nuclei_segmentation = lifted
    mods["plantseg.functionals"].segmentation = seg
    mods["plantseg.functionals"].prediction = mods[
        "plantseg.functionals.prediction"
    ]
    mods["plantseg.core.zoo"].model_zoo = zoo
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return calls


def test_gasp_pipeline_on_the_membrane(fake_plantseg):
    from patchworks.plugins.plantseg import segment

    tile, _ = epithelium()
    labels = segment(tile[0], rescale=False, device="cpu")
    assert labels.shape == tile.shape[1:] and labels.dtype == np.int32
    p = fake_plantseg["predict"]
    assert p["layout"] == "ZYX" and p["device"] == "cpu"
    assert p["model"] == "generic_confocal_3D_unet"
    assert fake_plantseg["gasp"]["beta"] == 0.6
    assert len(np.unique(labels[labels > 0])) >= 4


def test_tiles_resampled_to_the_model_resolution(fake_plantseg):
    from patchworks.plugins.plantseg import segment

    tile, _ = epithelium()
    cal = {"z": 0.47, "y": 0.3, "x": 0.3}  # twice the model's voxel size
    labels = segment(tile[0], voxel_size=cal, device="cpu")
    assert fake_plantseg["predict"]["shape"] == (16, 80, 112)
    assert labels.shape == tile.shape[1:]  # back at the tile's own shape
    # The supervoxel watershed measures distances on the image's sampling
    pitch = fake_plantseg["dt_watershed"]["pixel_pitch"]
    assert pitch == pytest.approx((0.47 / 0.3, 1.0, 1.0))


def test_nuclei_watershed_gives_one_cell_per_nucleus(fake_plantseg):
    from patchworks.plugins.plantseg import segment

    tile, truth = epithelium(gap=True)
    labels = segment(
        tile,
        segmentation="nuclei_watershed",
        foreground="otsu",
        nuclei_min_size=20,
        rescale=False,
        device="cpu",
    )
    assert len(np.unique(labels[labels > 0])) == 4
    assert not labels[:, :, 44:].any()  # masked beyond the tissue
    assert "dt_watershed" not in fake_plantseg  # no supervoxels needed


def test_lifted_multicut_gets_the_nuclei(fake_plantseg):
    from patchworks.plugins.plantseg import segment

    tile, _ = epithelium()
    segment(
        tile,
        segmentation="lifted_multicut",
        nuclei_min_size=20,
        rescale=False,
        device="cpu",
    )
    assert fake_plantseg["lifted"]["n_nuclei"] == 4


def test_options_checked_before_any_tile(fake_plantseg):
    from patchworks.plugins import plantseg

    with pytest.raises(ValueError, match="segmentation must be one of"):
        plantseg.plantseg_fn(segmentation="watershed")
    with pytest.raises(ValueError, match="voxel_size"):
        plantseg.plantseg_fn(segmentation="nuclei_watershed", max_radius_um=10)
    with pytest.raises(ValueError, match="nuclear channel"):
        plantseg.segment(epithelium()[0][0], segmentation="lifted_multicut")
    assert plantseg.available_models() == ["generic_confocal_3D_unet"]
    plantseg.fetch_model("generic_confocal_3D_unet")
    assert fake_plantseg["fetched"] == "generic_confocal_3D_unet"
    assert plantseg.segment.patchworks_kwargs_target is plantseg.plantseg_fn


def test_actionable_error_without_plantseg(monkeypatch):
    from patchworks.plugins import plantseg

    monkeypatch.setitem(sys.modules, "plantseg", None)
    with pytest.raises(ImportError, match="conda-forge"):
        plantseg.plantseg_fn()
    assert plantseg.available_models() == []
