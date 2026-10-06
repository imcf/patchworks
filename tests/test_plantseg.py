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
        get_model_patch_size=lambda name: [80, 160, 160],
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
    # The patch that fitted is remembered per process: not across tests.
    from patchworks.plugins import plantseg as pl

    monkeypatch.setattr(pl, "_fitting_patch", {})
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
    # 8 planes, extended to MIN_DEPTH (16), then twice as fine: 32
    assert fake_plantseg["predict"]["shape"] == (32, 80, 112)
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
    # The depth padding repeats the last plane: no extra (mirrored) nuclei
    assert fake_plantseg["lifted"]["n_nuclei"] == 4


def test_options_checked_before_any_tile(fake_plantseg):
    from patchworks.plugins import plantseg

    with pytest.raises(ValueError, match="segmentation must be one of"):
        plantseg.plantseg_fn(segmentation="watershed")
    with pytest.raises(ValueError, match="voxel_size"):
        plantseg.plantseg_fn(segmentation="nuclei_watershed", max_radius_um=10)
    with pytest.raises(ValueError, match="seed_labels"):
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


def test_nuclei_watershed_from_given_labels(fake_plantseg):
    from patchworks.plugins.plantseg import segment

    tile, truth = epithelium()
    labels = np.zeros(tile.shape[1:], "uint32")
    for i, (y, x) in enumerate(((10, 10), (10, 30), (30, 10), (30, 30))):
        labels[2:6, y - 3 : y + 3, x - 3 : x + 3] = 500 + i
    out = segment(
        np.stack([tile[0].astype("uint32"), labels]),
        segmentation="nuclei_watershed",
        seeds="labels",
        rescale=False,
        device="cpu",
    )
    assert len(np.unique(out[out > 0])) == 4
    with pytest.raises(ValueError, match="would ignore them"):
        segment(np.stack([tile[0], labels]), seeds="labels")


def _cfg(**over):
    cfg = dict(
        model="generic_confocal_3D_unet",
        model_id=None,
        config_path=None,
        weights_path=None,
        rescale=None,
        patch=None,
        device="cuda",
        boundary_channel=0,
    )
    cfg.update(over)
    return cfg


def test_patch_from_the_zoo_not_plantsegs_own_search(fake_plantseg):
    """PlantSeg's own search (patch=None) probed up to 416-voxel cubes and,
    on a cluster GPU, ended at 24 -- under its 64-pixel minimum."""
    from patchworks.plugins import plantseg as pl

    tile = np.random.default_rng(0).random((129, 430, 430), "float32")
    out = pl.predict_boundaries(tile, _cfg())
    assert fake_plantseg["predict"]["patch"] == (80, 160, 160)
    assert out.shape == tile.shape


def test_thin_edge_tiles_are_padded_to_plantsegs_minimum(fake_plantseg):
    from patchworks.plugins import plantseg as pl

    tile = np.random.default_rng(0).random((8, 40, 30), "float32")
    out = pl.predict_boundaries(tile, _cfg())
    assert fake_plantseg["predict"]["shape"] == (8, 64, 64)
    assert fake_plantseg["predict"]["patch"] == (8, 64, 64)
    assert out.shape == (8, 40, 30)


def test_gpu_out_of_memory_shrinks_the_patch(fake_plantseg, monkeypatch):
    from patchworks.plugins import plantseg as pl

    tried = []
    predict = sys.modules["plantseg.functionals.prediction"].unet_prediction

    def tight_gpu(raw, input_layout, model_name, model_id, **kw):
        tried.append(kw["patch"])
        if kw["patch"][1] > 90:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
        return predict(raw, input_layout, model_name, model_id, **kw)

    monkeypatch.setattr(
        sys.modules["plantseg.functionals.prediction"],
        "unet_prediction",
        tight_gpu,
    )
    tile = np.random.default_rng(0).random((40, 300, 300), "float32")
    out = pl.predict_boundaries(tile, _cfg())
    assert tried == [(40, 160, 160), (40, 120, 120), (40, 90, 90)]
    assert out.shape == tile.shape


def test_out_of_memory_at_the_smallest_patch_says_so(
    fake_plantseg, monkeypatch
):
    from patchworks.plugins import plantseg as pl

    def no_gpu(*a, **kw):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(
        sys.modules["plantseg.functionals.prediction"],
        "unet_prediction",
        no_gpu,
    )
    monkeypatch.setattr(pl, "retry_on_oom", lambda call, **kw: call())
    with pytest.raises(RuntimeError, match="even at patch \\(8, 64, 64\\)"):
        pl.predict_boundaries(np.zeros((40, 100, 100), "float32"), _cfg())

    # Anything else is not retried
    def broken(*a, **kw):
        raise ValueError("bad weights")

    monkeypatch.setattr(
        sys.modules["plantseg.functionals.prediction"],
        "unet_prediction",
        broken,
    )
    with pytest.raises(ValueError, match="bad weights"):
        pl.predict_boundaries(np.zeros((40, 100, 100), "float32"), _cfg())


def test_thin_z_tiles_are_extended_to_min_depth(fake_plantseg):
    """Real PlantSeg reads 1-2 planes as a 2-D image and vigra's smoothing
    needs ~8 planes: the last z-tile of a stack failed. Found running the
    plugin against plant-seg 2.0.0rc14."""
    from patchworks.plugins import plantseg as pl

    tile = np.random.default_rng(0).random((3, 70, 70), "float32") * 100
    out = pl.segment(tile, segmentation="gasp", rescale=False, device="cpu")
    assert fake_plantseg["predict"]["shape"] == (pl.MIN_DEPTH, 70, 70)
    assert out.shape == (3, 70, 70)


def test_two_plane_tile_is_not_taken_for_a_channel_pair(fake_plantseg):
    from patchworks.plugins import plantseg as pl

    tile = np.random.default_rng(0).random((2, 70, 70), "float32") * 100
    out = pl.segment(tile, segmentation="gasp", rescale=False, device="cpu")
    assert out.shape == (2, 70, 70)  # one 3-D tile, not [membrane, nuclei]
    with pytest.raises(ValueError, match="seed_labels"):
        pl.segment(tile[0], segmentation="nuclei_watershed", device="cpu")


def test_plantsegs_own_oom_check_shrinks_the_patch_and_is_remembered(
    fake_plantseg, monkeypatch
):
    """On a 24 GB RTX 4090, PlantSeg refused the zoo patch before running:
    "OOM error will happen. Please reduce the patch size/halo." -- a
    message without "out of memory", so the patch never shrank. The patch
    that fits is then where the batch's next tile starts."""
    from patchworks.plugins import plantseg as pl

    tried = []
    predict = sys.modules["plantseg.functionals.prediction"].unet_prediction

    def plantseg_check(raw, input_layout, model_name, model_id, **kw):
        tried.append(kw["patch"])
        if kw["patch"][1] > 120:  # what ArrayPredictor raises
            raise RuntimeError(
                "OOM error will happen. Please reduce the patch size/halo."
            )
        return predict(raw, input_layout, model_name, model_id, **kw)

    monkeypatch.setattr(
        sys.modules["plantseg.functionals.prediction"],
        "unet_prediction",
        plantseg_check,
    )
    tile = np.random.default_rng(0).random((129, 430, 430), "float32")
    pl.predict_boundaries(tile, _cfg())
    assert tried == [(80, 160, 160), (80, 120, 120)]
    tried.clear()
    pl.predict_boundaries(tile, _cfg())  # next tile of the batch
    assert tried == [(80, 120, 120)]


def test_gpu_report_names_the_processes_on_the_gpu(monkeypatch):
    """5.7 of 25.3 GB free before PlantSeg loaded anything, then 0.8: the
    report lists every process on the GPU, ours marked, to say whose."""
    import os
    import subprocess

    from patchworks.plugins import plantseg as pl

    rows = f"{os.getpid()}, python, 900 MiB, GPU-1\n4242, python, 19400 MiB, GPU-1\n"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(stdout=rows),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    report = pl.gpu_report()
    assert "CUDA_VISIBLE_DEVICES=3" in report
    assert f"  * {os.getpid()}, python, 900 MiB" in report
    assert "    4242, python, 19400 MiB" in report


def test_smallest_patch_failure_carries_the_gpu_report(
    fake_plantseg, monkeypatch
):
    from patchworks.plugins import plantseg as pl

    def no_room(*a, **kw):
        raise RuntimeError(
            "OOM error will happen. Please reduce the patch size/halo."
        )

    monkeypatch.setattr(
        sys.modules["plantseg.functionals.prediction"],
        "unet_prediction",
        no_room,
    )
    monkeypatch.setattr(pl, "retry_on_oom", lambda call, **kw: call())
    monkeypatch.setattr(pl, "gpu_report", lambda: "4242, python, 19400 MiB")
    with pytest.raises(RuntimeError, match="another process holds it") as err:
        pl.predict_boundaries(np.zeros((40, 100, 100), "float32"), _cfg())
    assert "19400 MiB" in str(err.value)
