"""Denoising before segmentation (CAREamics Noise2Void)."""

import sys
import types

import numpy as np
import pytest

from patchworks.plugins import careamics as cam
from patchworks.plugins.ome_zarr import to_ome_zarr


@pytest.fixture
def noisy_store(tmp_path):
    """A (z, y, x) = (8, 64, 96) image of bright squares in Poisson noise,
    two channels, tissue only on the left 64 columns."""
    rng = np.random.default_rng(0)
    clean = np.zeros((2, 8, 64, 96), "float32")
    clean[:, :, :, :64] = 20
    for y in range(4, 64, 16):
        for x in range(4, 64, 16):
            clean[:, 2:6, y : y + 8, x : x + 8] = 200
    img = rng.poisson(clean).astype("uint16")
    store = to_ome_zarr(img, tmp_path / "n.zarr", axes="czyx", n_levels=3)
    return store, clean


def test_training_crops_pick_the_tissue(noisy_store):
    store, _ = noisy_store
    crops = cam.training_crops(
        store, channel=0, crop_shape=(8, 32, 32), n_crops=2
    )
    assert len(crops) == 2 and crops[0].shape == (8, 32, 32)
    # The brightest crops: never the empty right-hand side
    assert all(c.mean() > 15 for c in crops)


@pytest.fixture
def fake_careamist(monkeypatch, tmp_path):
    """A CAREamist whose "denoising" halves the image, recording calls."""
    calls = []

    class FakeCAREamist:
        def __init__(self, *a, **kw):
            calls.append(("init", kw))

        def predict(self, pred_data, **kw):
            calls.append(("predict", pred_data.shape, kw))
            return [pred_data[None, None] / 2], ["array"]

    mod = types.ModuleType("careamics")
    mod.CAREamist = FakeCAREamist
    monkeypatch.setitem(sys.modules, "careamics", mod)
    monkeypatch.setattr(cam, "_models", {})
    model = tmp_path / "n2v.ckpt"
    model.write_bytes(b"")
    return calls, str(model)


def test_denoise_runs_before_the_segmentation(fake_careamist):
    calls, model = fake_careamist
    seen = []

    def seg(tile):
        seen.append(tile)
        return np.zeros(tile.shape[-3:], "int32")

    fn = cam.denoise_fn(seg, model, tile_size=(8, 32, 32))
    tile = np.full((2, 8, 40, 40), 100, "uint16")
    fn(tile)
    fn(tile)
    # One model load per process, both tiles through it
    assert [c[0] for c in calls].count("init") == 1
    _, shape, kw = calls[1]
    assert shape == (8, 40, 40) and kw["axes"] == "ZYX"
    assert kw["tile_size"] == (8, 32, 32) and kw["tile_overlap"] == (2, 8, 8)
    # Membrane denoised, nuclei (no nuclei_model) left as they were
    assert seen[0].shape == (2, 8, 40, 40)
    assert (seen[0][0] == 50).all() and (seen[0][1] == 100).all()
    assert calls[0][1]["checkpoint_path"].name == "n2v.ckpt"


def test_small_tiles_skip_careamics_tiling(fake_careamist):
    calls, model = fake_careamist
    out = cam.denoise(np.ones((8, 64, 64), "float32"), model=model)
    assert out.shape == (8, 64, 64)
    assert calls[-1][2]["tile_size"] is None


def test_missing_model_fails_before_any_tile(fake_careamist, tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        cam.denoise_fn(lambda t: t, tmp_path / "nope.ckpt")


def test_actionable_error_without_careamics(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "careamics", None)
    with pytest.raises(ImportError, match="patchworks\\[careamics\\]"):
        cam.denoise_fn(lambda t: t, tmp_path / "m.ckpt")


def test_noise2void_trains_and_denoises(noisy_store, tmp_path):
    """The real thing, one short epoch on the CPU."""
    pytest.importorskip("careamics")
    store, clean = noisy_store
    model = cam.train_n2v(
        store,
        tmp_path / "n2v.ckpt",
        channel=0,
        crop_shape=(8, 64, 64),
        n_crops=1,
        patch_size=(8, 32, 32),
        batch_size=4,
        epochs=1,
        work_dir=tmp_path / "work",
    )
    assert model.exists()
    cam._models.clear()
    raw = np.asarray(
        __import__("patchworks").load_ome_zarr(store, channel=0)
    ).astype("float32")
    out = cam.denoise(raw[:, :, :64], model=str(model))
    assert out.shape == (8, 64, 64) and np.isfinite(out).all()


def test_cli_segment_denoises_first(fake_careamist, noisy_store, capsys):
    from patchworks.cli import main

    calls, model = fake_careamist
    store, _ = noisy_store
    args = ["segment", store, "--method", "threshold", "--threshold", "50"]
    args += ["--tile-shape", "8,64,96", "--overlap", "0", "--denoise", model]
    assert main(args) == 0
    assert any(c[0] == "predict" for c in calls)


def test_cli_denoise_train_arguments(monkeypatch, tmp_path, capsys):
    from patchworks import cli

    seen = {}
    monkeypatch.setattr(
        cam, "train_n2v", lambda image, out, **kw: seen.update(kw) or out
    )
    out = tmp_path / "m.ckpt"
    args = ["denoise-train", "s.zarr", "--out", str(out), "--channel", "1"]
    args += ["--crop-shape", "16,256,256", "--epochs", "5"]
    assert cli.main(args) == 0
    assert seen["channel"] == 1 and seen["crop_shape"] == (16, 256, 256)
    assert seen["epochs"] == 5 and seen["patch_size"] is None


def test_workflow_build_fn_denoises_before_the_method(fake_careamist, tmp_path):
    import pathlib

    sys.path.insert(
        0, str(pathlib.Path(__file__).resolve().parents[1] / "workflow/scripts")
    )
    from _pw import build_fn

    calls, model = fake_careamist
    fn = build_fn(
        {
            "method": "threshold",
            "work_dir": str(tmp_path),
            "denoise": {"model": model},
        }
    )
    tile = np.zeros((8, 32, 32), "float32")
    tile[2:6, 8:16, 8:16] = 100
    labels = fn(tile)
    assert labels.max() == 1 and calls[-1][0] == "predict"
