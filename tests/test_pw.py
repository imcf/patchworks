"""Tests for workflow/scripts/_pw.py's config-to-segmentation-function wiring."""

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "workflow" / "scripts")
)

import numpy as np  # noqa: E402


def test_with_voxel_size_fills_in_cellposes_calibration(tmp_path):
    """The same auto-fill a custom function's voxel_size gets (see

    _with_voxel_size's docstring) must also reach cellpose_fn, or do_3D
    silently assumes isotropic voxels -- fragmenting objects across z for
    any real (anisotropic) calibration. This checks the actual calibration
    read + injection against a real store, not just that the parameter
    exists (see test_cellpose.py for that).
    """
    from _pw import _with_voxel_size
    from patchworks.plugins.cellpose import cellpose_fn
    from patchworks.plugins.ome_zarr import to_ome_zarr

    arr = np.zeros((4, 8, 8), dtype="uint16")
    to_ome_zarr(
        arr,
        str(tmp_path / "image.zarr"),
        axes="zyx",
        pixel_size={"z": 0.24, "y": 0.10833, "x": 0.10833},
        n_levels=1,
        progress=False,
    )
    cfg = {"work_dir": str(tmp_path)}

    kwargs = _with_voxel_size(cellpose_fn, {}, cfg)

    assert kwargs["voxel_size"]["z"] == 0.24
    assert kwargs["voxel_size"]["x"] == 0.10833


def test_with_voxel_size_never_overrides_an_explicit_value(tmp_path):
    """An explicit voxel_size must win, and win *cheaply*: no image.zarr

    exists in work_dir at all here, so if this read past the early return it
    would raise, not just return the wrong value.
    """
    from _pw import _with_voxel_size
    from patchworks.plugins.cellpose import cellpose_fn

    cfg = {"work_dir": str(tmp_path)}
    explicit = {"z": 1.0, "y": 1.0, "x": 1.0}

    kwargs = _with_voxel_size(cellpose_fn, {"voxel_size": explicit}, cfg)

    assert kwargs["voxel_size"] == explicit


def test_validate_config_accepts_a_positive_min_volume():
    from _pw import validate_config

    validate_config({"method": "threshold", "min_volume": 5.0})


def test_validate_config_accepts_no_min_volume():
    from _pw import validate_config

    validate_config({"method": "threshold"})
    validate_config({"method": "threshold", "min_volume": None})


def test_validate_config_rejects_a_non_positive_min_volume():
    import pytest
    from _pw import validate_config

    with pytest.raises(ValueError, match="min_volume"):
        validate_config({"method": "threshold", "min_volume": 0})
    with pytest.raises(ValueError, match="min_volume"):
        validate_config({"method": "threshold", "min_volume": -1.0})
    with pytest.raises(ValueError, match="min_volume"):
        validate_config({"method": "threshold", "min_volume": "5"})


def test_validate_config_accepts_a_positive_max_volume():
    from _pw import validate_config

    validate_config({"method": "threshold", "max_volume": 500.0})
    validate_config({"method": "threshold", "max_volume": None})


def test_validate_config_rejects_a_non_positive_max_volume():
    import pytest
    from _pw import validate_config

    with pytest.raises(ValueError, match="max_volume"):
        validate_config({"method": "threshold", "max_volume": 0})
    with pytest.raises(ValueError, match="max_volume"):
        validate_config({"method": "threshold", "max_volume": -1.0})
    with pytest.raises(ValueError, match="max_volume"):
        validate_config({"method": "threshold", "max_volume": "5"})


def test_validate_config_accepts_max_volume_above_min_volume():
    from _pw import validate_config

    validate_config(
        {"method": "threshold", "min_volume": 5.0, "max_volume": 500.0}
    )


def test_validate_config_rejects_max_volume_at_or_below_min_volume():
    import pytest
    from _pw import validate_config

    with pytest.raises(ValueError, match="max_volume"):
        validate_config(
            {"method": "threshold", "min_volume": 5.0, "max_volume": 5.0}
        )
    with pytest.raises(ValueError, match="max_volume"):
        validate_config(
            {"method": "threshold", "min_volume": 500.0, "max_volume": 5.0}
        )


def test_validate_config_rejects_a_model_the_install_does_not_have(
    monkeypatch,
):
    """Cellpose does not raise on an unknown model name -- it logs and loads

    its default. A v3 name against a v4 install therefore segments every
    tile with a model nobody chose, silently. That has to fail in prepare.
    """
    import pytest
    from _pw import validate_config

    import patchworks.plugins.cellpose as cp

    monkeypatch.setattr(cp, "available_models", lambda: ["cpsam", "cpsam_v2"])
    with pytest.raises(ValueError, match="cyto3"):
        validate_config({"method": "cellpose", "cellpose": {"model": "cyto3"}})


def test_validate_config_accepts_a_model_the_install_has(monkeypatch):
    from _pw import validate_config

    import patchworks.plugins.cellpose as cp

    monkeypatch.setattr(cp, "available_models", lambda: ["cpsam", "cpsam_v2"])
    validate_config({"method": "cellpose", "cellpose": {"model": "cpsam"}})


def test_validate_config_leaves_a_custom_model_path_alone(
    monkeypatch, tmp_path
):
    """A path is a custom-trained model; no name list can vouch for it."""
    from _pw import validate_config

    import patchworks.plugins.cellpose as cp

    monkeypatch.setattr(cp, "available_models", lambda: ["cpsam"])
    custom = tmp_path / "my_model.pth"
    custom.write_text("")
    validate_config({"method": "cellpose", "cellpose": {"model": str(custom)}})


def test_validate_config_accepts_shard_and_shard_labels():
    """Both take true/false or an explicit shard shape."""
    from _pw import validate_config

    for key in ("shard", "shard_labels"):
        validate_config({"method": "threshold", key: True})
        validate_config({"method": "threshold", key: False})
        validate_config({"method": "threshold", key: None})
        validate_config({"method": "threshold", key: [16, 512, 512]})


def test_validate_config_rejects_a_malformed_shard_shape():
    """A bad shape would otherwise surface hours later, inside the merge."""
    import pytest
    from _pw import validate_config

    for key in ("shard", "shard_labels"):
        with pytest.raises(ValueError, match=key):
            validate_config({"method": "threshold", key: "auto"})
        with pytest.raises(ValueError, match=key):
            validate_config({"method": "threshold", key: [16, 0, 512]})
        with pytest.raises(ValueError, match=key):
            validate_config({"method": "threshold", key: []})


def test_validate_config_accepts_the_supported_ngff_versions():
    from _pw import validate_config

    for value in ("auto", "0.4", "0.5", None):
        validate_config({"method": "threshold", "ngff_version": value})


def test_validate_config_rejects_an_unwritable_ngff_version():
    """0.6 is released; naming it must explain why it is still refused."""
    import pytest
    from _pw import validate_config

    with pytest.raises(ValueError, match="coordinateSystems"):
        validate_config({"method": "threshold", "ngff_version": "0.6"})
    with pytest.raises(ValueError, match="ngff_version"):
        validate_config({"method": "threshold", "ngff_version": "latest"})


def test_validate_config_rejects_sharding_on_ngff_04():
    """Zarr v2 has no sharding codec, so the pair is a contradiction.

    Silently ignoring `shard` here would be the worst outcome: the whole
    point of turning it on is a filesystem that cannot take the file count.
    """
    import pytest
    from _pw import validate_config

    with pytest.raises(ValueError, match="no sharding codec"):
        validate_config(
            {"method": "threshold", "ngff_version": "0.4", "shard": True}
        )
    # ...but 0.4 on its own is fine.
    validate_config({"method": "threshold", "ngff_version": "0.4"})


def test_segment_progress_resumes_only_the_same_batch(tmp_path):
    """A retried batch skips the tiles an earlier attempt already staged."""
    import zarr

    from _pw import (
        load_segment_progress,
        save_segment_progress,
        segment_progress_path,
    )

    from patchworks import create_stage

    stage = create_stage(tmp_path / "stage.zarr", (4, 8, 8), (1, 8, 8))
    path = segment_progress_path(stage, 3)
    assert load_segment_progress(path, [0, 1, 2], (1, 8, 8)) == {}

    save_segment_progress(path, [0, 1, 2], (1, 8, 8), {0: 5, 1: 0})
    assert load_segment_progress(path, [0, 1, 2], (1, 8, 8)) == {0: 5, 1: 0}
    # A different batch layout or tile shape is stale, not resumable.
    assert load_segment_progress(path, [0, 1], (1, 8, 8)) == {}
    assert load_segment_progress(path, [0, 1, 2], (2, 8, 8)) == {}
    # The checkpoint sits inside the store without confusing zarr ...
    assert zarr.open_group(stage, mode="r")["staged"].shape == (4, 8, 8)
    # ... and dies with it when prepare recreates the stage.
    create_stage(stage, (4, 8, 8), (1, 8, 8))
    assert load_segment_progress(path, [0, 1, 2], (1, 8, 8)) == {}


def test_validate_config_checks_stitch_and_iou_threshold():
    import pytest
    from _pw import validate_config

    validate_config({"method": "threshold", "stitch": "iou"})
    validate_config({"method": "threshold", "iou_threshold": 0.3})
    with pytest.raises(ValueError, match="stitch"):
        validate_config({"method": "threshold", "stitch": "glue"})
    with pytest.raises(ValueError, match="iou_threshold"):
        validate_config({"method": "threshold", "iou_threshold": 0})


def test_build_fn_applies_fill_holes_and_opening():
    import numpy as np
    from _pw import build_fn

    tile = np.zeros((1, 12, 12), "uint16")
    tile[0, 2:9, 2:9] = 1000
    tile[0, 5, 5] = 0  # a hole the threshold leaves
    fn = build_fn(
        {"method": "threshold", "fill_holes": "per_plane", "open_radius": 1}
    )
    out = fn(tile)
    assert out[0, 5, 5] == out[0, 3, 3] != 0


def test_validate_config_checks_the_denoise_block(tmp_path, monkeypatch):
    import _pw
    import pytest
    from _pw import validate_config

    monkeypatch.setattr(_pw, "has_module", lambda name: True)

    model = tmp_path / "n2v.ckpt"
    model.write_bytes(b"")
    validate_config({"method": "threshold", "denoise": {"model": str(model)}})
    with pytest.raises(ValueError) as err:
        validate_config(
            {
                "method": "threshold",
                "denoise": {
                    "model": str(tmp_path / "missing.ckpt"),
                    "tile": [16, 256, 256],
                    "tile_size": "big",
                },
            }
        )
    msg = str(err.value)
    assert "missing.ckpt" in msg and "unknown denoise keys ['tile']" in msg
    assert "denoise.tile_size" in msg
    with pytest.raises(ValueError, match="needs a model"):
        validate_config({"method": "threshold", "denoise": {}})


def test_custom_plugins_validate_through_their_factories():
    import pytest
    from _pw import validate_config

    for module in (
        "patchworks.plugins.watershed",
        "patchworks.plugins.plantseg",
    ):
        with pytest.raises(ValueError, match="unknown custom.kwargs"):
            validate_config(
                {
                    "method": "custom",
                    "custom": {"module": module, "kwargs": {"bogus": 1}},
                }
            )


def _seeded_store(tmp_path):
    """image.zarr with a membrane-like channel 0 and a nuclei label image."""
    from patchworks.plugins.ome_zarr import to_ome_zarr, write_labels

    img = np.zeros((2, 4, 32, 32), "uint16")
    img[0, :, 15:17, :] = 500  # a wall across y = 16
    store = to_ome_zarr(img, tmp_path / "image.zarr", axes="czyx", n_levels=2)
    nuclei = np.zeros((4, 32, 32), "uint32")
    nuclei[1:3, 5:9, 10:14] = 70_001
    nuclei[1:3, 22:26, 10:14] = 70_002
    write_labels(store, nuclei, name="nuclei_labels", progress=False)
    return nuclei


def test_open_image_stacks_the_seed_labels(tmp_path):
    from _pw import check_seed_labels, open_image

    nuclei = _seeded_store(tmp_path)
    arr = open_image(tmp_path, 0, 0, seed_labels="nuclei_labels")
    assert arr.shape == (2, 4, 32, 32)
    np.testing.assert_array_equal(np.asarray(arr[1]), nuclei)  # ids exact
    assert np.asarray(arr[0])[0, 16, 0] == 500
    check_seed_labels(
        tmp_path, {"seed_labels": "nuclei_labels", "channel": 0, "level": 0}
    )

    import pytest

    # A label group being segmented in place has no pyramid yet: refused
    import zarr

    zarr.open_group(str(tmp_path / "image.zarr"), mode="a").require_group(
        "labels/unfinished"
    ).create_array("0", shape=(4, 32, 32), dtype="uint32")
    with pytest.raises(ValueError, match="still being made"):
        check_seed_labels(
            tmp_path, {"seed_labels": "unfinished", "channel": 0, "level": 0}
        )
    with pytest.raises(ValueError, match="does not exist"):
        check_seed_labels(
            tmp_path, {"seed_labels": "cells", "channel": 0, "level": 0}
        )
    with pytest.raises(ValueError, match="same level"):
        open_image(tmp_path, 0, 1, seed_labels="nuclei_labels")
    with pytest.raises(ValueError, match="set one"):
        open_image(
            tmp_path, 0, 0, nuclei_channel=1, seed_labels="nuclei_labels"
        )


def test_seed_labels_grow_one_cell_per_given_nucleus(tmp_path):
    from _pw import build_fn, open_image

    nuclei = _seeded_store(tmp_path)
    cfg = {
        "method": "custom",
        "work_dir": str(tmp_path),
        "seed_labels": "nuclei_labels",
        "custom": {
            "module": "patchworks.plugins.watershed",
            "kwargs": {"foreground": None},
        },
    }
    fn = build_fn(cfg)
    assert fn.keywords["seeds"] == "labels"
    cells = fn(
        np.asarray(open_image(tmp_path, 0, 0, seed_labels="nuclei_labels"))
    )
    # The wall at y = 16 parts the two cells, one per nucleus
    assert len(np.unique(cells[cells > 0])) == 2
    assert cells[2, 7, 12] != cells[2, 24, 12]
    assert (cells[:, :15] == cells[2, 7, 12]).all()
    assert nuclei.max() > cells.max()  # renumbered per tile, merged later


def test_validate_config_checks_seed_labels():
    import pytest
    from _pw import validate_config

    ok = {
        "method": "custom",
        "label_name": "cyto_labels",
        "seed_labels": "nuclei_labels",
        "stitch": "iou",
        "custom": {"module": "patchworks.plugins.watershed"},
    }
    validate_config(ok)
    cases = {
        "own label_name": {**ok, "seed_labels": "cyto_labels"},
        "set one": {**ok, "nuclei_channel": 1},
        'needs method: "custom"': {
            **ok,
            "method": "cellpose",
            "cellpose": {"model": "cyto3"},
        },
        "takes no `seeds`": {
            **ok,
            "custom": {"module": "patchworks.plugins.dog"},
        },
        "custom.kwargs.seeds": {
            **ok,
            "custom": {
                "module": "patchworks.plugins.watershed",
                "kwargs": {"seeds": "channel"},
            },
        },
        "name of a label image": {**ok, "seed_labels": 3},
    }
    for message, cfg in cases.items():
        with pytest.raises(ValueError, match=message):
            validate_config(cfg)


def test_space_filling_plugins_require_iou_stitching(monkeypatch):
    import _pw
    import pytest
    from _pw import validate_config

    monkeypatch.setattr(_pw, "has_module", lambda name: True)

    for module in (
        "patchworks.plugins.watershed",
        "patchworks.plugins.plantseg",
    ):
        cfg = {
            "method": "custom",
            "nuclei_channel": 1,
            "custom": {"module": module},
        }
        with pytest.raises(ValueError, match='stitch: "iou"'):
            validate_config(cfg)  # "touch" is the default
        validate_config({**cfg, "stitch": "iou"})
    # Other custom functions are left alone
    validate_config(
        {
            "method": "custom",
            "custom": {
                "module": "patchworks.plugins.dog",
                "kwargs": {"low_sigma": 1, "high_sigma": 2, "threshold": 1},
            },
        }
    )


def test_seeded_cells_across_tiles_one_per_nucleus(tmp_path):
    """The real segment path on six tiles: open_image with seed_labels, the
    config's function, stage_tile, then the merge -- 12 nuclei, 12 cells,
    with IoU stitching; "touch" joins neighbours at the seams."""
    import dask.array as da
    from _pw import build_fn, open_image
    from patchworks import merge_tile_labels
    from patchworks._distributed import create_stage, spatial_tiles, stage_tile
    from patchworks.plugins.ome_zarr import to_ome_zarr, write_labels

    rng = np.random.default_rng(1)
    shape = (6, 60, 80)
    mem = rng.normal(20, 3, shape).astype("float32")
    mem[:, ::20, :] = 300  # walls: a 3 x 4 grid of cells
    mem[:, :, ::20] = 300
    nuclei = np.zeros(shape, "uint32")
    for k, (cy, cx) in enumerate(
        (y, x) for y in range(10, 60, 20) for x in range(10, 80, 20)
    ):
        nuclei[1:5, cy - 3 : cy + 3, cx - 3 : cx + 3] = 1000 + k
    img = np.stack([mem.astype("uint16"), np.zeros(shape, "uint16")])
    store = to_ome_zarr(img, tmp_path / "image.zarr", axes="czyx", n_levels=1)
    write_labels(store, nuclei, name="nuclei_labels", progress=False)
    cfg = {
        "method": "custom",
        "work_dir": str(tmp_path),
        "seed_labels": "nuclei_labels",
        "custom": {
            "module": "patchworks.plugins.watershed",
            "kwargs": {"foreground": None},
        },
    }
    image = open_image(tmp_path, 0, 0, seed_labels="nuclei_labels")
    fn = build_fn(cfg)
    tile_shape = (6, 32, 32)  # tile seams cut through cells

    def cells(stitch):
        stage = str(tmp_path / f"stage_{stitch}.zarr")
        halo = tmp_path / f"halo_{stitch}" if stitch == "iou" else None
        create_stage(stage, shape, tile_shape)
        for i in range(len(spatial_tiles(shape, tile_shape))):
            stage_tile(
                image,
                fn,
                stage,
                i,
                tile_shape=tile_shape,
                overlap=(0, 12, 12),
                channel_axis=0,
                halo_dir=halo,
            )
        merged = np.asarray(
            merge_tile_labels(
                da.from_zarr(stage, component="staged"),
                sequential_labels=True,
                halo_dir=halo,
            )
        )
        pairs = {
            (int(n), int(c))
            for n, c in zip(nuclei.ravel(), merged.ravel())
            if n
        }
        return len({c for _, c in pairs}), len(pairs)

    assert cells("iou") == (12, 12)  # each nucleus in its own cell
    assert cells("touch")[0] < 12  # why the workflow requires "iou"


def test_missing_packages_are_named_with_the_environment_to_use(
    tmp_path, monkeypatch
):
    """Run from the default environment, a PlantSeg config failed in its
    first GPU job; it is now refused up front, saying which environment."""
    import _pw

    missing = {"plantseg", "careamics"}
    monkeypatch.setattr(_pw, "has_module", lambda name: name not in missing)
    model = tmp_path / "n2v.ckpt"
    model.write_bytes(b"")
    cfg = {
        "method": "custom",
        "label_name": "cyto_labels_plantseg",
        "stitch": "iou",
        "nuclei_channel": 1,
        "custom": {"module": "patchworks.plugins.plantseg"},
        "denoise": {"model": str(model)},
    }
    problems = _pw.environment_problems(cfg)
    assert len(problems) == 2
    assert "pixi run -e plantseg" in problems[0]
    assert "cyto_labels_plantseg needs 'plantseg'" in problems[0]
    assert "-e careamics" in problems[1]
    missing.clear()
    assert _pw.environment_problems(cfg) == []
    # Functions that need nothing extra are left alone
    assert _pw.environment_problems({"method": "threshold"}) == []


def test_gpu_dog_needs_cupy_only_when_it_uses_the_gpu(monkeypatch):
    """config_cilia.yaml with use_gpu: true, run from the plantseg
    environment without cupy: every segment job failed on import."""
    import _pw

    monkeypatch.setattr(_pw, "has_module", lambda name: name != "cupy")
    cfg = {
        "method": "custom",
        "label_name": "cilia_labels",
        "custom": {
            "module": "patchworks.plugins.dog",
            "kwargs": {"use_gpu": True, "decon_kwargs": {"psf": "p.tif"}},
        },
    }
    problems = _pw.environment_problems(cfg)
    assert len(problems) == 1
    assert "cilia_labels needs 'cupy'" in problems[0]
    assert "-e cuda12" in problems[0]
    cfg["custom"]["kwargs"]["use_gpu"] = False
    assert _pw.environment_problems(cfg) == []
    assert len(_pw.environment_problems({"dilate_gpu": True})) == 1


def test_pin_slurm_gpus_only_where_the_node_does_not_isolate():
    """Unset CUDA_VISIBLE_DEVICES on an unisolated node put every job on
    GPU 0. Pin to SLURM's GPUs then -- and only then."""
    from _pw import pin_slurm_gpus

    env = {"SLURM_JOB_GPUS": "3"}
    msg = pin_slurm_gpus(env, visible=8)  # sees the node's 8 GPUs
    assert env["CUDA_VISIBLE_DEVICES"] == "3" and "SLURM_JOB_GPUS=3" in msg
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"

    isolated = {"SLURM_JOB_GPUS": "3"}  # cgroup: its one GPU shows as 0
    assert pin_slurm_gpus(isolated, visible=1) is None
    assert "CUDA_VISIBLE_DEVICES" not in isolated

    already = {"SLURM_JOB_GPUS": "3", "CUDA_VISIBLE_DEVICES": "0"}
    assert pin_slurm_gpus(already, visible=8) is None
    assert already["CUDA_VISIBLE_DEVICES"] == "0"

    assert pin_slurm_gpus({}, visible=8) is None  # not a GPU job
    step = {"SLURM_STEP_GPUS": "1,2", "SLURM_JOB_GPUS": "1,2"}
    assert pin_slurm_gpus(step, visible=8)
    assert step["CUDA_VISIBLE_DEVICES"] == "1,2"


def test_connectivity_is_one_value_shared_by_labelling_and_merge(tmp_path):
    """The tiles and the merge must use the same neighbourhood: one value
    per config, from the DoG kwargs or the top level, checked up front."""
    import _pw
    import numpy as np
    import pytest

    assert _pw.merge_connectivity({"method": "threshold"}) == 1
    assert _pw.merge_connectivity({"connectivity": 3}) == 3
    dog = {
        "method": "custom",
        "custom": {
            "module": "patchworks.plugins.dog",
            "kwargs": {"connectivity": 2},
        },
    }
    assert _pw.merge_connectivity(dog) == 2
    with pytest.raises(ValueError, match="disagree"):
        _pw.validate_config({**dog, "connectivity": 3})
    with pytest.raises(ValueError, match="1, 2 or 3"):
        _pw.validate_config({"method": "threshold", "connectivity": 5})

    # The threshold method labels with it: a diagonal pair of voxels
    tile = np.zeros((4, 4), "float32")
    tile[1, 1] = tile[2, 2] = 10.0
    faces = _pw.build_fn({"method": "threshold"})(tile)
    corners = _pw.build_fn({"method": "threshold", "connectivity": 2})(tile)
    assert faces.max() == 2 and corners.max() == 1

    # ... and so does the DoG plugin, given it at the top level
    fn = _pw.build_fn(
        {
            "method": "custom",
            "connectivity": 3,
            "work_dir": str(tmp_path),
            "custom": {
                "module": "patchworks.plugins.dog",
                "kwargs": {
                    "low_sigma": 0,
                    "high_sigma": 2,
                    "threshold": 1,
                    "voxel_size": {"z": 1.0, "y": 1.0, "x": 1.0},
                },
            },
        }
    )
    assert fn.keywords["connectivity"] == 3


def test_cellpose_scales_tiles_with_one_image_range_by_default():
    """Cellpose would stretch each tile from its own percentiles, giving
    neighbouring tiles different contrast; the workflow measures one range
    unless told otherwise."""
    from _pw import uses_image_range

    cp = {"model": "cpsam"}
    assert uses_image_range({"method": "cellpose", "cellpose": cp})
    assert not uses_image_range(
        {"method": "cellpose", "cellpose": cp, "normalize": "tile"}
    )
    # Cellpose's own normalize, set explicitly, is respected.
    assert not uses_image_range(
        {
            "method": "cellpose",
            "cellpose": {**cp, "normalize": {"percentile": [1, 99]}},
        }
    )
    assert not uses_image_range({"method": "custom"})


def test_validate_config_rejects_an_unknown_normalize():
    import pytest
    from _pw import validate_config

    with pytest.raises(ValueError, match="normalize"):
        validate_config({"method": "threshold", "normalize": "global"})


def _two_channel_store(path, membrane_level):
    """Membrane (ch 0) bright on the left half only, nuclei (ch 1) spots."""
    from patchworks.plugins.ome_zarr import to_ome_zarr

    rng = np.random.default_rng(0)
    img = rng.normal(100, 10, (2, 8, 64, 64)).clip(1).astype("uint16")
    img[0, :, :, :32] += membrane_level
    img[1, 2:6, 10:20, 10:20] += 2000
    to_ome_zarr(img, str(path), axes="czyx", n_levels=1, progress=False)


def test_image_wide_kwargs_resolve_per_tile_otsu_once(tmp_path):
    """foreground: "otsu" and an unset nuclei_threshold are each measured
    once over the image, so every tile gets the same number -- per tile they
    move with each tile's content and the mask changes at every seam."""
    from _pw import image_wide_kwargs

    store = tmp_path / "image.zarr"
    _two_channel_store(store, 1000)
    cfg = {
        "method": "custom",
        "channel": 0,
        "nuclei_channel": 1,
        "custom": {"module": "patchworks.plugins.watershed", "kwargs": {}},
    }
    out = image_wide_kwargs(cfg, str(store))
    assert set(out) == {"foreground", "nuclei_threshold"}
    assert 110 < out["foreground"] < 1100  # between the two halves
    assert 110 < out["nuclei_threshold"] < 2100
    # Explicit numbers, seeds from labels, other methods: nothing to do.
    cfg["custom"]["kwargs"] = {"foreground": 500, "nuclei_threshold": 900}
    assert image_wide_kwargs(cfg, str(store)) == {}
    assert image_wide_kwargs({**cfg, "method": "cellpose"}, str(store)) == {}
    plantseg = {**cfg, "seed_labels": "nuclei", "nuclei_channel": None}
    plantseg["custom"] = {"module": "patchworks.plugins.plantseg", "kwargs": {}}
    # PlantSeg: its standardization always; foreground only when asked.
    out = image_wide_kwargs(plantseg, str(store))
    assert set(out) == {"intensity_stats"}
    mean, std = out["intensity_stats"]
    assert 100 < mean < 1100 and std > 0
    plantseg["custom"]["kwargs"] = {"foreground": "otsu"}
    assert set(image_wide_kwargs(plantseg, str(store))) == {
        "foreground",
        "intensity_stats",
    }


def test_build_fn_applies_the_image_wide_kwargs(monkeypatch):
    import _pw

    seen = {}

    def fake(cfg, intensity_range=None):
        seen.update(cfg["custom"]["kwargs"])
        return lambda tile: tile

    monkeypatch.setattr(_pw, "_build_method_fn", fake)
    cfg = {
        "method": "custom",
        "custom": {"module": "m", "kwargs": {"foreground": "otsu", "a": 1}},
    }
    _pw.build_fn(cfg, kwargs_overrides={"foreground": 812.5})
    assert seen == {"foreground": 812.5, "a": 1}
    assert cfg["custom"]["kwargs"]["foreground"] == "otsu"  # not mutated
