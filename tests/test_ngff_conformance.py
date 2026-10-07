"""What the OME-Zarr spec requires of what patchworks writes.

Checked against an independent validator (ome-zarr-models) when this was
written; these tests pin the specific requirements it found broken.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import zarr

from patchworks import create_stage, provenance
from patchworks.plugins.ome_zarr import (
    fix_ngff_metadata,
    read_ngff_attr,
    to_ome_zarr,
    write_labels,
)


def _store(tmp_path, version, n_levels=3):
    img = np.random.default_rng(0).integers(0, 999, (2, 8, 64, 80))
    lab = np.zeros((8, 64, 80), "int32")
    lab[2:6, 10:30, 20:50] = 1
    path = to_ome_zarr(
        img.astype("uint16"),
        tmp_path / f"img_{version}.zarr",
        axes="czyx",
        n_levels=n_levels,
        pixel_size=(0.5, 0.1, 0.1),
        ngff_version=version,
        progress=False,
    )
    # n_levels asked for differs from the image's: the image's must win
    write_labels(
        path,
        lab,
        name="cells",
        n_levels=5,
        progress=False,
        provenance=provenance(level=0),
    )
    write_labels(
        path,
        lab[:, ::2, ::2].copy(),
        name="coarse",
        progress=False,
        provenance=provenance(level=1),
        level=1,
    )
    return str(path)


def _datasets(group):
    return [
        d["path"]
        for d in read_ngff_attr(group.attrs, "multiscales")[0]["datasets"]
    ]


def test_v05_arrays_carry_dimension_names(tmp_path):
    """0.5 requires every array's dimension_names to match the axes."""
    root = zarr.open_group(_store(tmp_path, "0.5"), mode="r")
    for path in _datasets(root):
        assert root[path].metadata.dimension_names == ("c", "z", "y", "x")
    for name in ("cells", "coarse"):
        group = root["labels"][name]
        for path in _datasets(group):
            assert group[path].metadata.dimension_names == ("z", "y", "x")


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_label_images_have_as_many_levels_as_the_image(tmp_path, version):
    """'the datasets key MUST have the same number of entries (scale
    levels) as the original unlabeled image' -- also for labels made at a
    coarser level, and whatever n_levels asked for."""
    root = zarr.open_group(_store(tmp_path, version), mode="r")
    n = len(_datasets(root))
    assert n == 3
    for name in ("cells", "coarse"):
        assert len(_datasets(root["labels"][name])) == n


def test_staging_into_a_v2_label_group_keeps_v2(tmp_path):
    """Segmenting in place stages into labels/<name>: a v3 group there made
    a 0.4 store no reader could open."""
    path = _store(tmp_path, "0.4")
    group = f"{path}/labels/new"
    zarr.open_group(path, mode="a")["labels"].require_group("new")
    create_stage(group, (8, 64, 80), (8, 32, 40), component="0", zarr_format=2)
    assert (Path(group) / ".zgroup").is_file()
    assert not (Path(group) / "zarr.json").exists()
    assert (Path(group) / "0" / ".zarray").is_file()


def test_an_existing_empty_directory_is_written_in_the_asked_version(tmp_path):
    """Snakemake creates the output directory before the job: taken for an
    existing store, it made ngff_version 0.4 write zarr v3."""
    out = tmp_path / "pre.zarr"
    out.mkdir()
    to_ome_zarr(
        np.zeros((4, 16, 16), "uint16"),
        out,
        axes="zyx",
        n_levels=1,
        ngff_version="0.4",
        progress=False,
        overwrite=True,
    )
    assert (out / ".zgroup").is_file() and not (out / "zarr.json").exists()


def test_fix_metadata_repairs_stores_from_older_versions(tmp_path):
    """No dimension_names, more label levels than the image: repaired in
    place, metadata only, and a second run finds nothing."""
    from zarr.core.sync import sync

    path = _store(tmp_path, "0.5")
    root = zarr.open_group(path, mode="r+")
    # As older versions wrote it: no names, and a surplus label level
    for group in (root, root["labels"]["cells"]):
        for p in _datasets(group):
            inner = group[p]._async_array
            sync(
                inner._save_metadata(
                    dataclasses.replace(inner.metadata, dimension_names=None)
                )
            )
    cells = root["labels"]["cells"]
    ms = read_ngff_attr(cells.attrs, "multiscales")
    extra = dict(ms[0]["datasets"][-1], path="9")
    cells.create_array("9", shape=(8, 4, 5), dtype="int32")
    from patchworks.plugins.ome_zarr import write_ngff_attrs

    write_ngff_attrs(
        cells, multiscales=[dict(ms[0], datasets=[*ms[0]["datasets"], extra])]
    )
    before = np.asarray(cells["0"])

    changes = fix_ngff_metadata(path)
    assert any("dimension_names" in c for c in changes)
    assert any("dropped 1 coarse level(s) (9)" in c for c in changes)
    root = zarr.open_group(path, mode="r")
    assert root["0"].metadata.dimension_names == ("c", "z", "y", "x")
    assert _datasets(root["labels"]["cells"]) == _datasets(root)
    assert "9" not in root["labels"]["cells"]
    assert np.array_equal(np.asarray(root["labels"]["cells"]["0"]), before)
    assert fix_ngff_metadata(path) == []
