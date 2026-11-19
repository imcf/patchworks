"""`patchworks validate`: the spec, and what patchworks writes."""

import numpy as np
import pytest
import zarr

from patchworks._validate import patchworks_findings, spec_findings, validate
from patchworks.cli import main
from patchworks.plugins.ome_zarr import (
    read_ngff_attr,
    to_ome_zarr,
    write_ngff_attrs,
)

pytest.importorskip("yaozarrs")
pytest.importorskip("pandas")
from test_ngff_conformance import _store  # noqa: E402
from test_position import polarity_scene  # noqa: E402


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_what_patchworks_writes_is_valid(tmp_path, version):
    """Images, label images (also at a coarser level), tables and the
    objects' values: valid for the spec validator, nothing to repair."""
    path = _store(tmp_path, version)
    assert spec_findings(path) == []
    findings, summary, checked = validate(path)
    # Two channels written without names: the one thing left to say
    assert checked
    assert [f.code for f in findings] == ["pw-channel-names"]
    assert not findings[0].error
    assert any(line.startswith("labels/cells") for line in summary)


def test_tables_and_values_stay_valid(tmp_path, capsys):
    store = polarity_scene(tmp_path / "p.zarr")
    assert main(["fix-metadata", store, "--properties"]) == 0
    capsys.readouterr()
    assert main(["validate", store]) == 0
    out = capsys.readouterr().out
    assert "valid: the OME-Zarr spec and what patchworks writes" in out
    assert "object table current; values of 5 objects in its metadata" in out
    assert "made " in out and "level 0: (48, 32, 160) int32" in out


def test_problems_and_their_repairs(tmp_path, capsys):
    path = _store(tmp_path, "0.5")
    cells = zarr.open_group(path, mode="r+")["labels"]["cells"]
    image_label = read_ngff_attr(cells.attrs, "image-label")
    write_ngff_attrs(
        cells, **{"image-label": {"version": image_label["version"]}}
    )
    # A label image whose axes no longer match its arrays: readers refuse it
    coarse = zarr.open_group(path, mode="r+")["labels"]["coarse"]
    ms = read_ngff_attr(coarse.attrs, "multiscales")
    write_ngff_attrs(coarse, multiscales=[dict(ms[0], axes=ms[0]["axes"][1:])])

    assert main(["validate", path, "-q"]) == 1
    out = capsys.readouterr().out
    assert "ERROR   labels/coarse [label_image_invalid]" in out
    assert (
        "warning labels/cells [pw-label-source]: does not name its image "
        "(source; patchworks fix-metadata)" in out
    )
    assert out.rstrip().endswith("1 error(s), 2 warning(s)")
    assert "level 0" not in out  # --quiet: no summary


def test_without_the_validator(tmp_path, monkeypatch, capsys):
    import builtins

    path = to_ome_zarr(
        np.zeros((4, 16, 16), "uint16"),
        tmp_path / "a.zarr",
        axes="zyx",
        n_levels=1,
        progress=False,
    )
    real = builtins.__import__

    def no_yaozarrs(name, *args, **kwargs):
        if name == "yaozarrs":
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_yaozarrs)
    assert spec_findings(path) is None
    assert main(["validate", str(path)]) == 0
    out = capsys.readouterr().out
    assert "OME-Zarr spec not checked" in out
    assert "warning image [pw-calibration]" in out
    findings, _ = patchworks_findings(path)
    assert [f.code for f in findings] == ["pw-calibration"]
