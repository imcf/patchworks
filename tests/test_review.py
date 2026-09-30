"""The review: flags, decisions, the corrected view."""

import numpy as np
import pytest
import zarr

from patchworks._review import Review, wilson_interval, write_rules
from patchworks._tables import read_table

pd = pytest.importorskip("pandas")

from test_tables import make_scene  # noqa: E402

EXPECT = {"cyto_labels": {"nuclei_labels": 1, "cilia_labels": (1, 1)}}


@pytest.fixture
def scene(tmp_path):
    return make_scene(tmp_path / "s.zarr")


def test_flags_find_every_planted_mistake(scene):
    rv = Review(scene, expect=EXPECT)
    assert rv.parents["cilia_labels"] == ["cyto_labels"]
    assert set(rv.children["cyto_labels"]) == {"nuclei_labels", "cilia_labels"}

    cells = {f.label: f.reasons for f in rv.flags("cyto_labels")}
    assert cells[4] == ["0 nuclei (expected 1)", "0 cilia (expected 1)"]
    assert cells[1] == ["2 nuclei (expected 1)"]  # the split nucleus
    assert cells[2] == ["2 cilia (expected 1)"]  # cilium 5 counts there

    nuclei = {f.label: f for f in rv.flags("nuclei_labels")}
    assert set(nuclei) == {5, 6}
    assert nuclei[5].partner == 6 and "tile seam (x)" in nuclei[5].reasons[0]

    cilia = rv.flags("cilia_labels")
    assert [f.label for f in cilia] == [4, 5]  # most suspicious first
    assert cilia[0].reasons == ["not inside any cyto"]
    assert cilia[1].reasons == ["only 25% inside cyto #2"]


def test_decisions_apply_everywhere_and_persist(scene):
    rv = Review(scene, expect=EXPECT)
    rv.decide("nuclei_labels", 5, "merge", into=6)
    rv.decide("cilia_labels", 4, "wrong")
    rv.decide("cilia_labels", 5, "parent", parent="cyto_labels", parent_id=3)

    rv = Review(scene, expect=EXPECT)  # from disk
    nuclei = rv.effective("nuclei_labels")
    assert nuclei.index.tolist() == [2, 3, 6]
    merged = nuclei.loc[6]
    assert merged["area_voxels"] == 960
    assert merged["centroid_x"] == pytest.approx(31.5)
    assert (merged["bbox_min_x"], merged["bbox_max_x"]) == (26, 37)
    assert merged["qc"] == "fixed" and merged["cyto_labels_id"] == 1

    cilia = rv.effective("cilia_labels")
    assert 4 not in cilia.index
    assert cilia.loc[5, "cyto_labels_id"] == 3
    assert np.isnan(cilia.loc[5, "cyto_labels_overlap"])  # set by hand

    cells = rv.effective("cyto_labels")
    assert cells["n_nuclei_labels"].to_dict() == {1: 1, 2: 1, 3: 1, 4: 0}
    assert cells["n_cilia_labels"].to_dict() == {1: 1, 2: 1, 3: 2, 4: 0}
    assert rv.queue("nuclei_labels") == [] and rv.queue("cilia_labels") == []
    assert {f.label for f in rv.flags("cyto_labels")} == {3, 4}

    rv.undo("cilia_labels", 4)
    assert 4 in Review(scene).effective("cilia_labels").index


def test_merging_a_parent_moves_its_children(scene):
    rv = Review(scene, expect=EXPECT)
    rv.decide("cyto_labels", 2, "merge", into=1)
    cilia = rv.effective("cilia_labels")
    assert cilia.loc[2, "cyto_labels_id"] == 1  # was in cell 2
    assert rv.effective("cyto_labels").loc[1, "n_cilia_labels"] == 3


def test_decide_rejects_nonsense(scene):
    rv = Review(scene)
    with pytest.raises(KeyError):
        rv.decide("cilia_labels", 999, "ok")
    with pytest.raises(ValueError):
        rv.decide("cilia_labels", 1, "maybe")
    with pytest.raises(ValueError):
        rv.decide("cilia_labels", 1, "parent", parent="nuclei_labels")
    with pytest.raises(KeyError):
        rv.decide("cilia_labels", 1, "merge", into=1)


def test_random_sample_gives_an_error_rate(scene):
    rv = Review(scene)
    order = rv.random_order("cilia_labels")
    assert sorted(order) == [1, 2, 3, 4, 5]
    assert rv.queue("cilia_labels", "random") == order
    rv.decide("cilia_labels", order[0], "ok", queue="random")
    rv.decide("cilia_labels", order[1], "wrong", queue="random")
    s = rv.summary("cilia_labels")
    assert (s["sample"], s["sample_errors"], s["error_rate"]) == (2, 1, 0.5)
    assert s["error_ci"] == pytest.approx(wilson_interval(1, 2))
    assert wilson_interval(3, 100) == pytest.approx((0.0103, 0.0845), abs=1e-3)


def test_rules_recorded_in_the_store(scene):
    write_rules(scene, {"expect": {"cyto_labels": {"cilia_labels": [0, 3]}}})
    rv = Review(scene)
    assert rv.expect == {"cyto_labels": {"cilia_labels": (0, 3)}}
    assert all(
        "cilia" not in r for f in rv.flags("cyto_labels") for r in f.reasons
    )
    with pytest.raises(ValueError, match="unknown"):
        write_rules(scene, {"expekt": {}})


def test_exports_and_relation_workbook(scene, tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    rv = Review(scene, expect=EXPECT)
    rv.decide("cilia_labels", 4, "wrong")
    paths = rv.export(tmp_path / "out", "csv")
    cilia = pd.read_csv(
        tmp_path / "out" / "cilia_labels.csv", index_col="label"
    )
    assert len(paths) == 3 and 4 not in cilia.index

    book = rv.relation_workbook(
        "cilia_labels", "cyto_labels", tmp_path / "rel.xlsx"
    )
    wb = openpyxl.load_workbook(book)
    rows = list(wb["cilia_labels"].iter_rows(values_only=True))
    assert rows[0] == (
        "cilia_labels_id",
        "cyto_labels_id",
        "overlap_voxels",
        "overlap_fraction",
        "qc",
    )
    assert [r[0] for r in rows[1:]] == [1, 2, 3, 5]
    counts = {
        r[0]: r[1]
        for r in wb["cyto_labels"].iter_rows(min_row=2, values_only=True)
    }
    assert counts == {1: 1, 2: 2, 3: 1, 4: 0}


def test_workbook_falls_back_to_csv_beyond_excel(scene, tmp_path, monkeypatch):
    import patchworks._review as review

    monkeypatch.setattr(review, "EXCEL_MAX_ROWS", 3)
    rv = Review(scene)
    out = rv.relation_workbook(
        "cilia_labels", "cyto_labels", tmp_path / "r.xlsx"
    )
    assert out.suffix == ".csv" and not (tmp_path / "r.xlsx").exists()
    assert (tmp_path / "r_cyto_labels.csv").exists()


def test_write_reviewed_labels(scene):
    rv = Review(scene)
    rv.decide("nuclei_labels", 5, "merge", into=6)
    rv.decide("cilia_labels", 4, "wrong")
    out = rv.write_reviewed_labels("nuclei_labels")
    arr = zarr.open_group(f"{scene}/labels/{out}", mode="r")["0"][...]
    assert set(np.unique(arr)) == {0, 2, 3, 6}
    assert read_table(f"{scene}/labels/{out}").loc[6, "area_voxels"] == 960
    cil = rv.write_reviewed_labels("cilia_labels")
    arr = zarr.open_group(f"{scene}/labels/{cil}", mode="r")["0"][...]
    assert 4 not in np.unique(arr)
