"""Shape measures, and where objects sit in their parent (apical, basal...)."""

import numpy as np
import pytest

from patchworks import provenance
from patchworks._review import Review
from patchworks._tables import compute_table, relate_tables
from patchworks.plugins.ome_zarr import to_ome_zarr, write_labels

pd = pytest.importorskip("pandas")

RULE = {"cilia_labels": {"parent": "cyto_labels", "apical": "nuclei_labels"}}


def polarity_scene(path):
    """Five tall cells (z 2..41, 24x24 in y/x), anisotropic voxels.

    Cells 1-4 have a nucleus at the bottom (basal), so apical is +z; each
    holds one cilium, placed: 1 at the top (apical, pointing up), 2 at the
    bottom (basal), 3 through the side wall (lateral, horizontal), 4 in the
    middle (central). Cell 5 has a cilium but no nucleus: no axis.
    """
    shape = (48, 32, 160)
    cells = np.zeros(shape, "int32")
    nuc = np.zeros(shape, "int32")
    cil = np.zeros(shape, "int32")
    for k in range(5):
        x0 = 4 + 32 * k
        cells[2:42, 4:28, x0 : x0 + 24] = k + 1
        if k < 4:
            nuc[4:14, 10:22, x0 + 6 : x0 + 18] = k + 1
    cx = [4 + 32 * k + 12 for k in range(5)]
    cil[36:46, 16, cx[0]] = 1  # up through the top
    cil[3:9, 16, cx[1]] = 2  # at the bottom
    cil[22, 16, cx[2] + 8 : cx[2] + 15] = 3  # out through the side
    cil[20:25, 16, cx[3]] = 4  # in the middle
    cil[36:46, 16, cx[4]] = 5  # top of the cell without a nucleus
    img = ((cells > 0) * 100).astype("uint16")
    store = to_ome_zarr(
        img, path, axes="zyx", n_levels=1, pixel_size=(0.5, 0.25, 0.25)
    )
    rec = provenance(tile_shape=[48, 32, 160], level=0)
    for name, lab in (
        ("cyto_labels", cells),
        ("nuclei_labels", nuc),
        ("cilia_labels", cil),
    ):
        write_labels(
            store,
            lab,
            name=name,
            overwrite=True,
            progress=False,
            provenance=rec,
        )
        compute_table(store, name)
    relate_tables(store, "nuclei_labels", "cyto_labels")
    relate_tables(store, "cilia_labels", "cyto_labels")
    return store


@pytest.fixture
def store(tmp_path):
    return polarity_scene(tmp_path / "p.zarr")


def test_shape_measures_in_micrometres(store):
    cilia = Review(store).effective("cilia_labels")
    # 10 voxels along z at 0.5 um: a rod of ~5 um (sqrt(n^2-1) * spacing)
    assert cilia.loc[1, "length_um"] == pytest.approx(0.5 * np.sqrt(99))
    assert cilia.loc[1, "axis_z"] == pytest.approx(1.0)
    assert cilia.loc[3, "axis_x"] == pytest.approx(1.0)  # horizontal
    assert cilia.loc[3, "length_um"] == pytest.approx(0.25 * np.sqrt(48))
    assert np.isinf(cilia.loc[1, "elongation"])  # a line: no second axis


def test_positions_from_the_nucleus(store):
    rv = Review(store, position=RULE)
    cilia = rv.effective("cilia_labels")
    assert cilia["position"].to_dict() == {
        1: "apical",
        2: "basal",
        3: "lateral",
        4: "central",
        5: "unknown",
    }
    assert cilia.loc[1, "angle_to_axis_deg"] == pytest.approx(0, abs=1e-6)
    assert cilia.loc[3, "angle_to_axis_deg"] == pytest.approx(90, abs=1e-6)
    assert cilia.loc[1, "position_axial"] > 0.5 > cilia.loc[4, "position_axial"]
    flagged = {f.label: f.reasons for f in rv.flags("cilia_labels")}
    assert flagged[5] == [
        "position unclear: its cyto has no nuclei to orient it"
    ]

    cells = rv.effective("cyto_labels")
    assert cells.loc[1, "n_cilia_labels_apical"] == 1
    assert cells.loc[3, "n_cilia_labels_lateral"] == 1
    assert cells["n_cilia_labels_basal"].sum() == 1


def test_fixed_direction_and_corrections(store):
    rv = Review(
        store,
        position={"cilia_labels": {"parent": "cyto_labels", "apical": "+z"}},
    )
    assert rv.effective("cilia_labels").loc[5, "position"] == "apical"
    rv.decide("cilia_labels", 4, "position", position="basal")
    cilia = Review(store, position=RULE).effective("cilia_labels")
    assert cilia.loc[4, "position"] == "basal" and cilia.loc[4, "qc"] == "fixed"
    with pytest.raises(ValueError):
        rv.decide("cilia_labels", 4, "position", position="upside")
    with pytest.raises(ValueError, match="needs"):
        Review(
            store,
            position={
                "cilia_labels": {"parent": "cyto_labels", "apical": "lumen"}
            },
        ).effective("cilia_labels")


def test_merged_cells_keep_exact_spread(store):
    """Joining two objects combines their spread exactly (as if measured
    together), so the shape and position of the result stay right."""
    rv = Review(store)
    before = rv.effective("cyto_labels")
    rv.decide("cyto_labels", 2, "merge", into=1)
    merged = rv.effective("cyto_labels").loc[1]
    width = before.loc[1, "bbox_max_x"] - before.loc[1, "bbox_min_x"] + 1
    gap = 32  # cell 2 starts 32 voxels further along x
    xs = np.r_[np.arange(width), np.arange(width) + gap].astype(float)
    assert merged["cov_xx"] == pytest.approx(xs.var())
    assert merged["cov_zz"] == pytest.approx(before.loc[1, "cov_zz"])


def test_relation_workbook_has_positions(store, tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    rv = Review(store, position=RULE)
    book = rv.relation_workbook(
        "cilia_labels", "cyto_labels", tmp_path / "r.xlsx"
    )
    wb = openpyxl.load_workbook(book)
    header = next(wb["cilia_labels"].iter_rows(values_only=True))
    assert "position" in header
    cells = list(wb["cyto_labels"].iter_rows(values_only=True))
    assert "cilia_labels_apical" in cells[0]


def test_tables_from_before_moments_classify_as_unknown(store, caplog):
    """A table written before the spread columns existed must not break the
    review: positions are unknown, with a message saying what to do."""
    import zarr

    for name in ("cilia_labels", "cyto_labels"):
        table = zarr.open_group(f"{store}/labels/{name}/table", mode="r+")
        for key in [k for k in table.array_keys() if k.startswith("cov_")]:
            del table[key]
        meta = dict(table.attrs["patchworks_table"])
        meta["columns"] = [
            c for c in meta["columns"] if not c.startswith("cov_")
        ]
        table.attrs["patchworks_table"] = meta
    cilia = Review(store, position=RULE).effective("cilia_labels")
    assert set(cilia["position"]) == {"unknown"}
    assert "length_um" not in cilia
    assert "recompute" in caplog.text
