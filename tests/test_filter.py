"""Object sizes and filtering a label image by volume, after a run."""

import numpy as np

from patchworks._filter import filter_labels, size_report
from patchworks._tables import compute_table, read_columns, relate_tables
from patchworks.plugins.ome_zarr import to_ome_zarr, write_labels


def _scene(path):
    """Two cells; one cilium each (27 voxels) and 30 one-voxel specks."""
    shape = (12, 64, 64)
    cells = np.zeros(shape, "int32")
    cells[:, :, :32] = 1
    cells[:, :, 32:] = 2
    cil = np.zeros(shape, "int32")
    cil[4:7, 10:13, 10:13] = 1
    cil[4:7, 40:43, 40:43] = 2
    rng = np.random.default_rng(0)
    for k, (z, y, x) in enumerate(rng.integers(0, 60, (30, 3)), start=3):
        cil[z % 12, y, x] = k
    store = to_ome_zarr(
        (cells * 10).astype("uint16"),
        path,
        axes="zyx",
        n_levels=1,
        pixel_size={"z": 0.5, "y": 0.5, "x": 0.5},
        progress=False,
    )
    for name, lab in (("cyto_labels", cells), ("cilia_labels", cil)):
        write_labels(
            store,
            lab,
            name=name,
            progress=False,
            n_objects=int(np.count_nonzero(np.unique(lab))),
        )
        compute_table(store, name)
    relate_tables(store, "cilia_labels", "cyto_labels")
    return store, cil


def test_sizes_suggest_the_cut_between_specks_and_objects(tmp_path):
    store, _ = _scene(tmp_path / "s.zarr")
    report = size_report(store, "cilia_labels")
    assert report["unit"] == "µm³" and report["n"] == 32
    assert 0.125 < report["suggested"] <= 27 * 0.125
    at = {r["min_volume"]: r for r in report["cuts"]}
    row = at[report["suggested"]]
    assert row["kept"] == 2
    assert row["per cyto_labels"] == 1.0  # one cilium per cell
    assert row["cyto_labels with one or more"] == 1.0


def test_filter_writes_a_new_label_image_without_the_specks(tmp_path):
    import zarr

    store, cil = _scene(tmp_path / "s.zarr")
    out, kept, dropped = filter_labels(store, "cilia_labels", min_volume=0.5)
    assert (out, kept, dropped) == ("cilia_labels_filtered", 2, 30)
    new = np.asarray(zarr.open_group(f"{store}/labels/{out}")["0"])
    assert sorted(np.unique(new)) == [0, 1, 2]
    assert (new > 0).sum() == 2 * 27
    table = read_columns(f"{store}/labels/{out}")
    assert table["label"].tolist() == [1, 2]
    assert table["cyto_labels_id"].tolist() == [1, 2]
    # The original is untouched.
    old = np.asarray(zarr.open_group(f"{store}/labels/cilia_labels")["0"])
    assert np.array_equal(old, cil)
