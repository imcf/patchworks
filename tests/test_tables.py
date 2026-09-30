"""Object tables: measured once, stored with the labels they describe."""

import numpy as np
import pytest
import scipy.ndimage as ndi
import zarr

from patchworks import provenance
from patchworks._tables import (
    StaleTableError,
    compute_table,
    measure_objects,
    read_table,
    relate_tables,
)
from patchworks.plugins.ome_zarr import to_ome_zarr, write_labels

pd = pytest.importorskip("pandas")


def make_scene(path, tile=(8, 32, 32)):
    """Four cells; each planted mistake is one the review must find.

    - cell 4 holds no nucleus and no cilium;
    - the nucleus of cell 1 is split in two exactly at the x=32 tile seam
      (ids 5 and 6);
    - cilium 4 lies in the gap between cells (inside none);
    - cilium 5 is only 1/4 inside cell 2.
    """
    shape = (16, 96, 96)
    cells = np.zeros(shape, "int32")
    nuc = np.zeros(shape, "int32")
    cil = np.zeros(shape, "int32")
    k = 0
    for y in (4, 52):
        for x in (4, 52):
            k += 1
            cells[2:14, y : y + 40, x : x + 40] = k
            if k in (2, 3):
                nuc[4:12, y + 10 : y + 20, x + 10 : x + 20] = k
    nuc[4:12, 14:24, 26:32] = 5
    nuc[4:12, 14:24, 32:38] = 6
    cil[6:8, 30:33, 30:33] = 1
    cil[6:8, 30:33, 70:73] = 2
    cil[6:8, 70:73, 20:23] = 3
    cil[6:8, 46:49, 46:49] = 4
    cil[6:8, 43:47, 60:63] = 5  # row 43 inside cell 2 (y 4..43), 44-46 out
    img = ((cells > 0) * 100 + (nuc > 0) * 200 + (cil > 0) * 400).astype(
        "uint16"
    )
    store = to_ome_zarr(img, path, axes="zyx", n_levels=1)
    rec = provenance(tile_shape=list(tile), level=0)
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
            n_objects=int(np.count_nonzero(np.unique(lab))),
        )
        compute_table(store, name, channels=[0])
    relate_tables(store, "nuclei_labels", "cyto_labels")
    relate_tables(store, "cilia_labels", "cyto_labels")
    return store


@pytest.fixture
def scene(tmp_path):
    return make_scene(tmp_path / "s.zarr")


def test_measure_objects_matches_scipy_across_blocks():
    rng = np.random.default_rng(0)
    lab = ndi.label(ndi.gaussian_filter(rng.random((20, 70, 90)), 2) > 0.52)[
        0
    ].astype("int32")
    img = (rng.random(lab.shape) * 100).astype("float32")
    z = zarr.create_array(
        store={}, shape=lab.shape, chunks=(7, 16, 32), dtype="int32"
    )
    z[...] = lab
    t = measure_objects(
        z, images={"a": img}, pixel_size={"z": 2, "y": 0.5, "x": 0.5}
    )
    ids = np.unique(lab)[1:]
    assert (t["label"] == ids).all()
    np.testing.assert_array_equal(
        t["area_voxels"], ndi.sum_labels(np.ones_like(lab), lab, ids)
    )
    com = np.array(ndi.center_of_mass(np.ones_like(lab), lab, ids))
    np.testing.assert_allclose(
        np.stack([t["centroid_z"], t["centroid_y"], t["centroid_x"]], 1), com
    )
    boxes = ndi.find_objects(lab)
    for i, label in enumerate(ids):
        box = boxes[label - 1]
        for ax, sl in zip("zyx", box):
            assert t[f"bbox_min_{ax}"][i] == sl.start
            assert t[f"bbox_max_{ax}"][i] == sl.stop - 1
    np.testing.assert_allclose(t["mean_intensity_a"], ndi.mean(img, lab, ids))
    np.testing.assert_allclose(
        t["std_intensity_a"], ndi.standard_deviation(img, lab, ids), rtol=1e-6
    )
    np.testing.assert_allclose(t["area_um3"], t["area_voxels"] * 0.5)


def test_table_lives_with_its_labels(scene):
    group = f"{scene}/labels/cilia_labels"
    table = read_table(group)
    assert table.index.name == "label" and len(table) == 5
    assert {
        "cyto_labels_id",
        "cyto_labels_overlap",
        "mean_intensity_ch0",
    } <= set(table.columns)
    assert table.loc[1, "cyto_labels_id"] == 1
    assert table.loc[4, "cyto_labels_id"] == 0  # in the gap
    assert table.loc[5, "cyto_labels_overlap"] == pytest.approx(0.25)

    # New labels under the table: it must not be read as theirs.
    grp = zarr.open_group(group, mode="r+")
    grp.attrs["n_objects"] = 99
    with pytest.raises(StaleTableError):
        read_table(group)
