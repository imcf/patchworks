"""Tests for choosing the overlap from the data."""

import numpy as np

from patchworks import object_f1, suggest_overlap


def test_object_f1():
    a = np.zeros((8, 8), "int32")
    a[1:4, 1:4] = 1
    a[5:8, 5:8] = 2
    b = a.copy()
    assert object_f1(a, b) == 1.0
    b[5:8, 5:8] = 0  # one object missed
    assert object_f1(a, b) == 2 / 3
    assert object_f1(np.zeros((3, 3), int), np.zeros((3, 3), int)) == 1.0


def _contextual(tile):
    """A method that needs context: keeps only objects with >= 60 voxels
    in its view, so an object cut small by a tile edge is lost unless the
    halo shows enough of it."""
    from scipy import ndimage as ndi

    lab, _ = ndi.label(tile > 0)
    sizes = np.bincount(lab.ravel())
    keep = sizes >= 60
    keep[0] = False
    return ndi.label(keep[lab])[0].astype("int32")


def test_suggest_overlap_finds_the_halo_objects_need():
    img = np.zeros((1, 64, 64), "uint16")
    # 8x8 squares straddling the x=32 and y=32 seams: each tile sees at
    # most half of one without a halo, too small for the method.
    img[0, 4:12, 28:36] = 1
    img[0, 28:36, 4:12] = 1
    img[0, 44:52, 26:34] = 1
    out = suggest_overlap(
        img, _contextual, (1, 32, 32), candidates=(0, 2, 4, 8), crop_tiles=2
    )
    assert out["reference_objects"] == 3
    assert out["scores"][0] < 0.99
    assert out["overlap"] in (4, 8)
    assert max(out["scores"]) == out["overlap"]  # stopped at the first hit
