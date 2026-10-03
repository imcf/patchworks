"""Nuclei-seeded watershed: one cell per nucleus, walls from the membrane."""

import numpy as np
import pytest

pytest.importorskip("skimage")

from patchworks.plugins import watershed as ws  # noqa: E402


def epithelium(seed=0, gap=False):
    """Four cells in a 2 x 2 grid, (z, y, x) = (8, 40, 40), separated by
    bright walls, each with a nucleus; background beyond x = 40.

    With *gap*, the wall between the two upper cells is missing on half its
    length -- where a boundary-only method merges them.
    """
    rng = np.random.default_rng(seed)
    shape = (8, 40, 56)
    membrane = rng.normal(10, 2, shape).astype("float32")
    nuclei = rng.normal(10, 2, shape).astype("float32")
    tissue = (slice(None), slice(0, 40), slice(0, 40))
    membrane[tissue] += 20  # cytoplasm a little above background
    for line in (0, 19, 20, 39):
        membrane[:, line, :40] = 200
        membrane[:, :40, line] = 200
    if gap:
        membrane[:, 2:10, 19:21] = 30
    truth = np.zeros(shape[1:], "int32")
    for i, (y, x) in enumerate(((10, 10), (10, 30), (30, 10), (30, 30)), 1):
        nuclei[2:6, y - 3 : y + 3, x - 3 : x + 3] = 150
        truth[y - 8 : y + 8, x - 8 : x + 8] = i
    return np.stack([membrane, nuclei]), truth


def test_one_cell_per_nucleus_even_across_a_broken_wall():
    tile, truth = epithelium(gap=True)
    labels = ws.segment(tile, nuclei_min_size=20)
    assert labels.shape == tile.shape[1:] and labels.dtype == np.int32
    # Four cells, each the one holding its own nucleus, none merged
    assert len(np.unique(labels[labels > 0])) == 4
    for z in range(2, 6):
        cells = {int(labels[z, y, x]) for y, x in np.argwhere(truth > 0)}
        assert len(cells) == 4
    # Each cell fills its own quadrant, walls apart
    inner = labels[4, 3:17, 3:17]
    assert (inner == inner[7, 7]).all()


def test_cells_stop_at_the_tissue_edge():
    tile, _ = epithelium()
    labels = ws.segment(tile, nuclei_min_size=20)
    assert not labels[:, :, 44:].any()  # empty space beyond the tissue
    flooded = ws.segment(tile, nuclei_min_size=20, foreground=None)
    assert flooded[:, :, 44:].all()  # without a mask a cell takes it


def test_max_radius_limits_reach_from_the_nucleus():
    tile, _ = epithelium()
    labels = ws.segment(
        tile,
        foreground=None,
        max_radius_um=2.0,
        voxel_size={"z": 1.0, "y": 0.5, "x": 0.5},
        nuclei_min_size=20,
    )
    # 2 um = 4 px around the 6 x 6 nucleus at (10, 10): rows 3..16 at most
    rows = np.flatnonzero(labels[4].any(axis=1) & (np.arange(40) < 20))
    assert rows.min() >= 3 and rows.max() <= 16


def test_needs_the_nuclear_channel_and_valid_options():
    tile, _ = epithelium()
    with pytest.raises(ValueError, match="nuclei_channel"):
        ws.segment(tile[0])
    with pytest.raises(ValueError, match="voxel_size"):
        ws.watershed_fn(max_radius_um=3)
    with pytest.raises(ValueError, match="foreground"):
        ws.watershed_fn(foreground="triangle")
    blank = np.zeros_like(tile)
    assert not ws.segment(blank).any()


def test_workflow_validates_kwargs_against_the_factory():
    from patchworks.plugins.watershed import segment, watershed_fn

    assert segment.patchworks_kwargs_target is watershed_fn
    import inspect

    assert "voxel_size" in inspect.signature(watershed_fn).parameters
