"""The napari review panel, driven like a user would (needs a display)."""

import os

import numpy as np
import pytest

pytest.importorskip("napari")
pytest.importorskip("pandas")
if not os.environ.get("DISPLAY"):
    pytest.skip(
        "needs a display (napari draws with OpenGL)", allow_module_level=True
    )

from test_review import EXPECT  # noqa: E402
from test_tables import make_scene  # noqa: E402


@pytest.fixture
def panel(tmp_path):
    from patchworks.plugins.review import review_in_napari

    store = make_scene(tmp_path / "s.zarr")
    viewer, widget = review_in_napari(store, expect=EXPECT, show=False)
    yield viewer, widget
    viewer.close()


def visible(layer):
    cmap = layer.colormap.color_dict
    return sorted(k for k in cmap if k and cmap[k][3] > 0)


def test_panel_walks_the_queue_and_fixes_things(panel):
    viewer, w = panel
    assert "cyto_labels: to review" in viewer.layers
    assert w.title.text().startswith("cyto #4")
    assert "0 nuclei (expected 1)" in w.reasons.text()
    assert visible(viewer.layers["cyto_labels"]) == [4]  # only this cell
    from patchworks.plugins.review import _camera

    np.testing.assert_allclose(_camera(viewer).center[-2:], (71.5, 71.5))

    w.name_box.setCurrentText("nuclei_labels")
    assert w.btn_join.text() == "Join with #6  [J]"
    w.join()  # the split nucleus: one object again
    assert w.rv.effective("nuclei_labels").index.tolist() == [2, 3, 6]

    w.name_box.setCurrentText("cilia_labels")
    assert w.title.text().startswith("cilia #4")
    # cilium 4 is shown with no cell (it is in none), cilium 5 with its cell
    w.decide("wrong")
    assert w.title.text().startswith("cilia #5")
    assert visible(viewer.layers["cyto_labels"]) == [2]
    assert viewer.layers["cyto_labels"].contour == 2  # the parent, outlined
    w.start("parent")
    assert "Click the right cyto" in w.prompt.text()
    w.pick((0, 3))  # a click on cell 3 (multiscale layers give (level, id))
    assert w.rv.effective("cilia_labels").loc[5, "cyto_labels_id"] == 3

    w.by_parent_box.setChecked(True)
    by = viewer.layers["cilia_labels by cyto_labels"]
    assert by.metadata["lut"].tolist() == [0, 1, 2, 3, 0, 3]

    w.focus_box.setChecked(False)
    assert (
        type(viewer.layers["cyto_labels"].colormap).__name__
        != ("DirectLabelColormap")
        or len(viewer.layers["cyto_labels"].colormap.color_dict) > 3
    )

    from patchworks._review import Review

    saved = Review(w.rv.store).decisions
    assert sorted(saved["nuclei_labels"]) == [5]
    assert sorted(saved["cilia_labels"]) == [4, 5]


def test_skip_back_and_undo(panel):
    viewer, w = panel
    first = w.current
    w.next()
    assert w.current != first
    w.back()
    assert w.current == first
    w.decide("ok")
    w.back()
    assert w.current == first and "reviewed: ok" in w.title.text()
    w.undo()
    assert first not in w.rv.decisions["cyto_labels"]
