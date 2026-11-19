"""Each object's values beside its label image: napari and Neuroglancer."""

import json
import urllib.error
import urllib.request

import numpy as np
import pytest
import zarr

from patchworks._properties import (
    label_properties,
    neuroglancer_properties,
    object_properties,
    write_label_properties,
)
from patchworks._provenance import PROVENANCE_KEY
from patchworks._review import Review, write_rules
from patchworks._tables import is_stale, label_fingerprint
from patchworks.cli import main
from patchworks.plugins.ome_zarr import read_ngff_attr

pytest.importorskip("pandas")
from test_position import RULE, polarity_scene  # noqa: E402


@pytest.fixture
def store(tmp_path):
    path = polarity_scene(tmp_path / "p.zarr")
    write_rules(path, {"position": RULE})
    return path


def test_the_values_of_each_object(store):
    df = object_properties(store, "cilia_labels")
    assert list(df.columns) == [
        "area_um3",
        "length_um",
        "elongation",
        "cyto_labels_id",
        "position",
        "qc",
    ]
    assert df.loc[1, "position"] == "apical"
    assert df.loc[3, "cyto_labels_id"] == 3
    cells = object_properties(store, "cyto_labels")
    assert "n_cilia_labels" in cells and "n_nuclei_labels" in cells
    assert object_properties(store, "missing") is None


def test_label_metadata_gets_the_values_and_keeps_the_rest(store):
    group = f"{store}/labels/cilia_labels"
    before = zarr.open_group(group, mode="r")
    record = dict(before.attrs)[PROVENANCE_KEY]
    fingerprint = label_fingerprint(before)

    line = write_label_properties(store, "cilia_labels")
    assert line.startswith("labels/cilia_labels: values of 5 objects")
    after = zarr.open_group(group, mode="r")
    image_label = read_ngff_attr(after.attrs, "image-label")
    assert image_label["source"] == {"image": "../../"}
    first = image_label["properties"][0]
    assert first["label-value"] == 1 and first["position"] == "apical"
    assert first["cyto_labels_id"] == 1
    assert first["length_um"] == pytest.approx(0.5 * np.sqrt(99), rel=1e-3)
    # The labels, their record and so their table are as they were
    assert dict(after.attrs)[PROVENANCE_KEY] == record
    assert label_fingerprint(after) == fingerprint
    assert not is_stale(group)

    # Too many objects: none, and an older list is not left behind
    line = write_label_properties(store, "cilia_labels", limit=3)
    assert "5 objects, more than 3" in line
    image_label = read_ngff_attr(
        zarr.open_group(group, mode="r").attrs, "image-label"
    )
    assert "properties" not in image_label and "source" in image_label


def test_fix_metadata_and_tables_write_them(store, capsys):
    assert main(["fix-metadata", store, "--properties"]) == 0
    out = capsys.readouterr().out
    for name in ("cyto_labels", "nuclei_labels", "cilia_labels"):
        assert f"labels/{name}: values of" in out
    nuclei = zarr.open_group(f"{store}/labels/nuclei_labels", mode="r")
    entries = read_ngff_attr(nuclei.attrs, "image-label")["properties"]
    assert [e["cyto_labels_id"] for e in entries] == [1, 2, 3, 4]

    assert main(["tables", store, "--names", "cyto_labels"]) == 0
    assert "labels/cyto_labels: values of 5 objects" in capsys.readouterr().out


def test_missing_values_are_left_out():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {"length_um": [np.inf, 2.0], "position": ["", "basal"]}, index=[1, 2]
    )
    assert label_properties(df) == [
        {"label-value": 1},
        {"label-value": 2, "length_um": 2.0, "position": "basal"},
    ]
    info = neuroglancer_properties(df)
    numbers = info["inline"]["properties"][1]
    assert numbers["id"] == "length_um" and numbers["values"] == ["NaN", 2.0]
    json.dumps(info, allow_nan=False)  # valid JSON: no bare NaN


def test_neuroglancer_reads_them_through_serve(store):
    """The state points each label image with a table at its values, and
    the server answers them."""
    from patchworks._serve import neuroglancer_state

    state = neuroglancer_state(store, "http://localhost:1/tok/p.zarr")
    cilia = next(la for la in state["layers"] if la["name"] == "cilia_labels")
    assert cilia["source"] == [
        "zarr3://http://localhost:1/tok/p.zarr/labels/cilia_labels",
        "precomputed://http://localhost:1/tok/p.zarr/.properties/cilia_labels",
    ]

    base = _serving(store)
    got = json.load(
        urllib.request.urlopen(f"{base}/.properties/cilia_labels/info")
    )
    assert got["@type"] == "neuroglancer_segment_properties"
    assert got["inline"]["ids"] == ["1", "2", "3", "4", "5"]
    summary = got["inline"]["properties"][0]["values"][0]
    assert "µm³" in summary and "in cyto_labels 1" in summary
    assert "apical" in summary
    tags = got["inline"]["properties"][-1]
    assert tags["id"] == "tags" and "apical" in tags["tags"]
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(f"{base}/.properties/nope/info")


def _serving(store):
    """Serve *store* in a thread, as `serve` does; the store's URL."""
    import threading
    from functools import partial
    from http.server import ThreadingHTTPServer
    from pathlib import Path

    from patchworks._serve import _Handler, _Properties

    attrs = {
        "token": "tok",
        "store_name": Path(store).name,
        "properties": _Properties(store),
    }
    handler = partial(
        type("H", (_Handler,), attrs), directory=str(Path(store).parent)
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{httpd.server_address[1]}/tok/{Path(store).name}"


def test_review_decisions_show(store):
    review = Review(store)
    review.decide("cilia_labels", 2, "wrong")
    df = object_properties(store, "cilia_labels", review)
    assert 2 not in df.index
