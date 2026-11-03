"""patchworks serve: a store over HTTP for Neuroglancer."""

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from patchworks._serve import _Handler, neuroglancer_link, neuroglancer_state
from patchworks.plugins.ome_zarr import (
    to_ome_zarr,
    write_channel_names,
    write_labels,
)


@pytest.fixture
def store(tmp_path):
    out = to_ome_zarr(
        np.arange(2 * 4 * 32 * 32, dtype="uint16").reshape(2, 4, 32, 32),
        tmp_path / "image.zarr",
        axes="czyx",
        n_levels=1,
        progress=False,
    )
    write_labels(
        out, np.ones((4, 32, 32), "int32"), name="cells", progress=False
    )
    write_channel_names(out, ["DAPI", "Cilia"])
    return out


@pytest.fixture
def server(store):
    from pathlib import Path

    handler = partial(
        type("H", (_Handler,), {"token": "secret", "store_name": "image.zarr"}),
        directory=str(Path(store).parent),
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _get(url, **headers):
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers))


def test_files_with_cors_and_byte_ranges(server, store):
    from pathlib import Path

    url = f"{server}/secret/image.zarr/zarr.json"
    whole = (Path(store) / "zarr.json").read_bytes()
    r = _get(url)
    assert r.read() == whole
    assert r.headers["Access-Control-Allow-Origin"] == "*"
    r = _get(url, Range="bytes=2-9")
    assert r.status == 206 and r.read() == whole[2:10]
    assert r.headers["Content-Range"] == f"bytes 2-9/{len(whole)}"
    # The last N bytes: how a shard's index is read
    r = _get(url, Range="bytes=-5")
    assert r.status == 206 and r.read() == whole[-5:]
    # The CORS preflight Neuroglancer's Range requests need
    pre = urllib.request.Request(url, method="OPTIONS")
    pre = urllib.request.urlopen(pre)
    assert pre.status == 204
    # Chrome's Private Network Access: a public page reading localhost
    assert pre.headers["Access-Control-Allow-Private-Network"] == "true"


def test_nothing_without_the_token_and_no_listings(server, store):
    """Only the store, under the token: not the files beside it either."""
    from pathlib import Path

    (Path(store).parent / "beside.txt").write_text("private")
    for path in (
        "/image.zarr/zarr.json",
        "/wrong/image.zarr/zarr.json",
        "/secret/image.zarr/",
        "/secret/beside.txt",
        "/secret/image.zarr/../beside.txt",
        "/secret/image.zarr/../../../etc/passwd",
    ):
        with pytest.raises(urllib.error.HTTPError) as err:
            _get(server + path)
        assert err.value.code == 404, path


def test_neuroglancer_layers_named_after_the_channels(store):
    state = neuroglancer_state(store, "http://localhost:1/t/image.zarr")
    image = [x for x in state["layers"] if x["type"] == "image"]
    assert [x["name"] for x in image] == ["DAPI", "Cilia"]
    assert [x["localPosition"] for x in image] == [[0], [1]]
    assert [x["visible"] for x in image] == [True, False]
    assert image[0]["source"] == "zarr3://http://localhost:1/t/image.zarr"
    # Each channel's own contrast, not the dtype's 0-65535 (black)
    lo, hi = image[0]["shaderControls"]["normalized"]["range"]
    assert 0 < lo < hi < 65535
    # x, y first, so "xy" is an xy section (z-y in the store's order)
    assert state["displayDimensions"] == ["x", "y", "z"]
    assert list(state["dimensions"]) == ["x", "y", "z"]
    (seg,) = [x for x in state["layers"] if x["type"] == "segmentation"]
    assert seg["name"] == "cells"
    assert seg["source"].endswith("/image.zarr/labels/cells")
    link = neuroglancer_link(state)
    assert link.startswith("https://neuroglancer-demo.appspot.com/#!")
    assert json.loads(urllib.parse.unquote(link.split("#!", 1)[1])) == state


def test_a_bundle_is_refused(tmp_path):
    from patchworks._serve import serve

    with pytest.raises(SystemExit, match="unzipped"):
        serve(tmp_path / "image.zarr.zip")
