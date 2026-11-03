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
    # Room for the fine levels of a whole slide, through the tunnel
    assert state["gpuMemoryLimit"] >= 2e9 and state["concurrentDownloads"] > 32
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


def test_neuroglancer_runs_behind_the_same_port(store):
    """With Neuroglancer's own server beside it, every path outside the
    token goes to that server (the viewer, its state stream, its POSTs), so
    the page and the store come from one address: no public page reading
    localhost, which browsers block."""
    from http.server import BaseHTTPRequestHandler
    from pathlib import Path

    class Viewer(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            """Answer with the path asked for."""
            body = f"viewer {self.path}".encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802
            """Echo the body back."""
            body = self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            """Quiet."""

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Viewer)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    handler = partial(
        type(
            "H",
            (_Handler,),
            {
                "token": "secret",
                "store_name": "image.zarr",
                "upstream": f"127.0.0.1:{upstream.server_address[1]}",
            },
        ),
        directory=str(Path(store).parent),
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        assert _get(f"{url}/v/abc/").read() == b"viewer /v/abc/"
        post = urllib.request.Request(
            f"{url}/state/abc", data=b'{"x": 1}', method="POST"
        )
        assert urllib.request.urlopen(post).read() == b'{"x": 1}'
        # The store is still the store's
        whole = (Path(store) / "zarr.json").read_bytes()
        assert _get(f"{url}/secret/image.zarr/zarr.json").read() == whole
    finally:
        for s in (httpd, upstream):
            s.shutdown()
            s.server_close()


def test_the_viewer_holds_the_state():
    """Neuroglancer's own server, when installed: it serves its viewer and
    keeps the state, so the link is the viewer's URL alone."""
    pytest.importorskip("neuroglancer")
    from patchworks._serve import start_viewer

    state = {"dimensions": {"x": [1e-6, "m"]}, "layers": [], "layout": "xy"}
    netloc, path = start_viewer(state)
    assert netloc.startswith("127.0.0.1:") and path.startswith("/v/")
    page = urllib.request.urlopen(f"http://{netloc}{path}").read()
    assert b"<html" in page.lower()


def test_one_connection_serves_many_chunks(server, store):
    """Keep-alive: through the SSH tunnel a new connection is a new channel
    to the node, so the chunks share one (HTTP/1.0 closed it each time)."""
    import http.client
    from pathlib import Path

    whole = (Path(store) / "zarr.json").read_bytes()
    conn = http.client.HTTPConnection(server.removeprefix("http://"))
    for headers in ({}, {"Range": "bytes=0-3"}, {"Range": "bytes=-2"}):
        conn.request("GET", "/secret/image.zarr/zarr.json", headers=headers)
        r = conn.getresponse()
        body = r.read()
        assert r.version == 11 and not r.will_close
        assert body in (whole, whole[:4], whole[-2:])
    conn.request("OPTIONS", "/secret/image.zarr/zarr.json")
    r = conn.getresponse()
    r.read()
    assert r.status == 204
    conn.request("GET", "/secret/image.zarr/missing")
    r = conn.getresponse()
    r.read()
    assert r.status == 404
    conn.request("GET", "/secret/image.zarr/zarr.json")
    assert conn.getresponse().read() == whole  # still the same connection
    conn.close()


@pytest.mark.parametrize("shard", [False, True])
def test_small_chunks_read_back_as_the_store(tmp_path, shard):
    """Neuroglancer is offered every array in chunks one plane deep and at
    most 256 wide, cut from the real ones: written out as a store of their
    own, they read back as exactly the original arrays."""
    import itertools

    import zarr

    from patchworks._serve import _Rechunked

    rng = np.random.default_rng(0)
    img = rng.integers(0, 4000, (2, 5, 300, 520)).astype("uint16")
    store = to_ome_zarr(
        img,
        tmp_path / "image.zarr",
        axes="czyx",
        n_levels=2,
        chunks=(1, 3, 200, 300),
        shard=shard,
        progress=False,
    )
    lab = rng.integers(0, 50, (5, 300, 520)).astype("int32")
    write_labels(store, lab, name="cells", chunks=(4, 150, 170), progress=False)
    from pathlib import Path

    re = _Rechunked(Path(store), cache_bytes=1 << 20)  # a small cache: evicts
    assert {"0", "1", "labels/cells/0"} <= set(re.arrays)
    out = tmp_path / "view.zarr"
    for rel, doc in re.arrays.items():
        meta = json.loads(re.metadata(rel))
        view = meta["chunk_grid"]["configuration"]["chunk_shape"]
        assert view[:-2] == [1] * (len(view) - 2) and max(view[-2:]) <= 256
        (out / rel).mkdir(parents=True)
        (out / rel / "zarr.json").write_text(json.dumps(meta))
        grid = [range(-(-n // c)) for n, c in zip(doc["shape"], view)]
        for idx in itertools.product(*grid):
            key = out / rel / "c" / "/".join(map(str, idx))
            key.parent.mkdir(parents=True, exist_ok=True)
            key.write_bytes(re.chunk(rel, idx))
        original = zarr.open_array(str(Path(store) / rel), mode="r")[...]
        assert np.array_equal(
            zarr.open_array(str(out / rel), mode="r")[...], original
        )
    with pytest.raises(IndexError):
        re.chunk("labels/cells/0", (99, 0, 0))


def test_the_server_offers_the_small_chunks(store):
    """Through HTTP: an array's metadata names the small chunks, and a chunk
    is there to read; a file that is not an array's is served as it is."""
    from pathlib import Path

    from numcodecs import Zstd

    from patchworks._serve import _Rechunked

    handler = partial(
        type(
            "H",
            (_Handler,),
            {
                "token": "secret",
                "store_name": "image.zarr",
                "rechunked": _Rechunked(Path(store)),
            },
        ),
        directory=str(Path(store).parent),
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}/secret/image.zarr"
    try:
        meta = json.loads(_get(f"{base}/0/zarr.json").read())
        assert meta["chunk_grid"]["configuration"]["chunk_shape"] == [
            1,
            1,
            32,
            32,
        ]
        raw = Zstd().decode(_get(f"{base}/0/c/1/2/0/0").read())
        plane = np.frombuffer(raw, "<u2").reshape(32, 32)
        expected = np.arange(2 * 4 * 32 * 32, dtype="uint16").reshape(
            2, 4, 32, 32
        )
        assert np.array_equal(plane, expected[1, 2])
        group = (Path(store) / "zarr.json").read_bytes()
        assert _get(f"{base}/zarr.json").read() == group
        with pytest.raises(urllib.error.HTTPError) as err:
            _get(f"{base}/0/c/9/0/0/0")
        assert err.value.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
