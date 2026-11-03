"""Serve an OME-Zarr store over HTTP for Neuroglancer, through an SSH tunnel.

Neuroglancer runs in the browser and reads the store chunk by chunk over
HTTP: the server must allow requests from its page (CORS) and serve byte
ranges (a chunk inside a shard is a range of the shard's file), which
Python's own `http.server` does not. Every URL starts with a random token,
so other users of the cluster cannot browse the store by guessing a port.
"""

from __future__ import annotations

import getpass
import json
import os
import re
import secrets
import socket
import urllib.parse
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Union

NEUROGLANCER = "https://neuroglancer-demo.appspot.com/"

#: The line `serve` prints for a program (the web launcher) to read:
#: `PATCHWORKS_SERVE {"host", "port", "base", "state"}`.
SERVE_TAG = "PATCHWORKS_SERVE"

# Each channel's colour in Neuroglancer when several are shown.
_COLORS = ("#00ff00", "#ff00ff", "#00ffff", "#ffff00", "#ffffff", "#ff0000")


#: The chunks Neuroglancer is given, per spatial axis: one plane deep and
#: this wide in y and x (less where the store's own are smaller).
VIEW_CHUNK = 256


class _Rechunked:
    """A store's arrays as Neuroglancer is shown them: in small chunks.

    A store written for napari and the workflow has chunks one plane deep
    but 1024 wide (the image), or 14 planes deep and 839 wide (labels,
    40 MB once unpacked). Neuroglancer fetches a whole chunk for any part
    of it, through the SSH tunnel: the labels' fourteen planes for the one
    shown, and for a side view 126 whole planes. Each array is offered
    instead in chunks one plane deep and :data:`VIEW_CHUNK` wide, cut on
    request from the real ones, which are kept decoded in memory a while.

    Parameters
    ----------
    store : Path
        The OME-Zarr image (zarr v3; a v2 store is served as it is).
    cache_bytes : int, optional
        Memory for decoded real chunks; default a third of what the
        process may use, at most 2 GB.
    """

    def __init__(self, store: Path, cache_bytes: int | None = None) -> None:
        import threading
        from collections import OrderedDict

        self.store = store
        self.arrays: dict[str, dict[str, Any]] = {}
        self._open: dict[str, Any] = {}
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self._used = 0
        if cache_bytes is None:
            from ._chunks import _get_available_memory

            cache_bytes = min(2 << 30, _get_available_memory() // 3)
        self.cache_bytes = int(cache_bytes)
        for meta in store.rglob("zarr.json"):
            try:
                doc = json.loads(meta.read_text())
            except (OSError, ValueError):
                continue
            if doc.get("node_type") != "array" or doc.get("zarr_format") != 3:
                continue
            rel = meta.parent.relative_to(store).as_posix()
            self.arrays[rel] = doc

    def view_chunks(self, rel: str) -> tuple[int, ...]:
        """The chunk shape Neuroglancer is given for array *rel*.

        Parameters
        ----------
        rel : str
            The array's path in the store, e.g. `"labels/cells/0"`.

        Returns
        -------
        tuple of int
            1 on every axis but the last two, which are at most
            :data:`VIEW_CHUNK`.
        """
        doc = self.arrays[rel]
        real = _real_chunks(doc)
        n = len(doc["shape"])
        return tuple(
            min(VIEW_CHUNK, c) if i >= n - 2 else 1 for i, c in enumerate(real)
        )

    def metadata(self, rel: str) -> bytes:
        """Array *rel*'s `zarr.json`, with the small chunks and plain zstd.

        Parameters
        ----------
        rel : str
            The array's path in the store.

        Returns
        -------
        bytes
        """
        doc = dict(self.arrays[rel])
        doc["chunk_grid"] = {
            "name": "regular",
            "configuration": {"chunk_shape": list(self.view_chunks(rel))},
        }
        doc["chunk_key_encoding"] = {
            "name": "default",
            "configuration": {"separator": "/"},
        }
        doc["codecs"] = [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "zstd", "configuration": {"level": 1, "checksum": False}},
        ]
        doc.pop("storage_transformers", None)
        return json.dumps(doc).encode()

    def _array(self, rel: str):
        """The zarr array *rel*, opened once.

        Parameters
        ----------
        rel : str
            The array's path in the store.

        Returns
        -------
        zarr.Array
        """
        if rel not in self._open:
            import zarr

            self._open[rel] = zarr.open_array(
                str(self.store / rel), mode="r", zarr_format=3
            )
        return self._open[rel]

    def _real_chunk(self, rel: str, index: tuple[int, ...]):
        """Real chunk *index* of array *rel*, decoded (cached).

        Parameters
        ----------
        rel : str
            The array's path in the store.
        index : tuple of int
            The real chunk's index on each axis.

        Returns
        -------
        np.ndarray
        """
        import numpy as np

        key = (rel, index)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        arr = self._array(rel)
        real = _real_chunks(self.arrays[rel])
        sl = tuple(
            slice(i * c, min((i + 1) * c, n))
            for i, c, n in zip(index, real, arr.shape)
        )
        block = np.asarray(arr[sl])
        with self._lock:
            self._cache[key] = block
            self._used += block.nbytes
            while self._used > self.cache_bytes and len(self._cache) > 1:
                _, old = self._cache.popitem(last=False)
                self._used -= old.nbytes
        return block

    def chunk(self, rel: str, index: tuple[int, ...]) -> bytes:
        """Small chunk *index* of array *rel*, encoded as its metadata says.

        Parameters
        ----------
        rel : str
            The array's path in the store.
        index : tuple of int
            The small chunk's index on each axis.

        Returns
        -------
        bytes
            zstd of the chunk's little-endian bytes, padded to the full
            chunk shape at the array's edge (as zarr stores it).

        Raises
        ------
        IndexError
            For an index outside the array.
        """
        import numpy as np
        from numcodecs import Zstd

        doc = self.arrays[rel]
        shape = doc["shape"]
        view = self.view_chunks(rel)
        real = _real_chunks(doc)
        if len(index) != len(shape) or any(
            i < 0 or i * v >= n for i, v, n in zip(index, view, shape)
        ):
            raise IndexError(index)
        lo = [i * v for i, v in zip(index, view)]
        hi = [min(a + v, n) for a, v, n in zip(lo, view, shape)]
        dtype = np.dtype(doc["data_type"]).newbyteorder("<")
        fill = doc.get("fill_value") or 0
        out = np.full(view, fill, dtype=dtype)
        # The real chunks this one overlaps (one, unless they do not align).
        first = [a // c for a, c in zip(lo, real)]
        last = [(b - 1) // c for b, c in zip(hi, real)]
        for ridx in np.ndindex(*[b - a + 1 for a, b in zip(first, last)]):
            r = tuple(a + k for a, k in zip(first, ridx))
            block = self._real_chunk(rel, r)
            start = [i * c for i, c in zip(r, real)]
            src, dst = [], []
            for a, b, s0, n in zip(lo, hi, start, block.shape):
                a2, b2 = max(a, s0), min(b, s0 + n)
                src.append(slice(a2 - s0, b2 - s0))
                dst.append(slice(a2 - a, b2 - a))
            out[tuple(dst)] = block[tuple(src)]
        return Zstd(level=1).encode(np.ascontiguousarray(out).tobytes())


def _real_chunks(doc: dict) -> list[int]:
    """The chunk shape a zarr v3 array is read in: its shard's inner
    chunks when sharded, else its chunk grid.

    Parameters
    ----------
    doc : dict
        The array's `zarr.json`.

    Returns
    -------
    list of int
    """
    for codec in doc.get("codecs") or []:
        if codec.get("name") == "sharding_indexed":
            return list(codec["configuration"]["chunk_shape"])
    return list(doc["chunk_grid"]["configuration"]["chunk_shape"])


class _Handler(SimpleHTTPRequestHandler):
    """Read-only files under *directory*, below a secret path prefix, with
    CORS and single byte ranges."""

    token = ""
    store_name = ""
    # Keep-alive: through the SSH tunnel every new connection is a new
    # channel to the node, and the browser opens only six at a time.
    # HTTP/1.0 closed one after every chunk.
    protocol_version = "HTTP/1.1"
    #: `host:port` of the Neuroglancer server running beside this one, to
    #: which every path outside `/<token>/` is passed; "" for none.
    upstream = ""
    #: The store's arrays in small chunks; None serves the files as they are.
    rechunked: _Rechunked | None = None

    def end_headers(self) -> None:
        """Add the CORS and range headers to every response."""
        self.send_header("Access-Control-Allow-Origin", "*")
        # A public page (Neuroglancer's) reading from localhost: Chrome asks
        # the server's leave first (Private Network Access), and since
        # Chrome 142 the user's too ("Local network access").
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Access-Control-Allow-Headers", "Range")
        self.send_header(
            "Access-Control-Expose-Headers",
            "Content-Range, Content-Length, Accept-Ranges",
        )
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802 - the CORS preflight
        """Answer the browser's CORS preflight."""
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def translate_path(self, path: str) -> str:
        """The file a URL path names, or a missing one outside the store.

        Parameters
        ----------
        path : str
            The URL path, which must start with `/<token>/<store_name>/`.

        Returns
        -------
        str
            The file's path on disk; a path that does not exist (a 404) for
            anything outside the token's prefix or the store.
        """
        # Only the store, only below the token: not the files beside it.
        prefix = f"/{self.token}/{self.store_name}/"
        real = super().translate_path("/" + path[len(f"/{self.token}/") :])
        inside = os.path.join(self.directory, self.store_name) + os.sep
        if not path.startswith(prefix) or not real.startswith(inside):
            # Somewhere that is not a file: a 404, whatever it asks for.
            return os.path.join(self.directory, f".{self.token}", "missing")
        return real

    def _to_viewer(self) -> bool:
        """Whether this request is the Neuroglancer server's.

        Returns
        -------
        bool
            True for any path outside `/<token>/`, when one runs.
        """
        return bool(self.upstream) and not self.path.startswith(
            f"/{self.token}/"
        )

    def _proxy(self) -> None:
        """Pass the request to the Neuroglancer server, and stream its answer
        back (its state updates are one long-lived response)."""
        import http.client

        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        conn = http.client.HTTPConnection(self.upstream, timeout=3600)
        headers = {
            k: v
            for k, v in self.headers.items()
            if k.lower() not in ("host", "connection", "accept-encoding")
        }
        try:
            conn.request(self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()
        except OSError:
            self.send_error(HTTPStatus.BAD_GATEWAY)
            return
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() not in (
                "transfer-encoding",
                "connection",
                "content-length",
                "access-control-allow-origin",
            ):
                self.send_header(k, v)
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        try:
            while True:
                block = resp.read1(65536)
                if not block:
                    break
                self.wfile.write(block)
                self.wfile.flush()
        except OSError:
            pass  # the browser went away
        finally:
            conn.close()

    def do_GET(self) -> None:  # noqa: N802
        """A file of the store, a small chunk, or the Neuroglancer server's
        answer."""
        if self._to_viewer():
            return self._proxy()
        if self._rechunked():
            return None
        return super().do_GET()

    def _rechunked(self) -> bool:
        """Answer from :attr:`rechunked` when the path is an array's
        metadata or one of its chunks.

        Returns
        -------
        bool
            Whether it answered.
        """
        if self.rechunked is None:
            return False
        prefix = f"/{self.token}/{self.store_name}/"
        path = urllib.parse.unquote(self.path.split("?", 1)[0])
        if not path.startswith(prefix):
            return False
        rel = path[len(prefix) :]
        arrays = self.rechunked.arrays
        if rel.endswith("/zarr.json") and rel[: -len("/zarr.json")] in arrays:
            body = self.rechunked.metadata(rel[: -len("/zarr.json")])
            ctype = "application/json"
        elif "/c/" in rel and rel.split("/c/", 1)[0] in arrays:
            name, key = rel.split("/c/", 1)
            try:
                index = tuple(int(k) for k in key.split("/"))
                body = self.rechunked.chunk(name, index)
            except (ValueError, IndexError):
                self.send_error(HTTPStatus.NOT_FOUND)
                return True
            ctype = "application/octet-stream"
        else:
            return False
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return True

    def do_POST(self) -> None:  # noqa: N802
        """Neuroglancer's state and actions, for its server."""
        if self._to_viewer():
            return self._proxy()
        self.send_error(HTTPStatus.NOT_FOUND)

    def list_directory(self, path):  # no listings: only what is asked for
        """Refuse directory listings: only what is asked for is served.

        Parameters
        ----------
        path : str
            The directory asked for (unused).
        """
        self.send_error(HTTPStatus.NOT_FOUND)
        return None

    def send_head(self):
        """Send the headers, honouring a single byte range.

        Returns
        -------
        file or None
            The open file, positioned at the range's start, for
            :meth:`copyfile`; None after an error.
        """
        path = self.translate_path(self.path)
        match = re.fullmatch(
            r"bytes=(\d*)-(\d*)", self.headers.get("Range", "").strip()
        )
        if os.path.isdir(path) or not match or not any(match.groups()):
            return super().send_head()
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return None
        size = os.fstat(f.fileno()).st_size
        first, last = match.groups()
        if first:
            start, end = int(first), int(last) if last else size - 1
        else:  # the last N bytes (a shard's index)
            start, end = max(0, size - int(last)), size - 1
        end = min(end, size - 1)
        if start > end:
            f.close()
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        f.seek(start)
        self._remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile) -> None:
        """Copy the response body: only the range's bytes after a range request.

        Parameters
        ----------
        source : file
            The file :meth:`send_head` opened.
        outputfile : file
            The connection.
        """
        remaining = getattr(self, "_remaining", None)
        if remaining is None:
            return super().copyfile(source, outputfile)
        while remaining > 0:
            block = source.read(min(remaining, 1 << 20))
            if not block:
                break
            outputfile.write(block)
            remaining -= len(block)
        self._remaining = None

    def log_message(self, format: str, *args: Any) -> None:
        """Log nothing: a request per chunk is far too many to print.

        Parameters
        ----------
        format : str
            The message's format (unused).
        *args
            Its arguments (unused).
        """
        pass  # a request per chunk: far too many to print


def _zarr_scheme(store: Path) -> str:
    """Neuroglancer's data source scheme for *store*.

    Parameters
    ----------
    store : Path
        The zarr store.

    Returns
    -------
    str
        `"zarr3"` for a zarr v3 store, else `"zarr"`.
    """
    return "zarr3" if (store / "zarr.json").exists() else "zarr"


def neuroglancer_state(store: Union[str, Path], base_url: str) -> dict:
    """A Neuroglancer state showing *store*.

    Each channel is an image layer (named, coloured, the first shown) and
    each label image a segmentation layer (hidden).

    Parameters
    ----------
    store : str or Path
        The OME-Zarr image on disk.
    base_url : str
        The URL it is served at, as Neuroglancer will reach it (through the
        tunnel: `http://localhost:<port>/<token>/<name>`).

    Returns
    -------
    dict
        The state: `dimensions` (x, y, z with the voxel size),
        `displayDimensions`, `layers` and `layout`.
    """
    from .plugins.napari import _inner_label_names, _store_axes
    from .plugins.ome_zarr import _multiscale_meta, channel_names

    store = Path(store)
    scheme = _zarr_scheme(store)
    source = f"{scheme}://{base_url}"
    meta = _multiscale_meta(str(store)) or {}
    axes = _store_axes(str(store), len(meta.get("axes") or []))
    layers: list[dict[str, Any]] = []
    if "c" in axes:
        import zarr

        group = zarr.open_group(str(store), mode="r")
        level0 = group[meta["datasets"][0]["path"]]
        n = level0.shape[axes.index("c")]
        names = channel_names(str(store)) or [f"channel {i}" for i in range(n)]
        coarse = group[meta["datasets"][-1]["path"]]
        for i, name in enumerate(names):
            color = _COLORS[i % len(_COLORS)]
            lo, hi = _contrast(coarse, axes, i)
            layers.append(
                {
                    "type": "image",
                    "source": source,
                    "name": name,
                    "localDimensions": {"c'": [1, ""]},
                    "localPosition": [i],
                    "blend": "additive",
                    "visible": i == 0,
                    "shader": (
                        f'#uicontrol vec3 color color(default="{color}")\n'
                        "#uicontrol invlerp normalized\n"
                        "void main() { emitRGB(color * normalized()); }\n"
                    ),
                    # The data's own range: the dtype's (0-65535) is black.
                    "shaderControls": {"normalized": {"range": [lo, hi]}},
                }
            )
    else:
        layers.append({"type": "image", "source": source, "name": "image"})
    for name in _inner_label_names(str(store)):
        layers.append(
            {
                "type": "segmentation",
                "source": f"{source}/labels/{name}",
                "name": name,
                "visible": False,
            }
        )
    # x, y first: "xy" is the first two displayed axes, and the store's
    # order (z, y, x) made it a z-y section.
    # The axes declared up front, in that order and with the voxel size:
    # displayDimensions alone is dropped while the store is still loading.
    from .plugins.ome_zarr import read_pixel_size

    shown = [a for a in "xyz" if a in axes]
    size = read_pixel_size(str(store))
    dimensions = {
        a: [
            float(size.get(a, 1.0)) * (1e-6 if size else 1.0),
            "m" if size else "",
        ]
        for a in shown
    }
    return {
        "dimensions": dimensions,
        "displayDimensions": shown,
        "layers": layers,
        "layout": "xy",
        # Neuroglancer's defaults (1 GB of GPU memory, 32 downloads at a
        # time) leave a whole-slide image at a coarse level, and every
        # request crosses the SSH tunnel: more room, more in flight.
        "gpuMemoryLimit": 2_000_000_000,
        "systemMemoryLimit": 4_000_000_000,
        "concurrentDownloads": 64,
    }


def _contrast(coarse, axes: str, channel: int) -> tuple[float, float]:
    """A channel's display range: its 1st-99.9th percentile.

    From the middle plane at the coarsest level (one small read); zeros are
    left out.

    Parameters
    ----------
    coarse : zarr.Array
        The coarsest pyramid level.
    axes : str
        Its axis names, e.g. `"czyx"`.
    channel : int
        The channel.

    Returns
    -------
    tuple of float
        `(low, high)`; `(0, 1)` for an empty plane.
    """
    import numpy as np

    key = tuple(
        channel if a == "c" else n // 2 if a in "tz" else slice(None)
        for a, n in zip(axes, coarse.shape)
    )
    values = np.asarray(coarse[key]).ravel()
    values = values[values != 0]
    if not values.size:
        return 0.0, 1.0
    lo, hi = (float(v) for v in np.percentile(values, [1, 99.9]))
    return lo, max(hi, lo + 1)


def start_viewer(state: dict) -> tuple[str, str] | None:
    """Start Neuroglancer's own server, showing *state*, if installed.

    Served through :func:`serve`'s port (every path outside its token),
    the page and its data come from one address: a public page reading
    `localhost` is what browsers now block (Chrome's "Local network
    access", Firefox's too).

    Parameters
    ----------
    state : dict
        The Neuroglancer state, from :func:`neuroglancer_state`.

    Returns
    -------
    tuple of str or None
        `(host:port of its server, the viewer's path)`, e.g.
        `("127.0.0.1:40123", "/v/abc/")`; None without the `neuroglancer`
        package.
    """
    try:
        import neuroglancer
    except ImportError:
        return None
    neuroglancer.set_server_bind_address("127.0.0.1", 0)
    viewer = neuroglancer.Viewer()
    viewer.set_state(neuroglancer.ViewerState(state))
    url = urllib.parse.urlsplit(viewer.get_viewer_url())
    _VIEWERS.append(viewer)  # kept alive with the server
    return url.netloc, url.path


_VIEWERS: list = []


def neuroglancer_link(state: dict, viewer: str = NEUROGLANCER) -> str:
    """The link opening Neuroglancer with *state*.

    Parameters
    ----------
    state : dict
        The state, from :func:`neuroglancer_state`.
    viewer : str, optional
        The Neuroglancer to open: the one :func:`serve` hosts
        (`http://localhost:<port>/<token>/neuroglancer/`), or the public
        one (default).

    Returns
    -------
    str
        *viewer*'s URL with the state in its fragment.
    """
    return (
        viewer
        + "#!"
        + urllib.parse.quote(json.dumps(state, separators=(",", ":")))
    )


def _free_port(host: str) -> int:
    """A port free on *host*, as the system picks one.

    Parameters
    ----------
    host : str
        The address to bind.

    Returns
    -------
    int
    """
    with socket.socket() as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def serve(
    store: Union[str, Path],
    *,
    port: int = 0,
    host: str | None = None,
    link_port: int | None = None,
) -> None:
    """Serve *store* until interrupted, printing the SSH tunnel to open and
    the Neuroglancer link.

    Parameters
    ----------
    store : str or Path
        The OME-Zarr image directory (`image.zarr`); not a `.zip` bundle,
        whose chunks are not files a browser can ask for.
    port : int, optional
        The port to listen on; 0 (default) picks a free one.
    host : str, optional
        The address to listen on. Default: 127.0.0.1 on a login node; in a
        SLURM job, every address, for the tunnel to reach it through the
        login node.
    link_port : int, optional
        The port on the browser's computer the tunnel comes out at, when
        not *port* (the launcher picks a free one there first): the link
        and the store's addresses in it use it.
    """
    store = Path(store).resolve()
    if str(store).lower().endswith(".zip"):
        raise SystemExit(
            f"{store} is a .zip bundle: serve the unzipped image.zarr (the "
            "bundle's chunks are not files a browser can ask for)"
        )
    if not (store / "zarr.json").exists() and not (store / ".zattrs").exists():
        raise SystemExit(f"{store} is not a zarr store")
    in_job = bool(os.environ.get("SLURM_JOB_ID"))
    host = host or ("0.0.0.0" if in_job else "127.0.0.1")
    port = port or _free_port(host)
    token = secrets.token_urlsafe(12)
    handler = partial(
        type(
            "Handler",
            (_Handler,),
            {
                "token": token,
                "store_name": store.name,
                "rechunked": _Rechunked(store),
            },
        ),
        directory=str(store.parent),
    )
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True

    # The addresses the browser uses: the local end of the tunnel.
    local = link_port or port
    base = f"http://localhost:{local}/{token}/{store.name}"
    user = getpass.getuser()
    here = socket.gethostname()
    if in_job:
        login = os.environ.get("SLURM_SUBMIT_HOST", "<login node>")
        tunnel = f"ssh -N -L {local}:{here}:{port} {user}@{login}"
    else:
        tunnel = f"ssh -N -L {local}:localhost:{port} {user}@{here}"
    state = neuroglancer_state(store, base)
    # Neuroglancer through this server when installed: the public page
    # cannot read localhost in today's browsers without the user's leave.
    own = start_viewer(state)
    viewer = NEUROGLANCER
    if own is not None:
        handler.func.upstream = own[0]  # type: ignore[attr-defined]
        viewer = f"http://localhost:{local}{own[1]}"
    print(
        SERVE_TAG,
        json.dumps(
            {
                "host": here,
                "port": port,
                "link_port": local,
                "base": base,
                "state": state,
                "viewer": viewer,
            }
        ),
        flush=True,
    )
    print(f"serving {store} on {here}:{port} (Ctrl+C to stop)")
    print("\n1. On your computer, open the tunnel (and keep it open):")
    print(f"   {tunnel}")
    print("\n2. Open this link in Chrome or Firefox:")
    print(f"   {viewer if own is not None else neuroglancer_link(state)}")
    print(
        "\n   Each channel is a layer (the first shown); the label images "
        "are segmentation layers, switched off: tick them in the layer list.",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
