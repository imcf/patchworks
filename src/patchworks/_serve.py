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


class _Handler(SimpleHTTPRequestHandler):
    """Read-only files under *directory*, below a secret path prefix, with
    CORS and single byte ranges."""

    token = ""
    store_name = ""
    #: `host:port` of the Neuroglancer server running beside this one, to
    #: which every path outside `/<token>/` is passed; "" for none.
    upstream = ""

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
        """A file of the store, or the Neuroglancer server's answer."""
        if self._to_viewer():
            return self._proxy()
        return super().do_GET()

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
            {"token": token, "store_name": store.name},
        ),
        directory=str(store.parent),
    )
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True

    base = f"http://localhost:{port}/{token}/{store.name}"
    user = getpass.getuser()
    here = socket.gethostname()
    if in_job:
        login = os.environ.get("SLURM_SUBMIT_HOST", "<login node>")
        tunnel = f"ssh -N -L {port}:{here}:{port} {user}@{login}"
    else:
        tunnel = f"ssh -N -L {port}:localhost:{port} {user}@{here}"
    state = neuroglancer_state(store, base)
    # Neuroglancer through this server when installed: the public page
    # cannot read localhost in today's browsers without the user's leave.
    own = start_viewer(state)
    viewer = NEUROGLANCER
    if own is not None:
        handler.func.upstream = own[0]  # type: ignore[attr-defined]
        viewer = f"http://localhost:{port}{own[1]}"
    print(
        SERVE_TAG,
        json.dumps(
            {
                "host": here,
                "port": port,
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
