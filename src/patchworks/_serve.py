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

    def end_headers(self) -> None:
        """Add the CORS and range headers to every response."""
        self.send_header("Access-Control-Allow-Origin", "*")
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


def neuroglancer_link(state: dict) -> str:
    """The link opening Neuroglancer with *state*.

    Parameters
    ----------
    state : dict
        The state, from :func:`neuroglancer_state`.

    Returns
    -------
    str
        The public Neuroglancer's URL with the state in its fragment.
    """
    return (
        NEUROGLANCER
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
            "Handler", (_Handler,), {"token": token, "store_name": store.name}
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
    print(
        SERVE_TAG,
        json.dumps({"host": here, "port": port, "base": base, "state": state}),
        flush=True,
    )
    print(f"serving {store} on {here}:{port} (Ctrl+C to stop)")
    print("\n1. On your computer, open the tunnel (and keep it open):")
    print(f"   {tunnel}")
    print("\n2. Open this link in Chrome or Firefox:")
    print(f"   {neuroglancer_link(state)}")
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
