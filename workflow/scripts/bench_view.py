"""Time what napari does when you zoom, pan or change channel.

Usage:
    cd workflow/viewer
    pixi run bench /path/to/image.zarr          # or image.zarr.zip
    pixi run bench U:\\...\\image.zarr.zip --render

Reads one typical screen (a plane through the middle, 2048 x 2048 pixels)
at full resolution and two levels down, from the image and every label
image, and says where the time goes:

- how the arrays are chunked, and how many planes a chunk makes zarr
  decompress to show one (a chunk 126 planes deep: 126 times the data);
- the time for one small read (the latency of the store: network, zip);
- each view read straight with zarr, read again (the operating system's
  cache), and the way the viewer reads it (dask);
- with --render, napari drawing a label plane (needs napari).

Nothing is written. Send the whole output.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

# As view.py: napari compiles with numba, whose cache next to a package on a
# network share can pass Windows' 260-character limit.
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "patchworks-numba-cache"),
)

import numpy as np

from patchworks._io import open_group_any, split_zip_path
from patchworks.plugins.ome_zarr import read_ngff_attr

SCREEN = 2048  # pixels across one view


def _drive(path: str) -> str:
    """Where the store lives: local disk, network, or unknown."""
    if os.name == "nt":
        import ctypes

        root = os.path.splitdrive(os.path.abspath(path))[0] + "\\"
        kind = ctypes.windll.kernel32.GetDriveTypeW(root)
        return {2: "removable", 3: "local disk", 4: "network drive"}.get(
            kind, f"drive type {kind}"
        )
    try:
        import psutil

        best = max(
            (
                p
                for p in psutil.disk_partitions(all=True)
                if os.path.abspath(path).startswith(p.mountpoint)
            ),
            key=lambda p: len(p.mountpoint),
        )
        net = best.fstype in ("nfs", "nfs4", "cifs", "smbfs", "fuse.sshfs")
        return f"{'network' if net else 'local'} ({best.fstype})"
    except Exception:  # noqa: BLE001 - only informative
        return "unknown"


def _levels(group_path: str) -> tuple[list, list[str]]:
    """The pyramid's zarr arrays, finest first, and the axis names."""
    root = open_group_any(group_path)
    ms = read_ngff_attr(root.attrs, "multiscales")[0]
    axes = [a["name"] if isinstance(a, dict) else a for a in ms["axes"]]
    return [root[d["path"]] for d in ms["datasets"]], axes


def _describe(name: str, arrays: list, axes: list[str]) -> None:
    print(f"\n{name}: axes {''.join(axes)}, {len(arrays)} level(s)")
    for i, arr in enumerate(arrays):
        shards = getattr(arr, "shards", None)
        z = axes.index("z") if "z" in axes else None
        depth = arr.chunks[z] if z is not None else 1
        print(
            f"  level {i}: shape {tuple(arr.shape)} {arr.dtype}, chunks "
            f"{tuple(arr.chunks)}"
            + (f" in shards {tuple(shards)}" if shards else "")
            + (
                f" -> one plane decompresses {depth} plane(s)"
                if z is not None
                else ""
            )
        )


def _view(arr, axes: list[str], level: int) -> tuple:
    """One screen: the middle plane (channel 0), SCREEN pixels across."""
    sl = []
    for ax, n in zip(axes, arr.shape):
        if ax in ("c", "t"):
            sl.append(0)
        elif ax == "z":
            sl.append(n // 2)
        else:
            w = min(SCREEN, n)
            lo = (n - w) // 2
            sl.append(slice(lo, lo + w))
    return tuple(sl)


def _chunks_touched(arr, sl: tuple) -> int:
    n = 1
    for s, c, size in zip(sl, arr.chunks, arr.shape):
        if isinstance(s, int):
            continue
        n *= -(-s.stop // c) - s.start // c
    return n


def _timed(fn) -> tuple[float, object]:
    t = time.perf_counter()
    out = fn()
    return time.perf_counter() - t, out


def _bench(name: str, arrays: list, axes: list[str], open_dask) -> None:
    opened, dask_levels = _timed(open_dask)
    print(f"  {name}: opened as the viewer does in {opened:.2f} s")
    for level in sorted({0, min(2, len(arrays) - 1)}):
        arr = arrays[level]
        sl = _view(arr, axes, level)
        # Another plane for the viewer's read: not one the OS has cached.
        other = tuple(
            (i + 1) % n if isinstance(i, int) and ax == "z" else i
            for i, ax, n in zip(sl, axes, arr.shape)
        )
        touched = _chunks_touched(arr, sl)
        chunk_mb = np.prod(arr.chunks) * arr.dtype.itemsize / 1e6
        cold, data = _timed(lambda: np.asarray(arr[sl]))
        warm, _ = _timed(lambda: np.asarray(arr[sl]))
        lazy = dask_levels[level]
        if lazy.ndim < len(axes):  # one channel's view: no channel index
            other = tuple(
                k for k, ax in zip(other, axes) if ax not in ("c", "t")
            )
        via_dask, _ = _timed(lambda: np.asarray(lazy[other]))
        dask_warm, _ = _timed(lambda: np.asarray(lazy[other]))
        print(
            f"  {name} level {level}: view {data.shape}, {touched} chunk(s) "
            f"= {touched * chunk_mb:,.0f} MB decompressed for "
            f"{data.nbytes / 1e6:,.0f} MB shown | zarr {cold:.2f} s, again "
            f"{warm:.2f} s | viewer (next plane) {via_dask:.2f} s, again "
            f"{dask_warm:.2f} s"
        )


def _latency(arr) -> float:
    """The best of 5 reads of one voxel in different chunks."""
    best = float("inf")
    for k in range(5):
        idx = tuple(
            min(n - 1, (k * 7 + 3) * c % max(n, 1))
            for n, c in zip(arr.shape, arr.chunks)
        )
        t, _ = _timed(lambda: arr[idx])
        best = min(best, t)
    return best


def _render(arrays: list, axes: list[str]) -> None:
    try:
        import napari
    except ImportError:
        print("\n--render: napari is not installed here")
        return
    arr = arrays[0]
    plane = np.asarray(arr[_view(arr, axes, 0)]).astype("int32")
    print(
        f"\nnapari drawing one {plane.shape} label plane "
        f"(ids up to {int(plane.max()):,})"
    )
    viewer = napari.Viewer(show=False)
    t, layer = _timed(lambda: viewer.add_labels(plane))
    print(f"  add the layer: {t:.2f} s")
    for k in range(3):
        layer.data = np.roll(plane, k + 1, axis=0)
        t, _ = _timed(layer.refresh)
        print(f"  redraw {k + 1}: {t:.2f} s")
    viewer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", help="image.zarr, or an image.zarr.zip bundle")
    parser.add_argument(
        "--render", action="store_true", help="also time napari's drawing"
    )
    args = parser.parse_args()
    store = str(args.store).rstrip("/\\")

    zipped = split_zip_path(store) is not None
    print(f"store: {store}")
    print(
        f"  {'a .zip bundle' if zipped else 'a directory'} on "
        f"{_drive(split_zip_path(store)[0] if zipped else store)}"
    )

    image, axes = _levels(store)
    _describe("image", image, axes)
    labels = []
    try:
        names = sorted(
            open_group_any(f"{store}/labels").attrs.get("labels")
            or (open_group_any(f"{store}/labels").attrs.get("ome") or {}).get(
                "labels", []
            )
        )
    except Exception:  # noqa: BLE001 - no labels
        names = []
    for name in names:
        arrays, lab_axes = _levels(f"{store}/labels/{name}")
        _describe(f"labels/{name}", arrays, lab_axes)
        labels.append((name, arrays, lab_axes))

    print(
        f"\none voxel read (a whole chunk fetched and decompressed): "
        f"{1000 * _latency(image[0]):.0f} ms"
    )

    from patchworks.plugins.napari import _label_levels, image_layers

    print(f"\none view ({SCREEN} x {SCREEN} pixels, middle plane):")
    c = 0 if "c" in axes else None
    _bench(
        "image",
        image,
        axes,
        lambda: image_layers(store, c)[0][0],
    )
    for name, arrays, lab_axes in labels:
        group = f"{store}/labels/{name}"
        _bench(
            f"labels/{name}",
            arrays,
            lab_axes,
            lambda g=group: _label_levels(g),
        )

    if args.render and labels:
        # The label image with the most objects in view: drawing an empty
        # plane measures nothing.
        def filled(entry):
            arr = entry[1][0]
            return int(np.count_nonzero(arr[_view(arr, entry[2], 0)]))

        busiest = max(labels, key=filled)
        _render(busiest[1], busiest[2])
    print(
        "\nTimes over ~0.5 s per view are felt. 'again' much faster than the "
        "first read: the store's latency or the network is the cost; both "
        "slow: decompressing (chunks much deeper than one plane)."
    )


if __name__ == "__main__":
    sys.exit(main())
