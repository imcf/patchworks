"""Open an OME-ZARR store in napari, auto-loading every label group.

Usage:
    pixi run -e viewer napari /path/to/image.zarr
    pixi run -e viewer napari /path/to/image.zarr.zip     # a `pixi run zip` bundle
    pixi run -e viewer napari /path/to/image.zarr --channel 1

A `.zip` bundle is read in place, without unpacking: same layers, same
calibration, same auto-loaded label groups as the directory it was packed
from.

Needs the `viewer` pixi environment (`pixi install -e viewer`) -- napari's
Qt/GUI dependencies live there, not in the default headless workflow env --
and a display: an X11-forwarded SSH session (`ssh -X`) or a remote-desktop/
VNC session. It cannot open over a plain headless SSH session.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

# Before napari imports numba: its cache path on a network share can
# pass Windows' 260-character limit.
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "patchworks-numba-cache"),
)

# --async: load views in the background (napari's own setting, for this
# session only). The window stays responsive, but a coarse level shows
# until the fine one is read, so zooming in felt slower: off by default.
if "--async" in sys.argv:
    os.environ.setdefault("NAPARI_ASYNC", "1")

from patchworks.plugins.napari import view_in_napari  # noqa: E402


def _warn_if_slow_to_read(path: str) -> None:
    """A bundle on a network drive is read chunk by chunk over the network,
    and one chunk at a time: copy it to this computer and unzip it."""
    from bench_view import _drive

    from patchworks._io import split_zip_path

    split = split_zip_path(path)
    if split and "network" in _drive(split[0]):
        print(
            "[view] this .zip is on a network drive: every zoom and pan "
            "reads it over the network, one chunk at a time. For a fast "
            "viewer, copy it to this computer and unzip it (or open the "
            "unzipped image.zarr directly).",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        help=(
            "OME-ZARR store to open, e.g. work_dir/image.zarr -- or a "
            "work_dir/image.zarr.zip bundle, read without unpacking"
        ),
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=None,
        help="show only this channel (default: every channel)",
    )
    parser.add_argument(
        "--no-glasbey",
        action="store_true",
        help="use napari's default label colours instead of glasbey",
    )
    parser.add_argument(
        "--async",
        dest="async_",
        action="store_true",
        help="load views in the background: a coarse level shows first",
    )
    parser.add_argument(
        "--split-channels",
        action="store_true",
        help="each channel its own layer, shown together (reads them all)",
    )
    args = parser.parse_args()
    _warn_if_slow_to_read(args.image)

    # Leaving the labels argument at its default auto-loads every label
    # group under <image>/labels/<name>/ as its own Labels layer -- exactly
    # what's wanted here, so there is nothing further to wire up.
    view_in_napari(
        args.image,
        channel=args.channel,
        glasbey=not args.no_glasbey,
        split_channels=args.split_channels,
    )


if __name__ == "__main__":
    main()
