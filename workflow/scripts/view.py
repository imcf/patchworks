"""Open an OME-ZARR store in napari, auto-loading every label group.

Usage:
    pixi run -e viewer napari /path/to/image.zarr
    pixi run -e viewer napari /path/to/image.zarr --channel 1

Needs the `viewer` pixi environment (`pixi install -e viewer`) -- napari's
Qt/GUI dependencies live there, not in the default headless workflow env --
and a display: an X11-forwarded SSH session (`ssh -X`) or a remote-desktop/
VNC session. It cannot open over a plain headless SSH session.
"""

from __future__ import annotations

import argparse

from patchworks.plugins.napari import view_in_napari


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image", help="OME-ZARR store to open, e.g. work_dir/image.zarr"
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
    args = parser.parse_args()

    # Leaving the labels argument at its default auto-loads every label
    # group under <image>/labels/<name>/ as its own Labels layer -- exactly
    # what's wanted here, so there is nothing further to wire up.
    view_in_napari(
        args.image, channel=args.channel, glasbey=not args.no_glasbey
    )


if __name__ == "__main__":
    main()
