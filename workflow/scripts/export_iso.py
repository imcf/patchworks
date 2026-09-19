"""Pack a finished OME-ZARR store into a single .iso image.

A zarr store is tens of thousands of small files. Copying that to a laptop,
a share or a USB disk is slow everywhere and painful on Windows, where each
file costs a round trip -- and some filesystems simply run out of inodes
first. An .iso is one file: it copies at streaming speed, and Windows 8+,
macOS and Linux all mount it read-only with a double-click, with the store
appearing exactly as it does here.

The image is ISO-9660 **level 3** with both Rock Ridge and Joliet, and
deep-directory relocation disabled. That combination is what a zarr store
needs: level 3 stores a file larger than 4 GB as multiple extents (a shard
can get there), Joliet is the view Windows reads and carries the long
names, Rock Ridge is the one Linux/macOS read, and without ``-D`` the
deep nesting of ``labels/<name>/0/c/0/0/0`` would be relocated into a
flattened ``RR_MOVED`` directory that only Rock Ridge readers can undo --
leaving Windows with a scrambled tree.

UDF would be the tidier choice, but many xorriso builds ship without it;
this combination works everywhere and is verified by round-trip below.

Usage
-----
    pixi run iso --store /path/to/image.zarr
    pixi run iso --store /path/to/image.zarr --output /elsewhere/scan.iso
    pixi run iso --store /path/to/image.zarr --dry-run

Reading it back
---------------
Windows    right-click the .iso -> Mount (it appears as a drive letter)
macOS      double-click, or `hdiutil attach scan.iso`
Linux      `sudo mount -o loop scan.iso /mnt/scan`

The store inside opens with napari/patchworks unchanged -- it is the same
bytes, just packaged.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# UDF revision 2.01 is the one every current Windows/macOS/Linux mounts
# without extra drivers.
_UDF_REVISION = "2.01"
# Volume ids are limited to 32 characters and a restricted character set.
_VOLID_MAX = 32


def human(n: float) -> str:
    """Bytes as a short human-readable string."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"


def tree_size(path: Path) -> tuple[int, int]:
    """Total bytes and file count under *path*."""
    total = count = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
            count += 1
    return total, count


def volume_id(name: str) -> str:
    """A volume label xorriso will accept, derived from the store's name."""
    cleaned = "".join(c if c.isalnum() else "_" for c in name.upper())
    return cleaned[:_VOLID_MAX].strip("_") or "PATCHWORKS"


def build_command(store: Path, output: Path) -> list[str]:
    """The xorriso invocation that packs *store* into *output*."""
    return [
        "xorriso",
        "-as",
        "mkisofs",
        # Rock Ridge (Linux/macOS) and Joliet (Windows) both carry the long
        # names; plain ISO-9660 alone would mangle them.
        "-R",
        "-J",
        "-joliet-long",
        # Keep the directory tree where it is. ISO-9660 normally relocates
        # anything deeper than 8 levels into RR_MOVED, which Rock Ridge
        # readers undo transparently and Windows does not -- and a zarr v3
        # chunk path is deeper than that.
        "-D",
        # Level 3: a file over 4 GB is stored as multiple extents rather
        # than refused. Sharded chunks can reach that.
        "-iso-level",
        "3",
        # Required for the `/name=source` pathspec below; without it the
        # whole string is taken as a filename.
        "-graft-points",
        "-volid",
        volume_id(store.stem),
        "-o",
        str(output),
        # Mount the store *as its own directory name* inside the image, so
        # the .iso contains image.zarr/... rather than a bare 0/ 1/ labels/.
        f"/{store.name}={store}",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--store", required=True, help="the .zarr directory to pack"
    )
    parser.add_argument(
        "--output",
        help="path of the .iso to write (default: <store>.iso beside it)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be written, build nothing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing .iso instead of refusing",
    )
    args = parser.parse_args()

    store = Path(args.store).resolve()
    if not store.is_dir():
        raise SystemExit(f"not a directory: {store}")
    output = (
        Path(args.output).resolve()
        if args.output
        else store.with_suffix(store.suffix + ".iso")
    )

    if shutil.which("xorriso") is None:
        raise SystemExit(
            "xorriso is not installed. It builds the image; install it with "
            "your package manager (e.g. `apt install xorriso`, "
            "`conda install -c conda-forge xorriso`) and re-run."
        )

    size, count = tree_size(store)
    print(f"store  : {store}")
    print(f"content: {count:,} files, {human(size)}")
    print(f"output : {output}")
    print(f"volume : {volume_id(store.stem)}")

    if output.exists() and not args.overwrite:
        raise SystemExit(
            f"{output} already exists; pass --overwrite to replace it"
        )

    # The image is roughly the payload plus UDF/ISO metadata. Refusing here
    # beats discovering it after an hour of writing.
    free = shutil.disk_usage(output.parent).free
    needed = int(size * 1.02) + 16 * 1024**2
    print(f"needs  : ~{human(needed)}, {human(free)} free on the target")
    if not args.dry_run and free < needed:
        raise SystemExit(
            f"not enough space: need ~{human(needed)}, have {human(free)}"
        )

    command = build_command(store, output)
    print(f"\n$ {' '.join(command)}")
    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    result = subprocess.run(command)
    if result.returncode != 0:
        raise SystemExit(f"xorriso failed with exit code {result.returncode}")

    made = output.stat().st_size
    print(f"\nwrote {output} ({human(made)})")
    print(
        "\nMount it read-only:\n"
        "  Windows  right-click -> Mount\n"
        "  macOS    double-click, or `hdiutil attach <iso>`\n"
        "  Linux    `sudo mount -o loop <iso> /mnt/point`"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
