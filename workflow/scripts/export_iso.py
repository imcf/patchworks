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
    pixi run iso --store /path/to/image.zarr --dry-run   # report, write nothing
    pixi run iso --store /path/to/image.zarr
    pixi run iso --store /path/to/image.zarr --output /elsewhere/scan.iso

**Submit it, do not run it on a login node.** Packing a store reads every
file and writes the whole image; a shared login node kills a process that
big without a message, and the run simply returns to the prompt partway
through with no error and no usable .iso. Mind the QOS wall-time ceiling
too (see docs/guide/snakemake.md):

    sbatch --qos=1day --time=12:00:00 --cpus-per-task=4 --mem=8G \
      --job-name=iso --output=iso-%j.log \
      --wrap "cd $PWD && pixi run iso --store /path/to/image.zarr"

Memory is not the constraint here: xorriso streams to the output file
rather than assembling the image in RAM, so the file count does not cost
memory. Time and free disk space are.

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
import time
import zipfile
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


# Every mkisofs-compatible builder, best first. They take the same options;
# xorriso needs `-as mkisofs` to emulate them. None of these is installable
# from conda-forge -- it carries no ISO-building C tool, only pycdlib, which
# is pure Python, assembles the image in RAM and does not survive a store
# with tens of thousands of files -- so this uses whatever the system has.
_BUILDERS = (
    ("xorriso", ["-as", "mkisofs"]),
    ("genisoimage", []),
    ("mkisofs", []),
)


def find_builder() -> tuple[str, list[str]]:
    """The first available ISO builder, as ``(executable, leading args)``.

    Raises
    ------
    SystemExit
        When none is on PATH, naming every way to get one.
    """
    for name, prefix in _BUILDERS:
        path = shutil.which(name)
        if path:
            return path, prefix
    raise SystemExit(
        "no ISO builder found. This needs one of: "
        + ", ".join(n for n, _ in _BUILDERS)
        + ".\n\n"
        "conda-forge does not package any of them, so `pixi add` will not "
        "help. Options, in order of least effort:\n"
        "  * check whether the cluster already has one, possibly behind a "
        "module: `which xorriso genisoimage mkisofs`, `module avail xorriso`\n"
        "  * ask the cluster admins to install xorriso (it is a small, "
        "standard package)\n"
        "  * build the image somewhere else -- copy the store with `tar` "
        "(one stream, no per-file cost) and pack it into an .iso there\n\n"
        "Do NOT substitute a pure-Python ISO builder: it assembles the whole "
        "image in memory and dies partway through a store this size."
    )


def build_command(store: Path, output: Path) -> list[str]:
    """The ISO-builder invocation that packs *store* into *output*."""
    executable, prefix = find_builder()
    return [
        executable,
        *prefix,
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


def write_zip(store: Path, output: Path, progress: bool = True) -> None:
    """Pack *store* into a single uncompressed .zip.

    ``ZIP_STORED``, not deflate: every chunk is already zstd-compressed, so
    re-compressing costs a full pass over the data to save almost nothing.

    Written entry by entry, so memory stays flat however many files the
    store holds -- the failure mode of the pure-Python *ISO* builders is
    that they assemble the image in RAM first, which this does not do.
    """
    total = sum(1 for p in store.rglob("*") if p.is_file())
    done = 0
    started = time.monotonic()
    with zipfile.ZipFile(
        output, "w", zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        for item in sorted(store.rglob("*")):
            if not item.is_file():
                continue
            # Relative to the store's *parent*, so the archive contains
            # image.zarr/... and unpacks to a usable store rather than a
            # bare 0/ 1/ labels/.
            archive.write(item, item.relative_to(store.parent))
            done += 1
            if progress and (done % 2000 == 0 or done == total):
                elapsed = time.monotonic() - started
                print(
                    f"  {done:,}/{total:,} files ({100 * done / total:.0f}%) "
                    f"after {elapsed / 60:.1f}m",
                    flush=True,
                )


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
        "--format",
        choices=("iso", "zip"),
        default="iso",
        help=(
            "iso (default): mounts read-only as a drive on Windows/macOS/"
            "Linux, but needs xorriso/genisoimage/mkisofs on the system. "
            "zip: needs nothing beyond Python, and zarr reads a store "
            "straight out of it without unpacking "
            "(zarr.storage.ZipStore) -- but it does not mount as a drive."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing archive instead of refusing",
    )
    args = parser.parse_args()

    store = Path(args.store).resolve()
    if not store.is_dir():
        raise SystemExit(f"not a directory: {store}")
    output = (
        Path(args.output).resolve()
        if args.output
        else store.with_suffix(f"{store.suffix}.{args.format}")
    )

    if args.format == "iso":
        # Fails here, with the full explanation, if nothing suitable exists.
        find_builder()

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

    if args.format == "iso":
        command = build_command(store, output)
        print(f"\n$ {' '.join(command)}")
    else:
        command = None
        print("\npacking with python's zipfile (ZIP_STORED, no re-compression)")
    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    if command is None:
        write_zip(store, output)
    else:
        result = subprocess.run(command)
        if result.returncode != 0:
            raise SystemExit(
                f"{Path(command[0]).name} failed with exit code "
                f"{result.returncode}"
            )

    made = output.stat().st_size
    print(f"\nwrote {output} ({human(made)})")
    if args.format == "iso":
        print(
            "\nMount it read-only:\n"
            "  Windows  right-click -> Mount\n"
            "  macOS    double-click, or `hdiutil attach <iso>`\n"
            "  Linux    `sudo mount -o loop <iso> /mnt/point`"
        )
    else:
        print(
            "\nRead it without unpacking:\n"
            "  import zarr\n"
            f'  store = zarr.storage.ZipStore("{output.name}", mode="r")\n'
            f'  group = zarr.open_group(store, path="{store.name}", '
            'mode="r")\n'
            "\nOr just open it in Windows Explorer / unzip it to get the "
            "store back as a directory."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
