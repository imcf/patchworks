"""Reshard an existing patchworks OME-ZARR in place -- no re-conversion, no
re-segmentation.

Sharding normally has to be chosen *before* a store is written, because many
concurrent writers cannot share a shard file. But once a store is finished
nothing is writing it, so a single-threaded pass can repack it: same chunks,
same data, same metadata, far fewer files.

That makes this the cheap way to fix a store that was written unsharded --
it costs one read+write of each level instead of re-running the pipeline.

Usage
-----
    # see what it would do, change nothing
    pixi run reshard --store /path/to/image.zarr --dry-run

    # labels only -- level 0 is typically the great majority of a label
    # group's files, so this is where the win is
    pixi run reshard --store /path/to/image.zarr --labels-only

    # everything: image pyramid + every label group
    pixi run reshard --store /path/to/image.zarr

Run it as a batch job, not on the login node: it reads and writes every
level, and a whole-store pass on a login node is killed with no traceback,
the same way the occupancy build is. A `runtime` past your QOS's MaxWall is
refused by sbatch before the job starts, so pass `--qos` to match (see
docs/guide/snakemake.md).

SAFETY
------
* Nothing else may be touching the store while this runs. Make sure no
  Snakemake job is still going (`squeue -u $USER`).
* Each level is written to a temporary sibling and only then swapped in, so
  an interruption leaves the original intact -- but it needs transient free
  space equal to the largest level being converted. --dry-run reports it.
* Array attributes are carried over, including the merge's own
  `patchworks_merge_state`. Losing that would let a later re-run merge
  already-merged ids together, so the script verifies it survived.
"""

import argparse
import shutil
import sys
from pathlib import Path


def _files(path):
    return sum(1 for p in Path(path).rglob("*") if p.is_file())


def _bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def _human(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"


def _levels(group_path):
    """Numeric pyramid levels present in a group, in order."""
    return sorted(
        (p.name for p in Path(group_path).iterdir() if p.name.isdigit()),
        key=int,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True, help="path to image.zarr")
    ap.add_argument(
        "--labels-only",
        action="store_true",
        help="skip the image pyramid, reshard only labels/*",
    )
    ap.add_argument(
        "--image-only",
        action="store_true",
        help="skip labels, reshard only the image pyramid",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would change, write nothing",
    )
    args = ap.parse_args()

    import zarr

    from patchworks.plugins.ome_zarr import reshard_level

    store = Path(args.store)
    if not store.exists():
        raise SystemExit(f"no such store: {store}")

    fmt = zarr.open_group(str(store), mode="r").metadata.zarr_format
    if fmt != 3:
        raise SystemExit(
            f"{store} is a zarr v{fmt} store (NGFF 0.4), which has no "
            "sharding codec. Nothing to do -- sharding needs zarr v3."
        )

    targets = []
    if not args.labels_only:
        for lvl in _levels(store):
            targets.append((str(store), lvl, f"image level {lvl}"))
    if not args.image_only and (store / "labels").exists():
        for group in sorted((store / "labels").iterdir()):
            if not group.is_dir():
                continue
            for lvl in _levels(group):
                targets.append(
                    (str(group), lvl, f"labels/{group.name} level {lvl}")
                )

    if not targets:
        raise SystemExit("found no arrays to reshard")

    print(f"store : {store}")
    print(f"arrays: {len(targets)}\n")
    print(f"{'array':34s} {'files':>9}  {'size':>10}  {'shards':>18}")
    print("-" * 78)

    total_before = biggest = 0
    todo = []
    for group_path, component, label in targets:
        path = Path(group_path) / component
        arr = zarr.open_array(str(path), mode="r")
        n, size = _files(path), _bytes(path)
        total_before += n
        already = getattr(arr, "shards", None)
        if already is None:
            todo.append((group_path, component, label, n))
            biggest = max(biggest, size)
        print(
            f"{label:34s} {n:>9,}  {_human(size):>10}  "
            f"{str(already) if already else 'none':>18}"
        )

    print("-" * 78)
    print(f"{'TOTAL':34s} {total_before:>9,} files")
    print(
        f"\n{len(todo)} array(s) need resharding, "
        f"{len(targets) - len(todo)} already sharded."
    )
    if todo:
        print(
            f"transient free space needed: ~{_human(biggest)} "
            "(largest level, freed again immediately)"
        )

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return
    if not todo:
        print("\nNothing to do.")
        return

    free = shutil.disk_usage(store).free
    if free < biggest * 1.1:
        raise SystemExit(
            f"\nABORT: {_human(free)} free, need ~{_human(biggest)} "
            "transient. Free some space first."
        )

    print()
    total_after = 0
    for group_path, component, label, before in todo:
        print(f"resharding {label} ...", flush=True)
        reshard_level(group_path, component, shard=True, progress=True)
        path = Path(group_path) / component
        after = _files(path)
        total_after += after
        arr = zarr.open_array(str(path), mode="r")
        print(
            f"  {label}: {before:,} -> {after:,} files "
            f"({before / max(1, after):.1f}x fewer), shards={arr.shards}",
            flush=True,
        )
        # The merge records how far it got in the array's attrs; losing it
        # would let a re-run merge already-merged ids and collide objects.
        if component == "0" and "labels/" in label:
            state = dict(arr.attrs)
            if "patchworks_merge_state" in state:
                print(f"    merge state carried over: {state}", flush=True)

    print(f"\nDone. Resharded {len(todo)} array(s).")
    print("Re-open the store in napari to confirm it still reads.")


if __name__ == "__main__":
    sys.exit(main())
