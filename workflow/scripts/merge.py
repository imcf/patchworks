"""Snakemake script: merge the staged tiles into one labelled OME-ZARR.

Runs patchworks' zarr-native boundary merge (stitches labels across tile
boundaries, optionally renumbers them) and writes the result back into the
image store under ``labels/<name>/`` as a calibrated, multi-scale pyramid.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import zarr
from patchworks import (
    capped_output_chunks,
    cpu_allocation,
    merge_tile_labels,
    provenance,
    safe_worker_count,
    set_compression,
)
from patchworks._chunks import _get_available_memory
from patchworks._volume_filter import (
    filter_labels_by_size,
    max_voxels_for_volume,
    min_voxels_for_volume,
)
from patchworks.plugins.ome_zarr import (
    read_pixel_size,
    register_labels,
    reshard_level,
)

from _pw import halo_path, load_tiles_json, stage_path, start_log

start_log(snakemake.log[0])  # noqa: F821
# Codec for every array this step creates (config `compression:`).
set_compression(snakemake.config.get("compression", "zstd"))  # noqa: F821
cfg = snakemake.config  # noqa: F821
work_dir = cfg["work_dir"]
label_name = cfg.get("label_name", "labels")
# Level 0 keeps napari-friendly chunks even when the tile is much larger; the
# cap has to divide the tile so workers still write whole chunks.
LABEL_CHUNK_CAP = (16, 1024, 1024)
image_store = str(Path(work_dir) / "image.zarr")
label_group = f"{image_store}/labels/{label_name}"

# prepare recorded where the segment jobs wrote: the label group's level 0
# (merged in place) or a scratch stage store.
manifest = load_tiles_json(snakemake.input.tiles)  # noqa: F821
target_path = manifest["target_path"]
target_component = manifest.get("target_component", "staged")
in_place = bool(manifest.get("in_place", False))

staged = zarr.open_group(target_path, mode="r")[target_component]

# Size the relabel pool against what this job was actually granted, not the
# node. Each worker holds roughly a few copies of one chunk, so the RAM budget
# -- and not the core count -- is what has to bound it: merge_workers: null
# used to leave merge_tile_labels capping itself at 4, while the profile's
# comment claimed the full allocation was in use.
chunk_nbytes = int(np.prod(staged.chunks)) * staged.dtype.itemsize
default_workers = min(
    cpu_allocation(), safe_worker_count(chunk_nbytes, fn_overhead=3)
)
print(
    f"[patchworks] merge: {cpu_allocation()} cpu(s), "
    f"{_get_available_memory() / 1024**3:.0f} GiB budget, "
    f"{default_workers} worker(s) for {chunk_nbytes / 1024**2:.0f} MB chunks"
)
# Each segment job recorded how many labels every tile wrote. Feeding those
# counts in lets the merge compute global id ranges by a cumulative sum,
# replacing a full read+write of the store that existed only to renumber it.
label_counts = {}
# Derive the marker paths from the manifest rather than reading
# snakemake.input: when the `prepare` checkpoint cannot be resolved (see the
# STAGE_OK note at the end of this file), Snakemake substitutes a placeholder
# for a checkpoint-dependent input function, and `markers` then points at
# tiles.json instead of the seg markers. The paths are deterministic, so
# building them here is both simpler and immune to that.
seg_dir = Path(work_dir) / label_name / "seg"
for batch in range(len(manifest["batches"])):
    marker = seg_dir / f"{batch}.done"
    for index, n in json.loads(marker.read_text())["counts"].items():
        label_counts[int(index)] = int(n)

if in_place:
    # The tiles already sit in labels/<name>/0, so the merge rewrites them
    # where they are: no scratch store, and one full write of the volume less.
    # Safe because the boundary scan finishes before any chunk is rewritten.
    out_chunks = None
else:
    # Level 0 keeps napari-friendly chunks even when tiles are much larger;
    # the cap must divide the tile so workers still write whole chunks.
    root = zarr.open_group(image_store, mode="a")
    parent = root.require_group("labels")
    if label_name in parent:
        del parent[label_name]
    parent.require_group(label_name)
    out_chunks = capped_output_chunks(staged.chunks, LABEL_CHUNK_CAP)

_, n_objects = merge_tile_labels(
    target_path,
    write_to=label_group if not in_place else target_path,
    input_component=target_component,
    output_component="0" if not in_place else target_component,
    output_chunks=out_chunks,
    sequential_labels=cfg.get("sequential_labels", True),
    n_workers=cfg.get("merge_workers") or default_workers,
    # Periodic log lines rather than a bar (see convert.py).
    progress=True,
    return_count=True,
    label_counts=label_counts,
    # stitch: iou joins labels across a seam only where both tiles' views of
    # the overlap agree, so touching cells stay apart.
    halo_dir=(
        halo_path(work_dir, label_name)
        if cfg.get("stitch", "touch") == "iou"
        else None
    ),
    iou_threshold=float(cfg.get("iou_threshold", 0.5)),
)
shutil.rmtree(halo_path(work_dir, label_name), ignore_errors=True)

# Global, exact volume filter -- runs once on the fully merged array so an
# object's size is never judged from just the fragment one tile happened to
# see. Runs before the pyramid so every level reflects the filtered result.
min_volume = cfg.get("min_volume")
max_volume = cfg.get("max_volume")
if min_volume or max_volume:
    # The labels are at the segmented level's resolution, so voxel counts
    # must be converted with that level's voxel size.
    voxel_size = read_pixel_size(image_store, level=int(cfg.get("level", 0)))
    if not voxel_size:
        raise RuntimeError(
            f"min_volume/max_volume filtering needs calibration in "
            f"{image_store}, which has none -- set both to null, or make "
            "sure the source carries a pixel size at conversion time"
        )
    min_voxels = (
        min_voxels_for_volume(min_volume, voxel_size) if min_volume else None
    )
    max_voxels = (
        max_voxels_for_volume(max_volume, voxel_size) if max_volume else None
    )
    n_objects, n_removed = filter_labels_by_size(
        label_group,
        "0",
        min_voxels,
        max_voxels,
        relabel=cfg.get("sequential_labels", True),
    )
    print(
        f"[patchworks] volume filter: dropped {n_removed} object(s) outside "
        f"[{min_volume or 0}, {max_volume or 'inf'}] µm³ "
        f"([{min_voxels or 0}, {max_voxels or 'inf'}] voxels), "
        f"{n_objects} remain"
    )

# Level 0 could not be sharded while it was being written: the segment jobs
# (or the merge's own pool) fill it a chunk at a time from several processes,
# and concurrent writers into one shard read-modify-write the same file and
# silently drop each other's chunks. Now that every writer is done, one
# single-threaded pass can rewrite it sharded -- cutting the file count of the
# largest level by the shard/chunk ratio, which matters on a filesystem that
# dislikes many small files. It costs a full extra read+write of level 0, so
# it stays opt-in.
# Say what the label store will cost in files, and why. The chunk shape is
# inherited from tile_shape, so an auto-sized tile that is not a multiple of
# the cap (e.g. 729 against 1024) quietly multiplies the file count -- which
# is invisible from the config and only shows up as a slow `ls` weeks later.
level0 = zarr.open_array(f"{label_group}/0", mode="r")
n_chunks = int(
    np.prod([-(-s // c) for s, c in zip(level0.shape, level0.chunks)])
)
print(
    f"[patchworks] {label_name} level 0: shape={level0.shape} "
    f"chunks={level0.chunks} -> {n_chunks:,} chunks"
)

shard_labels = cfg.get("shard_labels", False)
# Echo what was actually read, not just what it leads to. A key set in the
# wrong config file is silent otherwise: the run behaves as if it were never
# written, and the log gives no way to tell that from the feature failing.
print(
    f"[patchworks] sharding: shard={cfg.get('shard', False)!r} "
    f"shard_labels={shard_labels!r}"
)
if (not shard_labels or not cfg.get("shard")) and n_chunks > 2000:
    # Both keys, and level 0 is the one that matters. `shard` sends the
    # pyramid down the dask path, where each level is rechunked to the cap
    # and so shrinks fourfold; level 0 keeps tile_shape's chunks and is the
    # level `shard` cannot reach. On a real store it was 96% of the group's
    # files, so `shard` alone barely moved the count.
    missing = " and ".join(
        k
        for k, on in (
            ("shard", cfg.get("shard")),
            ("shard_labels", shard_labels),
        )
        if not on
    )
    print(
        f"[patchworks] NOTE: level 0 is {n_chunks:,} chunks, and typically "
        "the great majority of this label group's files -- the pyramid "
        "levels above it shrink fourfold each. Set `" + missing + ": true` "
        "to pack them into shards: `shard` covers levels 1..N, "
        "`shard_labels` covers level 0, which is the one that counts."
    )
if shard_labels:
    # `true` reuses whatever `shard` asks the conversion for; a list overrides
    # it with an explicit shard shape. `shard: false` does not veto this --
    # opting in here is the whole request, and an unsharded raw image with
    # sharded labels is a perfectly reasonable combination.
    spec = (cfg.get("shard") or True) if shard_labels is True else shard_labels
    reshard_level(label_group, "0", shard=spec, progress=True)

# How these labels were made, stored with them (read_provenance()).
_SETTINGS = (
    "input",
    "channel",
    "nuclei_channel",
    "level",
    "method",
    "cellpose",
    "custom",
    "dilate",
    "min_volume",
    "max_volume",
    "stitch",
    "iou_threshold",
    "sequential_labels",
    "compression",
    "skip_empty",
    "empty_threshold",
)
record = provenance(
    label_name=label_name,
    tile_shape=manifest["tile_shape"],
    overlap=manifest["overlap"],
    **{k: cfg.get(k) for k in _SETTINGS if k in cfg},
)

group = register_labels(
    image_store,
    label_name,
    provenance=record,
    n_levels=int(cfg.get("pyramid_levels", 5)),
    downscale=int(cfg.get("pyramid_downscale", 2)),
    progress=True,
    n_objects=n_objects,
    # Same `shard` the conversion uses, so one setting covers the whole
    # store. This one only reaches levels 1..N -- each is written by a single
    # dask pass, so one writer owns every shard. Level 0 needs the separate
    # `shard_labels` pass above for the reason given there.
    shard=cfg.get("shard", False),
    ngff_version=cfg.get("ngff_version", "auto"),
    # Segmented at `level`: calibrate (and offset) the labels as that level,
    # or they are drawn shrunk towards the origin of the image.
    level=int(cfg.get("level", 0)),
)

if not in_place:
    # Only the scratch route creates a store to clean up -- and only it has to
    # drop the checkpoint's completion sentinel, because that sentinel would
    # otherwise outlive the store it claims exists and a rerun would skip
    # "prepare" and segment into something already deleted.
    #
    # Deleting a checkpoint output is not free: it leaves `prepare`
    # permanently unresolvable, so a later DAG evaluation cannot expand
    # batch_done and hands dependent rules a placeholder input instead. That
    # is why the label counts above are read by path, not from snakemake.input.
    shutil.rmtree(stage_path(work_dir, label_name), ignore_errors=True)
    Path(f"{stage_path(work_dir, label_name)}.done").unlink(missing_ok=True)
# Did the tiling leave marks? Compares how often objects end exactly on a
# seam against planes inside the tiles; cheap (a sample of thin slabs) and
# written next to the run for later comparison between configs.
if cfg.get("seam_report", True):
    from patchworks import seam_report

    report = seam_report(
        group,
        manifest["tile_shape"],
        max_faces=int(cfg.get("seam_report_faces", 64)),
    )
    Path(work_dir, label_name, "seams.json").write_text(
        json.dumps(report, indent=2)
    )
    for ax, row in report["axes"].items():
        interior = row["interior_rate"]
        print(
            f"[patchworks] seams axis {ax}: {100 * row['seam_rate']:.1f}% of "
            f"labels end on a seam vs "
            f"{'n/a' if interior is None else f'{100 * interior:.1f}%'} "
            "inside tiles"
        )
print(f"[patchworks] labels written to {group}")
open(snakemake.output[0], "w").close()  # noqa: F821
