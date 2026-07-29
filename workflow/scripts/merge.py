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
    safe_worker_count,
)
from patchworks._chunks import _get_available_memory
from patchworks.plugins.ome_zarr import register_labels

from _pw import load_tiles_json, stage_path, start_log

start_log(snakemake.log[0])  # noqa: F821
cfg = snakemake.config  # noqa: F821
work_dir = cfg["work_dir"]
label_name = cfg.get("label_name", "labels")
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
    out_chunks = capped_output_chunks(staged.chunks, (16, 1024, 1024))

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
)
group = register_labels(
    image_store,
    label_name,
    n_levels=int(cfg.get("pyramid_levels", 5)),
    downscale=int(cfg.get("pyramid_downscale", 2)),
    progress=True,
    n_objects=n_objects,
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
print(f"[patchworks] labels written to {group}")
open(snakemake.output[0], "w").close()  # noqa: F821
