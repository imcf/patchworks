"""Snakemake script: segment one batch of tiles (one GPU job).

The tiles run one after the other, sharing one model load; each writes its
own chunk, so batches never collide.
"""

import json
import time
from pathlib import Path

from patchworks import stage_tile

from _pw import (
    build_fn,
    halo_path,
    parts_path,
    load_segment_progress,
    load_tiles_json,
    open_image,
    pin_slurm_gpus,
    save_segment_progress,
    segment_progress_path,
    start_log,
)

start_log(snakemake.log[0])  # noqa: F821
# Before anything initialises CUDA: stay on the GPU SLURM gave this job.
pinned = pin_slurm_gpus()
if pinned:
    print(f"[patchworks] {pinned}", flush=True)
cfg = snakemake.config  # noqa: F821
batch = int(snakemake.wildcards.batch)  # noqa: F821
work_dir = cfg["work_dir"]
label_name = cfg.get("label_name", "labels")

manifest = load_tiles_json(snakemake.input.tiles)  # noqa: F821
# A second channel (nuclei_channel) or a label image to grow from
# (seed_labels) rides along on a leading axis, untiled.
nuclei_channel = cfg.get("nuclei_channel")
seed_labels = cfg.get("seed_labels")
image = open_image(
    work_dir, cfg["channel"], cfg["level"], nuclei_channel, seed_labels
)
stacked = nuclei_channel is not None or bool(seed_labels)
indices = manifest["batches"][batch]

fn = build_fn(
    cfg,
    intensity_range=manifest.get("intensity_range"),
    kwargs_overrides=manifest.get("kwargs_overrides"),
)
stage = manifest["target_path"]
component = manifest.get("target_component", "staged")
tile_shape = tuple(manifest["tile_shape"])

# A retried batch continues where the last attempt stopped.
progress_file = segment_progress_path(stage, batch)
counts = load_segment_progress(progress_file, indices, tile_shape)
if counts:
    print(
        f"[patchworks] batch {batch}: resuming, {len(counts)}/{len(indices)} "
        "tile(s) already staged by an earlier attempt",
        flush=True,
    )
batch_started = time.monotonic()
for n, index in enumerate(indices, 1):
    if index in counts:
        continue
    started = time.monotonic()
    counts[index] = stage_tile(
        image,
        fn,
        stage,
        index,
        tile_shape=tile_shape,
        overlap=manifest["overlap"],
        component=component,
        channel_axis=0 if stacked else None,
        halo_dir=(
            halo_path(work_dir, label_name)
            if cfg.get("stitch", "touch") == "iou"
            else None
        ),
        parts_dir=parts_path(work_dir, label_name),
    )
    save_segment_progress(progress_file, indices, tile_shape, counts)
    took = time.monotonic() - started
    print(
        f"[patchworks] tile {index} ({n}/{len(indices)}): "
        f"{counts[index]} label(s) in {took:.0f}s",
        flush=True,
    )

# Each tile's label count, for the merge's global ids.
with open(snakemake.output[0], "w") as fh:  # noqa: F821
    json.dump({"batch": batch, "counts": counts}, fh)
Path(progress_file).unlink(missing_ok=True)
print(
    f"[patchworks] batch {batch}: {len(indices)} tile(s) done in "
    f"{(time.monotonic() - batch_started) / 60:.1f}m",
    flush=True,
)
