"""Snakemake script: segment one batch of tiles (one GPU job).

The tiles run one after the other, sharing one model load; each writes its
own chunk, so batches never collide. While the GPU segments one tile, the
next is read and the previous one renumbered, measured and written, each in
a thread of its own: the GPU no longer waits on the filesystem.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from patchworks._distributed import finish_tile, read_tile

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
todo = [(n, index) for n, index in enumerate(indices, 1) if index not in counts]
halo_dir = (
    halo_path(work_dir, label_name)
    if cfg.get("stitch", "touch") == "iou"
    else None
)
fn_name = getattr(fn, "__name__", type(fn).__name__)


def _read(index):
    return read_tile(
        image,
        index,
        tile_shape=tile_shape,
        overlap=manifest["overlap"],
        channel_axis=0 if stacked else None,
    )


def _finish(tile, out):
    return finish_tile(
        tile,
        out,
        stage,
        component=component,
        halo_dir=halo_dir,
        parts_dir=parts_path(work_dir, label_name),
        fn_name=fn_name,
    )


def _done(n, index, started, future):
    """Record a staged tile, in order: the progress file resumes from it."""
    counts[index] = future.result()
    save_segment_progress(progress_file, indices, tile_shape, counts)
    print(
        f"[patchworks] tile {index} ({n}/{len(indices)}): "
        f"{counts[index]} label(s) in {time.monotonic() - started:.0f}s",
        flush=True,
    )


# At most one tile read ahead and one being written: about three tiles in
# memory, against one before.
with (
    ThreadPoolExecutor(max_workers=1) as reader,
    ThreadPoolExecutor(max_workers=1) as writer,
):
    reading = reader.submit(_read, todo[0][1]) if todo else None
    writing = None
    try:
        for k, (n, index) in enumerate(todo):
            started = time.monotonic()
            tile = reading.result()
            reading = (
                reader.submit(_read, todo[k + 1][1])
                if k + 1 < len(todo)
                else None
            )
            out = fn(tile.block)
            if writing is not None:
                pending, writing = writing, None
                _done(*pending)
            writing = (n, index, started, writer.submit(_finish, tile, out))
            del tile, out
    finally:
        # Also when a tile fails: the one before it is staged, and a retry
        # need not segment it again.
        if writing is not None:
            _done(*writing)

# Each tile's label count, for the merge's global ids.
with open(snakemake.output[0], "w") as fh:  # noqa: F821
    json.dump({"batch": batch, "counts": counts}, fh)
Path(progress_file).unlink(missing_ok=True)
print(
    f"[patchworks] batch {batch}: {len(indices)} tile(s) done in "
    f"{(time.monotonic() - batch_started) / 60:.1f}m",
    flush=True,
)
