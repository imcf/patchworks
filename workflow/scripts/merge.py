"""Snakemake script: merge the staged tiles into one labelled OME-ZARR.

Runs patchworks' zarr-native boundary merge (stitches labels across tile
boundaries, optionally renumbers them) and writes the result back into the
image store under ``labels/<name>/`` as a calibrated, multi-scale pyramid.
"""

import json
import os
import shutil
from pathlib import Path

import zarr
from patchworks import capped_output_chunks, merge_tile_labels
from patchworks.plugins.ome_zarr import register_labels

from _pw import stage_path, start_log

start_log(snakemake.log[0])  # noqa: F821
cfg = snakemake.config  # noqa: F821
work_dir = cfg["work_dir"]
label_name = cfg.get("label_name", "labels")
image_store = str(Path(work_dir) / "image.zarr")
label_group = f"{image_store}/labels/{label_name}"

# merge_tile_labels defaults to min(4, cpu_count) workers, so it ignores
# whatever cpus_per_task the "merge" rule was actually allocated in the SLURM
# profile. Read the real allocation (SLURM_CPUS_PER_TASK) so the job uses all
# the cores it's paying for; merge_workers: in config.yaml can still override.
default_workers = int(
    os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4)
)
# Each segment job recorded how many labels every tile wrote. Feeding those
# counts in lets the merge compute global id ranges by a cumulative sum,
# replacing a full read+write of the store that existed only to renumber it.
label_counts = {}
for marker in snakemake.input:  # noqa: F821
    for index, n in json.loads(Path(marker).read_text())["counts"].items():
        label_counts[int(index)] = int(n)

# Merge straight into the label group's level 0. Writing to a scratch
# _merged.zarr and letting write_labels copy it across cost a full extra
# read+write of the volume plus the scratch store's disk; register_labels
# already expects level 0 to exist and only adds the pyramid and metadata.
root = zarr.open_group(image_store, mode="a")
parent = root.require_group("labels")
if label_name in parent:
    del parent[label_name]
parent.require_group(label_name)

staged = zarr.open_group(stage_path(work_dir, label_name), mode="r")["staged"]
# Level 0 keeps napari-friendly chunks even when tiles are much larger; the
# cap has to divide the tile shape so merge workers still write whole chunks.
out_chunks = capped_output_chunks(staged.chunks, (16, 1024, 1024))

_, n_objects = merge_tile_labels(
    stage_path(work_dir, label_name),
    write_to=label_group,
    input_component="staged",
    output_component="0",
    output_chunks=out_chunks,
    sequential_labels=cfg.get("sequential_labels", True),
    n_workers=cfg.get("merge_workers", default_workers),
    progress=False,
    return_count=True,
    label_counts=label_counts,
)
group = register_labels(
    image_store,
    label_name,
    n_levels=int(cfg.get("pyramid_levels", 5)),
    downscale=int(cfg.get("pyramid_downscale", 2)),
    progress=False,
    n_objects=n_objects,
)

shutil.rmtree(stage_path(work_dir, label_name), ignore_errors=True)
# Also drop the checkpoint's completion sentinel (stage.zarr.done): the
# "prepare" rule's stage=touch(STAGE_OK) output must not outlive the store it
# claims exists, or a future rerun (e.g. re-segmenting for new labels) skips
# "prepare" and "segment" tries to open a stage.zarr that's already gone.
Path(f"{stage_path(work_dir, label_name)}.done").unlink(missing_ok=True)
print(f"[patchworks] labels written to {group}")
open(snakemake.output[0], "w").close()  # noqa: F821
