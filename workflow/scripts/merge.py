"""Snakemake script: merge the staged tiles into one labelled OME-ZARR.

Runs patchworks' zarr-native boundary merge (stitches labels across tile
boundaries, optionally renumbers them) and writes the result back into the
image store under ``labels/<name>/`` as a calibrated, multi-scale pyramid.
"""

import os
import shutil
from pathlib import Path

from patchworks import merge_tile_labels
from patchworks.plugins.ome_zarr import write_labels

from _pw import stage_path, start_log

start_log(snakemake.log[0])  # noqa: F821
cfg = snakemake.config  # noqa: F821
work_dir = cfg["work_dir"]
label_name = cfg.get("label_name", "labels")
image_store = str(Path(work_dir) / "image.zarr")
merged_store = str(Path(work_dir) / label_name / "_merged.zarr")

# merge_tile_labels defaults to min(4, cpu_count) workers, so it ignores
# whatever cpus_per_task the "merge" rule was actually allocated in the SLURM
# profile. Read the real allocation (SLURM_CPUS_PER_TASK) so the job uses all
# the cores it's paying for; merge_workers: in config.yaml can still override.
default_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
merged = merge_tile_labels(
    stage_path(work_dir, label_name),
    write_to=merged_store,
    input_component="staged",
    sequential_labels=cfg.get("sequential_labels", True),
    n_workers=cfg.get("merge_workers", default_workers),
    progress=False,
)
group = write_labels(
    image_store,
    merged,
    name=label_name,
    n_levels=int(cfg.get("pyramid_levels", 5)),
    downscale=int(cfg.get("pyramid_downscale", 2)),
    overwrite=True,
)

shutil.rmtree(merged_store, ignore_errors=True)
shutil.rmtree(stage_path(work_dir, label_name), ignore_errors=True)
# Also drop the checkpoint's completion sentinel (stage.zarr.done): the
# "prepare" rule's stage=touch(STAGE_OK) output must not outlive the store it
# claims exists, or a future rerun (e.g. re-segmenting for new labels) skips
# "prepare" and "segment" tries to open a stage.zarr that's already gone.
Path(f"{stage_path(work_dir, label_name)}.done").unlink(missing_ok=True)
print(f"[patchworks] labels written to {group}")
open(snakemake.output[0], "w").close()  # noqa: F821
