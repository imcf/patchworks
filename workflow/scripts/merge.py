"""Snakemake script: merge the staged tiles into one labelled OME-ZARR.

Runs patchworks' zarr-native boundary merge (stitches labels across tile
boundaries, optionally renumbers them) and writes the result back into the
image store under ``labels/<name>/`` as a calibrated, multi-scale pyramid.
"""

import shutil
from pathlib import Path

from patchworks import merge_tile_labels
from patchworks.plugins.ome_zarr import write_labels

from _pw import stage_path

cfg = snakemake.config  # noqa: F821
work_dir = cfg["work_dir"]
image_store = str(Path(work_dir) / "image.zarr")
merged_store = str(Path(work_dir) / "_merged.zarr")

merged = merge_tile_labels(
    stage_path(work_dir),
    write_to=merged_store,
    input_component="staged",
    sequential_labels=cfg.get("sequential_labels", True),
    progress=False,
)
group = write_labels(
    image_store,
    merged,
    name=cfg.get("label_name", "labels"),
    n_levels=int(cfg.get("pyramid_levels", 5)),
    downscale=int(cfg.get("pyramid_downscale", 2)),
    overwrite=True,
)

shutil.rmtree(merged_store, ignore_errors=True)
shutil.rmtree(stage_path(work_dir), ignore_errors=True)
print(f"[patchworks] labels written to {group}")
open(snakemake.output[0], "w").close()  # noqa: F821
