"""Snakemake script: segment ONE tile on a GPU and write it to the stage.

Scattered over tile indices by Snakemake, so each tile is its own SLURM job
and many GPUs run in parallel. Each job writes a disjoint chunk of
``stage.zarr/staged`` (safe to run concurrently).
"""

from patchworks.plugins.cellpose import cellpose_fn

from _pw import (
    load_tiles_json,
    open_image,
    open_stage,
    process_one_tile,
    spatial_tile_slices,
)

cfg = snakemake.config  # noqa: F821
index = int(snakemake.wildcards.index)  # noqa: F821
work_dir = cfg["work_dir"]

manifest = load_tiles_json(snakemake.input.tiles)  # noqa: F821
tile_shape = tuple(manifest["tile_shape"])
overlap = int(manifest["overlap"])

image = open_image(work_dir, cfg["channel"], cfg["level"])
sl = spatial_tile_slices(image.shape, tile_shape)[index]

cp = cfg["cellpose"]
extra = {
    k: v
    for k, v in cp.items()
    if k not in ("model", "diameter", "do_3D", "gpu")
}
fn = cellpose_fn(
    cp.get("model", "cyto3"),
    gpu=cp.get("gpu", True),
    diameter=cp.get("diameter"),
    do_3D=cp.get("do_3D", False),
    **extra,
)

labels = process_one_tile(image, sl, overlap, fn)
open_stage(work_dir, mode="r+")[sl] = labels.astype("int32")

# touch the per-tile done marker
open(snakemake.output[0], "w").close()  # noqa: F821
print(f"[patchworks] segmented tile {index}")
