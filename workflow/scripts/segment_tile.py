"""Snakemake script: segment ONE tile and write it to the shared stage.

Scattered over tile indices, so each tile is its own SLURM job and many GPUs
run in parallel. Each job writes a disjoint chunk of the stage store.
"""

from patchworks import stage_tile

from _pw import build_fn, load_tiles_json, open_image, stage_path, start_log

start_log(snakemake.log[0])  # noqa: F821
cfg = snakemake.config  # noqa: F821
index = int(snakemake.wildcards.index)  # noqa: F821
work_dir = cfg["work_dir"]
label_name = cfg.get("label_name", "labels")

manifest = load_tiles_json(snakemake.input.tiles)  # noqa: F821
image = open_image(work_dir, cfg["channel"], cfg["level"])

stage_tile(
    image,
    build_fn(cfg),
    stage_path(work_dir, label_name),
    index,
    tile_shape=tuple(manifest["tile_shape"]),
    overlap=int(manifest["overlap"]),
)

open(snakemake.output[0], "w").close()  # noqa: F821
print(f"[patchworks] segmented tile {index}")
