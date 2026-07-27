"""Snakemake script: segment one BATCH of tiles into the shared stage.

Scattered over batches, so several batches run as separate SLURM jobs and many
GPUs work in parallel. Within a batch the tiles run sequentially in this one
process: CUDA is initialised once and the Cellpose model is loaded once (see
``patchworks.plugins.cellpose._model_cache``), instead of once per tile. Each
batch writes disjoint chunks of the stage store, so batches never collide.
"""

import json

from patchworks import stage_tile

from _pw import build_fn, load_tiles_json, open_image, start_log

start_log(snakemake.log[0])  # noqa: F821
cfg = snakemake.config  # noqa: F821
batch = int(snakemake.wildcards.batch)  # noqa: F821
work_dir = cfg["work_dir"]
label_name = cfg.get("label_name", "labels")

manifest = load_tiles_json(snakemake.input.tiles)  # noqa: F821
image = open_image(work_dir, cfg["channel"], cfg["level"])
indices = manifest["batches"][batch]

# Built once for the whole batch: this is what makes the model load amortize.
fn = build_fn(cfg)
# prepare decides where tiles land: the label group's level 0 directly when
# the tile fits the chunk cap (the merge then relabels it in place), else a
# scratch stage store.
stage = manifest["target_path"]
component = manifest.get("target_component", "staged")
tile_shape = tuple(manifest["tile_shape"])

counts = {}
for index in indices:
    counts[index] = stage_tile(
        image,
        fn,
        stage,
        index,
        tile_shape=tile_shape,
        # Scalar (older manifests) or per-axis list; stage_tile normalizes both.
        overlap=manifest["overlap"],
        component=component,
    )
    print(f"[patchworks] segmented tile {index}: {counts[index]} label(s)")

# The marker carries each tile's label count. That is what lets merge derive
# every tile's global id range with a cumulative sum instead of streaming the
# whole store to renumber it -- the counts are free here, we just write them
# down instead of throwing them away.
with open(snakemake.output[0], "w") as fh:  # noqa: F821
    json.dump({"batch": batch, "counts": counts}, fh)
print(f"[patchworks] batch {batch}: {len(indices)} tile(s) done")
