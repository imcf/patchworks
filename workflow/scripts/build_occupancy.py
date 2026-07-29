"""Snakemake script: build the max-pooled occupancy map for one level.

Its own rule, rather than work done inside ``run_multi``, because it streams
the entire image: doing that in the driver process runs it on the **login
node**, where a multi-terabyte read is killed without a traceback. As a rule
it is submitted like any other job and gets a real allocation.

Built once per image and shared by every config that segments it -- otherwise
the concurrently-running ``prepare`` steps would each stream the whole volume
and all but one would throw the result away.
"""

from patchworks import block_for_tile, build_occupancy_map

from _pw import start_log

start_log(snakemake.log[0])  # noqa: F821
cfg = snakemake.config  # noqa: F821

image_store = str(snakemake.input[0]).removesuffix("/zarr.json")  # noqa: F821
level = int(cfg.get("level", 0))

# Sizing the block from the tile keeps the map discriminating: a block as
# coarse as the tile itself would make every tile test occupied.
tile_shape = cfg.get("tile_shape")
kwargs = (
    {"block": block_for_tile(tuple(tile_shape))}
    if isinstance(tile_shape, (list, tuple))
    else {}
)

path = build_occupancy_map(image_store, level=level, **kwargs)
print(f"[patchworks] occupancy map ready at {path}", flush=True)
