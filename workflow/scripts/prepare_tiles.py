"""Snakemake script: plan tiles, create the empty stage store, list work.

Writes ``tiles.json`` (tile shape, overlap, and the indices of *occupied*
tiles to segment) and an empty ``stage.zarr/staged`` array that the per-tile
segment jobs fill in parallel.
"""

import json
from functools import partial
from pathlib import Path

import numpy as np
import zarr

from patchworks import auto_tile_shape_cellpose, estimate_empty_tiles

from _pw import open_image, spatial_tile_slices, stage_path

cfg = snakemake.config  # noqa: F821
work_dir = cfg["work_dir"]
image = open_image(work_dir, cfg["channel"], cfg["level"])

# Resolve the tile shape (spatial, matches the loaded image's ndim).
ts = cfg.get("tile_shape", "auto")
if ts == "auto":
    cp = cfg["cellpose"]
    tile_shape = tuple(
        partial(
            auto_tile_shape_cellpose,
            do_3D=cp.get("do_3D", False),
            use_gpu=cp.get("gpu", True),
            diameter=cp.get("diameter"),
        )(image.shape, image.dtype)
    )
else:
    tile_shape = tuple(ts)

tiles = spatial_tile_slices(image.shape, tile_shape)
n_tiles = len(tiles)

# Decide which tiles to actually segment (skip background).
occupied = list(range(n_tiles))
if cfg.get("skip_empty", True):
    info = estimate_empty_tiles(
        image, tile_shape, threshold=cfg.get("empty_threshold")
    )
    occ = info["occupancy"].ravel()  # row-major, matches spatial_tile_slices
    occupied = [i for i in range(n_tiles) if occ[i]]

# Create the empty staged-labels array (zeros), one chunk per tile.
root = zarr.open_group(stage_path(work_dir), mode="w")
root.create_array(
    name="staged",
    shape=image.shape,
    chunks=tile_shape,
    dtype=np.int32,
)

Path(work_dir, "tiles.json").write_text(
    json.dumps(
        {
            "tile_shape": list(tile_shape),
            "overlap": int(cfg.get("overlap", 0)),
            "n_tiles": n_tiles,
            "occupied": occupied,
        },
        indent=2,
    )
)
print(f"[patchworks] {len(occupied)}/{n_tiles} tiles to segment")
