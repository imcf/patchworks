"""Snakemake script: plan tiles, create the empty stage store, list work."""

import json
from functools import partial
from pathlib import Path

from patchworks import (
    auto_tile_shape_cellpose,
    create_stage,
    estimate_empty_tiles,
    spatial_tiles,
)

from _pw import open_image, stage_path, start_log

start_log(snakemake.log[0])  # noqa: F821
cfg = snakemake.config  # noqa: F821
work_dir = cfg["work_dir"]
label_name = cfg.get("label_name", "labels")
Path(work_dir, label_name).mkdir(parents=True, exist_ok=True)
image = open_image(work_dir, cfg["channel"], cfg["level"])

ts = cfg.get("tile_shape", "auto")
if ts == "auto":
    cp = cfg["cellpose"]
    # prepare runs on a CPU node, so the segment GPU's VRAM can't be queried
    # here; pass gpu_memory_gb from the config to size tiles for it (avoids the
    # "GPU memory query failed" fallback). None => the built-in 8 GiB default.
    gpu_gb = cfg.get("gpu_memory_gb")
    tile_shape = tuple(
        partial(
            auto_tile_shape_cellpose,
            do_3D=cp.get("do_3D", False),
            use_gpu=cp.get("gpu", True),
            diameter=cp.get("diameter"),
            gpu_memory=int(gpu_gb * 1024**3) if gpu_gb else None,
        )(image.shape, image.dtype)
    )
else:
    tile_shape = tuple(ts)

tiles = spatial_tiles(image.shape, tile_shape)
occupied = list(range(len(tiles)))
if cfg.get("skip_empty", True):
    info = estimate_empty_tiles(
        image, tile_shape, threshold=cfg.get("empty_threshold")
    )
    occ = info["occupancy"].ravel()  # row-major, matches spatial_tiles
    occupied = [i for i in range(len(tiles)) if occ[i]]

create_stage(stage_path(work_dir, label_name), image.shape, tile_shape)

Path(work_dir, label_name, "tiles.json").write_text(
    json.dumps(
        {
            "tile_shape": list(tile_shape),
            "overlap": int(cfg.get("overlap", 0)),
            "n_tiles": len(tiles),
            "occupied": occupied,
        },
        indent=2,
    )
)
print(f"[patchworks] {len(occupied)}/{len(tiles)} tiles to segment")
