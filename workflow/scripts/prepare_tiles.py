"""Snakemake script: plan tiles, create the empty stage store, list work."""

import json
from functools import partial
from pathlib import Path

import numpy as np

from patchworks import (
    auto_empty_threshold,
    auto_tile_shape_cellpose,
    build_occupancy_map,
    create_stage,
    normalize_overlap,
    spatial_tiles,
    tile_occupancy,
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

# Which z regime are we in? auto_tile_shape_cellpose pins z to the full extent
# when do_3D is set, so "auto" gives whole-z tiles: no z tiling, no z-boundary
# stitching, and Cellpose sees each object's full depth. An explicit z smaller
# than the image tiles in z instead, which is faster but leaves the merge to
# stitch objects back together across z boundaries.
# tile_shape is zipped against image.shape (see spatial_tiles), so axis 0 of
# both is z for a 3-D stack.
if len(tile_shape) >= 3:
    n_z = image.shape[0]
    if tile_shape[0] >= n_z:
        print("[patchworks] z regime: whole-z tiles (no z-boundary stitching)")
    else:
        print(
            f"[patchworks] z regime: tiled in z ({tile_shape[0]} of {n_z} "
            "planes); objects spanning z boundaries are stitched by the merge"
        )

# A halo wider than the tile itself is not boundary context -- it means every
# tile re-reads and re-segments its neighbours. Catch it here rather than
# paying 5x the GPU time for results that get trimmed away.
overlap = normalize_overlap(cfg.get("overlap", 0), len(tile_shape))
for axis, (ov, extent) in enumerate(zip(overlap, tile_shape)):
    if ov >= extent:
        raise ValueError(
            f"overlap[{axis}]={ov} >= tile_shape[{axis}]={extent}: each tile "
            f"would read past its neighbours. Use a per-axis overlap, e.g. "
            f"overlap: {list(max(1, t // 4) for t in tile_shape)}"
        )
amplification = np.prod(
    [(t + 2 * o) for t, o in zip(tile_shape, overlap)]
) / np.prod(tile_shape)
print(f"[patchworks] halo read amplification: {amplification:.2f}x")

tiles = spatial_tiles(image.shape, tile_shape)
occupied = list(range(len(tiles)))
if cfg.get("skip_empty", True):
    # The occupancy map reduces every voxel of the image to per-brick maxima,
    # so testing a tile covers the whole tile instead of a centred sample. The
    # map is built once per image.zarr and shared by every config using it;
    # convert does not produce it, so already-converted stores build it here
    # on first use.
    build_occupancy_map(str(Path(work_dir) / "image.zarr"), level=cfg["level"])
    threshold = cfg.get("empty_threshold")
    if threshold is None:
        # Derive the cutoff from raw voxels, not from the pooled maxima: a
        # brick maximum exceeds the threshold exactly when some voxel in that
        # brick does, so the comparison stays equivalent to a full scan.
        threshold = auto_empty_threshold(image, cfg["channel"], cfg["level"])
    info = tile_occupancy(
        str(Path(work_dir) / "image.zarr"),
        tile_shape,
        channel=cfg["channel"],
        threshold=threshold,
        level=cfg["level"],
    )
    occ = info["occupancy"].ravel()  # row-major, matches spatial_tiles
    occupied = [i for i in range(len(tiles)) if occ[i]]

create_stage(stage_path(work_dir, label_name), image.shape, tile_shape)

Path(work_dir, label_name, "tiles.json").write_text(
    json.dumps(
        {
            "tile_shape": list(tile_shape),
            "overlap": list(overlap),
            "n_tiles": len(tiles),
            "occupied": occupied,
        },
        indent=2,
    )
)
print(f"[patchworks] {len(occupied)}/{len(tiles)} tiles to segment")
