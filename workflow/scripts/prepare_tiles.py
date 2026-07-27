"""Snakemake script: plan tiles, create the empty stage store, list work."""

import json
from functools import partial
from pathlib import Path

import numpy as np
import zarr

from patchworks import (
    auto_empty_threshold,
    auto_tile_shape,
    auto_tile_shape_cellpose,
    block_for_tile,
    build_occupancy_map,
    capped_output_chunks,
    create_stage,
    normalize_overlap,
    spatial_tiles,
    tile_occupancy,
)

from _pw import open_image, stage_path, start_log, validate_config

# Chunking ceiling for the written labels, so a viewer can page them lazily.
LABEL_CHUNK_CAP = (16, 1024, 1024)

start_log(snakemake.log[0])  # noqa: F821
cfg = snakemake.config  # noqa: F821
work_dir = cfg["work_dir"]
label_name = cfg.get("label_name", "labels")
Path(work_dir, label_name).mkdir(parents=True, exist_ok=True)
image = open_image(work_dir, cfg["channel"], cfg["level"])

# Fail here, on a cheap CPU job, rather than in the first GPU job hours after
# convert and prepare have already run.
validate_config(cfg)

method = cfg.get("method", "cellpose")
ts = cfg.get("tile_shape", "auto")
if ts == "auto":
    # prepare runs on a CPU node, so the segment GPU's VRAM can't be queried
    # here; pass gpu_memory_gb from the config to size tiles for it (avoids the
    # "GPU memory query failed" fallback). None => the built-in 8 GiB default.
    gpu_gb = cfg.get("gpu_memory_gb")
    gpu_bytes = int(gpu_gb * 1024**3) if gpu_gb else None
    if method == "cellpose":
        cp = cfg["cellpose"]
        sizer = partial(
            auto_tile_shape_cellpose,
            do_3D=cp.get("do_3D", False),
            use_gpu=cp.get("gpu", True),
            diameter=cp.get("diameter"),
            gpu_memory=gpu_bytes,
        )
    else:
        # cfg["cellpose"] used to be read unconditionally here, so a DoG or
        # threshold config with tile_shape: "auto" died with KeyError:
        # 'cellpose'. The Cellpose estimator's memory model wouldn't apply to
        # them anyway.
        sizer = partial(
            auto_tile_shape, use_gpu=gpu_bytes is not None, gpu_memory=gpu_bytes
        )
    tile_shape = tuple(sizer(image.shape, image.dtype))
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
overlap = normalize_overlap(
    cfg.get("overlap", 0), len(tile_shape), tile_shape=tile_shape
)
for axis, (ov, extent) in enumerate(zip(overlap, tile_shape)):
    if ov >= extent:
        raise ValueError(
            f"overlap[{axis}]={ov} >= tile_shape[{axis}]={extent}: each tile "
            f"would read past its neighbours. Use a per-axis overlap, e.g. "
            f"overlap: {list(max(1, t // 4) for t in tile_shape)}"
        )
# Clip the halo to the image the way stage_tile does, or a tile that already
# spans an axis reports a halo it will never actually read -- a tile covering
# the whole image would claim 5x while reading exactly 1x.
read = np.prod(
    [min(t + 2 * o, s) for t, o, s in zip(tile_shape, overlap, image.shape)]
)
amplification = read / np.prod(tile_shape)
print(f"[patchworks] halo read amplification: {amplification:.2f}x")

tiles = spatial_tiles(image.shape, tile_shape)
occupied = list(range(len(tiles)))
if cfg.get("skip_empty", True):
    # The occupancy map reduces every voxel of the image to per-brick maxima,
    # so testing a tile covers the whole tile instead of a centred sample. The
    # map is built once per image.zarr and shared by every config using it;
    # convert does not produce it, so already-converted stores build it here
    # on first use.
    build_occupancy_map(
        str(Path(work_dir) / "image.zarr"),
        level=cfg["level"],
        # Sized from the tile: a block as coarse as the tile would make every
        # tile over-cover the same block and test occupied.
        block=block_for_tile(tile_shape),
    )
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

# One SLURM job per tile means every tile pays CUDA init + a model load before
# its first voxel. Batching amortizes that over `tiles_per_job` tiles, which
# run sequentially in one process so they share the cached model and never
# contend for the GPU. 1 = the old one-job-per-tile behaviour.
tiles_per_job = max(1, int(cfg.get("tiles_per_job", 1)))
batches = [
    occupied[i : i + tiles_per_job]
    for i in range(0, len(occupied), tiles_per_job)
]

# Where the segment jobs write. When the tile is already within the label
# chunk cap, they can write straight into the label group's level 0 and the
# merge relabels it in place -- one full write of the volume less than
# staging to a scratch store and copying it across. An oversized tile keeps
# the scratch store, so level 0 can still be chunked for lazy viewing.
image_store = str(Path(work_dir) / "image.zarr")
in_place = capped_output_chunks(tile_shape, LABEL_CHUNK_CAP) == tuple(
    tile_shape
)
if in_place:
    target_path, target_component = f"{image_store}/labels/{label_name}", "0"
    root = zarr.open_group(image_store, mode="a")
    parent = root.require_group("labels")
    if label_name in parent:
        del parent[label_name]
    parent.require_group(label_name)
    print("[patchworks] segmenting straight into the label group (in place)")
else:
    target_path, target_component = stage_path(work_dir, label_name), "staged"
    print(
        f"[patchworks] tile {tuple(tile_shape)} exceeds the label chunk cap "
        f"{LABEL_CHUNK_CAP}; staging to a scratch store so level 0 stays "
        "chunked for viewing"
    )
create_stage(target_path, image.shape, tile_shape, component=target_component)

Path(work_dir, label_name, "tiles.json").write_text(
    json.dumps(
        {
            "tile_shape": list(tile_shape),
            "overlap": list(overlap),
            "n_tiles": len(tiles),
            "occupied": occupied,
            "tiles_per_job": tiles_per_job,
            "batches": batches,
            "target_path": target_path,
            "target_component": target_component,
            "in_place": in_place,
        },
        indent=2,
    )
)
print(
    f"[patchworks] {len(occupied)}/{len(tiles)} tiles to segment "
    f"in {len(batches)} job(s) of up to {tiles_per_job}"
)
