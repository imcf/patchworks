"""Snakemake script: plan the tiles, measure what every tile shares (empty
threshold, intensity range, thresholds), and create the store they go to."""

import json
import shutil
from functools import partial
from pathlib import Path

import numpy as np
import zarr

from patchworks import (
    set_compression,
    auto_empty_threshold,
    auto_tile_shape,
    auto_tile_shape_cellpose,
    block_for_tile,
    build_occupancy_map,
    capped_output_chunks,
    create_stage,
    intensity_range,
    normalize_overlap,
    spatial_tiles,
    tile_occupancy,
)

from _pw import (
    halo_path,
    check_seed_labels,
    open_image,
    stage_path,
    start_log,
    image_wide_kwargs,
    tile_channels,
    uses_image_range,
    validate_config,
)

# Largest label chunk, so a viewer can page the labels lazily.
LABEL_CHUNK_CAP = (16, 1024, 1024)

start_log(snakemake.log[0])  # noqa: F821
set_compression(snakemake.config.get("compression", "zstd"))  # noqa: F821
cfg = snakemake.config  # noqa: F821
work_dir = cfg["work_dir"]
label_name = cfg.get("label_name", "labels")
Path(work_dir, label_name).mkdir(parents=True, exist_ok=True)
image = open_image(work_dir, cfg["channel"], cfg["level"])

# Fail on this cheap CPU job, not in the first GPU job.
validate_config(cfg)
check_seed_labels(work_dir, cfg)

method = cfg.get("method", "cellpose")
ts = cfg.get("tile_shape", "auto")
if ts == "auto":
    # This is a CPU job: the segment GPU's memory comes from the config.
    gpu_gb = cfg.get("gpu_memory_gb")
    gpu_bytes = int(gpu_gb * 1024**3) if gpu_gb else None
    # A second channel doubles a tile's bytes.
    n_channels = tile_channels(cfg)
    if method == "cellpose":
        cp = cfg["cellpose"]
        # Cellpose resizes z by the anisotropy: budget for the resized tile.
        anisotropy = cp.get("anisotropy")
        if anisotropy is None and cp.get("do_3D", False):
            from patchworks.plugins.cellpose import cellpose_anisotropy
            from patchworks.plugins.ome_zarr import read_pixel_size

            anisotropy = cellpose_anisotropy(
                read_pixel_size(
                    str(Path(work_dir) / "image.zarr"),
                    level=int(cfg.get("level", 0)),
                )
            )
        sizer = partial(
            auto_tile_shape_cellpose,
            do_3D=cp.get("do_3D", False),
            use_gpu=cp.get("gpu", True),
            diameter=cp.get("diameter"),
            gpu_memory=gpu_bytes,
            n_channels=n_channels,
            anisotropy=anisotropy,
        )
    else:
        sizer = partial(
            auto_tile_shape,
            use_gpu=gpu_bytes is not None,
            gpu_memory=gpu_bytes,
            n_channels=n_channels,
        )
    tile_shape = tuple(sizer(image.shape, image.dtype))
else:
    tile_shape = tuple(ts)

if len(tile_shape) >= 3:
    n_z = image.shape[0]
    if tile_shape[0] >= n_z:
        print("[patchworks] z regime: whole-z tiles (no z-boundary stitching)")
    else:
        print(
            f"[patchworks] z regime: tiled in z ({tile_shape[0]} of {n_z} "
            "planes); objects spanning z boundaries are stitched by the merge"
        )

# A halo as wide as the tile re-segments the neighbours: refuse it.
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
# How much more each tile reads than it keeps (the halo, clipped to the image).
read = np.prod(
    [min(t + 2 * o, s) for t, o, s in zip(tile_shape, overlap, image.shape)]
)
amplification = read / np.prod(tile_shape)
print(f"[patchworks] halo read amplification: {amplification:.2f}x")

tiles = spatial_tiles(image.shape, tile_shape)
occupied = list(range(len(tiles)))
threshold = None
if cfg.get("skip_empty", True):
    # Per-brick maxima of the whole image (built once, shared by every
    # config): a tile is empty when none of its bricks reaches the threshold.
    build_occupancy_map(
        str(Path(work_dir) / "image.zarr"),
        level=cfg["level"],
        block=block_for_tile(tile_shape),
    )
    threshold = cfg.get("empty_threshold")
    if threshold is None:
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

# Several tiles per job share one model load.
tiles_per_job = max(1, int(cfg.get("tiles_per_job", 1)))
batches = [
    occupied[i : i + tiles_per_job]
    for i in range(0, len(occupied), tiles_per_job)
]

# Tiles within the chunk cap are written straight into labels/<name>/0 and
# merged in place; larger ones go to a scratch store, rechunked by the merge.
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
create_stage(
    target_path,
    image.shape,
    tile_shape,
    component=target_component,
    zarr_format=(
        zarr.open_group(image_store, mode="r").metadata.zarr_format
        if in_place
        else None
    ),
)
# One intensity range and one set of thresholds for every tile, measured on
# the tiles to segment: decided per tile, they change at every seam.
image_range = None
if uses_image_range(cfg):
    chans = [cfg["channel"]]
    if cfg.get("nuclei_channel") is not None:
        chans.append(cfg["nuclei_channel"])
    image_range = [
        list(r)
        for r in intensity_range(
            Path(work_dir) / "image.zarr",
            chans,
            level=cfg["level"],
            regions=[tiles[i] for i in occupied],
        )
    ]
    print(
        f"[patchworks] intensity range (1-99%) of channels {chans}: {image_range}"
    )

thresholds = image_wide_kwargs(
    cfg, str(Path(work_dir) / "image.zarr"), [tiles[i] for i in occupied]
)
if thresholds:
    print(f"[patchworks] image-wide thresholds for every tile: {thresholds}")

# Halo strips from an earlier run describe tiles that no longer exist.
shutil.rmtree(halo_path(work_dir, label_name), ignore_errors=True)

Path(work_dir, label_name, "tiles.json").write_text(
    json.dumps(
        {
            "tile_shape": list(tile_shape),
            "overlap": list(overlap),
            "n_tiles": len(tiles),
            "occupied": occupied,
            "empty_threshold": None if threshold is None else float(threshold),
            "intensity_range": image_range,
            "kwargs_overrides": thresholds,
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
    + ("" if threshold is None else f"; empty below {float(threshold):g}")
)
