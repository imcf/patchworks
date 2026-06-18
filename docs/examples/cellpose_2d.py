"""Cellpose 2-D segmentation of a 3-D z-stack.

Each z-slice is processed independently. Tiles overlap by 20 voxels so cells
near tile boundaries are fully visible to Cellpose.
"""
from functools import partial

from blockbuster import auto_tile_shape_cellpose, tile_process
from blockbuster.plugins.cellpose import cellpose_fn

IMAGE = "image.zarr"
OUTPUT = "labels.zarr"
CHANNEL = 0
DIAMETER = 30  # pixels

fn = cellpose_fn("cyto3", gpu=True, diameter=DIAMETER)

# Auto-size tiles based on GPU VRAM (z=1 per tile for 2-D Cellpose)
tile_fn = partial(auto_tile_shape_cellpose, diameter=DIAMETER, use_gpu=True)

tile_process(
    IMAGE, fn,
    channel=CHANNEL,
    tile_shape=tile_fn,
    overlap=20,
    skip_empty=True,        # skip background slices
    write_to=OUTPUT,
    progress=True,
)
