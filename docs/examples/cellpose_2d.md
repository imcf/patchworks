# Cellpose 2-D

Segment every z-slice of a 3-D z-stack independently with Cellpose 2-D.
Each tile is one z-slice `(1, y, x)`.

## Installation

```bash
pip install "patchworks[cellpose,gpu]"
```

## Code

```python
from functools import partial
from patchworks import auto_tile_shape_cellpose, estimate_empty_tiles, tile_process
from patchworks.plugins.cellpose import cellpose_fn

IMAGE = "image.zarr"
OUTPUT = "labels.zarr"
CHANNEL = 0  # channel to segment
DIAMETER = 30  # expected cell diameter in pixels

# 1. Create the Cellpose function
fn = cellpose_fn("cyto3", gpu=True, diameter=DIAMETER)

# 2. (Optional) Preview empty tiles and pick a threshold
info = estimate_empty_tiles(IMAGE, tile_shape=(1, 2048, 2048), channel=CHANNEL)
print(f"{info['empty_fraction']:.0%} of slices are background")

# 3. Run
tile_fn = partial(auto_tile_shape_cellpose, diameter=DIAMETER, use_gpu=True)
tile_process(
    IMAGE,
    fn,
    channel=CHANNEL,
    tile_shape=tile_fn,  # auto-sized for GPU VRAM
    overlap=20,  # 20-voxel halo for boundary cells
    skip_empty=True,
    empty_threshold=info["threshold"],
    write_to=OUTPUT,
    progress=True,
)
```

## How the tile shape works

`auto_tile_shape_cellpose(do_3D=False)` sets `z=1`. Each tile is a single
2-D image `(1, y_tile, x_tile)`. Cellpose processes it as greyscale 2-D,
squeezing the singleton z axis internally.

The y/x tile size is chosen so the raw tile fits within the GPU VRAM budget
after accounting for Cellpose's 20× memory overhead and the 2 GB model weights.

## Two-channel input (Cellpose 3)

If you have a cytoplasm + nucleus image, pass `channels=[1, 2]`:

```python
fn = cellpose_fn("cyto3", gpu=True, diameter=30, channels=[1, 2])
# channels[0]=1 → use channel index 1 as cytoplasm
# channels[1]=2 → use channel index 2 as nucleus
```

## Multi-channel input (Cellpose 4)

Cellpose 4 uses `channel_axis` instead:

```python
# array shape: (z, y, x, c) with 2 channels
fn = cellpose_fn("cyto3", gpu=True, diameter=30, channel_axis=3)
```
