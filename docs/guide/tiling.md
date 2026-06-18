# Tiling strategy

## Why tiles?

Segmentation tools run on NumPy arrays in RAM. A 250 GB microscopy image can't
fit in RAM (and wouldn't fit on a GPU either). Tiling solves this: split the
image into manageable pieces, process each independently, stitch the results.

blockbuster uses **dask** to manage the tiling. Each dask chunk becomes one tile.
Tiles are streamed one at a time through your function and written to disk —
peak RAM during segmentation is approximately one tile's worth of data.

## Choosing a tile size

The right tile size depends on:
- Your available RAM (or GPU VRAM)
- The minimum context your segmentation method needs (objects should fit fully
  inside a tile, or you need overlap)
- Overhead: very small tiles mean many boundary merges; very large tiles may OOM

### Fixed tile shape

```python
tile_process("image.zarr", fn, tile_shape=(1, 1024, 1024))
```

For 2-D methods (e.g. 2-D Cellpose), `z=1` makes each tile one z-slice.
For 3-D methods, use the full z extent, e.g. `tile_shape=(120, 512, 512)`.

### Auto sizing

```python
# General method (cubic tiles that fit in available RAM)
tile_process("image.zarr", fn, tile_shape="auto")

# GPU sizing (uses GPU VRAM instead of host RAM)
tile_process("image.zarr", fn, tile_shape="auto", use_gpu=True)
```

### Callable sizing (Cellpose example)

Cellpose's memory usage scales as `20× raw input bytes + 2 GB model`. The
`auto_tile_shape_cellpose` function accounts for this:

```python
from functools import partial
from blockbuster import auto_tile_shape_cellpose, tile_process
from blockbuster.plugins.cellpose import cellpose_fn

fn = cellpose_fn("cyto3", gpu=True, diameter=30)
tile_fn = partial(auto_tile_shape_cellpose, diameter=30, use_gpu=True)

tile_process("image.zarr", fn, tile_shape=tile_fn)
```

The callable is called with `(shape, dtype)` at runtime, after the image is
loaded — useful when you don't know the image shape in advance.

## Overlap

Methods that need spatial context (Cellpose, StarDist, U-Net) produce wrong
results near tile edges: objects at the boundary are cut off. Overlap fixes this
by expanding each tile by `overlap` voxels on every side.

```
No overlap:        With overlap=20:
┌──────────┐      ┌──────────────────┐
│          │      │  ░░░░░░░░░░░░░░  │
│  tile A  │      │░░│              │░░│
│          │      │░░│  tile A core │░░│
└──────────┘      │░░│              │░░│
                  │  ░░░░░░░░░░░░░░  │
                  └──────────────────┘
                       halo (trimmed before merge)
```

!!! tip "How much overlap?"
    Use the diameter of the largest object you expect. For Cellpose with
    `diameter=30`, an overlap of 20-30 voxels is typically sufficient.
    For StarDist 2D_versatile_fluo, 32 voxels is recommended.

## Overlap and tile size interaction

!!! warning
    The overlap depth must be smaller than the tile dimension. blockbuster
    automatically clips the depth per axis, so z-tiles of size 1 (typical in
    2-D Cellpose mode) get `depth=0` in z even if you pass `overlap=20`.

    Axes that are too small for the requested overlap simply get a smaller halo.
