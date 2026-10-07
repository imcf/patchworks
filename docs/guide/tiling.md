# Tiling

The image is cut into tiles, each tile is read with a halo (`overlap`) of its
neighbours for context, segmented, and the halo is trimmed off before the
tiles are stitched. Peak memory is about one tile.

## Tile size

```python
tile_process("image.zarr", fn, tile_shape=(1, 1024, 1024))   # 2-D method: one plane per tile
tile_process("image.zarr", fn, tile_shape=(120, 512, 512))   # 3-D method
tile_process("image.zarr", fn, tile_shape="auto")            # fits free RAM (or VRAM with use_gpu=True)
```

Larger tiles mean fewer seams; too large runs out of memory. For Cellpose,
`auto_tile_shape_cellpose` knows the model's memory needs:

```python
from functools import partial
from patchworks import auto_tile_shape_cellpose

tile_process("image.zarr", fn, tile_shape=partial(auto_tile_shape_cellpose, diameter=30, use_gpu=True))
```

Pass `n_channels=2` to it when each tile carries two channels (a cyto stain
with a nuclei channel), or the first tile runs out of memory.

## Overlap

Methods that need context (Cellpose, StarDist, U-Nets) get objects wrong at a
tile edge. The halo gives them the context; use about the diameter of the
largest object.

On anisotropic stacks give one value per axis: a scalar applies to every
axis, and a 30-plane z-halo on a 16-plane tile reads the neighbours entirely
for nothing (5× the voxels, against 1.7× for `[4, 30, 30]`).
`auto_overlap` derives it from the voxel size:

```python
from patchworks import auto_overlap

auto_overlap(30, voxel_size=(2.0, 0.1, 0.1))   # -> (2, 30, 30)
```

The halo is clipped to the tile size on each axis; the cluster workflow
rejects `overlap >= tile_shape` and logs the read amplification up front.

To measure rather than guess, see
[Choosing the overlap from the data](merging.md#choosing-the-overlap-from-the-data).
