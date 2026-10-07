# Cellpose 3-D

`do_3D=True` segments the xy, xz and yz planes and combines them, so each
tile holds a block of z as well. It is slower than 2-D: skip empty tiles.

```python
from functools import partial
from patchworks import auto_tile_shape_cellpose, tile_process
from patchworks.plugins.cellpose import cellpose_fn
from patchworks.plugins.ome_zarr import read_pixel_size

fn = cellpose_fn(
    "cyto3",
    gpu=True,
    do_3D=True,
    diameter=20,
    voxel_size=read_pixel_size("image.zarr"),  # anisotropy = z / lateral
)

tile_process(
    "image.zarr",
    fn,
    channel=0,
    tile_shape=partial(auto_tile_shape_cellpose, do_3D=True, use_gpu=True, diameter=20),
    overlap=[4, 20, 20],
    skip_empty=True,
)
```

Without `voxel_size` (or `anisotropy=`), Cellpose assumes isotropic voxels
and fragments objects along z. The cluster workflow passes the calibration
by itself ([Configure](../guide/snakemake.md#2-configure)).
