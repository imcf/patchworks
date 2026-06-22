# Cellpose 3-D

Run Cellpose in full 3-D mode (`do_3D=True`). Each tile contains the full
z extent and a sub-region in y/x. Cellpose segments on all three orthogonal
plane orientations and takes a 3-D consensus.

!!! warning "Slow"
    3-D mask reconstruction in Cellpose is CPU-bound. Expect minutes per tile
    even with a fast GPU. Use `skip_empty=True` to skip background tiles.

## Code

```python
from functools import partial
from patchworks import auto_tile_shape_cellpose, make_local_cluster, tile_process
from patchworks.plugins.cellpose import cellpose_fn

IMAGE = "image.zarr"
OUTPUT = "labels_3d.zarr"
CHANNEL = 0
DIAMETER = 20     # pixels
ANISOTROPY = 3.0  # z_spacing / xy_spacing

fn = cellpose_fn(
    "cyto3",
    gpu=True,
    do_3D=True,
    diameter=DIAMETER,
    anisotropy=ANISOTROPY,
)

# Tile shape: full z, xy tiled for memory
# The 3× plane orientation overhead is accounted for automatically
tile_fn = partial(
    auto_tile_shape_cellpose,
    do_3D=True,
    use_gpu=True,
    diameter=DIAMETER,
)

# Use a process-based cluster for distributed work
# (in-process clients break the label merge — see Pitfalls)
client, cluster = make_local_cluster(use_gpu=True)
print("Dashboard:", client.dashboard_link)

try:
    tile_process(
        IMAGE, fn,
        channel=CHANNEL,
        tile_shape=tile_fn,
        overlap=10,
        skip_empty=True,
        write_to=OUTPUT,
        progress=True,
    )
finally:
    client.close()
    cluster.close()
```

## Memory notes

In `do_3D=True` mode, each tile has shape `(z_full, y_tile, x_tile)`.
Cellpose internally runs 2-D segmentation on xy, xz, and yz planes —
3× the raw tile bytes before the model overhead.

`auto_tile_shape_cellpose(do_3D=True)` accounts for this 3× factor when
sizing the y/x dimensions.
