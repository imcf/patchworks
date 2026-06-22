"""Cellpose 3-D segmentation of an anisotropic z-stack.

Each tile contains the full z extent. Cellpose do_3D=True runs segmentation on
xy, xz, and yz planes and takes a 3-D consensus.
"""

from functools import partial

from patchworks import auto_tile_shape_cellpose, make_local_cluster, tile_process
from patchworks.plugins.cellpose import cellpose_fn

IMAGE = "image.zarr"
OUTPUT = "labels_3d.zarr"
CHANNEL = 0
DIAMETER = 20  # pixels
ANISOTROPY = 3.0  # z-spacing / xy-spacing

fn = cellpose_fn(
    "cyto3",
    gpu=True,
    do_3D=True,
    diameter=DIAMETER,
    anisotropy=ANISOTROPY,
)

tile_fn = partial(
    auto_tile_shape_cellpose,
    diameter=DIAMETER,
    do_3D=True,
    use_gpu=True,
)

# Process-based cluster (required — in-process workers break the merge)
client, cluster = make_local_cluster(use_gpu=True)
print("Dashboard:", client.dashboard_link)

try:
    tile_process(
        IMAGE,
        fn,
        channel=CHANNEL,
        tile_shape=tile_fn,
        overlap=10,
        write_to=OUTPUT,
        progress=True,
    )
finally:
    client.close()
    cluster.close()
