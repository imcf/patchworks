"""Difference-of-Gaussians blob/thread segmentation (e.g. cilia).

Blur twice at different sigmas, subtract, threshold, label the connected
components. CPU (scipy) by default.
"""

from patchworks import tile_process
from patchworks.plugins.dog import dog_label_fn

IMAGE = "image.zarr"
OUTPUT = "labels_dog.zarr"
CHANNEL = 1

fn = dog_label_fn(low_sigma=1.0, high_sigma=3.0, threshold=0.02)

tile_process(
    IMAGE,
    fn,
    channel=CHANNEL,
    tile_shape=(1, 1024, 1024),
    overlap=8,
    write_to=OUTPUT,
    progress=True,
)
