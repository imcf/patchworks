"""Difference-of-Gaussians blob/thread segmentation (e.g. cilia).

Blur twice at different sigmas, subtract, threshold, label the connected
components. CPU (scipy) by default.
"""

from patchworks import intensity_range, tile_process
from patchworks.plugins.dog import dog_label_fn

IMAGE = "image.zarr"
OUTPUT = "labels_dog.zarr"
CHANNEL = 1

# The threshold is a fraction of the channel's range, measured once.
rng = intensity_range(IMAGE, CHANNEL, percentiles=(1, 99.9))[0]
fn = dog_label_fn(
    low_sigma=1.0, high_sigma=3.0, threshold=0.05, intensity_range=rng
)

tile_process(
    IMAGE,
    fn,
    channel=CHANNEL,
    tile_shape=(1, 1024, 1024),
    overlap=8,
    write_to=OUTPUT,
    progress=True,
)
