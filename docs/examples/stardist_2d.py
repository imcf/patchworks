"""StarDist 2-D segmentation of a 3-D z-stack.

Each z-slice is processed as a 2-D tile. StarDist needs more overlap context
than Cellpose (32 voxels recommended for the default 2D_versatile_fluo model).
"""
import numpy as np
from stardist.models import StarDist2D

from blockbuster import tile_process

IMAGE = "image.zarr"
OUTPUT = "labels_sd.zarr"
CHANNEL = 0

# Load model once (outside fn to avoid re-downloading on every call)
_model = StarDist2D.from_pretrained("2D_versatile_fluo")


def stardist_fn(tile: np.ndarray) -> np.ndarray:
    """Normalise + segment one 2-D tile with StarDist."""
    if tile.ndim == 3 and tile.shape[0] == 1:
        img = tile[0]
        squeeze = True
    else:
        img = tile
        squeeze = False
    norm = img.astype("float32")
    if norm.max() > 0:
        norm = norm / norm.max()
    labels, _ = _model.predict_instances(norm)
    labels = labels.astype("int32")
    return labels[np.newaxis] if squeeze else labels


tile_process(
    IMAGE, stardist_fn,
    channel=CHANNEL,
    tile_shape=(1, 1024, 1024),
    overlap=32,
    write_to=OUTPUT,
    progress=True,
)
