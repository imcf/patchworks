"""Custom segmentation function — anything goes.

blockbuster doesn't care what's inside fn. Here's a more elaborate example
using scipy + skimage with preprocessing.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import remove_small_objects

from blockbuster import tile_process


def my_fn(tile: np.ndarray) -> np.ndarray:
    """Gaussian blur → Otsu → connected components → remove small objects."""
    # Work on float32 to avoid overflow in gaussian_filter
    img = tile.astype("float32")

    # Smooth
    smoothed = gaussian_filter(img, sigma=1.5)

    # Threshold
    thr = threshold_otsu(smoothed)
    binary = smoothed > thr

    # Label connected components
    labeled = label(binary).astype("int32")

    # Remove small objects (noise)
    mask = remove_small_objects(binary, min_size=100)
    labeled[~mask] = 0

    return labeled


# Pass any array or zarr path
result = tile_process(
    "image.zarr",
    my_fn,
    tile_shape=(1, 512, 512),
    overlap=16,
    compute=True,
    progress=True,
)

print(f"Found {result.max()} objects")
