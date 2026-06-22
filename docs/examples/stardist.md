# StarDist

StarDist 2-D on a z-stack: each slice is segmented independently.
StarDist needs more overlap than Cellpose — 32 voxels for the default
`2D_versatile_fluo` model.

## Code

```python
import numpy as np
from stardist.models import StarDist2D
from patchworks import tile_process

IMAGE = "image.zarr"
OUTPUT = "labels_sd.zarr"

# Load model once (not inside fn — would re-download every call)
model = StarDist2D.from_pretrained("2D_versatile_fluo")


def stardist_fn(tile: np.ndarray) -> np.ndarray:
    squeeze = tile.ndim == 3 and tile.shape[0] == 1
    img = tile[0] if squeeze else tile

    # StarDist expects normalised float32
    norm = img.astype("float32")
    if norm.max() > 0:
        norm = norm / norm.max()

    labels, _ = model.predict_instances(norm)
    labels = labels.astype("int32")
    return labels[np.newaxis] if squeeze else labels


tile_process(
    IMAGE,
    stardist_fn,
    channel=0,
    tile_shape=(1, 1024, 1024),
    overlap=32,  # StarDist receptive field is larger than Cellpose
    write_to=OUTPUT,
    progress=True,
)
```

!!! tip "Model loading"
    Load the model **outside** the `fn` closure. If you load it inside,
    it will be re-initialised (and potentially re-downloaded) once per tile.

    For distributed execution, use `functools.partial` with a cached model:

    ```python
    from functools import lru_cache


    @lru_cache(maxsize=1)
    def _get_model():
        return StarDist2D.from_pretrained("2D_versatile_fluo")


    def stardist_fn(tile):
        model = _get_model()
        ...
    ```
