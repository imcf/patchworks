# Custom segmentation function

patchworks is completely agnostic to what happens inside `fn`. Here are
examples with different tools and preprocessing steps.

## Otsu threshold + skimage

```python
import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label
from patchworks import tile_process


def threshold_fn(tile: np.ndarray) -> np.ndarray:
    thr = threshold_otsu(tile)
    return label(tile > thr).astype("int32")


result = tile_process("image.zarr", threshold_fn, compute=True)
```

## Gaussian + morphological operations

```python
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.morphology import remove_small_objects
from skimage.measure import label
from patchworks import tile_process


def smooth_and_label(tile: np.ndarray) -> np.ndarray:
    smoothed = gaussian_filter(tile.astype("float32"), sigma=1.5)
    binary = smoothed > smoothed.mean()
    cleaned = remove_small_objects(binary, min_size=100)
    return label(cleaned).astype("int32")


tile_process(
    "image.zarr",
    smooth_and_label,
    tile_shape=(1, 512, 512),
    overlap=16,
    write_to="labels.zarr",
    progress=True,
)
```

## PyTorch model

```python
import numpy as np
import torch
from patchworks import tile_process

# Load once, outside the function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MySegmentationModel().to(device).eval()


@torch.no_grad()
def torch_fn(tile: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(tile.astype("float32")).unsqueeze(0).unsqueeze(0).to(device)
    logits = model(t)  # shape: (1, n_classes, z, y, x)
    pred = logits.argmax(1).squeeze(0).cpu().numpy()
    return pred.astype("int32")


tile_process(
    "image.zarr",
    torch_fn,
    tile_shape=(1, 512, 512),
    use_gpu=True,
    write_to="labels.zarr",
    progress=True,
)
```

## Any array input, not just zarr

```python
import dask.array as da
import numpy as np
from patchworks import tile_process

# From any array-like source
arr = da.from_array(my_numpy_array, chunks=(1, 1024, 1024))
result = tile_process(arr, my_fn, compute=True)

# From tifffile
import tifffile
import dask.array as da

arr = da.from_array(tifffile.imread("image.tif", aszarr=True))
result = tile_process(arr, my_fn, compute=True)
```
