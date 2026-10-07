# Custom segmentation function

Any function from an image tile to labels works. You write it, and patchworks
does everything around it: tiling, halos, empty tiles, stitching, relabelling,
resuming, tables. Nothing in the package needs changing.

## The contract

```python
labels = segment(tile, **kwargs)
```

- `tile` is a NumPy array of one tile, halo included: `(z, y, x)` or `(y, x)`,
  in the image's dtype. With `nuclei_channel` set, it is `(2, z, y, x)`.
- Return integer labels of the **same shape**, `0` for background. They only
  need to be unique within the tile.

```python
# my_seg.py
import numpy as np
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label


def segment(tile: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    smooth = gaussian(tile, sigma=sigma, preserve_range=True)
    thr = threshold_otsu(smooth) if smooth.max() > smooth.min() else np.inf
    return label(smooth > thr).astype("int32")
```

```python
from patchworks import tile_process
from my_seg import segment

tile_process("image.zarr", segment)
```

Ready-made functions: `cellpose_fn`, `dog_label_fn` (spots, cilia), and for
cells from a membrane stain the watershed and PlantSeg plugins
([Cells from a membrane stain](membrane_cells.md)).

## Load models once

One process segments many tiles, so cache the model:

```python
from functools import cache


@cache
def _model():
    from stardist.models import StarDist3D
    return StarDist3D.from_pretrained("3D_demo")


def segment(tile, prob_thresh=0.5):
    from csbdeep.utils import normalize
    labels, _ = _model().predict_instances(normalize(tile), prob_thresh=prob_thresh)
    return labels.astype("int32")
```

## Physical units

Declare a `voxel_size` parameter and the workflow passes the image's own
calibration, `{"z": .., "y": .., "x": ..}` in µm:

```python
def segment(tile, *, voxel_size=None, min_diameter_um=5.0):
    min_px = min_diameter_um / voxel_size["x"]
    ...
```

## Growing labels

`dilate_labels` wraps any function and grows each label by a few pixels
before the tiles are stitched (`overlap` must cover it):

```python
from patchworks import dilate_labels

fn = dilate_labels(segment, iterations=2)   # use_gpu=True needs cupy
```

On the cluster: `dilate: 2` in the config.

## Test on one tile

```python
from patchworks import load_ome_zarr

tile = load_ome_zarr("results/image.zarr", channel=0)[:, :512, :512].compute()
out = segment(tile)
assert out.shape == tile.shape and out.dtype.kind in "iu"
```

## On the cluster

```yaml
method: "custom"
label_name: "my_labels"
custom:
  module: "my_seg"       # importable on the compute nodes
  function: "segment"
  kwargs:                # checked against the signature in `prepare`
    sigma: 1.5
```

Getting the module onto the compute nodes:
[Your own segmentation function](snakemake.md#your-own-segmentation-function).
