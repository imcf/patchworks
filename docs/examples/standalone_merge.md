# Standalone merge

Use `merge_tile_labels` when you already have per-tile labels from your own
pipeline and only need the boundary-stitching step.

## From a dask array

```python
import dask.array as da
import numpy as np
from patchworks import merge_tile_labels

# Your own tiling + segmentation
image = da.from_zarr("image.zarr").rechunk((1, 1024, 1024))


def my_fn(tile: np.ndarray) -> np.ndarray:
    ...  # your segmentation code
    return labels.astype("int32")


labeled = image.map_blocks(
    my_fn,
    dtype="int32",
    meta=np.empty((0,) * image.ndim, dtype="int32"),
)

# Merge: boundary scan → connected components → relabel
merged = merge_tile_labels(labeled, write_to="labels.zarr", progress=True)
```

## From a pre-written zarr

If your pipeline already wrote per-tile labels to zarr:

```python
from patchworks import merge_tile_labels

merged = merge_tile_labels(
    "my_staged_labels.zarr",
    input_component="raw_labels",  # component name inside the zarr
    write_to="merged.zarr",
    sequential_labels=True,  # renumber to 1..N
)
```

## With overlap halos

If your tiles were computed with `da.overlap.overlap(depth=20)`, trim the
halos before merging:

```python
from patchworks import merge_tile_labels

# labeled was computed with da.overlap.overlap(depth=20)
merged = merge_tile_labels(labeled, write_to="labels.zarr", overlap=20)
```

## Integration with other frameworks

patchworks's merge step is framework-agnostic. Any pipeline that produces a
dask array of integer labels (one value per voxel, distinct per tile) can
use `merge_tile_labels`:

```python
# zarr → your pipeline → per-tile labels
labeled = your_pipeline(da.from_zarr("image.zarr"))  # dask.array.Array

from patchworks import merge_tile_labels

merged = merge_tile_labels(labeled, write_to="final.zarr")
```
