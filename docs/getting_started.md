# Getting Started

## Installation

blockbuster can be installed from PyPI on all operating systems, for Python ≥ 3.9.

!!! tip "Virtual environment (recommended)"
    We recommend creating a dedicated environment:

    ```bash
    conda create -n blockbuster python=3.12
    conda activate blockbuster
    ```
    or with pixi:
    ```bash
    pixi init blockbuster && cd blockbuster
    pixi add python=3.12
    ```

=== "Minimal"

    ```bash
    pip install blockbuster
    ```

=== "With GPU VRAM sizing"

    ```bash
    pip install "blockbuster[gpu]"
    ```
    Installs `nvidia-ml-py` to query free GPU VRAM when auto-sizing tiles.

=== "With Cellpose plugin"

    ```bash
    pip install "blockbuster[cellpose]"
    ```

=== "Everything"

    ```bash
    pip install "blockbuster[all]"
    ```

---

## The one function you need

```python
from blockbuster import tile_process

result = tile_process(image, fn)
```

`tile_process(image, fn)` splits `image` into tiles, runs `fn` on each tile,
and returns a globally consistent label array.

- **`image`** — a dask array or a path to an OME-ZARR store
- **`fn`** — any callable `(ndarray) -> ndarray` returning integer labels

---

## Step 1: write your function

blockbuster is method-agnostic. Your function receives a NumPy array (one tile)
and must return an integer label array of the same shape:

```python
import numpy as np

def my_fn(tile: np.ndarray) -> np.ndarray:
    from skimage.filters import threshold_otsu
    from skimage.measure import label
    binary = tile > threshold_otsu(tile)
    return label(binary).astype("int32")
```

The function is called independently on every tile. blockbuster ensures that
objects spanning tile boundaries are merged into a single label.

---

## Step 2: run it

=== "From a zarr path"

    ```python
    from blockbuster import tile_process

    result = tile_process("image.zarr", my_fn, compute=True)
    print(result.shape)   # (z, y, x)
    print(result.max())   # number of objects found
    ```

=== "From a dask array"

    ```python
    import dask.array as da
    from blockbuster import tile_process

    arr = da.from_zarr("image.zarr")
    result = tile_process(arr, my_fn, compute=True)
    ```

=== "Stream to zarr (recommended for large images)"

    ```python
    from blockbuster import tile_process

    tile_process(
        "image.zarr", my_fn,
        write_to="labels.zarr",
        progress=True,
    )
    ```
    The output is written tile by tile — peak RAM is one tile, not the whole image.

---

## Set the tile size

=== "Fixed tile shape"

    ```python
    result = tile_process("image.zarr", my_fn,
                          tile_shape=(1, 1024, 1024))
    ```

=== "Auto from available memory"

    ```python
    result = tile_process("image.zarr", my_fn,
                          tile_shape="auto",
                          use_gpu=True)  # sizes against GPU VRAM
    ```

=== "Callable (computed at runtime)"

    ```python
    from functools import partial
    from blockbuster import auto_tile_shape_cellpose, tile_process

    tile_fn = partial(auto_tile_shape_cellpose, diameter=30, use_gpu=True)
    result = tile_process("image.zarr", my_fn, tile_shape=tile_fn)
    ```

---

## Add overlap

Methods like Cellpose and StarDist need spatial context at tile boundaries.
Use `overlap` (in voxels) so boundary objects are fully visible:

```python
result = tile_process(
    "image.zarr", my_fn,
    tile_shape=(1, 2048, 2048),
    overlap=20,  # 20-voxel halo on every side
)
```

!!! info "How overlap works"
    Each tile is expanded by `overlap` voxels on every side before calling `fn`.
    The halo is trimmed before merging — the final output has the original shape.
    Objects near boundaries have enough context to be segmented correctly.

---

## Use Cellpose

```python
from blockbuster import tile_process
from blockbuster.plugins.cellpose import cellpose_fn

fn = cellpose_fn("cyto3", gpu=True, diameter=30)

tile_process(
    "image.zarr", fn,
    channel=0,
    tile_shape=(1, 2048, 2048),
    overlap=20,
    write_to="labels.zarr",
    progress=True,
)
```

See the [Cellpose 2-D example](examples/cellpose_2d.md) for the full workflow.

---

## What's next?

- [Tiling strategy](guide/tiling.md) — how tiles are sized and overlapped
- [Merging labels](guide/merging.md) — how cross-boundary labels become one
- [Skip empty tiles](guide/skip_empty.md) — speed up sparse volumes
- [GPU & distributed](guide/gpu_distributed.md) — Dask clusters for GPU workloads
- [Common pitfalls](guide/pitfalls.md) — GIL traps, recompute traps, memory traps
