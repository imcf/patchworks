# GPU & distributed processing

## Single GPU (no distributed client)

For a single GPU, you don't need a Dask distributed client. patchworks
detects GPU usage and pins execution to a single thread, so multiple Cellpose
evals don't compete for the same CUDA context:

```python
from patchworks.plugins.cellpose import cellpose_fn
from patchworks import tile_process

fn = cellpose_fn("cyto3", gpu=True, diameter=30)
tile_process("image.zarr", fn,
             tile_shape=(1, 2048, 2048),
             overlap=20,
             use_gpu=True,         # sizes tiles against GPU VRAM
             write_to="labels.zarr",
             progress=True)
```

## Dask distributed cluster

For multi-GPU or multi-node work, use `make_local_cluster`:

```python
from patchworks import make_local_cluster, tile_process
from patchworks.plugins.cellpose import cellpose_fn

fn = cellpose_fn("cyto3", gpu=True, diameter=30)

client, cluster = make_local_cluster(use_gpu=True)  # 1 worker, processes=True
print("Dashboard:", client.dashboard_link)

try:
    tile_process("image.zarr", fn,
                 tile_shape=(1, 2048, 2048),
                 overlap=20,
                 write_to="labels.zarr",
                 progress=True)
finally:
    client.close()
    cluster.close()
```

`make_local_cluster` always uses `processes=True`. See
[Pitfalls](pitfalls.md#the-in-process-client-trap) for why in-process workers
break the label merge.

## Why `processes=True` is required

A `dask.distributed.Client(processes=False, ...)` runs the worker as a thread
in the same process as the kernel. When your segmentation function holds the
Python GIL (every PyTorch/CUDA `eval` does), the worker thread can't send
heartbeats. The scheduler declares it dead, and the merge fails:

```
FutureCancelledError: lost dependencies
```

`make_local_cluster` always uses subprocess workers to avoid this.

!!! warning "Never use `Client(processes=False)` with patchworks"
    patchworks detects in-process clients at startup and raises immediately
    with a clear error message and the fix.

## GPU memory sizing

When `use_gpu=True`, patchworks queries free GPU VRAM via `nvidia-ml-py`
(install: `pip install "patchworks[gpu]"`). Tile size is set so each tile
uses at most half the available VRAM.

Without `nvidia-ml-py`, a conservative 8 GiB default is used with a warning.
Install it for accurate sizing on large-VRAM cards (A100, H100):

```bash
pip install "patchworks[gpu]"
```
