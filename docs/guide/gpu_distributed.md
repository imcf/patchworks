# GPU & distributed processing

## Single GPU (no distributed client)

For a single GPU, you don't need a Dask distributed client. patchworks
detects GPU usage and pins execution to a single thread, so multiple Cellpose
evals don't compete for the same CUDA context:

```python
from patchworks.plugins.cellpose import cellpose_fn
from patchworks import tile_process

fn = cellpose_fn("cyto3", gpu=True, diameter=30)
tile_process(
    "image.zarr",
    fn,
    tile_shape=(1, 2048, 2048),
    overlap=20,
    use_gpu=True,  # sizes tiles against GPU VRAM
    write_to="labels.zarr",
    progress=True,
)
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
    tile_process(
        "image.zarr",
        fn,
        tile_shape=(1, 2048, 2048),
        overlap=20,
        write_to="labels.zarr",
        progress=True,
    )
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

```python
FutureCancelledError: lost dependencies
```

`make_local_cluster` always uses subprocess workers to avoid this.

!!! warning "Never use `Client(processes=False)` with patchworks"
    patchworks detects in-process clients at startup and raises immediately
    with a clear error message and the fix.

## GPU memory sizing

When `use_gpu=True`, patchworks queries free GPU VRAM via `nvidia-ml-py`
(install: `pip install "patchworks[gpu]"`) and keeps 20% of it as headroom,
because `info.free` is a point-in-time reading of a device you usually do not
own outright. `auto_tile_shape` then sizes each tile to at most half of that
budget; `auto_tile_shape_cellpose` uses Cellpose's own memory model instead
(roughly 20× the raw tile bytes, plus ~2 GiB for the model).

"Raw tile bytes" counts every channel a tile carries, so pass `n_channels=2`
when feeding Cellpose a cyto+nuclei pair — see
[Multi-channel tiles](tiling.md#multi-channel-tiles). The Snakemake workflow
does this for you whenever `nuclei_channel` is set.

The device is resolved from `CUDA_VISIBLE_DEVICES`. This matters on
multi-GPU nodes: NVML enumerates **every** GPU regardless of `--gres=gpu:1`,
so querying index 0 unconditionally would read a different card's free memory
than the one your job was granted.

Without `nvidia-ml-py`, a conservative 8 GiB default is used with a warning.
Install it for accurate sizing on large-VRAM cards (A100, H100):

```bash
pip install "patchworks[gpu]"
```

!!! tip "Sizing for a GPU you can't see"
    The Snakemake workflow plans tiles in `prepare`, which runs on a CPU
    node, so no GPU can be queried there. Set `gpu_memory_gb` in the config
    to the segment GPU's VRAM (e.g. `24` for an RTX 4090) instead of relying
    on the fallback.

## Surviving a shared GPU

An out-of-memory error on a shared device is often transient — a co-tenant
job grew, not a tile that doesn't fit. patchworks retries on the GPU with a
backoff rather than falling back to the CPU, where a single tile can take
over an hour. Before each backoff it releases its own device memory
(including any cached Cellpose model), since holding that is exactly what
starves the other job. Both torch and cupy OOM errors are recognised.
