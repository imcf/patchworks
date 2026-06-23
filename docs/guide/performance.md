# Performance & memory safety

`tile_process` is built so a run **adapts to whatever machine it lands on** and
can't run out of RAM/VRAM or freeze the box — without you tuning anything.

## Automatic, machine-aware concurrency

The staging step (running your `fn` once per tile to a temp store) and the
merge step are sized to the host automatically:

- **GPU** (`use_gpu=True`) → **one tile at a time**, so concurrent evaluations
  can never exhaust VRAM.
- **CPU** → as many tiles in flight as fit **80 % of available RAM** (estimated
  from the tile size), and always **leaving one core free** so the machine
  stays responsive — it never pins every core.

The RAM figure is read live via `psutil`; without it, a conservative default is
used instead of guessing high.

## Live progress dashboard (GPU runs)

A single-GPU run still gets a **Dask dashboard**: patchworks spins up a tiny
1-worker / 1-thread in-process cluster, which keeps GPU evaluations serial (no
VRAM contention) while exposing the dashboard so you can watch tiles stream
through. The URL is logged at the start of staging:

```text
INFO:patchworks._core:Dask dashboard for this run: http://127.0.0.1:8787/status
```

This needs `distributed` (and `bokeh` for the UI) installed; if they are
missing, patchworks logs a warning and falls back to the threaded scheduler
(no dashboard, same result). A cluster you start yourself
(`make_local_cluster`) is used as-is instead.

## Overriding the worker count

```python
from patchworks import tile_process

# let patchworks pick (recommended)
tile_process("scan.zarr", fn)

# or cap it yourself (staging threads + merge processes)
tile_process("scan.zarr", fn, max_workers=8)
```

`max_workers` bounds both staging and merging. A running **distributed client**
manages its own concurrency, so the override is skipped there — configure the
cluster's memory limits instead.

## Why it won't OOM or freeze

| Resource | Guard |
|----------|-------|
| RAM | concurrent tiles × tile size × overhead ≤ 80 % of available RAM |
| VRAM | GPU path runs one tile at a time |
| CPU | always leaves at least one core free |
| Disk I/O | each pyramid/stage level is streamed chunk-by-chunk; no whole volume in memory |

The staging graph itself is kept small — a single fused `map_overlap`
(halo → `fn` → trim) rather than three separate passes — and there is **no**
extra read-back of the staged data.

## Getting more speed

- `tile_shape="auto"` sizes tiles to free RAM (or VRAM with `use_gpu=True`).
- `skip_empty=True` with `estimate_empty_tiles()` skips background tiles.
- A Dask **distributed** cluster (`make_local_cluster`) parallelises across
  workers/GPUs; patchworks then defers concurrency to the cluster.

!!! note "What doesn't help here"
    The merge and relabel steps are already vectorised NumPy + SciPy (C-level)
    with no per-voxel Python loop, and the pipeline is I/O-bound — so `numba`,
    `cupy`, `arrow` and `xarray` bring essentially nothing. The real levers are
    tile size, concurrency (above) and zarr chunking.
