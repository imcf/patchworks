# Performance, GPUs and empty tiles

`tile_process` sizes itself to the machine it runs on: concurrency from the
CPUs and memory **this job** may use (SLURM allocation, cgroup limit), not
the node's totals; one tile at a time on a GPU; tiles sized to free memory
or VRAM with `tile_shape="auto"`. You rarely need to tune anything.

## Plan before running

```python
plan = tile_process("scan.zarr", fn, tile_shape=(16, 1024, 1024),
                    skip_empty=True, dry_run=True, plan_sample=3)
plan["tiles_with_signal"], plan["estimated_seconds"] / 3600
```

`dry_run=True` reports the tiles, how many hold signal, memory and output
size without segmenting anything; `plan_sample=3` times the method on three
real tiles and extrapolates. From the command line: `patchworks segment
... --plan --plan-sample 3`.

## Skip empty tiles

Microscopy volumes are often mostly background. `skip_empty=True` returns
empty labels for a tile without signal instead of running the method on it,
and the merge never writes it:

```python
from patchworks import estimate_empty_tiles, tile_process

info = estimate_empty_tiles("scan.zarr", tile_shape=(16, 1024, 1024))
print(f"{info['empty_fraction']:.0%} of tiles are background")
tile_process("scan.zarr", fn, tile_shape=(16, 1024, 1024), skip_empty=True)
```

The threshold is Otsu's unless you give `empty_threshold=`.
`estimate_empty_tiles` is a quick preview from each tile's centre; the
exact decision (what the cluster workflow uses) comes from a max-pooled
occupancy map: `build_occupancy_map` and `tile_occupancy`.

## GPUs

- **One GPU**: `use_gpu=True` runs one tile at a time and sizes tiles to
  the free VRAM (`pip install "patchworks[gpu]"` for an exact reading).
- **Several GPUs on one node**: `gpus=4` runs one worker per GPU (Linux):

  ```python
  tile_process("scan.zarr", fn, tile_shape=(16, 1024, 1024), use_gpu=True, gpus=4)
  ```

- **Many nodes**: the [cluster workflow](snakemake.md) runs tile batches
  as separate SLURM jobs.
- **A shared GPU**: an out-of-memory error is retried after freeing this
  process' GPU memory, rather than moving the tile to the CPU.
- **A Dask cluster**: use `make_local_cluster(use_gpu=True)`, never
  `Client(processes=False)`. A model holding the GIL starves an in-process
  worker and the run fails with `FutureCancelledError: lost dependencies`;
  patchworks refuses such a client up front.

## Compression

Everything is written with zstd (level 1) by default: labels compress about
65×, images about 1.25×. `compression="zstd:3"` makes images ~10% smaller at
half the write speed; `"blosc"` only for old readers without zstd. Set it
per call (`to_ome_zarr(..., compression=)`), with `with
patchworks.compression("zstd:3"):`, or with the workflow's `compression:`.
