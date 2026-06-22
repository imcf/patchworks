# Common pitfalls

These pitfalls were discovered (and fixed) while processing 250 GB
light-sheet volumes with ~2200 GPU tiles. patchworks handles all of them
automatically, but understanding them helps you debug unexpected behavior.

---

## The in-process client trap

**Symptom:** `FutureCancelledError: lost dependencies` — GPU is barely used,
error appears minutes into the run with no obvious cause.

**Cause:** `dask.distributed.Client(processes=False, ...)` runs the worker as
a thread inside the scheduler process. Segmentation functions (Cellpose,
PyTorch) hold the Python GIL during inference. With the GIL held, the worker
thread can't send scheduler heartbeats. The scheduler declares the worker
dead, the label merge's P2P barrier drops its inputs.

**Fix:** Use a subprocess-based cluster:

```python
from patchworks import make_local_cluster
client, cluster = make_local_cluster(use_gpu=True)
```

or drop the distributed client entirely (the threaded scheduler works for
single-GPU runs — patchworks pins it to 1 thread automatically).

patchworks detects in-process clients at startup and raises immediately:

```
RuntimeError: Active Dask client uses an in-process worker (processes=False).
This breaks the label merge when fn holds the GIL. Use a process-based
cluster instead:
    from patchworks import make_local_cluster
    client, cluster = make_local_cluster(use_gpu=True)
```

---

## The 3-4× fn recompute trap

**Symptom:** Cellpose is called 3-4× per tile instead of once. A 9-tile run
triggers 33 segmentation calls. Verified by counting calls.

**Cause:** The merge step (boundary scan, connected components, relabel)
reads the label array several times. If the label array is a lazy dask graph
that includes the segmentation call, each read re-evaluates the full pipeline
— including calling your function again.

**Fix:** patchworks always **stages** first: it writes each tile's labels to
a temporary zarr exactly once, then the zarr-native merge reads concrete
on-disk data. Your function is called exactly once per tile, always. There is
no configuration needed — and no way to accidentally disable it.

```python
# fn runs exactly once per tile
tile_process("image.zarr", fn, write_to="labels.zarr")
```

The temp stage store is deleted after a successful merge (pass
`keep_stage=True` to keep it for debugging or resuming).

---

## The O(n²) sequential relabelling trap

**Symptom:** Computation hangs for hours before any tiles are processed.
Dask dashboard shows the graph construction itself taking minutes at 1000+
tiles.

**Cause:**
`dask_image.ndmeasure.merge_labels_across_chunk_boundaries(produce_sequential_labels=True)`
builds a dask task graph that is O(n_tiles²). At 64 tiles: 54 seconds. At
2200 tiles: several hours — just for graph construction, before any data is
even read.

**Fix:** patchworks does not use this function. The zarr-native merge is
O(face_area × n_boundaries). Sequential relabelling uses a linear post-pass:
`np.unique` + lookup-table remap, O(voxels). Pass `sequential_labels=True`
to enable it.

---

## The overlap boundary trap

**Symptom:** The output array has the wrong shape. Extra voxels appear at the
image edges.

**Cause:** `da.overlap.overlap(image, boundary="reflect")` adds mirrored data
at the edges. When the merge step trims halos with
`da.overlap.trim_overlap(boundary="none")`, these two modes don't compose:
the halo remains in the output.

**Fix:** patchworks always uses `boundary="none"` for both overlap expansion
and trim. This is also scientifically correct — no fabricated mirror data is
added past the true image edges.

---

## The `persist()` trap

**Symptom:** Worker OOM after a few tiles. Memory usage ramps to 100s of GB.

**Cause:** Calling `da.persist()` on a large overlapped array before writing
tries to load the entire halo-expanded array into a single worker's RAM.
For a 250 GB image with a 20-voxel halo, this is ~300 GB on one worker.

**Fix:** patchworks never persists intermediate results. The overlap graph
stays lazy; each tile is computed, written, and freed.

---

## Summary table

| Pitfall | Symptom | How patchworks handles it |
|---|---|---|
| In-process client | `FutureCancelledError` | Detected at startup, raises immediately |
| 3-4× fn recompute | Cellpose runs 3× per tile | Always stages labels to disk once |
| O(n²) relabelling | Graph construction hangs | Linear post-pass O(voxels) |
| Wrong overlap boundary | Wrong output shape | Always uses `boundary="none"` |
| Persisting large arrays | Worker OOM | Never persists; keeps dask graph lazy |
