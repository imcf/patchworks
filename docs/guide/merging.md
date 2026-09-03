# Merging labels

## The split-label problem

After segmenting each tile independently, labels are only locally unique:
tile A has labels 1-500, tile B also has labels 1-500. Worse, an object
spanning the A-B boundary gets label 247 in tile A and label 83 in tile B,
even though it's the same cell.

patchworks solves this with a zarr-native merge algorithm:

```text
Tile A labels:        Tile B labels:        After merge:
┌────────────┐        ┌────────────┐        ┌──────────────────────┐
│  3   1   2 │        │  1   4   2 │        │  3   1   2 │ 501 5 502│
│  3   1   1 │   +    │  1   1   2 │   →    │  3   1   1 │ 501 1  502│
│  1   5   5 │        │  5   5   3 │        │  1   5   5 │  5  5   3 │
└────────────┘        └────────────┘        └──────────────────────┘
                                             cell "1" is now one object
```

## The algorithm

The merge is **zarr-native** — no dask task graph, scales to thousands of tiles.
This is the same approach used by
[skeleplex](https://github.com/kevinyamauchi/skeleplex) and
[cellpose distributed](https://github.com/MouseLand/cellpose).

### Step 1: stage

Each tile's labels are written to zarr once. This is critical: without
staging, any downstream operation that reads the label array re-runs your
segmentation function. The merge internally reads labels multiple times.

```text
tile_process calls fn once per tile → staged zarr
                                         │
                         merge reads from staged zarr (no fn calls)
```

The Snakemake workflow goes further and stages **directly into**
`image.zarr/labels/<name>/0`, then has the merge rewrite that array in place
— saving a whole extra write of the volume plus the scratch store's disk. It
falls back to a separate store when the tile is larger than the label chunk
cap, since in place the chunking cannot be changed and level 0 has to stay
pageable for a viewer.

### Step 2: make the ids globally unique

Tiles write local `1..n`, which collide, so the boundary scan could not
otherwise tell two different objects apart. If each tile's label **count** is
known, this is just an exclusive cumulative sum — global id is
`offset[tile] + local`, computed in `O(n_tiles)` with no read of the volume
at all. `stage_tile` returns that count for exactly this purpose; pass the
counts as `label_counts=`.

Without counts, the merge falls back to streaming every chunk and renumbering
it in place — correct, but a full read **and write** of the volume.

### Step 3: boundary scan

Only the two voxels on either side of each tile boundary are read. For any
pair of touching non-zero labels `(a, b)`, they must be the same object. The
per-tile offsets are applied here, on the fly.

I/O cost: `O(n_boundaries × face_area)`, not `O(full_volume)`. The columns
are read in parallel, and a boundary next to a chunk that holds no labels is
skipped outright — a pair needs a non-zero label on *both* sides, so it could
never produce one.

### Step 4: connected components

scipy sparse connected components on the touching pairs produces a relabeling
lookup table. All labels that transitively touch each other are mapped to the
same canonical label.

Cost: `O(n_touching_pairs)`.

With `sequential_labels=True` the contiguous renumbering is folded into this
same LUT. Because the id domain is dense by construction, the surviving ids
are exactly the distinct LUT values — a `np.unique` over an array the length
of the object count, with no scan of the volume.

### Step 5: parallel relabel

The LUT is applied to every tile in parallel via `multiprocessing.Pool`. The
LUT is shared via process initializer to avoid re-pickling it for every chunk
(LUTs can be hundreds of MB for dense label volumes).

Chunks whose tile wrote no labels are skipped entirely — not read, and not
written. Zarr never materialises an unwritten chunk and reads it back as the
fill value, so background regions cost neither I/O nor disk. On a sparse
image that is most of the volume.

When the merge's output *is* its input, this pass rewrites the array in
place. That is safe because the boundary scan (step 3) has already finished,
so nothing still needs the original ids.

Because an in-place merge destroys its own input, it records how far it got
on the array itself, and refuses to guess on a re-run:

| State found | What happens |
| --- | --- |
| nothing recorded | fresh tile-local ids — merge normally |
| `running` | a previous attempt died mid-relabel, so the array is part local and part global. **Refuses**: re-segment to rebuild it. |
| `done` | already merged — a no-op, so a failure *after* the relabel (the pyramid, say) can simply be retried |

Without that, a second pass would add each tile's offset to ids that are
already global, which can land two unrelated objects on the same id.

## Using the merge step standalone

You can call the merge step directly on any existing label array or zarr:

```python
import dask.array as da
import numpy as np
from patchworks import merge_tile_labels

# From a dask array (your own tiling pipeline)
image = da.from_zarr("image.zarr").rechunk((1, 1024, 1024))
labeled = image.map_blocks(
    my_fn, dtype="int32", meta=np.empty((0,) * image.ndim, dtype="int32")
)
merged = merge_tile_labels(labeled, write_to="labels.zarr")

# From a zarr your pipeline already wrote
merged = merge_tile_labels(
    "my_staged_labels.zarr",
    input_component="raw_labels",
    write_to="merged.zarr",
    sequential_labels=True,
)
```

## Filtering by size after merge

Once labels are globally consistent, [`filter_labels_by_size`](../api/volume_filter.md)
can drop objects smaller than a voxel count, in place:

```python
from patchworks import filter_labels_by_size, merge_tile_labels

merged = merge_tile_labels("stage.zarr", write_to="labels.zarr", sequential_labels=True)
n_kept, n_removed = filter_labels_by_size("labels.zarr", "labels", min_voxels=500)
```

This has to run **after** the merge, not per tile: a tile only sees whatever
fragment of an object landed inside its own bounds, so a per-tile filter would
judge (and possibly drop) an object crossing a tile boundary as if it were
only that fragment's size.

Like the merge itself, it is a two-pass streaming zarr scan — the array never
has to fit in RAM. `relabel=True` (the default) folds the size filter into
the same lookup table that renumbers survivors to a contiguous `1..N` range,
so dropping small objects costs no extra pass over the volume beyond the scan
that already counts them.

Physical thresholds (µm³) convert to a voxel count via
[`min_voxels_for_volume`](../api/volume_filter.md), using the same
`{"z": .., "y": .., "x": ..}` calibration deconvolution and Cellpose's
`anisotropy` are derived from:

```python
from patchworks import min_voxels_for_volume
from patchworks.plugins.ome_zarr import read_pixel_size

min_voxels = min_voxels_for_volume(5.0, read_pixel_size("image.zarr"))
n_kept, n_removed = filter_labels_by_size("labels.zarr", "labels", min_voxels)
```

On the cluster, set `min_volume: 5.0` in the config instead — see [Configure
the run](snakemake.md#3-configure-the-run). It runs automatically between
`merge` and the pyramid build.

## Sequential label numbering

By default, merged labels are globally unique but may be **gappy** — boundary
merging fuses ids, leaving holes where the absorbed ones were. This is fine
for counting, `regionprops`, and measurement — the IDs just aren't
consecutive.

For contiguous 1..N numbering, use `sequential_labels=True`:

```python
tile_process("image.zarr", fn, write_to="labels.zarr", sequential_labels=True)
```

This is free: it composes into the relabel LUT the merge already applies, so
it costs a `np.unique` over the object count rather than another pass over
the volume.

!!! warning "Do not use dask's built-in sequential relabel"
    `dask_image.ndmeasure.merge_labels_across_chunk_boundaries` has a
    `produce_sequential_labels=True` option that builds a task graph of O(n²)
    in the number of tiles. At 64 tiles this takes 54 seconds; at 2200 tiles
    it would take hours — just for graph construction. patchworks's approach
    is always linear in the number of voxels.
