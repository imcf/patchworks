# Merging labels

## The split-label problem

After segmenting each tile independently, labels are only locally unique:
tile A has labels 1-500, tile B also has labels 1-500. Worse, an object
spanning the A-B boundary gets label 247 in tile A and label 83 in tile B,
even though it's the same cell.

blockbuster solves this with a zarr-native merge algorithm:

```
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

Each tile's labels are written to a temporary zarr once. This is critical:
without staging, any downstream operation that reads the label array re-runs
your segmentation function. The merge internally reads labels multiple times.

```
tile_process calls fn once per tile → staged zarr
                                         │
                         merge reads from staged zarr (no fn calls)
```

### Step 2: boundary scan

Only the two voxels on either side of each tile boundary are read. For any
pair of touching non-zero labels `(a, b)`, they must be the same object.

I/O cost: `O(n_boundaries × face_area)`, not `O(full_volume)`.

### Step 3: connected components

scipy sparse connected components on the touching pairs produces a relabeling
lookup table. All labels that transitively touch each other are mapped to the
same canonical label.

Cost: `O(n_touching_pairs)`.

### Step 4: parallel relabel

The LUT is applied to every tile in parallel via `multiprocessing.Pool`. The
LUT is shared via process initializer to avoid re-pickling it for every chunk
(LUTs can be hundreds of MB for dense label volumes).

## Using the merge step standalone

You can call the merge step directly on any existing label array or zarr:

```python
import dask.array as da
import numpy as np
from blockbuster import merge_tile_labels

# From a dask array (your own tiling pipeline)
image = da.from_zarr("image.zarr").rechunk((1, 1024, 1024))
labeled = image.map_blocks(my_fn, dtype="int32",
                            meta=np.empty((0,) * image.ndim, dtype="int32"))
merged = merge_tile_labels(labeled, write_to="labels.zarr")

# From a zarr your pipeline already wrote
merged = merge_tile_labels(
    "my_staged_labels.zarr",
    input_component="raw_labels",
    write_to="merged.zarr",
    sequential_labels=True,
)
```

## Sequential label numbering

By default, merged labels are globally unique but may be **gappy**
(block-encoded IDs like 1, 2, 500001, 500002, …). This is fine for
counting, `regionprops`, and measurement — the IDs just aren't consecutive.

For contiguous 1..N numbering, use `sequential_labels=True`:

```python
tile_process("image.zarr", fn, write_to="labels.zarr",
             sequential_labels=True)
```

This runs a cheap linear post-pass: `np.unique` + lookup-table remap, O(voxels).

!!! warning "Do not use dask's built-in sequential relabel"
    `dask_image.ndmeasure.merge_labels_across_chunk_boundaries` has a
    `produce_sequential_labels=True` option that builds a task graph of O(n²)
    in the number of tiles. At 64 tiles this takes 54 seconds; at 2200 tiles
    it would take hours — just for graph construction. blockbuster's approach
    is always linear in the number of voxels.
