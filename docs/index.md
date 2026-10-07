# patchworks

<p align="center">
  <img src="assets/logo.png" alt="patchworks logo" width="200">
</p>

**Tiled processing of arbitrarily large images — any image, any function.**

```text
┌──────┬──────┬──────┐                    ┌──────┬──────┬──────┐
│      │      │      │   fn(tile) → IDs   │  1   │  2   │  3   │
├──────┼──────┼──────┤  ───────────────►  ├──────┼──────┼──────┤
│      │ 2 TB │      │                    │  4   │  5   │  6   │
├──────┼──────┼──────┤                    ├──────┼──────┼──────┤
│      │      │      │                    │  7   │  8   │  9   │
└──────┴──────┴──────┘                    └──────┴──────┴──────┘
         tiles                               globally consistent labels
```

Segmentation tools assume the image fits in memory; a light-sheet or
whole-slide volume does not. patchworks splits it into tiles, runs any
segmentation function on each, and stitches the results so that an object
crossing a tile boundary keeps one label. The labels are stored with the
image in one OME-Zarr, with a table of every object.

## Where to start

| You want to | Read |
| --- | --- |
| Try it on an image from Python | [Getting started](getting_started.md) |
| Segment a large image on a SLURM cluster, maybe several stains related to each other | [Cluster workflow](guide/snakemake.md) |
| Run one step from a terminal | [Command line](guide/cli.md) |
| Use your own model or function | [Custom segmentation function](guide/custom_segmentation.md) |
| Cells from a membrane stain | [Membrane cells](guide/membrane_cells.md) |
| Check and correct the results | [Reviewing](guide/review.md) |

## Building blocks

Each step is a function you can use on its own:

| Step | Function |
| --- | --- |
| Convert any image to a pyramidal, calibrated OME-Zarr | `to_ome_zarr` |
| Segment tile by tile, with a halo for context | `tile_process`, `stage_tile` |
| Stitch tiles into consistent labels | `merge_tile_labels` |
| Measure every object; relate two label images | `measure_objects`, `label_relations` |
| View and review in napari | `view_in_napari`, `patchworks review` |
