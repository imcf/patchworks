# Command line

The `patchworks` command runs the same building blocks as the Python API,
without a script or a workflow config.

```bash
# Convert anything bioio reads (or an existing store) to a pyramidal OME-ZARR
patchworks convert scan.czi scan.zarr

# Segment it; labels go into scan.zarr/labels/labels
patchworks segment scan.zarr --method cellpose --model cyto3 --diameter 30 --gpu

# What is in the store: levels, chunks, codecs, calibration, label images
patchworks info scan.zarr

# Did the tiling leave marks in the labels?
patchworks seams scan.zarr/labels/labels --tile-shape 16,1024,1024

# Look at the result (needs patchworks[napari])
patchworks view scan.zarr

# One row per object; which cell each nucleus is in
patchworks tables scan.zarr --relate nuclei:cells

# Look at the likely mistakes one by one and fix them (napari)
patchworks review scan.zarr --expect cells:nuclei=1

# Bring a store written by an older patchworks up to the OME-Zarr spec
patchworks fix-metadata scan.zarr
```

`fix-metadata` only touches metadata, in place, and is safe to run twice.
`review` also runs without a window: `--summary`, `--export DIR`,
`--workbooks DIR`, `--write-labels NAME` (see [Reviewing](review.md)).

## Segmentation methods

| `--method` | What it does | Main flags |
| --- | --- | --- |
| `threshold` (default) | one global threshold (Otsu over the image unless `--threshold`), then connected components | `--threshold` |
| `cellpose` | [Cellpose](../api/plugins/cellpose.md) | `--model`, `--diameter`, `--do-3d`, `--gpu` |
| `dog` | [difference of Gaussians](../api/plugins/dog.md) | `--low-sigma`, `--high-sigma`, `--threshold` |
| `custom` | any importable function | `--fn module:function`, `--fn-kwargs '{"k": 1}'` |

Voxel sizes (Cellpose's anisotropy, the DoG sigmas, a custom function's
`voxel_size`) come from the store's calibration. The membrane plugins fill
space, so they need `--stitch iou`:
`--method custom --fn patchworks.plugins.plantseg:segment --stitch iou`.
`--denoise MODEL` denoises tiles first ([Denoising](membrane_cells.md#denoising-first)).

Tiling and stitching take the same options as
[`tile_process`](../api/tile_process.md): `--tile-shape` (`z,y,x`, `auto` or
`none`), `--overlap` (`N` or `z,y,x`), `--stitch iou`, `--resume`,
`--skip-empty`, `--level`, `--channel` (an index, or `none` for every channel),
`--compression`. Run `patchworks <command> --help` for the full list.
