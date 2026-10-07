# Command line

Installing patchworks also installs a `patchworks` command: the same building
blocks as the Python API, for a one-off run without writing a script or a
Snakemake config.

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

`patchworks fix-metadata` changes metadata only, in place: it gives every
OME-Zarr 0.5 array its `dimension_names` (required by 0.5; versions before
this one wrote none, which strict readers such as `ome-zarr-models` refuse),
and drops surplus coarse levels from a label image with more pyramid levels
than its image (the spec requires the same number). Running it twice
changes nothing the second time.

`patchworks review` without a window: `--summary` (counts and error
estimate), `--export DIR --format csv|xlsx|parquet` (corrected tables),
`--workbooks DIR` (relation workbooks), `--write-labels NAME` (a label image
with the corrections applied). `--position CHILD:PARENT=APICAL` classifies
children as apical/basal/lateral/central (APICAL: `+z`, or a label image to
point away from, such as the nuclei); `patchworks tables --max-distance UM`
gives a child touching no parent the nearest one. See
[Reviewing and correcting results](review.md).

## Segmentation methods

| `--method` | What it does | Main flags |
| --- | --- | --- |
| `threshold` (default) | one global threshold (Otsu over the image unless `--threshold`), then connected components | `--threshold` |
| `cellpose` | [Cellpose](../api/plugins/cellpose.md) | `--model`, `--diameter`, `--do-3d`, `--gpu` |
| `dog` | [difference of Gaussians](../api/plugins/dog.md) | `--low-sigma`, `--high-sigma`, `--threshold` |
| `custom` | any importable function | `--fn module:function`, `--fn-kwargs '{"k": 1}'` |

Cellpose's anisotropy and the DoG plugin's voxel size are read from the
store's own calibration, at the level being segmented -- as is `voxel_size`
for a custom function that takes one, such as the PlantSeg and watershed
plugins: `--method custom --fn patchworks.plugins.plantseg:segment
--fn-kwargs '{"segmentation": "gasp"}' --stitch iou` (`iou`: these fill
space, so neighbouring cells touch at every seam).

`--denoise MODEL` denoises every tile with a CAREamics model before any
method segments it; `patchworks denoise-train STORE --channel 0 --out
n2v.ckpt` trains one (Noise2Void, no ground truth). See
[Cells from a membrane stain](membrane_cells.md).

Tiling and stitching take the same options as
[`tile_process`](../api/tile_process.md): `--tile-shape` (`z,y,x`, `auto` or
`none`), `--overlap` (`N` or `z,y,x`), `--stitch iou`, `--resume`,
`--skip-empty`, `--level`, `--channel` (an index, or `none` for every channel),
`--compression`. Run `patchworks <command> --help` for the full list.
