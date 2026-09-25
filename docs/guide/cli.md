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
```

## Segmentation methods

| `--method` | What it does | Main flags |
| --- | --- | --- |
| `threshold` (default) | one global threshold (Otsu over the image unless `--threshold`), then connected components | `--threshold` |
| `cellpose` | [Cellpose](../api/plugins/cellpose.md) | `--model`, `--diameter`, `--do-3d`, `--gpu` |
| `dog` | [difference of Gaussians](../api/plugins/dog.md) | `--low-sigma`, `--high-sigma`, `--threshold` |
| `custom` | any importable function | `--fn module:function`, `--fn-kwargs '{"k": 1}'` |

Cellpose's anisotropy and the DoG plugin's voxel size are read from the
store's own calibration, at the level being segmented.

Tiling and stitching take the same options as
[`tile_process`](../api/tile_process.md): `--tile-shape` (`z,y,x`, `auto` or
`none`), `--overlap` (`N` or `z,y,x`), `--stitch iou`, `--resume`,
`--skip-empty`, `--level`, `--channel` (an index, or `none` for every channel),
`--compression`. Run `patchworks <command> --help` for the full list.
