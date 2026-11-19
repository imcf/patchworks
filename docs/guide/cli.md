# Command line

The `patchworks` command runs the same building blocks as the Python API,
without a script or a workflow config.

```bash
# Convert anything bioio reads (or an existing store) to a pyramidal OME-ZARR
patchworks convert scan.czi scan.zarr

# Segment it; labels go into scan.zarr/labels/labels
patchworks segment scan.zarr --method cellpose --model cyto3 --diameter 30 --gpu

# What is in the store: levels, chunks, codecs, calibration, label images
# (and how each was made: when, by which version, method, channel, level)
patchworks info scan.zarr

# Did the tiling leave marks in the labels?
patchworks seams scan.zarr/labels/labels --tile-shape 16,1024,1024

# Look at the result (needs patchworks[napari])
patchworks view scan.zarr

# One row per object; which cell each nucleus is in
patchworks tables scan.zarr --relate nuclei:cells

# The objects' volumes and lengths, and what a minimum of each keeps
patchworks sizes scan.zarr cilia_labels

# Without the small ones, as a new label image (the original stays; a job
# on a cluster): by volume (µm³) and/or length along the main axis (µm)
patchworks filter scan.zarr cilia_labels --min-length 1

# In Neuroglancer, through an SSH tunnel (prints the tunnel and the link)
patchworks serve scan.zarr

# Look at the likely mistakes one by one and fix them (napari)
patchworks review scan.zarr --expect cells:nuclei=1

# Is it valid OME-Zarr, and what should be repaired? (exit status 1 on errors;
# the spec check needs pip install "patchworks[validate]")
patchworks validate scan.zarr

# Bring a store written by an older patchworks up to the OME-Zarr spec
patchworks fix-metadata scan.zarr

# Put each object's values (volume, length, parent, position) into its
# label image's metadata, for napari: after the relations or a review
patchworks fix-metadata scan.zarr --properties

# Name the channels (napari, Fiji show them): from the original file, or typed
patchworks fix-metadata scan.zarr --channels-from scan.czi
patchworks fix-metadata scan.zarr --channel-names DAPI,Membrane,Cilia,Actin
patchworks fix-metadata scan.zarr --channels-from "planes/*.tif" --sequence-pattern "_Z(?P<Z>\d+)_C(?P<C>\d+)"
```

`fix-metadata` only touches metadata, in place, and is safe to run twice.
`validate` checks the store with an independent OME-Zarr validator
([yaozarrs](https://github.com/imaging-formats/yaozarrs)), each problem
with the validator's code, then what patchworks writes and the store lacks
(a label image without its `source`, unnamed channels, an object table out
of date), each with its repair, and prints what the store holds.
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
