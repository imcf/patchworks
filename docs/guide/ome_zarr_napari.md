# OME-Zarr and napari

patchworks keeps everything in one OME-Zarr: the image as a pyramid, and each
segmentation under `labels/<name>/` as a pyramid of its own with its
[table](tables.md). napari, Fiji/MoBIE and NGFF validators read it as is.

## Convert

```python
from patchworks.plugins.ome_zarr import to_ome_zarr

to_ome_zarr("scan.czi", "scan.zarr")   # CZI, LIF, ND2, OME-TIFF, ... via bioio
to_ome_zarr("scan.ims", "scan.zarr")   # Imaris
to_ome_zarr(array, "scan.zarr", axes="czyx", pixel_size={"z": 2.0, "y": 0.32, "x": 0.32})
```

Inputs are read lazily and the pyramid is written level by level, so any size
converts in bounded memory. The voxel size is read from the file and stored
in µm. Useful options:

| Option | Does |
| --- | --- |
| `n_levels=5` | pyramid levels (X and Y halved each time; Z kept) |
| `axes=` | axis order of a bare array; from 4-D it is otherwise guessed (with a warning) |
| `reuse_pyramid=True` | copy an Imaris file's own pyramid instead of rebuilding it |
| `shard=True` | pack chunks into ~512 MB shard files: ~100× fewer files |
| `ngff_version="0.4"` | zarr v2 / NGFF 0.4 for older readers (default: 0.5 on zarr v3) |
| `compression=` | see [Performance](performance.md#compression) |

Install the readers you need: `patchworks[bioio]` (plus a native reader such
as `bioio-czi` for speed), `patchworks[imaris]`.

### A folder of single-plane TIFFs

One file per plane, such as `sample_T0_Z000_C0_V0.tif`: give a glob and a
regex whose named groups are the axes. A constant axis (`T0`) is dropped.

```python
to_ome_zarr(
    "sample/*.tif", "sample.zarr",
    sequence_pattern=r"_T(?P<T>\d+)_Z(?P<Z>\d+)_C(?P<C>\d+)_V\d+",
    shard=True,
)
```

On the cluster: `sequence_pattern:` in the config, with `input:` the glob.

## Labels in the store

`tile_process("scan.zarr", fn)` writes the labels into
`scan.zarr/labels/labels` (name it with `output_component="cells"`), with a
pyramid matching the image's. To add your own label array, or a pyramid to a
flat store:

```python
from patchworks.plugins.ome_zarr import add_pyramid, write_labels

write_labels("scan.zarr", my_labels, name="nuclei")
add_pyramid("flat.zarr", base="0", n_levels=5)
```

A store written by an older version, or edited by hand, can be checked and
repaired with `patchworks fix-metadata scan.zarr`.

## View in napari

```python
from patchworks.plugins.napari import view_in_napari

view_in_napari("scan.zarr")              # every channel and every label image
view_in_napari("scan.zarr", channel=0)   # one channel
```

Only what is on screen is read, so it opens at once even for terabytes.
From the cluster workflow: `pixi run -e viewer napari <work_dir>/image.zarr`
(needs a display). To check and correct the results, see
[Reviewing](review.md).

## Remote stores

Any fsspec URL works where a path does, for reading and for writing labels
back (`pip install "patchworks[remote]"`; credentials from the usual
AWS/GCP settings):

```python
tile_process("s3://bucket/scan.zarr", fn, tile_shape=(16, 1024, 1024))
```

Scratch data stays on local disk, so only the final labels travel.
