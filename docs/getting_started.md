# Getting started

## Install

```bash
pip install patchworks
```

Python 3.11 or later. Add what you need:

| Extra | For |
| --- | --- |
| `cellpose` (`cellpose3`, `cellpose4` to pin) | the Cellpose plugin |
| `bioio`, `imaris` | converting CZI, LIF, ND2, TIFF, `.ims` to OME-Zarr |
| `napari` | viewing and reviewing results |
| `review` | object tables as pandas, Excel export |
| `dog` | deconvolution before the difference-of-Gaussians plugin |
| `careamics` | Noise2Void denoising |
| `workflow` | the cluster workflow (Snakemake) |
| `gpu`, `distributed`, `remote` | GPU memory sizing; Dask clusters; S3/GCS/HTTP stores |
| `all` | everything |

```bash
pip install "patchworks[cellpose,bioio,napari]"
```

GPU options of the DoG plugin and of label dilation need `cupy` for your
CUDA version (`pip install cupy-cuda12x`). PlantSeg is on conda-forge only
(`conda install -c conda-forge plant-seg`).

## Your first run

```python
from patchworks import tile_process
from patchworks.plugins.ome_zarr import to_ome_zarr
from patchworks.plugins.cellpose import cellpose_fn

to_ome_zarr("scan.czi", "scan.zarr")  # once: a pyramidal OME-Zarr

tile_process(
    "scan.zarr",
    cellpose_fn("cyto3", gpu=True, diameter=30),
    channel=0,
    tile_shape=(1, 2048, 2048),  # or "auto", sized to your memory
    overlap=20,                  # halo: about one object diameter
)
```

The labels are written into `scan.zarr/labels/labels`, next to the image.
Look at them:

```python
from patchworks.plugins.napari import view_in_napari

view_in_napari("scan.zarr")
```

## Any function

`fn` is any callable from a tile (a NumPy array) to integer labels of the
same shape:

```python
from skimage.filters import threshold_otsu
from skimage.measure import label


def my_fn(tile):
    return label(tile > threshold_otsu(tile)).astype("int32")


tile_process("scan.zarr", my_fn)
```

Each tile is segmented independently; objects crossing tile boundaries are
joined into one label afterwards.

## Next

- Large images on a cluster, several segmentations related to each other:
  [Cluster workflow](guide/snakemake.md)
- How tiles and halos are sized: [Tiling](guide/tiling.md)
- Your own model: [Custom segmentation function](guide/custom_segmentation.md)
- The same steps without Python: [Command line](guide/cli.md)
