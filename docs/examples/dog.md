# Difference of Gaussians (spots, cilia, fibres)

For small structures Cellpose is not made for: blur at two sigmas, subtract,
threshold, label. CPU by default, GPU with `use_gpu=True` (needs `cupy`);
optionally deconvolve each tile first with
[pycudadecon](https://github.com/tlambert03/pycudadecon)
(`pip install "patchworks[dog]"`, CUDA only).

> Cilia DoG + deconvolution approach courtesy of
> [angelo-angonezi](https://github.com/angelo-angonezi).

```python
from patchworks import tile_process
from patchworks.plugins.dog import dog_label_fn

fn = dog_label_fn(low_sigma=1.0, high_sigma=3.0, threshold=0.02)
tile_process("image.zarr", fn, channel=1, tile_shape=(16, 1024, 1024), overlap=[4, 8, 8])
```

## Parameters

- `low_sigma`: about the object's radius; `high_sigma`: a few times larger
  (the background to subtract).
- `threshold` applies to the DoG image: start near its peak value on a
  known object and adjust.
- `connectivity=3` (3-D) joins voxels touching at an edge or corner, so an
  oblique cilium is one object rather than a row of fragments. With the API,
  pass the same value to `merge_tile_labels`; the workflow does it for you.
- `dilate_labels(fn, iterations=2)` grows thin labels
  ([Growing labels](../guide/custom_segmentation.md#growing-labels)).

## Deconvolution first

```python
from patchworks.plugins.ome_zarr import read_pixel_size

fn = dog_label_fn(
    low_sigma=1.0, high_sigma=3.0, threshold=0.02,
    decon_kwargs=dict(psf=psf, wavelength=525, na=1.4, nimm=1.515),
    voxel_size=read_pixel_size("image.zarr"),   # fills dxdata/dzdata/dxpsf/dzpsf
)
```

Anything set in `decon_kwargs` wins, such as `dxpsf`/`dzpsf` for a PSF sampled
differently from the data. A wrong voxel size does not fail, it gives a
subtly wrong image, so let it come from the image. Make `overlap` cover the
PSF as well as `high_sigma`.

## On the cluster

`workflow/config/config_cilia.yaml` is a complete example:

```yaml
channel: 2
overlap: [8, 30, 30]
method: "custom"
label_name: "cilia_labels"
custom:
  module: "patchworks.plugins.dog"
  kwargs:
    low_sigma: 1.0
    high_sigma: 3.0
    threshold: 0.02
    connectivity: 3          # keep oblique cilia whole
    decon_kwargs:            # optional; voxel sizes come from the image
      psf: "/path/to/psf.tif"
      wavelength: 525
      na: 1.4
      nimm: 1.515
```

Deconvolution needs a GPU in the segment jobs. List it in a multi config
next to the cells to get each cilium's cell
([Tables and relations](../guide/tables.md#relating-two-label-images)).
