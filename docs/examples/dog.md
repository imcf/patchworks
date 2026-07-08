# Difference of Gaussians (blobs, threads, cilia, …)

A lightweight blob/thread detector for structures Cellpose isn't shaped for
(cilia, spots, fibres): blur twice at different sigmas, subtract, threshold,
label the connected components. CPU (scipy) by default, GPU (cupy) optional.
Optionally deconvolve each tile first with
[pycudadecon](https://github.com/tlambert03/pycudadecon).

## Installation

`dog_label_fn` itself only needs patchworks' core deps (scipy). The
deconvolution step needs pycudadecon:

```bash
pip install "patchworks[dog]"
```

GPU blur/label (`use_gpu=True`) needs `cupy` too, matching your CUDA version
(e.g. `pip install cupy-cuda12x`) — not bundled in the `dog` extra since it's
CUDA-version-specific.

## Code

```python
import numpy as np
from patchworks import tile_process
from patchworks.plugins.dog import dog_label_fn

IMAGE = "image.zarr"
OUTPUT = "labels_dog.zarr"

fn = dog_label_fn(low_sigma=1.0, high_sigma=3.0, threshold=0.02)

tile_process(
    IMAGE,
    fn,
    channel=1,
    tile_shape=(1, 1024, 1024),
    overlap=8,  # just needs to cover one object + high_sigma
    write_to=OUTPUT,
    progress=True,
)
```

## Picking `low_sigma` / `high_sigma` / `threshold`

`dog = blur(low_sigma) - blur(high_sigma)`. `low_sigma` should be about the
object's radius (denoises without erasing it); `high_sigma` a few times
larger (models the background to subtract out). `threshold` is applied
directly to the DoG image — start near the DoG's typical peak value on a
known-positive region and adjust from there; there's no auto (Otsu-style)
option, since the DoG image isn't bimodal the way a raw intensity image is.

## GPU

```python
fn = dog_label_fn(low_sigma=1.0, high_sigma=3.0, threshold=0.02, use_gpu=True)
```

Requires `cupy` (matching your CUDA version, e.g. `pip install cupy-cuda12x`)
— not a patchworks dependency, install it separately.

## With deconvolution first

```python
fn = dog_label_fn(
    low_sigma=1.0, high_sigma=3.0, threshold=0.02,
    decon_kwargs=dict(
        psf=psf, dxpsf=xy_scale, dxdata=xy_scale,
        dzpsf=z_scale, dzdata=z_scale,
        wavelength=525, na=1.4, nimm=1.515,
    ),
)
result = tile_process(IMAGE, fn, tile_shape=(1, 1024, 1024), overlap=32)
```

!!! note "Deconvolution always needs a GPU"
    `pycudadecon` is CUDA-only, independent of `dog_label_fn`'s own `use_gpu`
    flag (which only picks the backend for the blur/label steps). A SLURM job
    running this needs a GPU allocated. Widen `overlap` past the PSF support
    so edge tiles keep enough context (a plain intensity/threshold halo is
    too thin).

## Using it in the Snakemake workflow

No dedicated wiring needed — `patchworks.plugins.dog` exposes a `segment(tile, **kwargs)`
adapter for the documented [`"custom"` method](../guide/snakemake.md#custom-segmentation-function):

```yaml
method: "custom"
label_name: "cilia_labels"
custom:
  module: "patchworks.plugins.dog"
  function: "segment"
  kwargs:
    low_sigma: 1.0
    high_sigma: 3.0
    threshold: 0.02
```

See `workflow/config/config_cilia.yaml` for a full example, including
deconvolution.

## Relating cilia to their cell

Segment the cell body with Cellpose and the cilia with `dog_label_fn` as two
separate `tile_process` runs (same image, same `tile_shape`), then use
[`label_relations`](../guide/snakemake.md#relating-labels-across-segmentations)
to map each cilium to the cell it belongs to — see
`workflow/config/multi.yaml` for the same thing wired up as a cluster job.
