# Difference of Gaussians (blobs, threads, cilia, …)

A lightweight blob/thread detector for structures Cellpose isn't shaped for
(cilia, spots, fibres): blur twice at different sigmas, subtract, threshold,
label the connected components. CPU (scipy) by default, GPU (cupy) optional.
Optionally deconvolve each tile first with
[pycudadecon](https://github.com/tlambert03/pycudadecon).

> Cilia DoG + deconvolution approach courtesy of
> [angelo-angonezi](https://github.com/angelo-angonezi).

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

Let the voxel sizes come from the image rather than retyping them — the
lateral ones from X/Y, the axial ones from Z:

```python
from patchworks.plugins.ome_zarr import read_pixel_size

fn = dog_label_fn(
    low_sigma=1.0, high_sigma=3.0, threshold=0.02,
    decon_kwargs=dict(psf=psf, wavelength=525, na=1.4, nimm=1.515),
    voxel_size=read_pixel_size(IMAGE),   # -> dxdata/dzdata/dxpsf/dzpsf
)
result = tile_process(IMAGE, fn, tile_shape=(1, 1024, 1024), overlap=32)
```

Anything you set in `decon_kwargs` yourself wins, so a PSF sampled
differently from the data keeps its own sizes:

```python
decon_kwargs=dict(psf=psf, dxpsf=0.05, dzpsf=0.1, wavelength=525, ...)
```

!!! warning "A wrong voxel size does not fail loudly"
    Deconvolution given the wrong sampling still runs and still returns an
    image — just a subtly wrong one. That is the reason to derive these from
    the store's own calibration instead of keeping a second copy in a config
    that can drift.

    `read_pixel_size` returns `{}` for an uncalibrated store; then nothing is
    filled in and you must supply the sizes yourself.

!!! note "Deconvolution always needs a GPU"
    `pycudadecon` is CUDA-only, independent of `dog_label_fn`'s own `use_gpu`
    flag (which only picks the backend for the blur/label steps). A SLURM job
    running this needs a GPU allocated. Widen `overlap` past the PSF support
    so edge tiles keep enough context (a plain intensity/threshold halo is
    too thin).

## Growing the labels afterwards

DoG spots/threads are often thin — grow each label by a few pixels with
[`dilate_labels`](../api/postprocess.md):

```python
from patchworks import tile_process, dilate_labels
from patchworks.plugins.dog import dog_label_fn

fn = dog_label_fn(low_sigma=1.0, high_sigma=3.0, threshold=0.02)
fn = dilate_labels(fn, iterations=2)
tile_process(IMAGE, fn, tile_shape=(1, 1024, 1024), overlap=8, write_to=OUTPUT)
```

On the cluster, set `dilate: 2` in the YAML config instead — it applies to
`method: "custom"` (this plugin) the same way it does for `cellpose`/
`threshold`, see [Growing labels afterwards](../guide/custom_segmentation.md#growing-labels-afterwards-dilation).

## Using it in the Snakemake workflow

No dedicated wiring needed — `patchworks.plugins.dog` exposes a `segment(tile, **kwargs)`
adapter for the documented [`"custom"` method](../guide/custom_segmentation.md):

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

### With deconvolution, on SLURM

Add `decon_kwargs` under `custom.kwargs` — same keys as the plain-Python
example above — and the segment job deconvolves each tile with
`pycudadecon` before running the DoG detector:

```yaml
# config/config_cilia.yaml (excerpt) — only what differs from common.yaml,
# which supplies the input, work_dir, tile_shape and skip_empty
channel: 2
# Per-axis halo [z, y, x], covering the PSF support (decon) + the DoG's
# high_sigma. A scalar 30 would expand a [16, 1024, 1024] tile to 5.3x the
# voxels it keeps, nearly all of it wasted z.
overlap: [8, 30, 30]

method: "custom"
label_name: "cilia_labels"
custom:
  module: "patchworks.plugins.dog"
  function: "segment"
  kwargs:
    low_sigma: 1.0
    high_sigma: 3.0
    threshold: 0.02
    decon_kwargs:
      psf: "/path/to/psf.tif"
      wavelength: 525
      na: 1.4
      nimm: 1.515
```

No voxel sizes: the workflow reads `image.zarr`'s own calibration and fills
in `dxdata`/`dxpsf` from X/Y and `dzdata`/`dzpsf` from Z. Set any of them in
`decon_kwargs` to override — for instance a PSF sampled finer than the data:

```yaml
      dxpsf: 0.05
      dzpsf: 0.1
```

!!! tip "This works for your own methods too"
    The injection is not DoG-specific. Any `custom` function that declares a
    `voxel_size` parameter receives `{"z": .., "y": .., "x": ..}` from the
    store, so a method needing physical units never has to keep a second copy
    of the calibration in its config. If the store is uncalibrated the
    workflow says so and passes nothing.

Run it exactly like a Cellpose config — the shared settings come from
`config/common.yaml`, merged in ahead of this one:

```bash
python -m snakemake --workflow-profile profile/slurm --configfile config/common.yaml config/config_cilia.yaml
```

Checklist specific to this config:

- **Env:** the segment job's environment needs `patchworks[dog]`
  (`pip install "patchworks[dog]"`) on top of whatever else it uses — plain
  `dog_label_fn` only needs scipy, but `decon_kwargs` pulls in
  `pycudadecon`.
- **GPU always required:** `pycudadecon` is CUDA-only regardless of the
  detector's own `use_gpu` flag, so `set-resources: segment:` in
  `profile/slurm/config.yaml` must request a GPU (`slurm_extra:
  "'--gres=gpu:1'"`) the same as for Cellpose.
- **`overlap`:** widen it past the PSF support, not just past `high_sigma` —
  a thin intensity/threshold halo isn't enough once deconvolution is in the
  loop.
- **`skip_empty`:** the `prepare` rule (`workflow/scripts/prepare_tiles.py`)
  builds a max-pooled occupancy map and reduces it over each tile's **full**
  footprint (`build_occupancy_map` + `tile_occupancy`) before submitting any
  `segment` jobs, regardless of `method`. Cilia are small and often sit near
  a tile's edge, which is precisely where the older centred-window preview
  could miss them — this decides every tile exactly. No extra config needed
  beyond `skip_empty: true` (the default), and the map is built once and
  shared by every config against that store.
- Run alongside `config_cyto.yaml`/`config_nuclei.yaml` via `config/multi.yaml`
  to also get the cilia→cell/nucleus relation — see *Relating cilia to their
  cell*, below.

## Relating cilia to their cell

Segment the cell body with Cellpose and the cilia with `dog_label_fn` as two
separate `tile_process` runs (same image, same `tile_shape`), then use
[`label_relations`](../guide/label_relations.md)
to map each cilium to the cell it belongs to — see
`workflow/config/multi.yaml` for the same thing wired up as a cluster job.
