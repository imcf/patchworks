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
# config/config_cilia.yaml (excerpt)
channel: 2
tile_shape: [16, 1024, 1024]
overlap: 30 # cover the PSF support (decon) + the DoG's high_sigma
skip_empty: true

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
      dxpsf: 0.1
      dxdata: 0.1
      dzpsf: 0.2
      dzdata: 0.2
      wavelength: 525
      na: 1.4
      nimm: 1.515
```

Run it exactly like a Cellpose config:

```bash
python -m snakemake --workflow-profile profile/slurm \
                    --configfile config/config_cilia.yaml
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
  calls `estimate_empty_tiles()` before submitting any `segment` jobs,
  regardless of `method`, so cilia/DoG runs skip background tiles exactly
  like Cellpose runs — no extra config needed beyond `skip_empty: true`
  (the default).
- Run alongside `config_cyto.yaml`/`config_nuclei.yaml` via `config/multi.yaml`
  to also get the cilia→cell/nucleus relation — see *Relating cilia to their
  cell*, below.

## Relating cilia to their cell

Segment the cell body with Cellpose and the cilia with `dog_label_fn` as two
separate `tile_process` runs (same image, same `tile_shape`), then use
[`label_relations`](../guide/label_relations.md)
to map each cilium to the cell it belongs to — see
`workflow/config/multi.yaml` for the same thing wired up as a cluster job.
