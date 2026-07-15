# Custom segmentation function

Not using Cellpose? Run **your own** per-tile function — no need to edit the
package. You write one function; patchworks handles everything around it
(tiling, halos, skipping empty tiles, the zarr-native merge, global
relabelling, resume, logs).

This applies the same whether you call the API directly (`tile_process`) or
run via the [Snakemake cluster workflow](snakemake.md) — see [Wiring it into
the cluster workflow](#wiring-it-into-the-cluster-workflow) below for the
config side.

## The contract

Your function is called **once per tile**:

```python
labels = segment(tile)          # plus any kwargs you configure
```

| | What you get / must return |
| --- | --- |
| **Input `tile`** | A NumPy array of **one** tile, with the overlap halo already included. The channel and pyramid level are already selected, so it is purely spatial: `(z, y, x)` for a 3-D run, `(y, x)` for 2-D. Dtype is the image's (e.g. `uint16`). |
| **Return** | An integer **label** array (not a boolean mask), **same shape** as `tile`. `0` = background; each object a distinct positive integer. |
| **Labels** | Only need to be unique **within the tile**. Don't try to make them globally unique — the merge step stitches objects across tile borders and renumbers everything to a contiguous `1..N` (`sequential_labels: true`). |
| **Shape** | Must match the input exactly — patchworks trims the halo off your output, so a wrong shape is an error. Don't crop or resize inside the function. |

That is the whole interface. Anything that turns an image tile into a label
image works: classic image processing, StarDist, a trained model, an external
binary you shell out to, …

## Minimal example (no GPU, no deps beyond scikit-image)

```python
# my_seg.py
import numpy as np
from skimage.measure import label

def segment(tile: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Threshold + connected components. Returns int32 labels (0 = bg)."""
    from skimage.filters import gaussian, threshold_otsu

    smooth = gaussian(tile, sigma=sigma, preserve_range=True)
    thr = threshold_otsu(smooth) if smooth.max() > smooth.min() else np.inf
    return label(smooth > thr).astype("int32")
```

```python
from patchworks import tile_process
from my_seg import segment

tile_process("image.zarr", segment, write_to="labels.zarr")
```

## Growing labels afterwards (dilation)

To grow every label by a few pixels after segmentation, wrap your function
with [`patchworks.dilate_labels`](../api/postprocess.md):

```python
from patchworks import tile_process, dilate_labels
from patchworks.plugins.dog import dog_label_fn

fn = dog_label_fn(low_sigma=1.0, high_sigma=3.0, threshold=0.02)
fn = dilate_labels(fn, iterations=2)   # grow each label by 2 px, then run
result = tile_process("image.zarr", fn, tile_shape=(1, 2048, 2048),
                       overlap=8, write_to="labels.zarr")
```

`dilate_labels` wraps any `(tile) -> labels` function — the same contract
above — so it works with `dog_label_fn`, `cellpose_fn`, or your own
`segment`. It dilates each tile's labels before the halo is trimmed and
tiles are merged, so `overlap` must still cover the dilation amount. On the
cluster, set `dilate: N` in the config instead — see [Configure the
run](snakemake.md#3-configure-the-run).

By default the dilation itself runs on CPU (scipy), independent of whatever
backend `fn` used — pass `use_gpu=True` to dilate via cupy instead:

```python
fn = dilate_labels(fn, iterations=2, use_gpu=True)  # needs cupy installed
```

On the cluster, this is the `dilate_gpu: true` config key.

## Real example: StarDist 3-D, with model caching

Heavy models must be loaded **once**, not per tile. On SLURM each tile is its
own process so this matters less, but for local runs one process segments many
tiles — cache the model at module level (or with `functools.lru_cache`):

```python
# stardist_seg.py
import numpy as np

_MODEL = None

def _model():
    global _MODEL
    if _MODEL is None:                       # loaded once per worker process
        from stardist.models import StarDist3D
        _MODEL = StarDist3D.from_pretrained("3D_demo")
    return _MODEL

def segment(tile: np.ndarray, prob_thresh: float = 0.5) -> np.ndarray:
    from csbdeep.utils import normalize

    labels, _ = _model().predict_instances(
        normalize(tile), prob_thresh=prob_thresh
    )
    return labels.astype("int32")
```

Using a GPU? Just let your framework see it — nothing extra needed.

## Test it before you submit

Run your function on one real tile first — it catches shape/dtype bugs in
seconds instead of after a queue wait (or a long local run). Output must be
integer, same shape, `0` for background:

```python
from patchworks import load_ome_zarr
from my_seg import segment

img = load_ome_zarr("results/image.zarr", channel=0, level=0)
tile = img[:, :512, :512].compute()      # a small spatial block
out = segment(tile)

assert out.shape == tile.shape, (out.shape, tile.shape)
assert out.dtype.kind in "iu"            # integer labels, not a float mask
print("objects in tile:", int(out.max()))
```

## Wiring it into the cluster workflow

Point the config at your module and function:

```yaml
method: "custom"
label_name: "my_labels"
custom:
  module: "my_seg"        # import name
  function: "segment"     # default is "segment"
  kwargs:                 # optional — forwarded as segment(tile, **kwargs)
    sigma: 1.5
```

Same for the StarDist example above:

```yaml
method: "custom"
label_name: "stardist"
custom:
  module: "stardist_seg"
  function: "segment"
  kwargs:
    prob_thresh: 0.5
```

For getting the module importable on a compute node, dependency/GPU
checklist, and troubleshooting, see [Custom functions on the
cluster](snakemake.md#custom-segmentation-function) in the cluster guide.
