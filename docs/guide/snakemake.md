# Cluster workflow (Snakemake + SLURM)

`tile_process` runs every tile **serially on one GPU**. For a large 3-D image
that can be days. The bundled Snakemake workflow instead submits **one GPU job
per tile**, so with *N* GPUs the segmentation is ~*N*× faster. This page walks
through running it from scratch.

```text
convert ──▶ prepare (checkpoint) ──▶ segment {tile}  ──▶ merge
                                     one GPU SLURM job per tile
```

## 1. Get the workflow

The workflow lives in the `workflow/` directory of the patchworks repository
(it is not shipped inside the pip package — it is a set of Snakemake files you
run):

```bash
git clone https://github.com/imcf/patchworks
cd patchworks/workflow
```

## 2. Install the dependencies

You need patchworks with the workflow + reader + segmentation extras, in the
environment Snakemake will use:

```bash
pip install "patchworks[workflow,cellpose,imaris,bioio]"
```

- `workflow` → Snakemake + the SLURM executor plugin
- `cellpose` → the segmentation model
- `imaris` / `bioio` → read your input format (`.ims`, `.czi`, `.lif`, …)

On a cluster, do this inside a conda/venv/pixi env that the compute nodes can
see, or let each rule activate a conda env. Prefer pixi? Skip this step — the
workflow ships a `pixi.toml`; see *pixi (instead of conda)*, below.

## 3. Configure the run

Copy and edit `config/config.yaml`. Every field:

```yaml
# input / output
input: "/data/scan.ims"        # .ims/.czi/.lif/.nd2/ome-tiff/.zarr
work_dir: "/scratch/results"   # everything is written here

# conversion (input → pyramidal OME-ZARR)
reuse_pyramid: true            # .ims: copy its own pyramid (fast)
convert_chunks: null           # null → bounded auto chunks; or [c,z,y,x]
shard: false                   # true → pack chunks into shards (fewer files)

# tiling
channel: 0                     # channel to segment (null = keep all)
level: 0                       # pyramid level (0 = full resolution)
tile_shape: "auto"             # "auto", or e.g. [16, 1024, 1024] (zyx)
gpu_memory_gb: null            # for "auto" on SLURM: your segment GPU's VRAM
overlap: 30                    # halo ≈ one object diameter
skip_empty: true               # skip background tiles
empty_threshold: null          # null → Otsu

# segmentation
method: "cellpose"             # "cellpose" (GPU) or "threshold" (no GPU)
label_name: "cellpose"         # name under image.zarr/labels/
cellpose:
  model: "cyto3"
  diameter: 30
  do_3D: true
  gpu: true
  # extra model.eval() kwargs, e.g. flow_threshold: 0.4

# label pyramid
pyramid_levels: 5
pyramid_downscale: 2
sequential_labels: true        # renumber labels to a contiguous 1..N
```

!!! tip "Tile size vs runtime"
    `tile_shape: "auto"` sizes each tile to your GPU's VRAM. Smaller tiles =
    more (faster) jobs; very large 3-D tiles are slow. Keep `do_3D: false` (2-D
    per slice) if your objects segment fine per slice — it is much faster.

    Tile planning runs on a **CPU** node, which cannot see the segment GPU, so
    it logs `GPU memory query failed … using 8 GiB default` and sizes tiles for
    8 GiB. Harmless, but to size for the real GPU set `gpu_memory_gb:` to its
    VRAM (e.g. `24`, `40`, `80`) — or just set `tile_shape` explicitly.

## 4. Dry-run (always do this first)

Check the plan without running anything:

```bash
python -m snakemake -s Snakefile --configfile config/config.yaml -n -p
```

You should see `convert`, `prepare`, and a note that the **checkpoint** will add
the `segment` jobs after `prepare` runs. (The number of segment jobs is only
known after `prepare` decides which tiles are non-empty.)

## 5a. Run locally (single machine)

```bash
python -m snakemake -s Snakefile --configfile config/config.yaml \
    --rerun-triggers mtime --cores 8
```

Tiles run on the local machine (one at a time on the GPU). Good for a small
image or a smoke test. `--rerun-triggers mtime` re-runs only steps whose output
is missing/stale — so upgrading patchworks won't redo the conversion (the SLURM
profile sets this for you).

## 5b. Run on SLURM (one GPU job per tile)

Edit `profile/slurm/config.yaml` for **your** cluster — partitions, account,
and the GPU request:

```yaml
executor: slurm
jobs: 64                       # max concurrent SLURM jobs ≈ GPUs you can grab
default-resources:
  slurm_partition: "cpu"       # your CPU partition
  # slurm_account: "my_account"
  mem_mb: 16000
  cpus_per_task: 4
  runtime: 60
set-resources:
  segment:                     # the GPU step
    slurm_partition: "gpu"     # your GPU partition
    slurm_extra: "'--gres=gpu:1'"
    mem_mb: 32000
    runtime: 120
  merge:
    mem_mb: 128000
    runtime: 240
```

Then launch (from a login node — Snakemake submits and watches the jobs):

```bash
python -m snakemake --workflow-profile profile/slurm \
                    --configfile config/config.yaml
```

Snakemake submits `convert`, then `prepare`, then **one `segment` job per
non-empty tile** (up to `jobs:` at once → that many GPUs in parallel), then
`merge`. Raise `jobs:` to use more GPUs.

!!! note "GPU request flag"
    Clusters differ. `--gres=gpu:1` is common; some need `--gpus=1` or a
    specific gres name (`--gres=gpu:a100:1`). Put whatever `sbatch` flag your
    cluster needs in `slurm_extra`.

## 6. Monitor

- **Snakemake** prints each job as it submits/finishes and a `X of Y steps`
  counter.
- **SLURM**: `squeue --me` shows your queued/running jobs (`smk-segment`, …);
  logs land where your profile/cluster sends them.
- **patchworks** logs (`processing tile k/N`, ETA) are inside each job's stdout.

## 7. Output

Everything is under `work_dir`:

```text
results/
  image.zarr/                 # converted, pyramidal OME-ZARR
  image.zarr/labels/<name>/   # the segmentation (multi-scale, calibrated)
```

The labels live **inside** the image store. View image + labels together:

```python
from patchworks.plugins.napari import view_in_napari
view_in_napari("/scratch/results/image.zarr")   # auto-loads the labels
```

## 8. Re-running and resuming

Snakemake is resumable — if jobs fail or you cancel, just relaunch the same
command and it picks up only the missing tiles. To force a clean rerun, delete
`work_dir` (or the relevant outputs).

The OME-ZARR conversion is **not redone** once `image.zarr` exists: the
`convert` rule's output is a marker file inside the store, so Snakemake skips it
on every later run. To force a fresh conversion, delete `image.zarr` or run
`snakemake --forcerun convert`.

This relies on `--rerun-triggers mtime` (set in the SLURM profile and the pixi
tasks; add it on the command line for ad-hoc local runs). Without it, Snakemake
also re-runs a step when its **code, params or software environment** change —
so upgrading patchworks would re-do the conversion and overwrite an existing
result. Keep `mtime` and reruns happen only when an output is missing or stale.

## Running two segmentations (e.g. nuclei + cytoplasm)

Every path the workflow writes — `tiles.json`, `stage.zarr`, per-tile `seg/`,
the cached model, `labels.done` — lives under `work_dir/<label_name>/`, so
running the workflow **twice with two configs against the same `work_dir`**
never collides: each run gets its own private subdirectory, and both reuse
the *same* already-converted `image.zarr` (conversion never re-runs).

```yaml
# config/config_nuclei.yaml
input: "/data/scan.ims"
work_dir: "/scratch/results"
label_name: "nuclei_labels"
channel: 1                # nuclear stain channel
tile_shape: [16, 1024, 1024]
cellpose:
  model: "nuclei"
  diameter: 15
  do_3D: true
```

```yaml
# config/config_cyto.yaml
input: "/data/scan.ims"
work_dir: "/scratch/results"   # same work_dir — image.zarr is reused
label_name: "cyto_labels"
channel: 0                # cytoplasm/membrane channel
tile_shape: [16, 1024, 1024]   # keep this identical across configs — see below
cellpose:
  model: "cyto3"
  diameter: 30
  do_3D: true
```

Run them one after another (or as two independent SLURM submissions, even
concurrently — they touch disjoint files):

```bash
snakemake --workflow-profile profile/slurm --configfile config/config_nuclei.yaml
snakemake --workflow-profile profile/slurm --configfile config/config_cyto.yaml
```

Both land side by side in the same store:

```text
results/image.zarr/labels/nuclei_labels/
results/image.zarr/labels/cyto_labels/
```

!!! tip "Keep `tile_shape` (and `level`) identical across configs"
    Different segmentations of the same image can use different `channel` and
    `cellpose:` settings freely, but keep `tile_shape`/`level` the same across
    configs — the label arrays then share the exact same chunk layout, which
    `label_relations()` (below) requires.

### Relating labels across segmentations

Once you have two segmentations of the same image (e.g. nuclei inside cells),
`label_relations()` maps each label in one to the label it overlaps most in
the other — by streaming both arrays chunk by chunk, so it scales to
hundreds of thousands of objects without loading anything fully into RAM:

```python
import dask.array as da
from patchworks import label_relations

nuclei = da.from_zarr("results/image.zarr", component="labels/nuclei_labels/0")
cells = da.from_zarr("results/image.zarr", component="labels/cyto_labels/0")

table = label_relations(nuclei, cells)
table[2]
# {'match': 3, 'overlap_voxels': 4821, 'overlap_fraction': 0.94}
# -> nucleus 2 belongs to cell 3, 94% of its voxels fall inside it
```

Save it as a table:

```python
import csv

with open("nuclei_to_cell.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["nucleus_id", "cell_id", "overlap_voxels", "overlap_fraction"])
    for nucleus_id, m in table.items():
        w.writerow([nucleus_id, m["match"], m["overlap_voxels"], m["overlap_fraction"]])
```

## Measurements (fast, whole-volume regionprops)

`skimage.measure.regionprops` needs the full labelled + intensity array in
RAM — fine for one tile, not for a hundred-thousand-object OME-ZARR. Use
[`dask-image`](https://image.dask.org)'s `ndmeasure`, which computes directly
on the dask/zarr-backed arrays, chunk-parallel, without materializing the
volume:

```bash
pip install dask-image
```

```python
import dask.array as da
from dask_image.ndmeasure import area, center_of_mass, mean, standard_deviation

labels = da.from_zarr("results/image.zarr", component="labels/cyto_labels/0")
image = da.from_zarr("results/image.zarr", component="0")[0]  # channel 0, level 0

ids = da.unique(labels[labels > 0]).compute()
areas = area(image, labels, ids).compute()               # voxel counts
means = mean(image, labels, ids).compute()                # mean intensity
stds = standard_deviation(image, labels, ids).compute()
centroids = center_of_mass(image, labels, ids).compute()  # voxel coords (z, y, x)
```

Multiply `areas` by the voxel's physical volume and `centroids` by the pixel
size (both read straight from the OME-ZARR's own `multiscales` metadata) to
get µm-scale measurements.

For interactively inspecting individual cells by clicking in the viewer
(not all objects at once), the
[napari-skimage-regionprops](https://github.com/haesleinhuepf/napari-skimage-regionprops)
plugin's table widget works well — point it at a cropped region rather than
the full volume, since it loads its input fully into memory.

## Custom segmentation function

Not using Cellpose? Run **your own** per-tile function — no need to edit the
package. You write one function; patchworks handles everything around it
(tiling, halos, skipping empty tiles, the zarr-native merge, global relabelling,
resume, logs).

### The contract

Your function is called **once per tile**:

```python
labels = segment(tile)          # plus any kwargs you configure
```

| | What you get / must return |
| --- | --- |
| **Input `tile`** | A NumPy array of **one** tile, with the overlap halo already included. The channel and pyramid level from the config are already selected, so it is purely spatial: `(z, y, x)` for a 3-D run, `(y, x)` for 2-D. Dtype is the image's (e.g. `uint16`). |
| **Return** | An integer **label** array (not a boolean mask), **same shape** as `tile`. `0` = background; each object a distinct positive integer. |
| **Labels** | Only need to be unique **within the tile**. Don't try to make them globally unique — the merge step stitches objects across tile borders and renumbers everything to a contiguous `1..N` (`sequential_labels: true`). |
| **Shape** | Must match the input exactly — patchworks trims the halo off your output, so a wrong shape is an error. Don't crop or resize inside the function. |

That is the whole interface. Anything that turns an image tile into a label
image works: classic image processing, StarDist, a trained model, an external
binary you shell out to, …

### Minimal example (no GPU, no deps beyond scikit-image)

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

```yaml
method: "custom"
label_name: "my_labels"
custom:
  module: "my_seg"        # import name (see "Make it importable")
  function: "segment"     # default is "segment"
  kwargs:                 # optional — forwarded as segment(tile, **kwargs)
    sigma: 1.5
```

### Real example: StarDist 3-D, with model caching

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

```yaml
method: "custom"
label_name: "stardist"
custom:
  module: "stardist_seg"
  function: "segment"
  kwargs:
    prob_thresh: 0.5
```

Using a GPU? The segment jobs already hold one (the `gres: "gpu:1"` request),
so just let your framework see it — nothing extra in the config.

### Test it before you submit

Run your function on one real tile first — it catches shape/dtype bugs in
seconds instead of after a queue wait. Output must be integer, same shape, `0`
for background:

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

### Make it importable on the cluster

The segment job runs `import <module>`, so the module must be on the path. Pick
one:

1. **Drop the file in `workflow/scripts/`** — Snakemake adds the script dir to
   `sys.path`, so `module: "my_seg"` just works. Simplest for a single file.
2. **Install it** into the run env (`pip install -e .`, `pixi add --pypi …`),
   then use its import name. Best for a real package with dependencies.
3. **Set `PYTHONPATH`** to the file's directory before launching Snakemake.

### Cluster checklist

- **Dependencies:** the env that runs the **segment** jobs must have everything
  your function imports (`pip`/`pixi add` it). A missing import or any crash
  shows up in `logs/segment/<index>.log`, not the (empty) SLURM log.
- **Offline GPU nodes:** the built-in `fetch_model` prefetch covers Cellpose
  only. If your function downloads weights/data on first use, fetch them once on
  the **login node** (network access) so they land in shared `$HOME`; otherwise
  the segment jobs fail with `Network is unreachable`. See *Troubleshooting*.
- **Memory / walltime:** tune `segment:` in `profile/slurm/config.yaml` for your
  model (`mem_mb`, `runtime`) just as for Cellpose.
- Everything else — tiling, halos, empty-tile skipping, the zarr-native merge,
  resume, and per-tile logs — is identical to a Cellpose run.

For full control (your own tiling/merge loop instead of the bundled rules), call
the public API directly — see *How it works* below.

## pixi (instead of conda)

Conda is **not** required — Snakemake runs in whatever environment launches it.
The workflow ships a `pixi.toml`, so the whole thing is:

```bash
cd workflow
pixi install          # builds the env (patchworks + snakemake + readers)
pixi run dry          # dry-run
pixi run go           # run locally (8 cores)
pixi run slurm        # submit to SLURM (edit profile/slurm/config.yaml first)
```

`pixi run …` activates the env, so the rule scripts execute in that env — do
**not** pass `--use-conda`. On a cluster, keep the `workflow/` directory on a
shared filesystem the compute nodes can read: the SLURM executor re-launches
Snakemake from this env's interpreter on each compute node.

## Conda (optional)

To have each rule run in a named conda env instead of the active one, add
`--use-conda` and point the rules at an env; or activate your env in a SLURM
prologue. The simplest path is a single shared env that the compute nodes see.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `snakemake: command not found` | use `python -m snakemake` |
| Segment jobs pend forever | wrong `slurm_partition`/GPU request; on scicore use `gres: "gpu:1"` |
| Segment dies, `Network is unreachable` | offline GPU nodes — the `fetch_model` localrule caches the model on the submit host first; if it still fails, your submit host has no network either (pre-download manually) |
| `cellpose is not installed` in a job | the job's env lacks `patchworks[cellpose]` |
| Reading the input fails | install the matching reader (`patchworks[imaris]`/`[bioio]` + a `bioio-*`) |
| Out of GPU memory | smaller `tile_shape`, or `do_3D: false` |
| A job fails with an empty SLURM log | read `logs/segment/<index>.log` (per tile) or `logs/steps.log` — the real traceback is there |
| Very slow | confirm GPU is used (`nvidia-smi`); try 2-D or a lower `level` |

## How it works (for the curious)

The rule scripts are thin wrappers over patchworks' public API, so you can build
the same per-tile distribution yourself:

```python
from patchworks import (
    load_ome_zarr, spatial_tiles, create_stage, stage_tile, merge_tile_labels
)
from patchworks.plugins.ome_zarr import write_labels

img = load_ome_zarr("image.zarr", channel=0)
tiles = spatial_tiles(img.shape, tile_shape=(16, 1024, 1024))
create_stage("stage.zarr", img.shape, (16, 1024, 1024))
# (distribute these across jobs:)
for i in range(len(tiles)):
    stage_tile(img, my_fn, "stage.zarr", i, tile_shape=(16, 1024, 1024), overlap=30)
merged = merge_tile_labels("stage.zarr", input_component="staged",
                           write_to="merged.zarr", sequential_labels=True)
write_labels("image.zarr", merged, name="cells")
```
