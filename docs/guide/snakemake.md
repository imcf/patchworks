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
| Segment jobs pend forever | wrong `slurm_partition`/`slurm_extra` GPU flag for your cluster |
| `cellpose is not installed` in a job | the job's env lacks `patchworks[cellpose]` |
| Reading the input fails | install the matching reader (`patchworks[imaris]`/`[bioio]` + a `bioio-*`) |
| Out of GPU memory | smaller `tile_shape`, or `do_3D: false` |
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
