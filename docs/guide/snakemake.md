# Cluster workflow (Snakemake + SLURM)

The `workflow/` directory runs patchworks on a SLURM cluster: the image is
converted once, its tiles are segmented as GPU jobs in parallel, and the
results are merged into the store, with an object table per label image.
Several segmentations of one image can run together and be related to each
other (which cell each nucleus is in).

```text
convert ──▶ prepare ──▶ segment (one GPU job per batch of tiles) ──▶ merge
```

!!! tip "Prefer a form to YAML?"
    The [web launcher](launcher.md) builds the config, submits the run and
    follows it from a browser.

## 1. Set up

```bash
git clone https://github.com/imcf/patchworks
cd patchworks/workflow
pixi install
```

Keep `workflow/` on a filesystem the compute nodes can read: jobs run the
environment's Python directly.

Optional environments add methods; every task works in them (`pixi run -e
<env> ...`):

| Environment | Adds |
| --- | --- |
| `cellpose3`, `cellpose4` | a pinned Cellpose version |
| `cuda12`, `cuda13` | cupy, for GPU DoG and `dilate_gpu` (match `nvidia-smi`) |
| `plantseg` | PlantSeg and cupy ([membrane cells](membrane_cells.md)) |
| `careamics` | Noise2Void denoising (`denoise:`) |
| `viewer` | napari, for looking at results |

A CUDA environment cannot be installed or run on a login node without a
GPU; tell pixi a driver is there: `export CONDA_OVERRIDE_CUDA=12.0`.

## 2. Configure

Edit `config/config.yaml` (or keep your own copy and add `--configfile
my_config.yaml` to the commands below); set at least:

```yaml
input: "/data/scan.ims"        # .ims/.czi/.lif/.nd2/OME-TIFF/.zarr
work_dir: "/scratch/results"   # everything is written here
channel: 0                     # channel to segment (0-based)
label_name: "cells"            # becomes image.zarr/labels/cells
method: "cellpose"             # "cellpose", "threshold" or "custom"
cellpose:
  model: "cyto3"
  diameter: 30
  do_3D: true
tile_shape: "auto"             # or e.g. [16, 1024, 1024] (z, y, x)
overlap: [4, 30, 30]           # halo: about one object diameter per axis
```

Every other setting is described in the file itself, next to it. Then edit
`profile/slurm/config.yaml` for your cluster: partitions, the GPU request,
`jobs:` (how many jobs at once).

!!! warning "`runtime` is capped by the QOS"
    A job asking for more time than its QOS allows is refused by `sbatch`
    and never starts; its log stays empty. Ask for a QOS that allows it:
    `qos: "1day"` next to the `runtime:`. `sacctmgr show qos
    format=name,maxwall` lists the ceilings.

## 3. Run

```bash
pixi run dry      # the plan, nothing run
pixi run go       # locally, on this machine
pixi run slurm    # on the cluster: submits and follows the jobs
```

Run it in `tmux` on the login node: it follows the jobs until the end. A
run that stops (or fails) resumes where it was when started again with the
same command; finished steps are not redone.

## Several segmentations and their relations

`config/multi.yaml` lists segmentation configs and which label images to
relate, and runs them all with one command:

```yaml
common: config/common.yaml        # shared: input, work_dir, tile_shape, ...
segmentations:
  - config/config_nuclei.yaml     # each holds only what differs
  - config/config_cyto.yaml
  - config/config_cilia.yaml
relations:
  - {a: nuclei_labels, b: cyto_labels, output: nuclei_to_cyto.xlsx}
  - {a: cilia_labels, b: cyto_labels, output: cilia_to_cyto.xlsx, max_distance_um: 1.0}
review:                           # optional: what `patchworks review` flags
  expect: {cyto_labels: {nuclei_labels: 1}}
bundle: {format: zip}             # optional: one file of the whole store at the end
```

```bash
pixi run multi-dry   --config my_multi.yaml   # check first
pixi run multi-slurm --config my_multi.yaml
```

What it does:

- **Checks everything first**: the configs agree on `work_dir`, `level` and
  conversion settings, label names are unique, relations name label images
  that exist, each environment has what its method needs.
- **Converts once**, then runs the segmentations **concurrently**. A config
  with `seed_labels:` (cells grown from nuclei) waits for the one it grows
  from.
- **Relates** each pair: one job per parent label image, reading it once
  for all its children. Each relation gives a workbook with one row per
  child (its parent, the overlap) and one per parent (how many children); a
  sheet too long for Excel is written as `.csv`. `max_distance_um` gives a
  child touching no parent the nearest one within that distance.
- **Bundles** the store into one `.zip` when everything succeeded.

Re-running the same command redoes only what is missing or out of date.
Only one run can drive a `work_dir` at a time. A relative `--config` is
found from where you run pixi; the paths inside it, next to it.

Relate jobs take `relate: {cpus: 16, mem: "32G", time: 180, qos: "1day"}`
in the multi config (the work is mostly reading; more CPUs help up to the
filesystem's speed).

## Outputs

```text
work_dir/
  image.zarr/                    the image, as a pyramidal OME-Zarr
  image.zarr/labels/<name>/      each label image (pyramid, calibrated)
  image.zarr/labels/<name>/table one row per object: size, position, shape, relations
  <relation>.xlsx                one workbook per relation
  image.zarr.zip                 with `bundle:`
  <name>/logs/                   one log per step and per tile batch
```

Look at the results with `pixi run -e viewer napari work_dir/image.zarr`,
and check them with `patchworks review` ([Reviewing](review.md)). The store
opens in any OME-Zarr reader; a store from an older version can be brought
up to the spec with `patchworks fix-metadata image.zarr`.

## Options worth knowing

**Two channels for Cellpose** — `nuclei_channel: 1` gives Cellpose 3 the
nuclear stain as a second input. Cellpose 4 (`cpsam`) next to a bright
nuclear channel may segment the nuclei instead of the cells: give it the
membrane alone, or grow cells from the nuclei ([membrane
cells](membrane_cells.md)).

**Cellpose model names** differ between versions: v3 has `cyto3`,
`nuclei`; v4 the `cpsam` family. A name the installed version lacks is
refused before anything runs. For 3-D, the anisotropy is taken from the
image's calibration; set `cellpose: {anisotropy: 1}` if results show rings
along z (faster, slightly less exact at the top and bottom).

**Size filter** — `min_volume` / `max_volume` (µm³) drop objects outside
that range after the merge, judged on whole objects, not tile fragments.

**Connectivity** — `connectivity: 3` (threshold method, DoG plugin) joins
voxels touching at edges and corners too: thin oblique structures such as
cilia stay whole. Tiles are joined the same way.

**Growing labels** — `dilate: N` grows every label by N pixels;
`dilate_gpu: true` does it on the GPU (needs a CUDA environment).

**Batching** — `tiles_per_job: 4` segments four tiles per job with one
model load. Keep the batch's run time under the job's `runtime`.

**Empty tiles** — `skip_empty: true` (default) skips background tiles,
found from a low-resolution occupancy map.

**Fewer files** — `shard: true` packs chunks into shards: far fewer files
on a filesystem that dislikes many small ones. `shard_labels: true` also
packs label level 0 after the merge (one extra pass). Choose before
converting, or repack a finished store with `pixi run reshard --store
image.zarr`.

**OME-Zarr version** — `ngff_version: "auto"` writes 0.5 (zarr v3). Set
`"0.4"` (zarr v2, no sharding) for a reader that cannot read 0.5 yet.

**One file to copy** — `pixi run zip --store image.zarr` (or `bundle:` in
the multi config). zarr, napari and patchworks read the store straight out
of the zip. `pixi run iso` makes a read-only disk image instead, if
`xorriso`, `genisoimage` or `mkisofs` is installed.

**Email** — `notify_email: "you@example.org"` mails you when the long steps
finish or fail, with the failing step's log. Test it with `python
scripts/run_multi.py --config my_multi.yaml --test-email`.

## Your own segmentation function

`method: "custom"` imports any function from a tile to labels:

```yaml
method: "custom"
custom:
  module: "my_seg"          # a file in workflow/scripts/, or an installed package
  function: "segment"
  kwargs: {threshold: 0.5}
```

The segment jobs' environment needs everything it imports. GPU nodes are
often offline: download model weights once on the login node. See
[Custom segmentation function](custom_segmentation.md).

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Directory cannot be locked` | a run was killed: add `--unlock` to the same command, then run again |
| `another run_multi ... is already driving` | an earlier run is still going: wait, or stop it (the message names its pid) |
| A job failed, its SLURM log is empty | read the step's own log under `work_dir/<label_name>/logs/` |
| A step looks hung | every step logs progress about once a minute: `tail -f` its log |
| Jobs pend forever | wrong partition or GPU request in the profile |
| `Network is unreachable` in a segment job | offline GPU node: fetch the model once on the login node |
| `Virtual package '__cuda' does not match` | `export CONDA_OVERRIDE_CUDA=12.0` on the login node |
| Out of GPU memory | smaller `tile_shape`, or `do_3D: false` |
| `No module named ...` in a job | the method needs another environment (`-e plantseg`, `-e cuda12`, ...); the run says which |
