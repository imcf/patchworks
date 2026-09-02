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
channel: 0                     # channel to segment, 0-based (null = keep all)
nuclei_channel: null           # optional 2nd channel for Cellpose (see below)
level: 0                       # pyramid level (0 = full resolution)
tile_shape: "auto"             # "auto", or e.g. [16, 1024, 1024] (zyx)
gpu_memory_gb: null            # for "auto" on SLURM: your segment GPU's VRAM
overlap: [4, 30, 30]           # halo ≈ one object diameter; scalar or [z,y,x]
skip_empty: true               # skip background tiles
empty_threshold: null          # null → Otsu
tiles_per_job: 4               # tiles per SLURM job (sequential, shared model)

# segmentation
method: "cellpose"             # "cellpose" (GPU), "threshold" (no GPU), "custom"
label_name: "cellpose"         # name under image.zarr/labels/
dilate: 0                      # optional: pixels to grow labels by, any method
dilate_gpu: false               # dilate via cupy instead of scipy (needs a GPU)
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

!!! tip "Growing labels after segmentation"
    `dilate: N` grows every label by `N` pixels once segmentation finishes,
    regardless of `method`. `0` (default) disables it. Runs on CPU (scipy)
    by default; set `dilate_gpu: true` to dilate via cupy instead — that
    needs `cupy` installed in the segment job's environment (matching your
    CUDA version, e.g. `pip install cupy-cuda12x`) and a GPU allocated for
    that job (`set-resources: segment:` in `profile/slurm/config.yaml`,
    same as for a GPU `method`). It's independent of whatever `method`
    itself runs on — you can dilate on GPU even with `method: "threshold"`
    (CPU), or on CPU even with `method: "cellpose"` (GPU). See [Growing
    labels afterwards](custom_segmentation.md#growing-labels-afterwards-dilation)
    for how it works and the equivalent direct-API call.

!!! tip "Tile size vs runtime"
    `tile_shape: "auto"` sizes each tile to your GPU's VRAM. Smaller tiles =
    more (faster) jobs; very large 3-D tiles are slow. Keep `do_3D: false` (2-D
    per slice) if your objects segment fine per slice — it is much faster.

    Tile planning runs on a **CPU** node, which cannot see the segment GPU, so
    it logs `GPU memory query failed … using 8 GiB default` and sizes tiles for
    8 GiB. Harmless, but to size for the real GPU set `gpu_memory_gb:` to its
    VRAM (e.g. `24`, `40`, `80`) — or just set `tile_shape` explicitly.

    With `do_3D: true`, `"auto"` pins z to the **full** extent: no z tiling,
    no z-boundary stitching, and Cellpose sees each object's whole depth. An
    explicit z (like `[16, 1024, 1024]`) tiles in z instead. `prepare` logs
    which regime it picked.

    `"auto"` also caps the tile to the **host** RAM available to the job, not
    just VRAM — a `do_3D` tile that comfortably fits a big GPU can still be
    too large for the SLURM/cgroup memory the job was actually granted, and
    that shows up as a plain `SIGKILL`, not a catchable CUDA-OOM error.
    Because `prepare` runs on a CPU node, it checks *its own* grant as a
    stand-in for `segment`'s — keep `prepare`'s and `segment`'s `mem_mb` in
    `profile/slurm/config.yaml` equal, or the estimate is sized against the
    wrong job's budget.

!!! tip "Use a per-axis `overlap`"
    A scalar halo is applied to every axis. On a `[16, 1024, 1024]` tile,
    `overlap: 30` reads `76 × 1084 × 1084` to keep `16 × 1024 × 1024` — 5.3×
    the voxels it uses, nearly all of it wasted z. `[4, 30, 30]` brings that
    to ~1.7×, i.e. roughly **3× less GPU time for identical results**.
    `prepare` logs the amplification and refuses a halo wider than the tile.

!!! tip "Batch tiles per job with `tiles_per_job`"
    Each job pays CUDA init plus a Cellpose weight load — tens of seconds —
    before its first voxel. `tiles_per_job: N` segments N tiles sequentially
    in one process, sharing one loaded model, which is most of the win on
    fast tiles.

    Tiles in a batch run **sequentially**, so job wall time is roughly
    `N × per-tile time` and must stay inside the profile's `runtime`/QOS
    ceiling. A retry re-runs the whole batch. Measure one tile with
    `seff <jobid>` and size N from that; `1` restores one job per tile.

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

## 5b. Run on SLURM (one GPU job per batch of tiles)

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
retries: 2                     # resubmit a failed job, asking for more memory
set-resources:
  segment:                     # the GPU step
    slurm_partition: "gpu"     # your GPU partition
    slurm_extra: "'--gres=gpu:1'"
    mem_mb: "attempt * 32000"  # grows on each retry
    runtime: 120
  merge:
    mem_mb: "attempt * 64000"
    runtime: 240
```

Then launch (from a login node — Snakemake submits and watches the jobs):

```bash
python -m snakemake --workflow-profile profile/slurm \
                    --configfile config/config.yaml
```

Snakemake submits `convert`, then `prepare`, then **one `segment` job per
batch of `tiles_per_job` non-empty tiles** (up to `jobs:` at once → that many
GPUs in parallel), then `merge`. Raise `jobs:` to use more GPUs.

!!! tip "Recognisable job names in `squeue`"
    The SLURM executor names every job after its run UUID and **rejects** a
    `--job-name` in `slurm_extra`, so by default `squeue` shows nothing you
    can identify. A prefix is the supported lever, and it goes first in the
    name (`<prefix>_<uuid>`) — the part a queue listing truncates to:

    ```yaml
    slurm-jobname-prefix: patchworks    # already in the shipped profile
    ```

    `run_multi` overrides it per config, so a three-way run shows
    `pw-convert`, then `pw-nuclei_labels` / `pw-cyto_labels` /
    `pw-cilia_labels` — telling the concurrent runs apart at a glance:

    ```bash
    squeue -u $USER -o '%.18i %.24j %.8T %.10M'
    ```

    Alphanumerics, underscores and hyphens only, 50 characters max; an
    invalid prefix fails the run, so `label_name` is sanitised before use.

!!! tip "Sizing memory"
    Every step now sizes its own worker counts from what SLURM actually
    granted (`SLURM_CPUS_PER_TASK`, `SLURM_MEM_PER_*`, the cgroup limit)
    rather than the node's totals, and `merge` logs the budget it detected.
    Compare that line against `seff <jobid>` when tuning `mem_mb`.

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

Every path the workflow writes — `tiles.json`, per-batch `seg/` markers, the
cached model, `labels.done` — lives under `work_dir/<label_name>/`, so
running the workflow **twice with two configs against the same `work_dir`**
never collides: each run gets its own private subdirectory, and both reuse
the *same* already-converted `image.zarr` (conversion never re-runs).

Most of what those configs contain is identical — the input, the `work_dir`,
the tiling, everything `convert` reads. Put it in **one** shared file and let
each config carry only what actually differs. Snakemake merges several
`--configfile` values in order, with the later one winning:

```yaml
# config/common.yaml — shared by every segmentation
input: "/data/scan.ims"
work_dir: "/scratch/results"
tile_shape: [16, 1024, 1024]
shard: false                   # true → far fewer files, same chunks
tiles_per_job: 4
```

```yaml
# config/config_nuclei.yaml — only the differences
label_name: "nuclei_labels"
channel: 1                # nuclear stain channel
overlap: [4, 30, 30]
method: "cellpose"
cellpose:
  model: "nuclei"
  diameter: 15
  do_3D: true
```

```yaml
# config/config_cyto.yaml — only the differences
label_name: "cyto_labels"
channel: 0                # cytoplasm/membrane channel
nuclei_channel: 1         # optional: nuclear stain, as Cellpose's 2nd input
overlap: [4, 30, 30]
method: "cellpose"
cellpose:
  model: "cyto3"
  diameter: 30
  do_3D: true
```

### Giving Cellpose a nuclei channel

`nuclei_channel` hands Cellpose a second channel — the nuclear stain — which
usually improves cytoplasm segmentation. Both indices are 0-based, like
`channel`.

Only the `segment` step reads it. The pair is stacked on a leading axis that
is *carried* into each tile rather than tiled, so the tile geometry, the
occupancy map and the staged labels are byte-for-byte what a single-channel
run produces, and `merge` and `label_relations` need no changes. Two things
follow from that:

- A tile holds twice the bytes. `tile_shape: "auto"` accounts for this — it is
  told the tile carries two channels and shrinks each spatial side by ~1/√2,
  so the tile still fits the same VRAM *and* host-RAM budget (see the "Tile
  size vs runtime" tip above). A **hand-set** `tile_shape` sized to fill a
  GPU has no such protection and needs halving yourself.
- The translation is version-specific. Cellpose 3 gets `channels: [1, 2]`
  (1-based into the channel axis, `0` = grayscale); Cellpose 4 (cpsam) dropped
  `channels` entirely and simply reads both. Either is overridable by setting
  `channels:` or `channel_axis:` in the `cellpose:` block.

!!! warning "In a `multi.yaml` run, pin `tile_shape` explicitly"

    `label_relations` requires its two label arrays to share a chunk layout,
    and that layout comes from `tile_shape`. Giving *one* config a
    `nuclei_channel` while the group uses `tile_shape: "auto"` produces a
    **smaller** tile for that config only — so the label groups end up chunked
    differently and the relations step fails, after every segmentation has
    already run.

    `run_multi`'s cross-config check compares the configured values, and
    `"auto" == "auto"`, so it flags this case specifically. Fix it by setting
    one explicit `tile_shape` in `common.yaml`, sized for the two-channel
    config (roughly each spatial side ÷ √2 versus what you would use for a
    single channel), so every config shares it.

If you do not need that segmentation related to the others, the alternative is
to run it on its own against the same `work_dir` and leave it out of
`multi.yaml`.

`nuclei_channel` applies to the SLURM/Snakemake path. The single-process
`tile_process` API still takes one `channel`.

Run them as two independent SLURM submissions — they touch disjoint files, so
they can run concurrently. Give each its own `--directory`, because
Snakemake's lock lives in the working directory, not in the config:

```bash
snakemake --workflow-profile profile/slurm --configfile config/common.yaml config/config_nuclei.yaml --directory /scratch/results/nuclei_labels/.snakemake
```

```bash
snakemake --workflow-profile profile/slurm --configfile config/common.yaml config/config_cyto.yaml --directory /scratch/results/cyto_labels/.snakemake
```

!!! warning "Conversion settings belong in the shared file"
    `convert` runs **once**, from the first config only. A `shard`, `input` or
    `pyramid_levels` set on the second config is therefore never read, and
    nothing logs that it was dropped. `run_multi` refuses to start when those
    keys disagree across configs and tells you which one — but if you drive
    the configs by hand, keep them in `common.yaml`.

    Splitting the configs is optional: a self-contained config still works,
    and `common:` can simply be left out of `multi.yaml`.

!!! tip "One command for several segmentations + relations"
    `config/multi.yaml` lists any number of segmentation configs plus which
    pairs to relate afterward; `pixi run multi` (or `multi-slurm`) converts
    once, then runs every config **concurrently** with its own state
    directory, and writes a workbook per pair — see *One command: multiple
    segmentations + relations* below for the config format.

Both land side by side in the same store:

```text
results/image.zarr/labels/nuclei_labels/
results/image.zarr/labels/cyto_labels/
```

!!! tip "Keep `tile_shape` (and `level`) identical across configs"
    Different segmentations of the same image can use different `channel` and
    `cellpose:` settings freely, but keep `tile_shape`/`level` the same across
    configs — the label arrays then share the exact same chunk layout, which
    [`label_relations()`](label_relations.md) requires.

See [Relating labels across segmentations](label_relations.md) for what
`label_relations()` returns and how to save it yourself — the cluster
workflow's own automation is below.

### One command: multiple segmentations + relations

`scripts/run_multi.py` (wired up as `pixi run multi`) drives the above: it
converts once, runs every segmentation config **concurrently**, then computes
and saves every configured relation — one command instead of juggling several
`snakemake` calls and a separate Python step.

```yaml
# config/multi.yaml
common: config/common.yaml # shared settings, merged under each config below

segmentations:
  - config/config_nuclei.yaml
  - config/config_cyto.yaml

relations:
  - a: nuclei_labels
    b: cyto_labels
    output: nuclei_to_cyto.xlsx # written into work_dir
```

`common:` is optional — leave it out and each config must be self-contained,
as before. With it, changing the input path or turning on `shard` is a
one-line edit in one file instead of the same edit repeated per config.

```bash
pixi run multi-dry    # dry-run every segmentation config (skips relations)
pixi run multi        # run locally
pixi run multi-slurm  # submit every segmentation to SLURM
```

Before anything is submitted, the script checks that every listed config
shares one `work_dir` (so `label_relations` has a single `image.zarr` to read
both label groups from), that `tile_shape` and `level` are identical (so the
label arrays share a chunk layout), and that `label_name` is unique — two
configs sharing one would silently overwrite each other's outputs. `relations`
is optional; omit it to just run segmentations without a relation step.

The configs then run at the same time, each with its own state directory, so
the GPU partition stays busy instead of idling through every config's
`prepare` and multi-hour `merge` in turn. A config that fails does **not**
abort the others; you get a per-config status and a non-zero exit.

!!! tip "The relate step runs on the cluster too, under `multi-slurm`"
    `label_relations()` streams every chunk of two full-resolution label
    volumes — real CPU/IO work, not orchestration. Under `multi-slurm` it is
    submitted as its own `srun` job (`scripts/relate.py`) instead of running
    in the driver process on the login node, the same fix already applied to
    the occupancy map. Tune its allocation with `--relate-partition`,
    `--relate-mem`, `--relate-cpus` and `--relate-time` (defaults: `scicore`,
    `32G`, `8`, `180` minutes) — these are wide-margin guesses, not measured
    numbers, so raise them for a very large or very object-dense pair. Under
    plain `multi` (no `--profile`), it still runs locally, in-process, as
    before.

!!! tip "After a killed run"
    Snakemake only releases its lock on a clean exit, so a run that was killed
    (Ctrl-C, an SSH drop, an OOM) leaves the directory locked. Each phase has
    its own state directory, so releasing them by hand means reconstructing
    several paths — use the flag instead:

    ```bash
    pixi run multi -- --unlock     # then re-run normally
    ```

    Running the orchestrator under `tmux` avoids most of these in the first
    place: it survives a dropped connection.

!!! warning "`jobs:` is per config"
    The profile's `jobs:` caps one Snakemake process. Running three configs
    concurrently can therefore have 3× that many jobs in flight — lower it if
    that would exceed your cluster quota.

Each `output:` is an Excel workbook (`openpyxl`, part of the `workflow`
extra) with two sheets:

| Sheet | One row per | Columns |
| --- | --- | --- |
| `<a>` | every non-background `a` label, **including unmatched ones** | `<a>_id`, `<b>_id` (blank if unmatched), `overlap_voxels`, `overlap_fraction` (0 if unmatched) |
| `<b>` | every non-background `b` label, **including ones with zero matches** | `<b>_id`, `<a>_count`, `total_overlap_voxels` |

Unlike calling [`label_relations()`](label_relations.md) directly (which
only returns matched `a` labels), the workbook always covers every object in
both segmentations, so counts (e.g. "how many nuclei have no matching cell",
"how many cells have zero cilia") aren't silently dropped.

Both lists are ordinary lists, so 3+ segmentations work the same way — add
more entries to `segmentations`, then list whichever pairs to relate. There's
no automatic "chain": list every pair explicitly, e.g. for nuclei + cyto +
membrane you'd add `nuclei_labels -> cyto_labels`, `nuclei_labels ->
membrane_labels`, and `cyto_labels -> membrane_labels` as three separate
entries under `relations`.

The shipped `config/multi.yaml` is actually a three-way example: nuclei +
cytoplasm (Cellpose) plus cilia (`method: "custom"` ->
[`patchworks.plugins.dog`](../examples/dog.md), deconvolution + a
difference-of-Gaussians detector), related both ways (`cilia_labels ->
cyto_labels` and `cilia_labels -> nuclei_labels`) so you can use whichever
fits a given dataset. See `config/config_cilia.yaml`. Its deconvolution step
needs `pip install "patchworks[dog]"` in the segment jobs' environment.

## Email notifications

Set an address and the workflow mails you when the long steps finish or fail:

```yaml
# config/common.yaml (or config/config.yaml for a single-config run)
notify_email: "you@unibas.ch"
notify_events: ["finish", "error"] # any of: start, finish, error
```

Leave `notify_email` empty (the default) and nothing is sent.

Per-job mail is **SLURM's own** `--mail-type`, not a message sent from inside
the job. That matters: the controller sends it, so it still arrives when a job
is OOM-killed or cancelled by the scheduler — exactly the failures worth
hearing about, and exactly the ones a notification sent from within the job
would miss.

It is applied to the long single-job steps only — `convert`, `occupancy` and
`merge`. `segment` is deliberately excluded: there is one job per tile batch,
so a thousand-tile run would mean hundreds of messages.

On top of that, the workflow itself sends:

| When | Mail |
| --- | --- |
| The run fails | subject `[patchworks] FAILED: <label_name>`, with the last 40 lines of the failing step's log — usually the traceback itself |
| The run succeeds | subject `[patchworks] done: <label_name>`, with the output label path |

These cover what SLURM cannot: a local run with no scheduler at all, and
failures where the useful content is the Python traceback rather than an exit
code.

Check it works without waiting for a multi-hour step:

```bash
python scripts/run_multi.py --config config/multi.yaml --test-email
```

It prints the address it actually resolved from the merged config, the exact
`sbatch` flags the jobs will carry, and whether a test message was accepted —
which separates "the address never reached the config" from "it did and the
mail was dropped downstream". Those look identical otherwise: no email either
way.

!!! warning "No mail from `segment`"
    `segment` is excluded on purpose, so a run that is only segmenting tiles
    sends nothing. Mail comes from `convert`, `occupancy` and `merge`, plus
    the workflow-level success/failure message — if those already completed
    before you set the address, there is nothing left in the run to mail you.

!!! note "Delivery is best-effort, by design"
    A notification can never fail a run. If no local `sendmail` exists and no
    SMTP server answers on localhost, the failure is logged as a warning and
    the pipeline carries on — a six-hour segmentation that worked must not be
    reported as failed because a mail host was down. If you get the warning
    but no mail, ask your cluster admins which relay host to use.

## Measurements

See [Measurements](measurements.md) for computing area/centroid/intensity
stats on a whole label store, interactively in napari or headless/scripted —
`skimage.measure.regionprops` alone doesn't scale to a store this size.

## Custom segmentation function

Not using Cellpose? See [Custom segmentation function](custom_segmentation.md)
for the function contract, examples (including label dilation), and how to
test it before submitting. The rest of this section covers what's specific to
running it **on the cluster**: getting the module importable and the
checklist below.

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
| `Directory cannot be locked` | a previous run was killed instead of exiting cleanly. For a multi-run: `pixi run multi -- --unlock` (it covers every state directory, including the conversion one). For a single config: add `--unlock --directory <the same one you ran with>` |
| `BioImage does not support the image: '.../*tif'` | a glob input needs `sequence_pattern` — see [A folder of single-plane TIFFs](ome_zarr_napari.md#a-folder-of-single-plane-tiffs) |
| Segment jobs pend forever | wrong `slurm_partition`/GPU request; on scicore use `gres: "gpu:1"` |
| Segment dies, `Network is unreachable` | offline GPU nodes — the `fetch_model` localrule caches the model on the submit host first; if it still fails, your submit host has no network either (pre-download manually) |
| `cellpose is not installed` in a job | the job's env lacks `patchworks[cellpose]` |
| Reading the input fails | install the matching reader (`patchworks[imaris]`/`[bioio]` + a `bioio-*`) |
| Out of GPU memory | smaller `tile_shape`, or `do_3D: false` |
| A job fails with an empty SLURM log | read the step's own log — `logs/convert.log`, `logs/prepare.log`, `logs/segment/<batch>.log`, `logs/merge.log` — the real traceback is there |
| A long step looks hung | every step logs progress (`… 4,200/8,064 (52%) after 31m, ~28m left`) roughly once a minute; `tail -f` the step's log above |
| Very slow | confirm GPU is used (`nvidia-smi`); try 2-D or a lower `level` |

## How it works (for the curious)

The rule scripts are thin wrappers over patchworks' public API, so you can build
the same per-tile distribution yourself:

```python
from patchworks import (
    load_ome_zarr, spatial_tiles, create_stage, stage_tile, merge_tile_labels
)
from patchworks.plugins.ome_zarr import write_labels

TILE = (16, 1024, 1024)
img = load_ome_zarr("image.zarr", channel=0)
tiles = spatial_tiles(img.shape, tile_shape=TILE)
create_stage("stage.zarr", img.shape, TILE)

# (distribute these across jobs:) each returns how many labels it wrote
counts = {}
for i in range(len(tiles)):
    counts[i] = stage_tile(
        img, my_fn, "stage.zarr", i, tile_shape=TILE, overlap=[4, 30, 30]
    )

# Handing those counts to the merge lets it derive each tile's global id range
# by a cumulative sum, instead of streaming the whole store to renumber it.
merged = merge_tile_labels("stage.zarr", input_component="staged",
                           write_to="merged.zarr", sequential_labels=True,
                           label_counts=counts)
write_labels("image.zarr", merged, name="cells")
```

The workflow goes two steps further. It stages **into**
`image.zarr/labels/cells/0` in the first place, then has the merge rewrite
that array **in place** (`write_to=` and `input_component=`/
`output_component=` all pointing at it) and calls `register_labels` to add
the pyramid. That removes both the scratch store and the full-volume copy
`write_labels` would otherwise do — three passes over the label volume become
two.

It falls back to the separate store above when the tile is larger than the
label chunk cap, because in place the chunking cannot be changed and level 0
has to stay pageable for a viewer.
