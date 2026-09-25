# Web launcher (Streamlit)

The [cluster workflow](snakemake.md) is driven by YAML files and shell
commands. The launcher is a small web app that does the same through a
form. It:

- logs in to the cluster over SSH
- builds the config for you
- uploads the config and starts the run
- shows the run's progress

It does nothing the command line can't do. It is there so people who don't
live in a terminal can run a segmentation, and so nobody starts a run with
a typo in a YAML key.

```text
your browser ──▶ launcher (Streamlit) ──SSH──▶ cluster login node
                                                 ├─ uploads config/launcher_*.yaml
                                                 └─ sbatch controller job ─▶ snakemake / run_multi.py
                                                                             ─▶ one SLURM job per tile
```

## Before you start

Set this up once per cluster:

1. **Install the workflow on the cluster**, as in steps 1–2 of the
   [cluster workflow](snakemake.md#1-get-the-workflow). You need a
   `patchworks/workflow` directory with an environment that has Snakemake
   and patchworks. The launcher runs what is already there; it installs
   nothing.
2. **Adapt `workflow/profile/slurm/config.yaml`** to your cluster's
   partitions and GPUs, as for a command-line run.
3. **Pin the cluster's host key** in `workflow/launcher/clusters.yaml`.
   This lets the launcher check that it is talking to the real cluster
   before it sends your password:

    ```bash
    ssh-keyscan login.example.org | ssh-keygen -lf -
    ```

    Put each `SHA256:…` value you get in `host_key_fingerprints:` of the
    cluster's entry, together with the defaults you want to offer:

    ```yaml
    mycluster:
      host: login.example.org
      host_key_fingerprints:
        - "SHA256:abc…"        # ed25519
        - "SHA256:def…"        # ecdsa / rsa, if the server offers them
      workflow_dir_hint: /home/<user>/patchworks/workflow
      setup_cmd: 'eval "$(pixi shell-hook)"'   # or: module load …; conda activate …
      controller_time: "3-00:00:00"
      controller_mem: "4G"
      controller_partition: ""                 # blank = cluster default
    ```

    `setup_cmd` runs in the workflow directory before every command. It
    must make `snakemake`, `patchworks` and `sbatch` available there.

## Start the app

On your own machine, or on a server inside your institute's network:

```bash
cd patchworks/workflow/launcher
pip install -r requirements.txt      # streamlit, paramiko, pyyaml
streamlit run app.py
```

The app opens at <http://localhost:8501>. To offer it to a team, run it
on an internal server (`streamlit run app.py --server.address 0.0.0.0`),
behind your usual reverse proxy or VPN. Everyone logs in with their own
cluster account, and sessions are isolated per browser.

!!! warning "Don't host it publicly"
    The login form takes real cluster passwords. Don't deploy it on a
    public service such as Streamlit Community Cloud. Keep it where only
    the people it is for can reach it.

## 1. Connect

In the sidebar:

1. Pick a **preset**, or *Custom* and type the host and port.
2. Enter your **username** and **password** and press **Connect**. The app
   compares the server's host key with the pinned fingerprints and refuses
   to connect if none match. A host with no pinned fingerprint is refused
   as well, unless you tick *allow an unverified host key*. Only tick that
   when you run the app yourself, on your own machine. The password is used
   only to open the connection. It isn't stored, and the form clears it
   straight away.
3. Enter the **workflow directory on the cluster**, e.g.
   `/home/<user>/patchworks/workflow`.
4. Check the **environment setup** line, which comes pre-filled from the
   preset.

## 2. Configure the run (Config tab)

First choose what to run:

- **One segmentation**: one label image, run with `snakemake`.
- **Several segmentations + relations (multi)**: for example nuclei, cells
  and cilia on the same image, followed by the tables relating them. This
  runs with `scripts/run_multi.py`, the command-line counterpart
  described in [One command: multiple segmentations +
  relations](snakemake.md#one-command-multiple-segmentations-relations).

### Shared settings

These apply to every segmentation in the run:

| Section | Settings |
| --- | --- |
| Input / output | `input` (a file on the cluster), `work_dir` (where everything is written) |
| Conversion | pyramid levels and chunking of the converted image, `compression`, `reuse_pyramid`, `shard`, `ngff_version` |
| Tiling | `level`, `tile_shape` (or `auto`), tiles per job, GPU memory, `skip_empty` |
| Merge and label pyramid | `stitch` (`touch` or `iou`), label pyramid, `sequential_labels`, `shard_labels`, `seam_report`, `merge_workers` |
| Notifications | `notify_email` |

The settings mean the same as in the [config
reference](snakemake.md#3-configure-the-run).

### Per segmentation

- **`label_name`**: the name of the label image, under
  `image.zarr/labels/<label_name>`. It must be unique within a run.
- **`channel`**, and optionally **`nuclei_channel`**: the second channel
  given to Cellpose.
- **`overlap`**: the halo around each tile. Use a single number, or
  `[z, y, x]`.
- **Post-processing**: `fill_holes` (off, 3-D, or per plane);
  `open_radius`, which removes thin spurs (leave it at 0 for cilia);
  `dilate`; `min_volume` / `max_volume`, in µm³.

The **`method`** is one of:

| Method | Settings |
| --- | --- |
| **cellpose** | Model, diameter (0 = estimate), 3-D, GPU. Other Cellpose arguments go in *extra cellpose kwargs* as YAML, e.g. `flow_threshold: 0.4`. |
| **dog** | Difference of Gaussians; see the [DoG example](../examples/dog.md). `sigma_units: um` gives the same physical blur along z and x/y. Filling in a **PSF path** turns on deconvolution first, which asks for wavelength, NA and immersion index. Leave `dup_rev_z` on *auto*: it mirrors thin tiles in z to avoid wrap-around ghosts that shift objects along z. |
| **threshold** | Otsu. Useful for a quick test. |
| **custom** | Your own function: give its module, function and keyword arguments. See [Make it importable on the cluster](snakemake.md#make-it-importable-on-the-cluster). |

### Relations (multi only)

Each row of the **Relations** table relates every object of `a` to the
`b` object it overlaps most. The result is written as an Excel workbook
in `work_dir`. For example:

| a | b | output (.xlsx) |
| --- | --- | --- |
| nuclei_labels | cyto_labels | nuclei_in_cells.xlsx |
| cilia_labels | cyto_labels | cilia_in_cells.xlsx |

Add rows with the **+** under the table. **Relate and bundle jobs (SLURM)**
sets the partition, memory, CPUs, time and QOS of the relate jobs; blank
or 0 means the defaults. It can also pack the finished store into one
`.zip` or `.iso` for download.

### What the form checks

As you type, the app shows:

- errors in the settings, using the same validation the workflow runs;
- for a multi run, whatever `run_multi.py` would refuse: duplicate label
  names; segmentations that differ in `work_dir`, `input`, `tile_shape`,
  `level` or conversion settings; relations naming an unknown label, or
  relating a label to itself; outputs that aren't `.xlsx` or that repeat;
- an **Effective config** panel for each segmentation. This is exactly
  what Snakemake will run with: the generated file overlaid on the
  cluster's `config/config.yaml`.

    A warning lists any key that comes from the cluster file rather than
    the form. Some of these differ from what you expect, such as a
    leftover `cellpose.flow_threshold`. Remove those keys from the
    cluster's `config/config.yaml`, or add them to *extra cellpose
    kwargs*.

## 3. Launch (Launch tab)

**Plan (tiles, memory, size -- no segmentation)** runs `patchworks segment
--plan`. It reports the number of tiles, the empty tiles, the memory per
tile and the output size, without segmenting anything. It needs the
converted image, so it only works once the workflow has converted the
input at least once. For a multi run, pick the segmentation under **plan
for**.

Then choose a **run mode** and press **Launch**:

| Mode | What happens | Use it for |
| --- | --- | --- |
| **Dry run** | `snakemake -n` (multi: `run_multi.py -n`). The output is shown right away. | Always first: it catches missing files and bad paths. |
| **Submit — controller as a SLURM job (recommended)** | A small 1-CPU job runs Snakemake (or `run_multi.py`). That job submits one GPU job per batch of tiles and waits for them. | Real runs. |
| **Submit — controller on the login node** | The controller runs detached (`setsid nohup`) on the login node. | Short runs, or clusters that forbid jobs submitting jobs. |

The controller job's **time** must cover the whole run, from conversion
to merge, plus the relations for a multi run. Its **mem**, **partition**,
**qos** and **account** come from the preset. Set **time** generously: if
the controller dies, Snakemake stops submitting new tiles, although jobs
already queued finish.

Nothing depends on your browser after launching. You can close the tab.

!!! tip "Resuming"
    Launching the same settings again resumes the run: Snakemake skips the
    tiles that are already done. Change `work_dir` to start fresh.

## 4. Follow the run (Jobs tab)

Every launch is recorded on the cluster, in
`<workflow dir>/.streamlit_launcher_logs/jobs.jsonl`. The Jobs tab lists
these jobs even after a reload or from another computer, as long as the
same workflow directory is used. Pick a job to see:

- its SLURM state (`PENDING` / `RUNNING` / `DONE`) or, for a login-node
  run, whether the process is alive;
- the last 300 lines of its log. Toggle **auto-refresh every 15 s** to
  follow it.

The per-tile GPU jobs show up in `squeue -u $USER` as usual. Their own logs
are under `work_dir`; see [Monitor](snakemake.md#6-monitor).

## Files the launcher writes

| Path (on the cluster) | Content |
| --- | --- |
| `<workflow dir>/config/launcher_<label>_<time>.yaml` | a single run's complete config |
| `<workflow dir>/config/launcher_multi_<time>/seg_<label>.yaml` | one complete config per segmentation of a multi run |
| `<workflow dir>/config/launcher_multi_<time>/multi.yaml` | the segmentations and relations of a multi run |
| `<workflow dir>/.streamlit_launcher_logs/<name>.log` | the controller's log |
| `<workflow dir>/.streamlit_launcher_logs/<name>.sbatch` | the controller job script |
| `<workflow dir>/.streamlit_launcher_logs/jobs.jsonl` | the job registry |

The configs are ordinary workflow configs. You can re-run one from a
terminal with `snakemake --configfile config/launcher_….yaml
--workflow-profile profile/slurm`, or `python scripts/run_multi.py --config
config/launcher_multi_…/multi.yaml --profile profile/slurm`. That makes
the launcher a convenient way to write a correct config even if you run
it by hand.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| *host key … matches none of the pinned fingerprints* | The fingerprint in `clusters.yaml` is wrong, or the server's key changed. Check with the admins before updating it. |
| *no pinned fingerprint* | Add `host_key_fingerprints` to the preset, as described in [Before you start](#before-you-start). |
| Connection fails although the password is right | The cluster requires 2FA or key-only login. The launcher only supports password login. |
| `sbatch: command not found` or `snakemake: command not found` | The **environment setup** line doesn't set up the environment. Test it in an SSH session: `cd <workflow dir> && <setup line> && which snakemake sbatch`. |
| *plan failed -- it needs the converted image* | Launch once (or run `convert`) first. The plan reads the converted store. |
| Job shows `DONE` right away | The controller failed at startup. The log in the Jobs tab shows why, often a bad path in `input` or `work_dir`. |
| Settings you didn't set show up in the run | See the **Effective config** warning in the Config tab. They come from the cluster's `config/config.yaml`. |
