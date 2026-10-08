# patchworks launcher (Streamlit)

A form-based UI that builds a run's config, uploads it to a cluster over SSH
and starts the workflow there, so you don't need a terminal. It handles
either one segmentation (`snakemake`) or several segmentations plus their
relations (`scripts/run_multi.py`). Pick a cluster preset from
`clusters.yaml` or type in a host, then log in with your own
username and password.

For the full walkthrough, see the documentation page *Web launcher
(Streamlit)* (`docs/guide/launcher.md`).

## Run it

```bash
cd workflow/launcher
pixi run start            # http://localhost:8501
```

`pixi.toml` here is a workspace of its own (like `../viewer/`): it solves
on Linux, macOS and Windows and holds only Streamlit, paramiko and PyYAML.
Without pixi: `pip install -r requirements.txt && streamlit run app.py`.

Run it on your own machine, or on a server inside the institute network
for a team. Each visitor logs in with their own cluster account, isolated
to their browser session. The login form takes real cluster passwords, so
don't host it on a public service such as Streamlit Community Cloud. Keep
it where only the people it's for can reach it.

## Security model

- **Host keys are verified.** A preset lists the server's host-key
  fingerprints (`host_key_fingerprints:`). The app refuses a key that
  matches none of them. For a host with no pinned fingerprint, it refuses
  to connect unless the user ticks *allow an unverified host key*. Only
  tick that when you run the app yourself. To get the fingerprints, run
  `ssh-keyscan <host> | ssh-keygen -lf -` or ask the cluster's admins.
- **The password** only opens the SSH connection. It's never written to
  disk, and the login form clears it on submit. The app doesn't use local
  SSH keys or an agent, so it can't log in as anyone but the person typing.

## Using it

0. **Sidebar.** Connect, then choose the workflow folder. A clone at
   `~/patchworks/workflow` is filled in by itself; the sidebar confirms it
   holds a `Snakefile`. Every path field (workflow folder, input image,
   work_dir, PSF) has a folder button that browses the cluster over SFTP:
   folders open, image files and `.zarr` stores are picked, and a folder
   field can also take a new folder inside the current one. A TIFF glob
   still has to be typed in.
1. **Configure tab.** Choose *One segmentation* or *Several
   segmentations + relations*. The form is split into blocks: image, tiling,
   segmentation(s) (one tab each), relations and review, outputs, and a
   final *Check* that lists anything still missing (an empty input or
   work_dir is refused) before it shows the configs it will upload.
   - Shared settings (input, work_dir, conversion, compression, tiling, GPU,
     merge/pyramid) are entered once.
   - Each segmentation gets its own label name, channel, method (cellpose,
     threshold, DoG with optional deconvolution, custom plugin), overlap
     and post-processing.
   - A multi run adds a relations table (`a` → `b` → `.xlsx`), plus
     optional SLURM settings for the relate jobs and a bundle format.
   - Each config is complete: the workflow fills in its defaults and
     reads nothing from the cluster's `config/config.yaml`. A multi run
     is checked the way `run_multi.py` checks it
     before anything is uploaded: unique label names; one work_dir, input,
     tile shape and level; relations between known labels.
2. **Plan & launch tab.**
   - *Environment*: the workflow's pixi environment the run starts from,
     read from its `pixi.toml`. The app picks the smallest one with what
     the config needs (PlantSeg: `plantseg`; `use_gpu` or `dilate_gpu`:
     cupy, `cuda12`; `denoise`: `careamics`) and swaps it into the
     `pixi shell-hook` of the sidebar's setup line.
   - *Plan* runs `patchworks segment --plan` on the converted image, for
     every segmentation of the run, and shows tiles, memory and size in
     one table, before any GPU time is spent. It needs
     `<work_dir>/image.zarr`: launch with *Convert only* first (the same
     conversion step a full run starts with, which a later run then
     skips), or plan after a full run has converted.
   - Then pick a mode:
     - **Dry run**: `snakemake -n`, or `run_multi.py -n`. The output is
       shown right away.
     - **Controller as a SLURM job** (recommended): a small 1-CPU job runs
       the Snakemake controller, or `run_multi.py`, which then submits the
       real jobs. It outlives login-node reboots and reapers. Its time,
       memory, partition, QOS and account default to the preset's
       `controller_*` values. The job clears its own `SLURM_*` variables,
       so every `srun`/`sbatch` it starts becomes a job of its own.
     - **Controller on the login node**: detached with `setsid nohup`.
       Fine for short runs only.
3. **Jobs tab.** Launched jobs are recorded on the cluster in
   `<workflow dir>/.streamlit_launcher_logs/jobs.jsonl`. After a reload, a
   new browser or another machine, the app finds them again. It shows the
   SLURM/process state and tails the log, with optional auto-refresh.

Generated configs go to `<workflow dir>/config/launcher_<name>.yaml`. A
multi run goes to `<workflow dir>/config/launcher_multi_<stamp>/`, with one
`seg_<label>.yaml` per segmentation plus `multi.yaml`.

## Known limitations

- **Password auth only.** Clusters that require 2FA or keys need
  `connect()` in `app.py` adapted.
- The shipped sciCORE preset has no fingerprints yet (see the TODO in
  `clusters.yaml`). Fill them in before relying on it.
