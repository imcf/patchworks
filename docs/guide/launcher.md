# Web launcher

A small web app that runs the [cluster workflow](snakemake.md) from a form:
it logs in over SSH, writes a valid config, starts the run and shows its
progress. It does nothing the command line cannot; it is for people who do
not live in a terminal, and it never starts a run with a typo in a key.

## Set up once per cluster

1. Install the workflow on the cluster ([Set up](snakemake.md#1-set-up)) and
   adapt `workflow/profile/slurm/config.yaml`. The launcher installs nothing.
2. Pin the cluster's host key in `workflow/launcher/clusters.yaml`, so the
   app refuses to send a password anywhere else:

    ```bash
    ssh-keyscan login.example.org | ssh-keygen -lf -
    ```

    ```yaml
    mycluster:
      host: login.example.org
      host_key_fingerprints: ["SHA256:abc…"]
      workflow_dir_hint: /home/<user>/patchworks/workflow
      setup_cmd: 'eval "$(pixi shell-hook)"'   # must put snakemake, patchworks, sbatch on PATH
      controller_time: "3-00:00:00"
    ```

3. Start it, on your machine or an internal server:

    ```bash
    cd patchworks/workflow/launcher
    pip install -r requirements.txt
    streamlit run app.py      # http://localhost:8501
    ```

!!! warning
    The form takes real cluster passwords. Never host it publicly.

## Use it

1. **Connect**: pick the cluster, log in, give the workflow directory.
2. **Config**: one segmentation, or several with the relations between them
   (the form of a [multi config](snakemake.md#several-segmentations-and-their-relations)).
   Settings mean the same as in the [config](snakemake.md#2-configure). The
   form validates as you type and shows the **effective config**, including
   any key inherited from the cluster's `config/config.yaml`.
3. **Launch**: *Plan* reports tiles, memory and output size; *Dry run* first,
   then *Submit*. The controller runs as a small SLURM job (or on the login
   node), so you can close the browser. Launching the same settings again
   resumes.
4. **Jobs**: the state and log of every launch, from any browser.

The configs it writes (`config/launcher_*.yaml`) are ordinary workflow
configs you can also run from a terminal.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| host key matches no pinned fingerprint / no pinned fingerprint | Check `host_key_fingerprints` with your admins. |
| Login fails with the right password | The cluster needs 2FA or keys; only password login is supported. |
| `snakemake` or `sbatch` not found | Fix the environment setup line; test it with `cd <dir> && <setup> && which snakemake sbatch`. |
| Plan fails | It needs the converted image: launch once first. |
| Job `DONE` at once | The controller failed at startup; its log in **Jobs** says why. |
