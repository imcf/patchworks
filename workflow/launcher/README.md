# patchworks launcher (Streamlit)

A form-based UI to build a `config.yaml`, upload it to a remote host over
SSH, and start the Snakemake workflow there — dry-run, local, or submitted to
SLURM — without touching a terminal.

Runs on your own machine (laptop), not on the cluster.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How it works

1. You log in with host/username/password (paramiko, password auth). The
   password is used only to open the SSH connection and is never written to
   disk.
2. You fill in the config form (mirrors `workflow/config/config.yaml`) and
   pick a run mode: dry run, run locally (`--cores`), or submit to SLURM
   (`--workflow-profile`).
3. On **Launch**, the app uploads the generated YAML via SFTP into
   `<remote workflow dir>/config/`, then runs `snakemake` remotely wrapped in
   `setsid nohup ... &`, redirected to a log file under
   `<remote workflow dir>/.streamlit_launcher_logs/`. Detaching this way
   means the run keeps going even if you close the browser tab, your laptop
   sleeps, or this app restarts — it's the same process tree either way,
   `snakemake` itself submits the SLURM jobs and stays alive to track them.
4. The **Monitor** tab polls `ps` and tails the remote log every 5 seconds.

## Known limitations

- **Host keys are auto-accepted** (`paramiko.AutoAddPolicy`), not checked
  against a `known_hosts` entry — fine for a first connection to a cluster
  you trust, but it means a MITM on that very first connection would go
  unnoticed. If that matters for your setup, switch to
  `paramiko.RejectPolicy` and pre-populate `~/.ssh/known_hosts`.
- **Password auth only.** If your cluster requires 2FA/MFA or key-based
  login, this won't authenticate — swap `connect()` in `app.py` for
  key-based or keyboard-interactive auth.
- **One connection = one browser session.** Reloading the Streamlit app in a
  new tab starts a fresh, unconnected session; it does not currently look up
  a job by PID/log path from an earlier session. The remote job itself is
  unaffected (see above) — you'd just need to note the PID/log path shown at
  launch time to check on it by hand (`ps -p <pid>`, `tail -f <log>`).
- **Single-config only.** `multi.yaml` (running several segmentations plus
  `label_relations`) isn't wired up — extend `build_config`/`launch` in
  `app.py` if you need it.
- **`method: custom`** needs its `custom:` block (module/function/kwargs)
  added by hand to the generated config; the form doesn't build it.
