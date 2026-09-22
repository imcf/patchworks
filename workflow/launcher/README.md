# patchworks launcher (Streamlit)

A form-based UI to build a `config.yaml`, upload it to a remote host over
SSH, and start the Snakemake workflow there — dry-run, local, or submitted to
SLURM — without touching a terminal. Pick a cluster preset (sciCORE is
shipped in `clusters.yaml`) or type in a custom host, then log in with your
own username/password.

Can run on your own machine, or be deployed once (e.g. on Streamlit
Community Cloud) and shared — each visitor authenticates with their own
cluster credentials, isolated to their own browser session; nobody's
password or SSH session is visible to another user of the same deployment.
See **Deploying for a team**, and its security notes, before sharing a link.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying for a team (Streamlit Community Cloud)

1. Push this repo (or your fork) to GitHub — the branch this file is on
   already has everything Community Cloud needs.
2. Go to <https://share.streamlit.io>, sign in with GitHub, and "New app"
   from this repository/branch with main file path
   `workflow/launcher/app.py`. Community Cloud picks up
   `workflow/launcher/requirements.txt` automatically (same directory as the
   entry point).
3. Under the app's settings, restrict **"Who can view this app"** to your
   organization/specific emails rather than leaving it public — this app's
   login form takes real cluster passwords, so treat the deployment the same
   way you'd treat any internal tool with cluster access.
4. Add or edit entries in `clusters.yaml` (a PR against this branch) for any
   cluster your team needs beyond sciCORE; each one can optionally pin an
   SSH host-key fingerprint (see the comments in that file) so the app
   verifies the cluster's identity instead of trusting it on first connect —
   worth doing before relying on a shared deployment for real logins.

## How it works

1. You pick a cluster (or enter a custom host) and log in with your own
   username/password (paramiko, password auth). The password is used only to
   open that SSH connection, for your browser session, and is never written
   to disk.
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

- **Host keys are only verified for clusters with a pinned fingerprint** in
  `clusters.yaml` (`host_key_fingerprint:`). Without one — the shipped
  sciCORE entry included, since its fingerprint isn't filled in — the app
  falls back to trust-on-first-use (`paramiko.AutoAddPolicy`), which means a
  MITM on that very first connection would go unnoticed. Fill in the
  fingerprint (`ssh-keyscan -t ed25519 <host> | ssh-keygen -lf -`, or ask the
  cluster's admins) before relying on a preset for real logins, especially
  on a shared deployment.
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
