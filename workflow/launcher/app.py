"""Streamlit launcher for the patchworks Snakemake workflow.

Runs locally (on your laptop, not on the cluster). It connects to a remote
host over SSH, lets you build a `config.yaml` through a form, uploads it, and
starts `snakemake` there — detached (`setsid` + `nohup`), so the run survives
you closing the browser tab or this app restarting. Progress is read back by
tailing the remote log over the same SSH connection.

Usage:
    pip install -r requirements.txt
    streamlit run app.py

See README.md for the security tradeoffs of password-based SSH login.
"""

from __future__ import annotations

import io
import shlex
import time
from datetime import datetime

import paramiko
import streamlit as st
import yaml

st.set_page_config(page_title="patchworks launcher", layout="wide")

# ---------------------------------------------------------------------------
# SSH connection
# ---------------------------------------------------------------------------


def get_client() -> paramiko.SSHClient | None:
    return st.session_state.get("ssh_client")


def connect(host: str, port: int, username: str, password: str) -> None:
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    # Unknown-host keys are accepted automatically rather than verified
    # against a known_hosts entry — convenient for a first connection to a
    # cluster login node, but it means a MITM on that first connection would
    # go unnoticed. See README.md.
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        host, port=port, username=username, password=password, timeout=15
    )
    st.session_state["ssh_client"] = client
    st.session_state["ssh_target"] = f"{username}@{host}:{port}"


def run_short(client: paramiko.SSHClient, command: str) -> tuple[int, str, str]:
    """Run *command* (via a login shell) and wait for it to finish."""
    stdin, stdout, stderr = client.exec_command(
        f"bash -lc {shlex.quote(command)}"
    )
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    rc = stdout.channel.recv_exit_status()
    return rc, out, err


# ---------------------------------------------------------------------------
# Config form -> YAML matching workflow/config/config.yaml
# ---------------------------------------------------------------------------


def build_config(values: dict) -> dict:
    cfg: dict = {
        "input": values["input"],
        "work_dir": values["work_dir"],
        "reuse_pyramid": values["reuse_pyramid"],
        "convert_chunks": yaml.safe_load(values["convert_chunks"] or "null"),
        "shard": values["shard"],
        "ngff_version": values["ngff_version"],
        "sequence_pattern": values["sequence_pattern"] or None,
        "channel": values["channel"],
        "level": values["level"],
        "tile_shape": yaml.safe_load(values["tile_shape"]),
        "gpu_memory_gb": values["gpu_memory_gb"] or None,
        "overlap": yaml.safe_load(values["overlap"]),
        "skip_empty": values["skip_empty"],
        "tiles_per_job": values["tiles_per_job"],
        "empty_threshold": values["empty_threshold"],
        "method": values["method"],
        "label_name": values["label_name"],
        "pyramid_levels": values["pyramid_levels"],
        "pyramid_downscale": values["pyramid_downscale"],
        "sequential_labels": values["sequential_labels"],
        "shard_labels": values["shard_labels"],
        "merge_workers": values["merge_workers"],
        "notify_email": values["notify_email"],
        "notify_events": values["notify_events"],
    }
    if values["method"] == "cellpose":
        cellpose = {
            "model": values["cp_model"],
            "diameter": values["cp_diameter"],
            "do_3D": values["cp_do_3d"],
            "gpu": values["cp_gpu"],
        }
        if values["cp_extra_kwargs"].strip():
            cellpose.update(yaml.safe_load(values["cp_extra_kwargs"]))
        cfg["cellpose"] = cellpose
    return cfg


def config_form() -> dict:
    values: dict = {}
    st.subheader("Input / output")
    c1, c2 = st.columns(2)
    values["input"] = c1.text_input(
        "input", help=".ims/.czi/.lif/.nd2/ome-tiff/.zarr, or a glob"
    )
    values["work_dir"] = c2.text_input(
        "work_dir", help="everything is written under here, on the remote host"
    )
    values["label_name"] = st.text_input("label_name", value="cellpose_labels")

    with st.expander("Conversion"):
        c1, c2, c3 = st.columns(3)
        values["reuse_pyramid"] = c1.checkbox("reuse_pyramid", value=False)
        values["shard"] = c2.checkbox("shard", value=False)
        values["ngff_version"] = c3.selectbox(
            "ngff_version", ["auto", "0.4", "0.5"], index=0
        )
        values["convert_chunks"] = st.text_input(
            "convert_chunks (YAML, e.g. [8, 512, 512], or blank for auto)",
            value="",
        )
        values["sequence_pattern"] = st.text_input(
            "sequence_pattern (regex, only for a glob input)", value=""
        )

    st.subheader("Tiling")
    c1, c2, c3 = st.columns(3)
    values["channel"] = c1.number_input("channel", min_value=0, value=0, step=1)
    values["level"] = c2.number_input("level", min_value=0, value=0, step=1)
    values["tile_shape"] = c3.text_input(
        "tile_shape", value="auto", help='"auto" or e.g. [16, 1024, 1024]'
    )
    c1, c2, c3 = st.columns(3)
    values["gpu_memory_gb"] = c1.number_input(
        "gpu_memory_gb (0 = unset)", min_value=0, value=0
    )
    values["overlap"] = c2.text_input("overlap", value="30")
    values["tiles_per_job"] = c3.number_input(
        "tiles_per_job", min_value=1, value=4, step=1
    )
    c1, c2 = st.columns(2)
    values["skip_empty"] = c1.checkbox("skip_empty", value=True)
    empty_threshold = c2.text_input(
        "empty_threshold (blank = Otsu)", value=""
    )
    values["empty_threshold"] = (
        float(empty_threshold) if empty_threshold.strip() else None
    )

    st.subheader("Segmentation")
    values["method"] = st.selectbox(
        "method", ["cellpose", "threshold", "custom"], index=0
    )
    if values["method"] == "cellpose":
        c1, c2, c3, c4 = st.columns(4)
        values["cp_model"] = c1.text_input("cellpose.model", value="nuclei")
        values["cp_diameter"] = c2.number_input(
            "cellpose.diameter", min_value=0.0, value=30.0
        )
        values["cp_do_3d"] = c3.checkbox("cellpose.do_3D", value=True)
        values["cp_gpu"] = c4.checkbox("cellpose.gpu", value=True)
        values["cp_extra_kwargs"] = st.text_area(
            "extra cellpose kwargs (YAML mapping, optional)",
            value="",
            help="e.g. flow_threshold: 0.4",
        )
    elif values["method"] == "custom":
        st.info(
            "method: custom needs a `custom:` block (module/function/kwargs) "
            "this form doesn't build — edit the uploaded config by hand, or "
            "extend this app."
        )

    with st.expander("Label pyramid / merge"):
        c1, c2, c3 = st.columns(3)
        values["pyramid_levels"] = c1.number_input(
            "pyramid_levels", min_value=1, value=5, step=1
        )
        values["pyramid_downscale"] = c2.number_input(
            "pyramid_downscale", min_value=2, value=2, step=1
        )
        values["sequential_labels"] = c3.checkbox(
            "sequential_labels", value=True
        )
        c1, c2 = st.columns(2)
        values["shard_labels"] = c1.checkbox("shard_labels", value=False)
        merge_workers = c2.text_input("merge_workers (blank = auto)", value="")
        values["merge_workers"] = (
            int(merge_workers) if merge_workers.strip() else None
        )

    with st.expander("Notifications"):
        values["notify_email"] = st.text_input("notify_email", value="")
        values["notify_events"] = st.multiselect(
            "notify_events",
            ["start", "finish", "error"],
            default=["finish", "error"],
        )

    return values


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------


def launch(
    client: paramiko.SSHClient,
    *,
    remote_workflow_dir: str,
    setup_cmd: str,
    config_yaml: str,
    run_mode: str,
    cores: int,
    profile: str,
    label_name: str,
) -> None:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_label = "".join(c if c.isalnum() else "-" for c in label_name) or "run"
    remote_config = f"{remote_workflow_dir}/config/streamlit_{safe_label}_{stamp}.yaml"
    remote_log_dir = f"{remote_workflow_dir}/.streamlit_launcher_logs"
    remote_log = f"{remote_log_dir}/{safe_label}_{stamp}.log"

    rc, _, err = run_short(client, f"mkdir -p {shlex.quote(remote_log_dir)}")
    if rc != 0:
        st.error(f"could not create remote log dir: {err}")
        return

    sftp = client.open_sftp()
    sftp.putfo(io.BytesIO(config_yaml.encode()), remote_config)
    sftp.close()

    snakemake_cmd = [
        "snakemake",
        "-s",
        "Snakefile",
        "--configfile",
        shlex.quote(remote_config),
        "--rerun-triggers",
        "mtime",
    ]
    if run_mode == "Dry run (local)":
        snakemake_cmd += ["--cores", str(cores), "-n", "-p"]
    elif run_mode == "Run locally":
        snakemake_cmd += ["--cores", str(cores)]
    else:  # Submit to SLURM
        snakemake_cmd += ["--workflow-profile", shlex.quote(profile)]

    inner = f"cd {shlex.quote(remote_workflow_dir)} && "
    if setup_cmd.strip():
        inner += f"{setup_cmd.strip()} && "
    inner += " ".join(snakemake_cmd)

    # setsid detaches the process from this SSH session's controlling
    # terminal/session so it keeps running after the channel closes; nohup
    # additionally ignores SIGHUP. Together a dropped connection (or this app
    # restarting) doesn't kill a multi-hour run.
    launcher = (
        f"setsid nohup bash -lc {shlex.quote(inner)} "
        f"> {shlex.quote(remote_log)} 2>&1 < /dev/null & echo LAUNCHER_PID:$!"
    )
    rc, out, err = run_short(client, launcher)
    pid = None
    for line in out.splitlines():
        if line.startswith("LAUNCHER_PID:"):
            pid = line.split(":", 1)[1].strip()
    if not pid:
        st.error(f"failed to start the job.\nstdout: {out}\nstderr: {err}")
        return

    st.session_state["job"] = {
        "pid": pid,
        "log": remote_log,
        "config": remote_config,
        "started": stamp,
        "mode": run_mode,
    }
    st.success(f"started (remote PID {pid}), logging to {remote_log}")


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------


@st.fragment(run_every=5)
def render_monitor():
    job = st.session_state.get("job")
    client = get_client()
    if not job or client is None:
        st.caption("No job started yet in this session.")
        return

    st.write(
        f"**PID** {job['pid']} · **started** {job['started']} · "
        f"**mode** {job['mode']}"
    )
    rc, out, _ = run_short(
        client,
        f"ps -p {shlex.quote(job['pid'])} > /dev/null && echo RUNNING || echo DONE",
    )
    status = "RUNNING" if "RUNNING" in out else "DONE / NOT FOUND"
    st.write(f"**status** {status}")

    rc, log_tail, err = run_short(
        client, f"tail -n 400 {shlex.quote(job['log'])}"
    )
    st.code(log_tail or "(log is empty so far)", language="text")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("patchworks launcher")

with st.sidebar:
    st.header("Remote host")
    if get_client() is None:
        host = st.text_input("host")
        port = st.number_input("port", value=22, min_value=1, max_value=65535)
        username = st.text_input("username")
        password = st.text_input("password", type="password")
        st.caption(
            "Password is used only for this SSH connection and is never "
            "written to disk."
        )
        if st.button("Connect", type="primary", disabled=not (host and username)):
            try:
                connect(host, int(port), username, password)
                st.rerun()
            except Exception as e:
                st.error(f"connection failed: {e}")
    else:
        st.success(f"connected as {st.session_state['ssh_target']}")
        if st.button("Disconnect"):
            get_client().close()
            del st.session_state["ssh_client"]
            st.session_state.pop("job", None)
            st.rerun()

    st.divider()
    st.header("Remote workflow")
    remote_workflow_dir = st.text_input(
        "workflow directory on the remote host",
        help="absolute path to patchworks/workflow there",
    )
    setup_cmd = st.text_area(
        "environment setup command (optional)",
        value="",
        help=(
            "run before snakemake, e.g. `module load pixi && eval \"$(pixi "
            "shell-hook)\"` or `source /path/to/venv/bin/activate`"
        ),
    )

if get_client() is None:
    st.info("Connect to a remote host in the sidebar to continue.")
    st.stop()

if not remote_workflow_dir:
    st.info("Set the remote workflow directory in the sidebar to continue.")
    st.stop()

tab_config, tab_launch, tab_monitor = st.tabs(["Config", "Launch", "Monitor"])

with tab_config:
    values = config_form()
    try:
        cfg_dict = build_config(values)
        cfg_yaml = yaml.safe_dump(cfg_dict, sort_keys=False)
        st.session_state["cfg_yaml"] = cfg_yaml
        with st.expander("Generated config.yaml", expanded=False):
            st.code(cfg_yaml, language="yaml")
    except Exception as e:
        st.session_state.pop("cfg_yaml", None)
        st.error(f"config is not valid YAML yet: {e}")

with tab_launch:
    cfg_yaml = st.session_state.get("cfg_yaml")
    if not cfg_yaml:
        st.warning("Fix the config in the Config tab first.")
    else:
        run_mode = st.radio(
            "run mode",
            ["Dry run (local)", "Run locally", "Submit to SLURM"],
            horizontal=True,
        )
        cores, profile = 8, "profile/slurm"
        if run_mode in ("Dry run (local)", "Run locally"):
            cores = st.number_input("--cores", min_value=1, value=8, step=1)
        if run_mode == "Submit to SLURM":
            profile = st.text_input(
                "--workflow-profile", value="profile/slurm"
            )

        if st.button("Launch", type="primary"):
            launch(
                get_client(),
                remote_workflow_dir=remote_workflow_dir,
                setup_cmd=setup_cmd,
                config_yaml=cfg_yaml,
                run_mode=run_mode,
                cores=int(cores),
                profile=profile,
                label_name=values.get("label_name", "run"),
            )

with tab_monitor:
    render_monitor()
