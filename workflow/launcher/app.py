"""Streamlit launcher for the patchworks Snakemake workflow.

Connects to a cluster over SSH, builds a run config through a form, shows
what Snakemake will *actually* run with (the uploaded file merged over the
cluster copy's config.yaml), and submits the workflow -- by default with the
Snakemake controller itself running as a small SLURM job. Jobs are recorded
on the cluster, so the Jobs tab finds them again after a reload.

Usage:
    pip install -r requirements.txt
    streamlit run app.py

The logic lives in launcher_core.py (no UI, tested); this file is the UI.
See README.md for the security model before sharing a deployment.
"""

from __future__ import annotations

import io
import shlex
from datetime import datetime
from pathlib import Path

import paramiko
import streamlit as st
import yaml

import launcher_core as core

st.set_page_config(page_title="patchworks launcher", layout="wide")

CLUSTERS = core.load_clusters(Path(__file__).with_name("clusters.yaml"))


# ---------------------------------------------------------------------------
# SSH
# ---------------------------------------------------------------------------


def get_client() -> paramiko.SSHClient | None:
    return st.session_state.get("ssh_client")


def connect(host, port, username, password, fingerprints, allow_unverified):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(
        core.make_host_key_policy(fingerprints, allow_unverified)
    )
    client.connect(
        host,
        port=port,
        username=username,
        password=password,
        timeout=15,
        # Only the password typed here: never keys or an agent that happen
        # to be available to whoever runs this app.
        look_for_keys=False,
        allow_agent=False,
    )
    st.session_state["ssh_client"] = client
    st.session_state["ssh_target"] = f"{username}@{host}:{port}"


def run(client, command: str, *, login: bool = True, timeout: float = 120):
    """Run *command* remotely; a login shell only when PATH needs it."""
    wrapped = f"bash -lc {shlex.quote(command)}" if login else command
    _, stdout, stderr = client.exec_command(wrapped, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    return stdout.channel.recv_exit_status(), out, err


def read_remote(client, path: str) -> str | None:
    try:
        with client.open_sftp() as sftp, sftp.open(path) as fh:
            return fh.read().decode(errors="replace")
    except OSError:
        return None


def write_remote(client, path: str, text: str, *, append=False) -> None:
    with client.open_sftp() as sftp:
        if append:
            with sftp.open(path, "a") as fh:
                fh.write(text)
        else:
            sftp.putfo(io.BytesIO(text.encode()), path)


# ---------------------------------------------------------------------------
# Config form
# ---------------------------------------------------------------------------


def _opt_float(label: str, container=st, help=None, key=None):
    text = container.text_input(label, value="", help=help, key=key)
    return float(text) if text.strip() else None


def shared_form() -> dict:
    """Settings every segmentation of a run shares (image, tiling, merge)."""
    v: dict = {}
    st.subheader("Input / output")
    c1, c2 = st.columns(2)
    v["input"] = c1.text_input(
        "input", help=".ims/.czi/.lif/.nd2/ome-tiff/.zarr, or a TIFF glob"
    )
    v["work_dir"] = c2.text_input(
        "work_dir", help="everything is written under here, on the cluster"
    )

    with st.expander("Conversion"):
        c1, c2, c3, c4 = st.columns(4)
        v["reuse_pyramid"] = c1.checkbox("reuse_pyramid", value=False)
        v["shard"] = c2.checkbox("shard", value=False)
        v["ngff_version"] = c3.selectbox("ngff_version", ["auto", "0.4", "0.5"])
        v["compression"] = c4.selectbox(
            "compression",
            ["zstd", "zstd:3", "blosc", "blosc:lz4", "none"],
            help="zstd: best for labels; zstd:3 ~10% smaller raw images; "
            "blosc for older Java viewers without zstd",
        )
        v["convert_chunks"] = st.text_input(
            "convert_chunks (YAML, e.g. [8, 512, 512]; blank = auto)", ""
        )
        v["sequence_pattern"] = st.text_input(
            "sequence_pattern (regex, only for a TIFF glob input)", ""
        )

    st.subheader("Tiling")
    c1, c2, c3, c4 = st.columns(4)
    v["level"] = c1.number_input("level", min_value=0, value=0, step=1)
    v["tile_shape"] = c2.text_input(
        "tile_shape", value="auto", help='"auto" or e.g. [16, 1024, 1024]'
    )
    v["tiles_per_job"] = c3.number_input(
        "tiles_per_job", min_value=1, value=4, step=1
    )
    v["gpu_memory_gb"] = c4.number_input(
        "gpu_memory_gb (0 = detect)", min_value=0, value=0
    )
    c1, c2 = st.columns(2)
    v["skip_empty"] = c1.checkbox("skip_empty", value=True)
    v["empty_threshold"] = _opt_float("empty_threshold (blank = Otsu)", c2)

    with st.expander("Merge and label pyramid"):
        c1, c2, c3 = st.columns(3)
        v["stitch"] = c1.selectbox(
            "stitch",
            ["touch", "iou"],
            help="iou keeps touching cells at a seam apart; needs overlap",
        )
        v["iou_threshold"] = c2.number_input(
            "iou_threshold", min_value=0.05, max_value=1.0, value=0.5
        )
        v["sequential_labels"] = c3.checkbox("sequential_labels", value=True)
        c1, c2, c3, c4 = st.columns(4)
        v["pyramid_levels"] = c1.number_input(
            "pyramid_levels", min_value=1, value=5
        )
        v["pyramid_downscale"] = c2.number_input(
            "pyramid_downscale", min_value=2, value=2
        )
        v["shard_labels"] = c3.checkbox("shard_labels", value=False)
        v["seam_report"] = c4.checkbox("seam_report", value=True)
        workers = st.text_input("merge_workers (blank = auto)", "")
        v["merge_workers"] = int(workers) if workers.strip() else None

    with st.expander("Notifications"):
        v["notify_email"] = st.text_input("notify_email", "")
        v["notify_events"] = st.multiselect(
            "notify_events",
            ["start", "finish", "error"],
            default=["finish", "error"],
        )
    return v


def segmentation_form(k: str = "", label: str = "labels") -> dict:
    """One segmentation: what to label, how, and how to post-process it.

    *k* prefixes every widget key, so several can sit on one page.
    """
    v: dict = {}
    c1, c2, c3, c4 = st.columns(4)
    v["label_name"] = c1.text_input("label_name", value=label, key=f"{k}lbl")
    v["channel"] = c2.number_input(
        "channel", min_value=0, value=0, step=1, key=f"{k}ch"
    )
    nuc = c3.text_input("nuclei_channel (blank = none)", "", key=f"{k}nuc")
    v["nuclei_channel"] = int(nuc) if nuc.strip() else None
    v["overlap"] = c4.text_input(
        "overlap", value="30", help="N, or per axis [z, y, x]", key=f"{k}ov"
    )
    v["method"] = st.selectbox(
        "method",
        ["cellpose", "dog", "threshold", "custom"],
        format_func=lambda m: {
            "dog": "dog (difference of Gaussians, optional deconvolution)"
        }.get(m, m),
        key=f"{k}method",
    )
    if v["method"] == "cellpose":
        c1, c2, c3, c4 = st.columns(4)
        v["cp_model"] = c1.text_input(
            "cellpose.model", value="cyto3", key=f"{k}cpm"
        )
        v["cp_diameter"] = c2.number_input(
            "cellpose.diameter (0 = estimate)",
            min_value=0.0,
            value=30.0,
            key=f"{k}cpd",
        )
        v["cp_do_3d"] = c3.checkbox("cellpose.do_3D", False, key=f"{k}cp3")
        v["cp_gpu"] = c4.checkbox("cellpose.gpu", True, key=f"{k}cpg")
        v["cp_extra_kwargs"] = st.text_area(
            "extra cellpose kwargs (YAML mapping)",
            "",
            help="e.g. flow_threshold: 0.4",
            key=f"{k}cpx",
        )
    elif v["method"] == "dog":
        c1, c2, c3, c4 = st.columns(4)
        v["dog_low_sigma"] = c1.number_input("low_sigma", 1.0, key=f"{k}dl")
        v["dog_high_sigma"] = c2.number_input("high_sigma", 3.0, key=f"{k}dh")
        v["dog_threshold"] = c3.number_input(
            "threshold", value=0.02, format="%.4f", key=f"{k}dt"
        )
        v["dog_sigma_units"] = c4.selectbox(
            "sigma_units",
            ["px", "um"],
            help="um: the same physical blur along z and x/y",
            key=f"{k}du",
        )
        v["dog_use_gpu"] = st.checkbox(
            "blur/label on GPU (cupy)", False, key=f"{k}dg"
        )
        v["dog_psf"] = st.text_input(
            "deconvolution PSF path on the cluster (blank = no deconvolution)",
            "",
            key=f"{k}psf",
        )
        if v["dog_psf"]:
            c1, c2, c3, c4 = st.columns(4)
            v["dog_wavelength"] = c1.number_input(
                "wavelength (nm)", value=525, key=f"{k}wl"
            )
            v["dog_na"] = c2.number_input("NA", value=1.4, key=f"{k}na")
            v["dog_nimm"] = c3.number_input(
                "immersion n", value=1.515, key=f"{k}ni"
            )
            v["dog_dup_rev_z"] = c4.selectbox(
                "dup_rev_z",
                ["auto", "on", "off"],
                help="mirror in z against FFT wrap-around ghosts on thin tiles",
                key=f"{k}dup",
            )
    elif v["method"] == "custom":
        c1, c2 = st.columns(2)
        v["custom_module"] = c1.text_input("custom.module", "", key=f"{k}cm")
        v["custom_function"] = c2.text_input(
            "custom.function", "segment", key=f"{k}cf"
        )
        v["custom_kwargs"] = st.text_area(
            "custom.kwargs (YAML mapping)", "", key=f"{k}ck"
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    holes = c1.selectbox(
        "fill_holes", ["off", "3-D", "per plane"], key=f"{k}holes"
    )
    v["fill_holes"] = {"off": False, "3-D": True, "per plane": "per_plane"}[
        holes
    ]
    v["open_radius"] = c2.number_input(
        "open_radius",
        min_value=0,
        value=0,
        help="cuts spurs thinner than ~2r+1 voxels -- not for cilia",
        key=f"{k}open",
    )
    v["dilate"] = c3.number_input("dilate (px)", 0, value=0, key=f"{k}dil")
    v["min_volume"] = _opt_float("min_volume (um³)", c4, key=f"{k}minv")
    v["max_volume"] = _opt_float("max_volume (um³)", c5, key=f"{k}maxv")
    return v


def multi_form() -> tuple[list[dict], list[dict], dict, dict]:
    """Segmentations, relations, and the relate/bundle job settings."""
    n = st.number_input(
        "segmentations", min_value=1, max_value=8, value=2, step=1
    )
    segs = []
    for i in range(int(n)):
        default = ["nuclei_labels", "cyto_labels", "cilia_labels"]
        with st.expander(f"Segmentation {i + 1}", expanded=i == 0):
            segs.append(
                segmentation_form(
                    f"s{i}_",
                    default[i] if i < len(default) else f"labels_{i + 1}",
                )
            )
    labels = [s["label_name"] for s in segs]
    st.subheader("Relations")
    st.caption(
        "Each row relates every object of `a` to the `b` object it overlaps "
        "most, written as a two-sheet Excel workbook in work_dir."
    )
    rows = st.data_editor(
        [{"a": labels[0], "b": labels[-1], "output": "relations.xlsx"}]
        if len(labels) > 1
        else [],
        num_rows="dynamic",
        column_config={
            "a": st.column_config.SelectboxColumn("a", options=labels),
            "b": st.column_config.SelectboxColumn("b", options=labels),
            "output": st.column_config.TextColumn("output (.xlsx)"),
        },
        key="relations",
    )
    relations = [
        r for r in rows if r.get("a") and r.get("b") and r.get("output")
    ]
    with st.expander("Relate and bundle jobs (SLURM)"):
        c1, c2, c3, c4, c5 = st.columns(5)
        relate = {
            "partition": c1.text_input("relate partition", ""),
            "mem": c2.text_input("relate mem", ""),
            "cpus": int(c3.number_input("relate cpus (0 = default)", 0))
            or None,
            "time": int(c4.number_input("relate minutes (0 = default)", 0))
            or None,
            "qos": c5.text_input("relate qos", ""),
        }
        fmt = st.selectbox("bundle the finished store", ["none", "zip", "iso"])
        bundle = {"format": None if fmt == "none" else fmt}
    return segs, relations, relate, bundle


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------


def launch(client, *, wf_dir, setup_cmd, files, name, mode, profile, slurm):
    """Upload *files* (remote path -> config dict) and start the run.

    A single config runs ``snakemake``; a multi run (whose files include a
    ``multi.yaml``) runs ``scripts/run_multi.py``, which segments, relates
    and bundles.
    """
    log_dir = f"{wf_dir}/{core.LOG_DIR}"
    log = f"{log_dir}/{name}.log"
    rc, _, err = run(client, f"mkdir -p {shlex.quote(log_dir)}", login=False)
    if rc:
        st.error(f"could not create {log_dir}: {err}")
        return
    for path in files:
        parent = path.rsplit("/", 1)[0]
        run(client, f"mkdir -p {shlex.quote(parent)}", login=False)
    for path, content in files.items():
        write_remote(client, path, yaml.safe_dump(content, sort_keys=False))
    multi = next((p for p in files if p.endswith("/multi.yaml")), None)
    command = (
        core.multi_command(multi, mode, profile=profile)
        if multi
        else core.snakemake_command(next(iter(files)), mode, profile=profile)
    )
    inner = core.inner_command(wf_dir, setup_cmd, command)

    if mode == core.DRY_RUN:
        with st.spinner("dry run…"):
            rc, out, err = run(client, inner, timeout=900)
        (st.success if rc == 0 else st.error)(f"dry run exited {rc}")
        st.code((out + err)[-20000:] or "(no output)", language="text")
        return

    label = name.rsplit("_", 1)[0]
    job = {
        "name": name,
        "config": multi or next(iter(files)),
        "log": log,
        "started": name.rsplit("_", 1)[-1],
        "mode": mode,
        "kind": "multi" if multi else "single",
    }
    if mode == core.SLURM_JOB:
        script = f"{log_dir}/{name}.sbatch"
        write_remote(client, script, core.controller_job_script(inner, **slurm))
        rc, out, err = run(
            client,
            core.inner_command(
                wf_dir,
                setup_cmd,
                core.sbatch_command(script, name=f"pw-{label}", log=log),
            ),
        )
        job["slurm_job"] = core.parse_sbatch(out)
        if rc or not job["slurm_job"]:
            st.error(f"sbatch failed ({rc}):\n{out}\n{err}")
            return
        st.success(f"controller submitted as SLURM job {job['slurm_job']}")
    else:
        rc, out, err = run(client, core.detached_command(inner, log))
        job["pid"] = core.parse_tag(out, "PID")
        if not job["pid"]:
            st.error(f"failed to start:\n{out}\n{err}")
            return
        st.success(f"started on the login node (PID {job['pid']})")
    write_remote(
        client,
        f"{wf_dir}/{core.REGISTRY}",
        core.registry_line(job),
        append=True,
    )
    st.caption(f"log: {log} · config: {job['config']}")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("patchworks launcher")

with st.sidebar:
    st.header("Cluster")
    if get_client() is None:
        name = st.selectbox("preset", ["Custom"] + sorted(CLUSTERS))
        preset = CLUSTERS.get(name, {})
        fps = preset.get("host_key_fingerprints", [])
        host = st.text_input("host", value=preset.get("host", ""))
        port = st.number_input(
            "port",
            value=int(preset.get("port", 22)),
            min_value=1,
            max_value=65535,
        )
        allow_unverified = False
        if not fps:
            st.warning(
                "No pinned host-key fingerprint for this host, so its "
                "identity cannot be verified. Pin it in clusters.yaml."
            )
            allow_unverified = st.checkbox(
                "allow an unverified host key (only if you run this app "
                "yourself)"
            )
        # A form that clears on submit: the password does not linger in
        # the session's widget state after the connection is made.
        with st.form("login", clear_on_submit=True):
            username = st.text_input("username")
            password = st.text_input("password", type="password")
            submitted = st.form_submit_button("Connect", type="primary")
        if submitted and host and username:
            try:
                connect(
                    host, int(port), username, password, fps, allow_unverified
                )
                st.session_state["preset"] = preset
                st.rerun()
            except Exception as exc:  # shown, not raised: a UI
                st.error(f"connection failed: {exc}")
    else:
        st.success(f"connected as {st.session_state['ssh_target']}")
        if st.button("Disconnect"):
            get_client().close()
            for key in ("ssh_client", "preset", "base_cfg"):
                st.session_state.pop(key, None)
            st.rerun()

    st.divider()
    st.header("Remote workflow")
    preset = st.session_state.get("preset", {})
    wf_dir = st.text_input(
        "workflow directory on the cluster",
        help=f"e.g. {preset.get('workflow_dir_hint', '/path/to/patchworks/workflow')}",
    )
    setup_cmd = st.text_area(
        "environment setup (run before snakemake)",
        value=preset.get("setup_cmd", ""),
        help='e.g. eval "$(pixi shell-hook)"',
    )

client = get_client()
if client is None:
    st.info("Connect to a cluster in the sidebar to continue.")
    st.stop()
if not wf_dir:
    st.info("Set the workflow directory in the sidebar to continue.")
    st.stop()

tab_config, tab_launch, tab_jobs = st.tabs(["Config", "Launch", "Jobs"])

with tab_config:
    kind = st.radio(
        "run",
        ["One segmentation", "Several segmentations + relations (multi)"],
        horizontal=True,
    )
    shared = shared_form()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    files: dict = {}
    try:
        if kind.startswith("One"):
            st.subheader("Segmentation")
            seg = segmentation_form()
            cfg = core.build_config({**shared, **seg})
            name = f"{core.safe_name(cfg['label_name'])}_{stamp}"
            files = {f"{wf_dir}/config/launcher_{name}.yaml": cfg}
        else:
            segs, relations, relate, bundle = multi_form()
            name = f"multi_{stamp}"
            files = core.build_multi(
                shared,
                segs,
                relations,
                directory=f"{wf_dir}/config/launcher_{name}",
                relate=relate,
                bundle=bundle,
            )
            for problem in core.multi_problems(files):
                st.error(problem)
            if core.multi_problems(files):
                files = {}
        st.session_state["files"] = files
        st.session_state["run_name"] = name
    except Exception as exc:
        st.session_state.pop("files", None)
        st.error(f"the config is not valid yet: {exc}")
    if files:
        cache = st.session_state.setdefault("base_cfg", {})
        if wf_dir not in cache:
            text = read_remote(client, f"{wf_dir}/config/config.yaml")
            cache[wf_dir] = yaml.safe_load(text) if text else None
        for path, content in files.items():
            if path.endswith("/multi.yaml"):
                with st.expander("multi.yaml"):
                    st.code(yaml.safe_dump(content, sort_keys=False), "yaml")
                continue
            merged, inherited = core.effective_config(cache[wf_dir], content)
            label = content["label_name"]
            if inherited:
                st.warning(
                    f"{label}: not set by this form, so taken from the "
                    f"cluster's config/config.yaml: {', '.join(inherited)}"
                )
            with st.expander(f"Effective config: {label}"):
                st.code(yaml.safe_dump(merged, sort_keys=False), "yaml")

with tab_launch:
    files = st.session_state.get("files")
    if not files:
        st.warning("Fix the config in the Config tab first.")
    else:
        seg_cfgs = [c for p, c in files.items() if not p.endswith("multi.yaml")]
        plan_for = st.selectbox(
            "plan for",
            range(len(seg_cfgs)),
            format_func=lambda i: seg_cfgs[i]["label_name"],
        )
        if st.button("Plan (tiles, memory, size -- no segmentation)"):
            cmd = core.plan_command(seg_cfgs[plan_for])
            rc, out, err = run(
                client,
                core.inner_command(wf_dir, setup_cmd, cmd),
                timeout=300,
            )
            if rc:
                st.error(
                    "plan failed -- it needs the converted image, so run the "
                    f"workflow (or a dry run + convert) once first.\n{err}"
                )
            else:
                st.code(out, language="json")

        mode = st.radio("run mode", core.RUN_MODES)
        profile = "profile/slurm"
        slurm: dict = {}
        if mode != core.DRY_RUN:
            profile = st.text_input("--workflow-profile", "profile/slurm")
        if mode == core.SLURM_JOB:
            st.caption(
                "The controller only submits and watches jobs: one CPU, "
                "little memory, but it must outlive the whole run."
            )
            c1, c2, c3, c4, c5 = st.columns(5)
            slurm = {
                "time": c1.text_input(
                    "time", preset.get("controller_time", "3-00:00:00")
                ),
                "mem": c2.text_input("mem", preset.get("controller_mem", "4G")),
                "partition": c3.text_input(
                    "partition", preset.get("controller_partition", "")
                ),
                "qos": c4.text_input("qos", preset.get("controller_qos", "")),
                "account": c5.text_input(
                    "account", preset.get("controller_account", "")
                ),
            }
        elif mode == core.LOGIN_NODE:
            st.warning(
                "The controller runs for the whole workflow on the login "
                "node, where a reboot or a process reaper ends it. Prefer "
                "the SLURM-job mode unless your cluster forbids it."
            )
        if st.button("Launch", type="primary"):
            launch(
                client,
                wf_dir=wf_dir,
                setup_cmd=setup_cmd,
                files=files,
                name=st.session_state["run_name"],
                mode=mode,
                profile=profile,
                slurm=slurm,
            )

with tab_jobs:
    text = read_remote(client, f"{wf_dir}/{core.REGISTRY}") or ""
    jobs = core.parse_registry(text)
    if not jobs:
        st.caption("No jobs launched from this workflow directory yet.")
    else:
        pick = st.selectbox(
            "job",
            range(len(jobs)),
            format_func=lambda i: f"{jobs[i]['name']} · {jobs[i]['mode']}",
        )
        auto = st.toggle("auto-refresh every 15 s", value=False)

        @st.fragment(run_every=15 if auto else None)
        def monitor(job=jobs[pick]):
            # squeue may live behind `module load`, so SLURM jobs are
            # checked through the setup line; a PID check and a tail need
            # neither it nor a login shell.
            _, state, _ = run(
                client,
                core.inner_command(wf_dir, setup_cmd, core.status_command(job))
                if job.get("slurm_job")
                else core.status_command(job),
                login=bool(job.get("slurm_job")),
                timeout=30,
            )
            ident = job.get("slurm_job") or job.get("pid")
            st.write(
                f"**{ident}** · {state.strip() or 'unknown'} · "
                f"started {job['started']}"
            )
            _, tail, _ = run(
                client,
                f"tail -n 300 {shlex.quote(job['log'])}",
                login=False,
                timeout=30,
            )
            st.code(tail or "(log is empty so far)", language="text")
            if st.button("refresh now"):
                st.rerun(scope="fragment")

        monitor()
