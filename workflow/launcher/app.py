"""Streamlit launcher for the patchworks Snakemake workflow.

Connects to a cluster over SSH, builds a run config through a form, shows
what Snakemake will *actually* run with (the uploaded file merged over the
cluster copy's config.yaml), and submits the workflow -- by default with the
Snakemake controller itself running as a small SLURM job. Jobs are recorded
on the cluster, so the Jobs tab finds them again after a reload.

Usage:
    pixi run start          # or: pip install -r requirements.txt
                            #     streamlit run app.py

The logic lives in launcher_core.py (no UI, tested); this file is the UI.
See README.md for the security model before sharing a deployment.
"""

from __future__ import annotations

import io
import shlex
import stat
from datetime import datetime
from pathlib import Path

import paramiko
import streamlit as st
import yaml

import launcher_core as core

st.set_page_config(
    page_title="patchworks launcher",
    page_icon=":material/grid_view:",
    layout="wide",
)

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
    # A clone in the home folder is the usual place: fill it in if found.
    try:
        sftp = get_sftp()
        guess = core.join_remote(sftp.normalize("."), "patchworks/workflow")
        sftp.stat(f"{guess}/Snakefile")
        st.session_state["wf_dir__picked"] = guess
    except OSError:
        pass


def get_sftp() -> paramiko.SFTPClient:
    """One SFTP channel per session, reopened if the server closed it."""
    sftp = st.session_state.get("sftp")
    if sftp is None or sftp.sock.closed:
        sftp = get_client().open_sftp()
        st.session_state["sftp"] = sftp
    return sftp


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


def remote_is_dir(path: str) -> bool:
    try:
        return stat.S_ISDIR(get_sftp().stat(path).st_mode or 0)
    except OSError:
        return False


def list_remote(path: str) -> list[tuple[str, bool]]:
    """``(name, is_dir)`` for each entry of a remote folder."""
    sftp = get_sftp()
    entries = []
    for attr in sftp.listdir_attr(path):
        mode = attr.st_mode or 0
        is_dir = stat.S_ISDIR(mode)
        if stat.S_ISLNK(mode):  # follow links to folders
            is_dir = remote_is_dir(core.join_remote(path, attr.filename))
        entries.append((attr.filename, is_dir))
    return entries


# ---------------------------------------------------------------------------
# Remote path picker
# ---------------------------------------------------------------------------

ICONS = {
    "dir": ":material/folder:",
    "store": ":material/deployed_code:",
    "image": ":material/image:",
    "file": ":material/description:",
}
MAX_ENTRIES = 300


def _go(cwd_key: str, path: str) -> None:
    st.session_state[cwd_key] = path


def _go_typed(cwd_key: str, typed_key: str) -> None:
    path = st.session_state.get(typed_key, "").strip()
    if path:
        st.session_state[cwd_key] = path


def _pick(key: str, path: str) -> None:
    # Applied before the input is drawn on the next run: a widget's value
    # cannot be changed once it is on the page.
    st.session_state[f"{key}__picked"] = path
    st.session_state[f"{key}__close"] = True


def _start_dir(value: str, dirs_only: bool) -> str:
    sftp = get_sftp()
    value = value.strip()
    if value.startswith("/"):
        if dirs_only and remote_is_dir(value):
            return value
        parent = core.parent_dir(value)
        if remote_is_dir(parent):
            return parent
    return sftp.normalize(".")


@st.dialog("Choose on the cluster", width="large")
def browse(key: str, label: str, dirs_only: bool) -> None:
    if st.session_state.pop(f"{key}__close", False):
        st.rerun()
    cwd_key = f"{key}__cwd"
    if not st.session_state.get(cwd_key):
        st.session_state[cwd_key] = _start_dir(
            st.session_state.get(key, ""), dirs_only
        )
    cwd = st.session_state[cwd_key]
    home = get_sftp().normalize(".")

    st.markdown(f"**{label}** — {'a folder' if dirs_only else 'a file'}")
    st.code(cwd, language=None)
    c1, c2, c3 = st.columns([1, 1, 3])
    c1.button(
        "Up",
        icon=":material/arrow_upward:",
        on_click=_go,
        args=(cwd_key, core.parent_dir(cwd)),
        disabled=cwd == "/",
    )
    c2.button(
        "Home", icon=":material/home:", on_click=_go, args=(cwd_key, home)
    )
    if dirs_only:
        c3.button(
            "Use this folder",
            icon=":material/check:",
            type="primary",
            on_click=_pick,
            args=(key, cwd),
        )
    c1, c2 = st.columns(2)
    c1.text_input(
        "go to",
        key=f"{key}__typed",
        placeholder="/scratch/…",
        on_change=_go_typed,
        args=(cwd_key, f"{key}__typed"),
    )
    needle = c2.text_input("filter", key=f"{key}__filter").strip().lower()

    try:
        entries = core.browse_entries(list_remote(cwd), dirs_only=dirs_only)
    except OSError as exc:
        st.error(f"cannot list {cwd}: {exc}")
        return
    if needle:
        entries = [e for e in entries if needle in e[0].lower()]
    with st.container(height=380):
        if not entries:
            st.caption("(nothing here)")
        for i, (name, kind) in enumerate(entries[:MAX_ENTRIES]):
            path = core.join_remote(cwd, name)
            # Folders open; stores, images and files are picked. In a
            # folder picker a store is just a folder.
            opens = kind == "dir" or dirs_only
            st.button(
                name,
                key=f"{key}__e{i}",
                icon=ICONS[kind],
                type="tertiary",
                on_click=_go if opens else _pick,
                args=(cwd_key, path) if opens else (key, path),
            )
        if len(entries) > MAX_ENTRIES:
            st.caption(
                f"… {len(entries) - MAX_ENTRIES} more: type in *filter* "
                "to narrow the list."
            )
    if dirs_only:
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        new = c1.text_input("or a new folder in here", key=f"{key}__new")
        c2.button(
            "Use it",
            disabled=not new.strip(),
            on_click=_pick,
            args=(key, core.join_remote(cwd, new.strip().strip("/"))),
        )


def remote_path(
    label: str,
    key: str,
    *,
    dirs_only: bool = False,
    help: str | None = None,
    narrow: bool = False,
) -> str:
    """A path on the cluster: typed in, or chosen with *Browse*."""
    picked = st.session_state.pop(f"{key}__picked", None)
    if picked is not None:
        st.session_state[key] = picked
    c1, c2 = st.columns(
        [3, 1] if narrow else [6, 1], vertical_alignment="bottom"
    )
    value = c1.text_input(label, key=key, help=help)
    if c2.button(
        ":material/folder_open:",
        key=f"{key}__browse",
        help="browse the cluster",
    ):
        st.session_state[f"{key}__cwd"] = None  # start from the value
        browse(key, label, dirs_only)
    return (value or "").strip()


# ---------------------------------------------------------------------------
# Config form
# ---------------------------------------------------------------------------


def section(title: str, caption: str | None = None):
    """A bordered block of the page, with a heading."""
    box = st.container(border=True)
    box.markdown(f"#### {title}")
    if caption:
        box.caption(caption)
    return box


def _opt_float(label: str, container=st, help=None, key=None):
    text = container.text_input(label, value="", help=help, key=key)
    return float(text) if text.strip() else None


def image_form() -> dict:
    """Where the image is, where results go, and how it is converted."""
    v: dict = {}
    c1, c2 = st.columns(2)
    with c1:
        v["input"] = remote_path(
            "input image",
            "input",
            help=".ims/.czi/.lif/.nd2/ome-tiff/.zarr, or a TIFF glob "
            "(type the glob in)",
        )
    with c2:
        v["work_dir"] = remote_path(
            "work_dir (results)",
            "work_dir",
            dirs_only=True,
            help="everything is written under here, on the cluster; the "
            "converted image goes to <work_dir>/image.zarr",
        )
    with st.expander("Conversion", icon=":material/transform:"):
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
        c1, c2 = st.columns(2)
        v["convert_chunks"] = c1.text_input(
            "convert_chunks (YAML, e.g. [8, 512, 512]; blank = auto)", ""
        )
        v["sequence_pattern"] = c2.text_input(
            "sequence_pattern (regex, only for a TIFF glob input)", ""
        )
    return v


def tiling_form() -> dict:
    v: dict = {}
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
    c1, c2, _, _ = st.columns(4)
    v["skip_empty"] = c1.checkbox("skip_empty", value=True)
    v["empty_threshold"] = _opt_float("empty_threshold (blank = Otsu)", c2)
    return v


def outputs_form() -> dict:
    """Merging, the label pyramid, the object table and mail."""
    v: dict = {}
    with st.expander("Merge and label pyramid", icon=":material/join:"):
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
    with st.expander("Object table", icon=":material/table:"):
        c1, c2 = st.columns(2)
        v["object_table"] = c1.checkbox(
            "object table (for `patchworks review`)",
            value=True,
            help="one row per object in labels/<name>/table: size, "
            "centroid, bounding box",
        )
        chans = c2.text_input(
            "table_channels (e.g. 0,2; blank = none)",
            "",
            help="adds each object's mean/std intensity in these channels",
        )
        v["table_channels"] = [
            int(c) for c in chans.replace(" ", "").split(",") if c
        ]
    with st.expander("Notifications", icon=":material/mail:"):
        c1, c2 = st.columns(2)
        v["notify_email"] = c1.text_input("notify_email", "")
        v["notify_events"] = c2.multiselect(
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
    st.markdown("**What to segment**")
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

    st.divider()
    st.markdown("**Method**")
    v["method"] = (
        st.segmented_control(
            "method",
            ["cellpose", "dog", "threshold", "custom"],
            default="cellpose",
            format_func=lambda m: {"dog": "DoG (+ deconvolution)"}.get(m, m),
            key=f"{k}method",
            label_visibility="collapsed",
        )
        or "cellpose"
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
            height=68,
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
        v["dog_psf"] = remote_path(
            "deconvolution PSF (blank = no deconvolution)", f"{k}psf"
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
    elif v["method"] == "threshold":
        st.caption("Otsu threshold and connected components: no settings.")
    elif v["method"] == "custom":
        c1, c2 = st.columns(2)
        v["custom_module"] = c1.text_input("custom.module", "", key=f"{k}cm")
        v["custom_function"] = c2.text_input(
            "custom.function", "segment", key=f"{k}cf"
        )
        v["custom_kwargs"] = st.text_area(
            "custom.kwargs (YAML mapping)", "", key=f"{k}ck", height=68
        )

    st.divider()
    st.markdown("**Clean-up**")
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


DEFAULT_LABELS = ["nuclei_labels", "cyto_labels", "cilia_labels"]


def segmentations_form() -> list[dict]:
    """Several segmentations, one tab each."""
    n = st.columns(4)[0].number_input(
        "how many segmentations", min_value=1, max_value=8, value=2, step=1
    )
    defaults = [
        DEFAULT_LABELS[i] if i < len(DEFAULT_LABELS) else f"labels_{i + 1}"
        for i in range(int(n))
    ]
    # A tab is named after its label_name as typed on the previous run.
    names = [
        f"{i + 1} · {st.session_state.get(f's{i}_lbl', defaults[i])}"
        for i in range(int(n))
    ]
    segs = []
    for i, tab in enumerate(st.tabs(names)):
        with tab:
            segs.append(segmentation_form(f"s{i}_", defaults[i]))
    return segs


def relations_form(labels: list[str]) -> tuple[list[dict], dict, dict]:
    """Relations between segmentations, their jobs and the review rules."""
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
            "max_distance_um": st.column_config.NumberColumn(
                "max distance (µm)",
                min_value=0.0,
                help="an `a` object touching no `b` gets the nearest one "
                "within this distance (blank: overlap only)",
            ),
        },
        key="relations",
    )
    relations = [
        r for r in rows if r.get("a") and r.get("b") and r.get("output")
    ]
    with st.expander("Relate and bundle jobs (SLURM)", icon=":material/dns:"):
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
        fmt = st.selectbox("bundle the finished store", ["zip", "iso", "none"])
        bundle = {"format": None if fmt == "none" else fmt}
    with st.expander(
        "Review rules (what `patchworks review` flags)",
        icon=":material/rule:",
    ):
        st.caption(
            "A child outside every parent, or less than min_overlap inside "
            "one, is always flagged. Add how many of each child a parent "
            "should hold: a cell with no nucleus, or 3 cilia, is then "
            "flagged too."
        )
        rules = st.data_editor(
            [],
            num_rows="dynamic",
            column_config={
                "parent": st.column_config.SelectboxColumn(
                    "parent", options=labels
                ),
                "child": st.column_config.SelectboxColumn(
                    "child", options=labels
                ),
                "min": st.column_config.NumberColumn(
                    "min", min_value=0, step=1
                ),
                "max": st.column_config.NumberColumn(
                    "max", min_value=0, step=1
                ),
            },
            key="review_rules",
        )
        min_overlap = st.number_input(
            "min_overlap", min_value=0.0, max_value=1.0, value=0.5, step=0.05
        )
        st.markdown("**Position** (apical / basal / lateral / central)")
        c1, c2, c3 = st.columns(3)
        pos_child = c1.selectbox("classify", ["(none)"] + labels)
        pos_parent = c2.selectbox("in", labels, index=min(1, len(labels) - 1))
        pos_apical = c3.selectbox(
            "apical is",
            [f"away from {n}" for n in labels] + ["+z", "-z"],
            help="away from the nuclei: for epithelia with basal nuclei; "
            "+z: apical is up the stack",
        )
    expect: dict = {}
    for r in rules:
        if r.get("parent") and r.get("child") and r.get("min") is not None:
            hi = r.get("max") if r.get("max") is not None else r["min"]
            expect.setdefault(r["parent"], {})[r["child"]] = (
                int(r["min"]) if hi == r["min"] else [int(r["min"]), int(hi)]
            )
    review = {
        "expect": expect,
        "min_overlap": None if min_overlap == 0.5 else float(min_overlap),
    }
    if pos_child != "(none)":
        review["position"] = {
            pos_child: {
                "parent": pos_parent,
                "apical": pos_apical.removeprefix("away from "),
            }
        }
    return relations, {"relate": relate, "bundle": bundle}, review


# ---------------------------------------------------------------------------
# Plan and launch
# ---------------------------------------------------------------------------


def plan_all(client, wf_dir, setup_cmd, cfgs: list[dict]) -> dict:
    """``patchworks segment --plan`` for each segmentation of the run.

    Every segmentation of a run reads the same converted image, so that is
    checked once up front: without it there is nothing to plan.
    """
    store = core.image_store(cfgs[0])
    _, out, _ = run(
        client,
        f"cd {shlex.quote(wf_dir)} && {core.store_exists_command(store)}",
        login=False,
        timeout=30,
    )
    if out.strip() != "YES":
        return {"missing": store}
    results = {}
    progress = st.progress(0.0)
    for i, cfg in enumerate(cfgs):
        label = cfg["label_name"]
        progress.progress(i / len(cfgs), text=f"planning {label}…")
        rc, out, err = run(
            client,
            core.inner_command(wf_dir, setup_cmd, core.plan_command(cfg)),
            timeout=300,
        )
        plan = core.parse_plan(out) if rc == 0 else None
        results[label] = {"plan": plan, "rc": rc, "out": out, "err": err}
    progress.empty()
    return {"results": results}


def show_plans(state: dict) -> None:
    if "missing" in state:
        st.warning(
            f"There is no converted image at `{state['missing']}` yet, and "
            "the plan reads it. Launch the run once (converting is its "
            "first step), then plan. Check that work_dir is the one you "
            "mean.",
            icon=":material/hourglass_empty:",
        )
        return
    results = state["results"]
    rows = [
        core.plan_row(label, r["plan"])
        for label, r in results.items()
        if r["plan"]
    ]
    if rows:
        st.dataframe(rows, hide_index=True)
    for label, r in results.items():
        if r["plan"] is None:
            reason = core.last_error_line(r["err"] or r["out"]) or (
                f"exit code {r['rc']}"
            )
            st.error(f"**{label}**: plan failed — {reason}")
        with st.expander(f"{label}: full output", icon=":material/terminal:"):
            st.code(
                ((r["out"] or "") + (r["err"] or ""))[-20000:] or "(no output)",
                language="text",
            )


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
        st.code(
            (out + err)[-20000:] or "(no output)", language="text", height=400
        )
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
    st.caption(f"log: {log} · config: {job['config']} — see the Jobs tab")


STATE_COLORS = {"RUNNING": "green", "PENDING": "orange", "DONE": "gray"}


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### :material/dns: Cluster")
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
        st.success(
            f"connected as **{st.session_state['ssh_target']}**",
            icon=":material/link:",
        )
        if st.button("Disconnect", icon=":material/link_off:"):
            get_client().close()
            for key in ("ssh_client", "sftp", "preset", "base_cfg", "plans"):
                st.session_state.pop(key, None)
            st.rerun()

    st.divider()
    st.markdown("### :material/folder_code: Workflow on the cluster")
    preset = st.session_state.get("preset", {})
    hint = preset.get("workflow_dir_hint", "/path/to/patchworks/workflow")
    if get_client() is None:
        wf_dir = ""
        st.caption("Connect first.")
    else:
        wf_dir = remote_path(
            "workflow folder",
            "wf_dir",
            dirs_only=True,
            help=f"the clone's workflow/ folder, e.g. {hint}",
            narrow=True,
        ).rstrip("/")
        if wf_dir:
            checked = st.session_state.setdefault("wf_checked", {})
            if wf_dir not in checked:
                try:
                    get_sftp().stat(f"{wf_dir}/Snakefile")
                    checked[wf_dir] = True
                except OSError:
                    checked[wf_dir] = False
            if checked[wf_dir]:
                st.caption(":green[:material/check_circle: Snakefile found]")
            else:
                st.warning(
                    "No Snakefile in this folder: pick the clone's "
                    "`workflow/` folder."
                )
    setup_cmd = st.text_area(
        "environment setup (run before snakemake)",
        value=preset.get("setup_cmd", ""),
        help='e.g. eval "$(pixi shell-hook)"',
        height=68,
    )

st.title(":material/grid_view: patchworks launcher")

client = get_client()
if client is None:
    st.info(
        "Connect to a cluster in the sidebar to continue.",
        icon=":material/arrow_back:",
    )
    st.stop()
if not wf_dir:
    st.info(
        "Choose the workflow folder in the sidebar to continue.",
        icon=":material/arrow_back:",
    )
    st.stop()

st.caption(f"{st.session_state['ssh_target']} · `{wf_dir}`")

tab_config, tab_launch, tab_jobs = st.tabs(
    [
        ":material/tune: 1 · Configure",
        ":material/rocket_launch: 2 · Plan & launch",
        ":material/monitor_heart: 3 · Jobs",
    ]
)

with tab_config:
    with section(":material/alt_route: Run"):
        kind = (
            st.segmented_control(
                "run",
                ["One segmentation", "Several segmentations + relations"],
                default="One segmentation",
                label_visibility="collapsed",
            )
            or "One segmentation"
        )
    single = kind.startswith("One")

    with section(
        ":material/image: Image",
        "The image to segment, and where every result goes.",
    ):
        shared = image_form()
    with section(
        ":material/grid_on: Tiling",
        "How the image is cut into tiles for the GPU jobs.",
    ):
        shared |= tiling_form()

    if single:
        with section(":material/category: Segmentation"):
            segs = [segmentation_form()]
    else:
        with section(
            ":material/category: Segmentations",
            "Each gets its own label name, channel and method; they share "
            "the image and tiling above.",
        ):
            segs = segmentations_form()
        labels = [s["label_name"] for s in segs]
        with section(":material/hub: Relations and review"):
            relations, jobs_cfg, review = relations_form(labels)

    with section(":material/output: Outputs"):
        shared |= outputs_form()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    files: dict = {}
    problems: list[str] = []
    try:
        if single:
            cfg = core.build_config({**shared, **segs[0]})
            name = f"{core.safe_name(cfg['label_name'])}_{stamp}"
            files = {f"{wf_dir}/config/launcher_{name}.yaml": cfg}
        else:
            name = f"multi_{stamp}"
            files = core.build_multi(
                shared,
                segs,
                relations,
                directory=f"{wf_dir}/config/launcher_{name}",
                relate=jobs_cfg["relate"],
                bundle=jobs_cfg["bundle"],
                review=review,
            )
            problems = core.multi_problems(files)
        seg_cfgs = [c for p, c in files.items() if not p.endswith("multi.yaml")]
        missing = sorted({m for c in seg_cfgs for m in core.missing_fields(c)})
        if missing:
            problems.insert(0, f"still to fill in: {', '.join(missing)}")
    except Exception as exc:
        problems = [f"the config is not valid yet: {exc}"]

    with section(":material/fact_check: Check"):
        if problems:
            files = {}
            for problem in problems:
                st.warning(problem, icon=":material/edit_note:")
        else:
            st.success(
                "The config is complete: go to **2 · Plan & launch**.",
                icon=":material/check_circle:",
            )
            cache = st.session_state.setdefault("base_cfg", {})
            if wf_dir not in cache:
                text = read_remote(client, f"{wf_dir}/config/config.yaml")
                cache[wf_dir] = yaml.safe_load(text) if text else None
            for path, content in files.items():
                if path.endswith("/multi.yaml"):
                    with st.expander("multi.yaml", icon=":material/code:"):
                        st.code(
                            yaml.safe_dump(content, sort_keys=False), "yaml"
                        )
                    continue
                merged, inherited = core.effective_config(
                    cache[wf_dir], content
                )
                label = content["label_name"]
                if inherited:
                    st.warning(
                        f"{label}: not set by this form, so taken from the "
                        f"cluster's config/config.yaml: {', '.join(inherited)}"
                    )
                with st.expander(
                    f"Effective config: {label}", icon=":material/code:"
                ):
                    st.code(yaml.safe_dump(merged, sort_keys=False), "yaml")
    st.session_state["files"] = files
    st.session_state["run_name"] = name if files else None

with tab_launch:
    files = st.session_state.get("files")
    if not files:
        st.info(
            "Finish the config in **1 · Configure** first.",
            icon=":material/arrow_back:",
        )
    else:
        seg_cfgs = [c for p, c in files.items() if not p.endswith("multi.yaml")]
        many = len(seg_cfgs) > 1
        with section(
            ":material/straighten: Plan",
            "Tiles, memory and output size, read from the converted image: "
            "no segmentation runs, so it is safe on the login node."
            + (" Every segmentation is planned." if many else ""),
        ):
            if st.button(
                "Plan all segmentations" if many else "Plan",
                icon=":material/play_arrow:",
            ):
                st.session_state["plans"] = plan_all(
                    client, wf_dir, setup_cmd, seg_cfgs
                )
            if st.session_state.get("plans"):
                show_plans(st.session_state["plans"])

        with section(
            ":material/rocket_launch: Launch",
            "Uploads the config"
            + (f"s of all {len(seg_cfgs)} segmentations" if many else "")
            + " and starts the run. Launching the same settings again "
            "resumes it.",
        ):
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
                    "mem": c2.text_input(
                        "mem", preset.get("controller_mem", "4G")
                    ),
                    "partition": c3.text_input(
                        "partition", preset.get("controller_partition", "")
                    ),
                    "qos": c4.text_input(
                        "qos", preset.get("controller_qos", "")
                    ),
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
            with st.expander("Files to upload", icon=":material/upload:"):
                st.code("\n".join(files), language=None)
            if st.button(
                "Dry run" if mode == core.DRY_RUN else "Launch",
                type="primary",
                icon=":material/rocket_launch:",
            ):
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
        st.info(
            "No jobs launched from this workflow folder yet.",
            icon=":material/inbox:",
        )
    else:
        with section(":material/monitor_heart: Job"):
            c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
            pick = c1.selectbox(
                "job",
                range(len(jobs)),
                format_func=lambda i: f"{jobs[i]['name']} · {jobs[i]['mode']}",
            )
            auto = c2.toggle("auto-refresh every 15 s", value=False)

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
            state = state.strip() or "unknown"
            ident = job.get("slurm_job") or job.get("pid")
            kind = "SLURM job" if job.get("slurm_job") else "PID"
            color = STATE_COLORS.get(state, "red")
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
                c1.markdown(f"**state**  \n:{color}[**{state}**]")
                c2.markdown(f"**{kind}**  \n{ident}")
                c3.markdown(f"**started**  \n{job['started']}")
                c4.button(
                    "Refresh",
                    icon=":material/refresh:",
                    key="refresh_job",
                )
                st.caption(f"log: `{job['log']}` · config: `{job['config']}`")
            _, tail, _ = run(
                client,
                f"tail -n 300 {shlex.quote(job['log'])}",
                login=False,
                timeout=30,
            )
            st.code(
                tail or "(log is empty so far)", language="text", height=500
            )

        monitor()
