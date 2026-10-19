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
import re
import shlex
import stat
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import paramiko
import streamlit as st
import yaml

import launcher_core as core
from help_text import HELP

st.set_page_config(
    page_title="patchworks launcher",
    page_icon=":material/grid_view:",
    layout="wide",
)

# While a part of the page re-runs (the Jobs tab's auto-refresh), Streamlit
# fades what it shows until redrawn: keep it readable and usable instead.
st.markdown(
    "<style>[data-stale='true']{opacity:1 !important;transition:none "
    "!important;}</style>",
    unsafe_allow_html=True,
)

CLUSTERS = core.load_clusters(Path(__file__).with_name("clusters.yaml"))


# ---------------------------------------------------------------------------
# SSH
# ---------------------------------------------------------------------------


def get_client() -> paramiko.SSHClient | None:
    """The session's SSH connection; None (log in again) once it dropped."""
    client = st.session_state.get("ssh_client")
    transport = client.get_transport() if client is not None else None
    if client is not None and (transport is None or not transport.is_active()):
        for key in ("ssh_client", "sftp"):
            st.session_state.pop(key, None)
        st.session_state["connection_lost"] = True
        return None
    return client


def connect(
    host,
    port,
    username,
    password,
    fingerprints,
    allow_unverified,
    use_keys=False,
):
    """Open the SSH connection: with the password typed, or (*use_keys*)
    with the keys and agent ``ssh`` itself would use -- *password* is then
    the key's passphrase, if any."""
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(
        core.make_host_key_policy(fingerprints, allow_unverified)
    )
    if use_keys:
        client.connect(
            host,
            port=port,
            username=username,
            passphrase=password or None,
            timeout=15,
            look_for_keys=True,
            allow_agent=True,
        )
    else:
        client.connect(
            host,
            port=port,
            username=username,
            password=password,
            timeout=15,
            # Only the password typed here: no key or agent unless asked.
            look_for_keys=False,
            allow_agent=False,
        )
    # A keepalive every 30 s: firewalls drop an idle connection, and the
    # page sits idle while you fill in the form.
    client.get_transport().set_keepalive(30)
    st.session_state["ssh_client"] = client
    st.session_state["ssh_target"] = f"{username}@{host}:{port}"
    st.session_state.pop("connection_lost", None)
    # The workflow folder used last time, else a clone in the home folder
    # (the usual place): fill it in if found.
    sftp = get_sftp()
    for guess in (
        core.load_settings().get("wf_dir"),
        core.join_remote(sftp.normalize("."), "patchworks/workflow"),
    ):
        try:
            if guess:
                sftp.stat(f"{guess}/Snakefile")
                st.session_state["wf_dir__picked"] = guess
                break
        except OSError:
            continue


def get_sftp() -> paramiko.SFTPClient:
    """One SFTP channel per session, reopened if the server closed it."""
    sftp = st.session_state.get("sftp")
    if sftp is None or sftp.sock.closed:
        sftp = get_client().open_sftp()
        st.session_state["sftp"] = sftp
    return sftp


class ConnectionLost(Exception):
    """The SSH connection dropped (network, server timeout): log in again."""


def _forget_connection() -> None:
    """Drop the dead connection; the sidebar then asks to log in again (or,
    with an SSH key, logs in again by itself, once)."""
    for key in ("ssh_client", "sftp", "auto_tried"):
        st.session_state.pop(key, None)
    st.session_state["connection_lost"] = True


def _connection_lost(why: str):
    """In the page's own thread: back to the log-in prompt at once. In a
    worker thread (the Jobs tab's parallel fetches), which cannot touch the
    page: raise :class:`ConnectionLost` for its caller to do so."""
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx(suppress_warning=True) is not None:
        _forget_connection()
        st.rerun()
    raise ConnectionLost(why)


def run(
    client,
    command: str,
    *,
    login: bool = True,
    timeout: float = 120,
    stdin_text: str | None = None,
):
    """Run *command* remotely; a login shell only when PATH needs it.

    A dropped connection sends the page back to the log-in prompt (see
    :func:`_connection_lost`).
    """
    wrapped = f"bash -lc {shlex.quote(command)}" if login else command
    transport = client.get_transport() if client is not None else None
    if transport is None or not transport.is_active():
        _connection_lost("the SSH connection to the cluster dropped")
    # The server caps the channels one connection may have open at once
    # (OpenSSH's MaxSessions): one refused while others close is retried.
    for attempt in range(4):
        try:
            stdin, stdout, stderr = client.exec_command(
                wrapped, timeout=timeout
            )
            if stdin_text is not None:
                stdin.write(stdin_text)
                stdin.channel.shutdown_write()
            break
        except paramiko.ChannelException:
            if attempt == 3:
                raise
            time.sleep(0.5 * 2**attempt)
        except EOFError as exc:  # the server closed the connection
            _connection_lost(str(exc) or "the server closed the connection")
        except (paramiko.SSHException, OSError) as exc:
            if not transport.is_active():
                _connection_lost(str(exc))
            raise
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


def _pick_tiffs(key: str, pattern_glob: str, pattern: str | None) -> None:
    _pick(key, pattern_glob)
    if pattern:
        st.session_state["sequence_pattern__picked"] = pattern


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
def browse(
    key: str, label: str, dirs_only: bool, tiff_folder: bool = False
) -> None:
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
        listing = list_remote(cwd)
    except OSError as exc:
        st.error(f"cannot list {cwd}: {exc}")
        return
    entries = core.browse_entries(listing, dirs_only=dirs_only)
    tiffs = core.tiff_names(listing) if tiff_folder else []
    if len(tiffs) > 1:
        # One image stored as a plane per file: the input is a glob.
        pattern = core.guess_sequence_pattern(tiffs)
        st.button(
            f"Use this folder's {len(tiffs)} TIFFs as one image",
            icon=":material/burst_mode:",
            type="primary",
            on_click=_pick_tiffs,
            args=(key, core.tiff_glob(cwd, tiffs), pattern),
            help="each file one plane; which plane is read from the file "
            "names (sequence_pattern)",
        )
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


def _update_workflow(wf_dir: str) -> None:
    rc, out, _ = run(get_client(), core.update_command(wf_dir), timeout=300)
    st.session_state["update_note"] = (rc, out.strip()[-2000:])
    # What the clone has, looked at again.
    st.session_state.pop("missing_scripts", None)
    st.session_state.pop("cellpose_models", None)


def workflow_version_check(wf_dir: str) -> None:
    """Warn when the cluster's clone is older than this launcher (it lacks
    scripts the launcher runs), and offer to update it."""
    missing = st.session_state.setdefault("missing_scripts", {})
    if wf_dir not in missing:
        absent = []
        for script in core.REQUIRED_SCRIPTS:
            try:
                get_sftp().stat(f"{wf_dir}/{script}")
            except OSError:
                absent.append(script)
        missing[wf_dir] = absent
    note = st.session_state.pop("update_note", None)
    if note:
        rc, out = note
        (st.success if rc == 0 else st.error)(
            ("updated: " if rc == 0 else "git pull failed: ")
            + (out.splitlines()[-1].strip() if out else "")
        )
    if missing[wf_dir]:
        st.warning(
            "The workflow on the cluster is older than this launcher: it has "
            f"no {', '.join(s.split('/')[-1] for s in missing[wf_dir])}. "
            "Update it (git pull in the clone).",
            icon=":material/update:",
        )
        st.button(
            "Update it",
            icon=":material/download:",
            key="update_workflow",
            on_click=_update_workflow,
            args=(wf_dir,),
            help="Runs `git pull --ff-only` in the clone on the cluster: "
            "it never merges, so local changes there stop it and say so.",
        )


def remote_path(
    label: str,
    key: str,
    *,
    dirs_only: bool = False,
    help: str | None = None,
    narrow: bool = False,
    tiff_folder: bool = False,
) -> str:
    """A path on the cluster: typed in, or chosen with *Browse*."""
    picked = st.session_state.pop(f"{key}__picked", None)
    if picked is not None:
        st.session_state[key] = picked
    c1, c2 = st.columns(
        [3, 1] if narrow else [6, 1], vertical_alignment="bottom"
    )
    value = c1.text_input(label, key=key, help=help or help_for(key))
    if c2.button(
        ":material/folder_open:",
        key=f"{key}__browse",
        help="browse the cluster",
    ):
        st.session_state[f"{key}__cwd"] = None  # start from the value
        browse(key, label, dirs_only, tiff_folder)
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


# A segmentation's fields have short keys (``s0_cpm``): what they explain.
SHORT_KEYS = {
    "lbl": "label_name",
    "ch": "channel",
    "nuc": "nuclei_channel",
    "ov": "overlap",
    "cpm": "cp_model",
    "cpd": "cp_diameter",
    "cp3": "cp_do_3d",
    "cpg": "cp_gpu",
    "cpx": "cp_extra_kwargs",
    "norm": "normalize",
    "psm": "ps_model",
    "pss": "ps_segmentation",
    "psb": "ps_beta",
    "psmin": "ps_post_minsize",
    "nmin": "ws_nuclei_min_size",
    "seed": "seed_labels",
    "rad": "max_radius_um",
    "fg": "foreground",
    "px": "extra_kwargs",
    "dl": "dog_low_sigma",
    "dh": "dog_high_sigma",
    "dt": "dog_threshold",
    "du": "dog_sigma_units",
    "dg": "dog_use_gpu",
    "psf": "dog_psf",
    "wl": "dog_wavelength",
    "na": "dog_na",
    "ni": "dog_nimm",
    "dup": "dog_dup_rev_z",
    "cm": "custom_module",
    "cf": "custom_function",
    "ck": "custom_kwargs",
    "stitch_True": "stitch",
    "stitch_False": "stitch",
    "iou": "iou_threshold",
    "holes": "fill_holes",
    "open": "open_radius",
    "dil": "dilate",
    "dgpu": "dilate_gpu",
    "minv": "min_volume",
    "maxv": "max_volume",
}


def help_for(key: str | None) -> str | None:
    """The explanation of the field with widget key *key* (help_text.HELP)."""
    if not key:
        return None
    if key in HELP:
        return HELP[key]
    short = re.sub(r"^s\d+_", "", key)
    return HELP.get(SHORT_KEYS.get(short, short))


def _init(key: str, help: str | None = None, **default) -> dict:
    """``key=``, its explanation (``help=``, the question mark beside the
    field) and the widget's default, unless the key already holds a value
    (one a past run's config put there): a widget given both warns."""
    out = {"key": key, "help": help or help_for(key)}
    if key in st.session_state:
        return out
    return {**out, **default}


def _opt_float(label: str, container=st, help=None, key=None):
    text = container.text_input(label, **_init(key, help=help, value=""))
    return float(text) if text.strip() else None


def image_form() -> dict:
    """Where the image is, where results go, and how it is converted."""
    v: dict = {}
    c1, c2 = st.columns(2)
    with c1:
        v["input"] = remote_path(
            "input image",
            "input",
            tiff_folder=True,
        )
    with c2:
        v["work_dir"] = remote_path(
            "work_dir (results)",
            "work_dir",
            dirs_only=True,
        )
    picked = st.session_state.pop("sequence_pattern__picked", None)
    if picked is not None:
        st.session_state["sequence_pattern"] = picked
    v["sequence_pattern"] = None
    if core.is_glob(v["input"]):
        v["sequence_pattern"] = (
            st.text_input(
                "sequence_pattern: where the plane's index is in each file name",
                key="sequence_pattern",
                placeholder=r"_Z(?P<Z>\d+).*?_C(?P<C>\d+)",
                help="a regex; each named group (Z, C, T) is an axis, read "
                "from the digits it matches. Guessed from the file names when "
                "you use a folder's TIFFs; check it.",
            ).strip()
            or None
        )
    if v["work_dir"]:
        channels_view(v["work_dir"])
    with st.expander("Conversion", icon=":material/transform:"):
        c1, c2, c3, c4 = st.columns(4)
        v["reuse_pyramid"] = c1.checkbox(
            "reuse_pyramid", **_init("reuse_pyramid", value=False)
        )
        v["shard"] = c2.checkbox("shard", **_init("shard", value=False))
        v["ngff_version"] = c3.selectbox(
            "ngff_version", ["auto", "0.4", "0.5"], **_init("ngff_version")
        )
        v["compression"] = c4.selectbox(
            "compression",
            core.COMPRESSIONS,
            **_init("compression"),
        )
        v["convert_chunks"] = st.text_input(
            "convert_chunks (YAML, e.g. [8, 512, 512]; blank = auto)",
            **_init("convert_chunks", value=""),
        )
    return v


def channels_view(work_dir: str) -> None:
    """Each channel of the converted image, to pick the channels by."""
    with st.expander("See the channels", icon=":material/palette:"):
        key = f"channels_{work_dir}"
        st.caption(
            "Each channel's maximum over z, once the image is converted "
            "(Plan & launch → Convert only); and its voxel size."
        )
        if st.button("Show", icon=":material/visibility:", key="show_channels"):
            # Any config of this work_dir will do: the script reads only
            # its work_dir. Written here, so this works before Configure
            # is complete.
            probe = f"{wf_dir}/config/launcher_channels_probe.yaml"
            run(
                get_client(),
                f"mkdir -p {shlex.quote(wf_dir)}/config",
                login=False,
            )
            write_remote(
                get_client(),
                probe,
                yaml.safe_dump({"work_dir": work_dir, "input": ""}),
            )
            rc, out, err = run(
                get_client(),
                core.inner_command(
                    wf_dir, setup_cmd, core.channels_command(probe)
                ),
                timeout=600,
            )
            st.session_state[key] = core.parse_json_output(out) or {
                "error": core.explain_remote_error(err or out) or f"exit {rc}"
            }
        info = st.session_state.get(key)
        if not info:
            return
        if "error" in info:
            st.warning(info["error"], icon=":material/hourglass_empty:")
            return
        voxel = ", ".join(
            f"{a} {v:g} µm" for a, v in (info.get("voxel_um") or {}).items()
        )
        st.caption(
            f"{info['axes']} {' × '.join(map(str, info['shape']))}"
            + (f" · voxel {voxel}" if voxel else " · voxel size unknown")
        )
        cols = st.columns(min(4, max(1, len(info["pictures"]))))
        for c, path in enumerate(info["pictures"]):
            try:
                with get_sftp().open(path) as fh:
                    cols[c % len(cols)].image(fh.read(), caption=f"channel {c}")
            except OSError:
                cols[c % len(cols)].caption(f"channel {c}: no picture")


def tiling_form() -> dict:
    v: dict = {}
    c1, c2, c3, c4 = st.columns(4)
    v["level"] = c1.number_input(
        "level", min_value=0, step=1, **_init("level", value=0)
    )
    v["tile_shape"] = c2.text_input(
        "tile_shape",
        **_init("tile_shape", value="auto"),
    )
    v["tiles_per_job"] = c3.number_input(
        "tiles_per_job", min_value=1, step=1, **_init("tiles_per_job", value=4)
    )
    v["gpu_memory_gb"] = c4.number_input(
        "gpu_memory_gb (0 = the segment GPU)",
        min_value=0,
        **_init("gpu_memory_gb", value=0),
    )
    c1, c2, _, _ = st.columns(4)
    v["skip_empty"] = c1.checkbox(
        "skip_empty", **_init("skip_empty", value=True)
    )
    v["empty_threshold"] = _opt_float(
        "empty_threshold (blank = Otsu)", c2, key="empty_threshold"
    )
    return v


def outputs_form() -> dict:
    """Merging, the label pyramid, the object table and mail."""
    v: dict = {}
    with st.expander("Merge and label pyramid", icon=":material/join:"):
        st.caption("How tiles are stitched is set per segmentation.")
        v["sequential_labels"] = st.checkbox(
            "sequential_labels", **_init("sequential_labels", value=True)
        )
        c1, c2, c3, c4 = st.columns(4)
        v["pyramid_levels"] = c1.number_input(
            "pyramid_levels", min_value=1, **_init("pyramid_levels", value=5)
        )
        v["pyramid_downscale"] = c2.number_input(
            "pyramid_downscale",
            min_value=2,
            **_init("pyramid_downscale", value=2),
        )
        v["shard_labels"] = c3.checkbox(
            "shard_labels", **_init("shard_labels", value=False)
        )
        v["seam_report"] = c4.checkbox(
            "seam_report", **_init("seam_report", value=True)
        )
        workers = st.text_input(
            "merge_workers (blank = auto)", **_init("merge_workers", value="")
        )
        v["merge_workers"] = int(workers) if workers.strip() else None
    with st.expander("Object table", icon=":material/table:"):
        c1, c2 = st.columns(2)
        v["object_table"] = c1.checkbox(
            "object table (for `patchworks review`)",
            **_init("object_table", value=True),
        )
        chans = c2.text_input(
            "table_channels (e.g. 0,2; blank = none)",
            **_init("table_channels", value=""),
        )
        v["table_channels"] = [
            int(c) for c in chans.replace(" ", "").split(",") if c
        ]
    with st.expander("Notifications", icon=":material/mail:"):
        c1, c2 = st.columns(2)
        v["notify_email"] = c1.text_input(
            "notify_email", **_init("notify_email", value="")
        )
        v["notify_events"] = c2.multiselect(
            "notify_events",
            ["start", "finish", "error"],
            **_init("notify_events", default=["finish", "error"]),
        )
    return v


def _remote_yaml(path: str) -> dict:
    text = read_remote(get_client(), path)
    if text is None:
        raise FileNotFoundError(path)
    return yaml.safe_load(text) or {}


def load_run(path: str, wf_dir: str) -> None:
    """Fill the form from a run's config (a callback: before the widgets).

    A multi config's segmentations and common file resolve as run_multi
    resolves them: next to the multi config, then in the workflow folder.
    """
    try:
        cfg = _remote_yaml(path)
        multi = None
        if "segmentations" in cfg:
            multi = cfg
            here = core.parent_dir(path)

            def resolve(p: str) -> str:
                if p.startswith("/"):
                    return p
                for base in (here, wf_dir):
                    candidate = core.join_remote(base, p)
                    if read_remote(get_client(), candidate) is not None:
                        return candidate
                return core.join_remote(here, p)

            common = (
                _remote_yaml(resolve(cfg["common"]))
                if cfg.get("common")
                else {}
            )
            segs = [
                {**common, **_remote_yaml(resolve(p))}
                for p in cfg["segmentations"]
            ]
        else:
            segs = [cfg]
        state, ignored = core.form_state(segs, multi)
    except Exception as exc:  # shown, not raised: a UI
        st.session_state["load_note"] = (
            "error",
            f"could not load {path}: {exc}",
        )
        return
    if not segs[0].get("input"):
        # A template: the form keeps its own image and work_dir.
        for key in core.IMAGE_KEYS:
            state.pop(key, None)
    for key in ("loaded_relations", "loaded_rules"):
        st.session_state.pop(key, None)
    st.session_state.update(state)
    st.session_state["loaded"] = st.session_state.get("loaded", 0) + 1
    st.session_state["load_note"] = (
        "ok",
        f"loaded {path}"
        + (
            f". Not in the form, so left out: {', '.join(ignored)}"
            if ignored
            else ""
        ),
    )


def templates_dir() -> str:
    """Where templates live: the preset's (a lab may share one), or the
    workflow's own config/templates/."""
    preset = st.session_state.get("preset", {})
    return preset.get("templates_dir") or f"{wf_dir}/config/templates"


def save_template_form(draft: dict) -> None:
    """Keep the form's settings under a name, to start the next run from."""
    with st.popover("Save as template", icon=":material/bookmark_add:"):
        name = st.text_input(
            "name",
            key="template_name",
            placeholder="nuclei_cells_cilia",
            help=HELP["template_name"],
        )
        with_image = st.checkbox(
            "with the image and work_dir",
            key="template_with_image",
            help=HELP["template_with_image"],
        )
        if st.button("Save", type="primary", disabled=not name.strip()):
            out = core.template_files(
                draft, name, templates_dir(), with_image=with_image
            )
            folder = core.parent_dir(next(iter(out)))
            run(get_client(), f"mkdir -p {shlex.quote(folder)}", login=False)
            for path, cfg in out.items():
                write_remote(
                    get_client(), path, yaml.safe_dump(cfg, sort_keys=False)
                )
            st.success(f"saved as {core.safe_name(name)} in {templates_dir()}")


def start_from_form(wf_dir: str, jobs: list[dict]) -> None:
    """Reopen a past launch's config, or any config file on the cluster."""
    past = list(
        dict.fromkeys(
            j["config"]
            for j in jobs
            if j.get("config") and j.get("kind") in ("single", "multi")
        )
    )
    directory = templates_dir()
    try:
        templates = core.template_choices(list_remote(directory), directory)
    except OSError:
        templates = {}
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    template = c1.selectbox(
        "a template",
        list(templates),
        index=None,
        placeholder="saved with Save as template" if not templates else "",
        help=f"{HELP['template']}\n\nFrom `{directory}`.",
    )
    c2.button(
        "Load",
        icon=":material/upload_file:",
        key="load_template",
        disabled=template is None,
        on_click=load_run,
        args=(templates.get(template), wf_dir),
    )
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    choice = c1.selectbox(
        "a past launch",
        past,
        index=None,
        placeholder="newest first" if past else "no launches yet",
        format_func=lambda p: p.rsplit("/config/", 1)[-1],
        help=HELP["past_launch"],
    )
    c2.button(
        "Load",
        icon=":material/upload_file:",
        key="load_past",
        disabled=choice is None,
        on_click=load_run,
        args=(choice, wf_dir),
    )
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    with c1:
        other = remote_path("or a config file (or a multi config)", "load_path")
    c2.button(
        "Load",
        icon=":material/upload_file:",
        key="load_file",
        disabled=not other,
        on_click=load_run,
        args=(other, wf_dir),
    )
    note = st.session_state.pop("load_note", None)
    if note:
        (st.success if note[0] == "ok" else st.error)(note[1])


def segmentation_form(
    k: str = "", label: str = "labels", others: list[str] | None = None
) -> dict:
    """One segmentation: what to label, how, and how to post-process it.

    *k* prefixes every widget key, so several can sit on one page; the keys
    are the ones :func:`launcher_core.segmentation_state` fills in.
    """
    v: dict = {}
    st.markdown("**What to segment**")
    c1, c2, c3, c4 = st.columns(4)
    v["label_name"] = c1.text_input(
        "label_name", **_init(f"{k}lbl", value=label)
    )
    v["channel"] = c2.number_input(
        "channel", min_value=0, step=1, **_init(f"{k}ch", value=0)
    )
    nuc = c3.text_input(
        "nuclei_channel (blank = none)", **_init(f"{k}nuc", value="")
    )
    v["nuclei_channel"] = int(nuc) if nuc.strip() else None
    v["overlap"] = c4.text_input(
        "overlap",
        **_init(f"{k}ov", value="30"),
    )

    st.divider()
    v["method"] = (
        st.segmented_control(
            "method",
            ["cellpose", "plantseg", "watershed", "dog", "threshold", "custom"],
            format_func=lambda m: {
                "dog": "DoG (+ deconvolution)",
                "plantseg": "PlantSeg (cells, membranes)",
                "watershed": "watershed (cells from nuclei)",
            }.get(m, m),
            **_init(f"{k}method", default="cellpose"),
        )
        or "cellpose"
    )
    if v["method"] == "cellpose":
        c1, c2, c3, c4 = st.columns(4)
        models = cellpose_models()
        default = "cyto3" if "cyto3" in models else models[0]
        if st.session_state.get(f"{k}cpm") not in (None, *models):
            models = [*models, st.session_state[f"{k}cpm"]]  # loaded, or own
        v["cp_model"] = c1.selectbox(
            "cellpose.model",
            models,
            accept_new_options=True,
            **_init(f"{k}cpm", index=models.index(default)),
        )
        v["cp_diameter"] = c2.number_input(
            "cellpose.diameter (0 = estimate)",
            min_value=0.0,
            **_init(f"{k}cpd", value=30.0),
        )
        v["cp_do_3d"] = c3.checkbox(
            "cellpose.do_3D", **_init(f"{k}cp3", value=False)
        )
        v["cp_gpu"] = c4.checkbox(
            "cellpose.gpu", **_init(f"{k}cpg", value=True)
        )
        v["cp_extra_kwargs"] = st.text_area(
            "extra cellpose kwargs (YAML mapping)",
            height=68,
            **_init(f"{k}cpx", value=""),
        )
        v["normalize"] = st.radio(
            "normalize",
            ["image", "tile"],
            horizontal=True,
            format_func=lambda n: {
                "image": "one intensity range for the whole image",
                "tile": "each tile its own (uneven staining)",
            }[n],
            **_init(f"{k}norm"),
        )
    elif v["method"] in core.SEEDED:
        plantseg_form(v, k, others or [])
    elif v["method"] == "dog":
        c1, c2, c3, c4 = st.columns(4)
        v["dog_low_sigma"] = c1.number_input(
            "low_sigma", min_value=0.0, **_init(f"{k}dl", value=1.0)
        )
        v["dog_high_sigma"] = c2.number_input(
            "high_sigma", min_value=0.0, **_init(f"{k}dh", value=3.0)
        )
        v["dog_threshold"] = c3.number_input(
            "threshold", format="%.4f", **_init(f"{k}dt", value=0.02)
        )
        v["dog_sigma_units"] = c4.selectbox(
            "sigma_units",
            ["px", "um"],
            **_init(f"{k}du"),
        )
        v["dog_use_gpu"] = st.checkbox(
            "blur/label on GPU (cupy)", **_init(f"{k}dg", value=False)
        )
        v["dog_psf"] = remote_path(
            "deconvolution PSF (blank = no deconvolution)", f"{k}psf"
        )
        if v["dog_psf"]:
            c1, c2, c3, c4 = st.columns(4)
            v["dog_wavelength"] = c1.number_input(
                "wavelength (nm)", **_init(f"{k}wl", value=525)
            )
            v["dog_na"] = c2.number_input("NA", **_init(f"{k}na", value=1.4))
            v["dog_nimm"] = c3.number_input(
                "immersion n", **_init(f"{k}ni", value=1.515)
            )
            v["dog_dup_rev_z"] = c4.selectbox(
                "dup_rev_z",
                ["auto", "on", "off"],
                **_init(f"{k}dup"),
            )
    elif v["method"] == "threshold":
        st.caption("Otsu threshold and connected components: no settings.")
    elif v["method"] == "custom":
        c1, c2 = st.columns(2)
        v["custom_module"] = c1.text_input(
            "custom.module", **_init(f"{k}cm", value="")
        )
        v["custom_function"] = c2.text_input(
            "custom.function", **_init(f"{k}cf", value="segment")
        )
        v["custom_kwargs"] = st.text_area(
            "custom.kwargs (YAML mapping)",
            height=68,
            **_init(f"{k}ck", value=""),
        )

    st.divider()
    st.markdown("**Stitching across tiles**")
    # PlantSeg and the watershed give every voxel to a cell, so cells touch
    # at every seam: they need iou. The key carries that, so the default
    # follows the method when it changes.
    fills = core.fills_space(v)
    c1, c2, _ = st.columns([1, 1, 2])
    v["stitch"] = c1.selectbox(
        "stitch",
        ["touch", "iou"],
        **_init(f"{k}stitch_{fills}", index=1 if fills else 0),
    )
    if v["stitch"] == "iou":
        v["iou_threshold"] = c2.number_input(
            "iou_threshold",
            min_value=0.05,
            max_value=1.0,
            **_init(f"{k}iou", value=0.5),
        )

    st.divider()
    st.markdown("**Clean-up**")
    c1, c2, c3, c4, c5 = st.columns(5)
    holes = c1.selectbox(
        "fill_holes", ["off", "3-D", "per plane"], **_init(f"{k}holes")
    )
    v["fill_holes"] = {"off": False, "3-D": True, "per plane": "per_plane"}[
        holes
    ]
    v["open_radius"] = c2.number_input(
        "open_radius",
        min_value=0,
        **_init(f"{k}open", value=0),
    )
    v["dilate"] = c3.number_input(
        "dilate (px)", min_value=0, **_init(f"{k}dil", value=0)
    )
    if v["dilate"]:
        v["dilate_gpu"] = c3.checkbox(
            "on the GPU", **_init(f"{k}dgpu", value=False)
        )
    v["min_volume"] = _opt_float("min_volume (um³)", c4, key=f"{k}minv")
    v["max_volume"] = _opt_float("max_volume (um³)", c5, key=f"{k}maxv")
    return v


def cellpose_models() -> list[str]:
    """The models the cluster's Cellpose accepts, asked once per session."""
    if "cellpose_models" not in st.session_state:
        with st.spinner("asking the cluster's Cellpose for its models…"):
            _, out, _ = run(
                get_client(),
                core.inner_command(
                    wf_dir, setup_cmd, core.CELLPOSE_MODELS_COMMAND
                ),
                timeout=120,
            )
        st.session_state["cellpose_models"] = core.parse_models(out)
    return st.session_state["cellpose_models"] or list(
        core.CELLPOSE_FALLBACK_MODELS
    )


def plantseg_form(v: dict, k: str, others: list[str]) -> None:
    """PlantSeg or the watershed: cells filling space, from a membrane stain.

    Both can grow one cell per nucleus: nuclei from a channel (the
    nuclei_channel above) or another segmentation's objects (seed_labels).
    """
    if v["method"] == "plantseg":
        c1, c2, c3, c4 = st.columns(4)
        models = list(core.PLANTSEG_MODELS)
        if st.session_state.get(f"{k}psm") not in (None, *models):
            models.append(st.session_state[f"{k}psm"])  # a loaded one
        v["ps_model"] = c1.selectbox(
            "model",
            models,
            accept_new_options=True,
            **_init(f"{k}psm"),
        )
        v["ps_segmentation"] = c2.selectbox(
            "segmentation",
            core.PLANTSEG_SEGMENTATIONS,
            **_init(f"{k}pss"),
        )
        v["ps_beta"] = c3.number_input(
            "beta",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            **_init(f"{k}psb", value=0.6),
        )
        v["ps_post_minsize"] = c4.number_input(
            "smallest cell (voxels)",
            min_value=0,
            **_init(f"{k}psmin", value=100),
        )
        uses_nuclei = v["ps_segmentation"] in core.PLANTSEG_NUCLEI
    else:
        v["ws_nuclei_min_size"] = st.number_input(
            "smallest nucleus (voxels)",
            min_value=0,
            **_init(f"{k}nmin", value=50),
        )
        uses_nuclei = True
    c1, c2, c3 = st.columns(3)
    if uses_nuclei:
        source = c1.radio(
            "nuclei from",
            ["channel", "labels"],
            format_func=lambda s: {
                "channel": "the nuclei channel (above)",
                "labels": "another segmentation",
            }[s],
            disabled=not others,
            **_init(f"{k}seed_from"),
        )
        if source == "labels" and others:
            v["seed_labels"] = c2.selectbox(
                "grow from", others, **_init(f"{k}seed")
            )
        v["max_radius_um"] = _opt_float(
            "at most this far from a nucleus (µm)",
            c3,
            help="blank: up to the walls",
            key=f"{k}rad",
        )
    c1, c2 = st.columns([1, 3])
    v["foreground"] = c1.selectbox(
        "foreground",
        ["otsu", "none"],
        **_init(f"{k}fg"),
    )
    v["extra_kwargs"] = c2.text_area(
        "more settings (YAML mapping)",
        height=68,
        **_init(f"{k}px", value=""),
    )


DEFAULT_LABELS = ["nuclei_labels", "cyto_labels", "cilia_labels"]


def segmentations_form() -> list[dict]:
    """Several segmentations, one tab each."""
    n = st.columns(4)[0].number_input(
        "how many segmentations",
        min_value=1,
        max_value=8,
        step=1,
        **_init("n_segs", value=2),
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
    labels = [
        st.session_state.get(f"s{i}_lbl", defaults[i]) for i in range(int(n))
    ]
    segs = []
    for i, tab in enumerate(st.tabs(names)):
        with tab:
            others = [x for j, x in enumerate(labels) if j != i]
            segs.append(segmentation_form(f"s{i}_", defaults[i], others))
    return segs


def relations_form(labels: list[str]) -> tuple[list[dict], dict, dict]:
    """Relations between segmentations, their jobs and the review rules."""
    st.caption(
        "Each row relates every object of `a` to the `b` object it overlaps "
        "most, written as a two-sheet Excel workbook in work_dir."
    )
    # A loaded config brings its own rows; the key changes with each load,
    # since an editor only takes its rows when first drawn.
    loaded = st.session_state.get("loaded", 0)
    rows = st.data_editor(
        st.session_state.get(
            "loaded_relations",
            [{"a": labels[0], "b": labels[-1], "output": "relations.xlsx"}]
            if len(labels) > 1
            else [],
        ),
        num_rows="dynamic",
        column_config={
            "a": st.column_config.SelectboxColumn(
                "a",
                options=labels,
                help="The objects to relate, one row each in the workbook "
                "(say the nuclei).",
            ),
            "b": st.column_config.SelectboxColumn(
                "b",
                options=labels,
                help="What each of them is related to: the b object it "
                "overlaps most (say the cell holding the nucleus).",
            ),
            "output": st.column_config.TextColumn(
                "output (.xlsx)",
                help="The workbook's file name, in work_dir: a sheet per "
                "a object and one per b object (with its count of a).\n\n"
                "Example: `nuclei_to_cells.xlsx`",
            ),
            "max_distance_um": st.column_config.NumberColumn(
                "max distance (µm)",
                min_value=0.0,
                help="an `a` object touching no `b` gets the nearest one "
                "within this distance (blank: overlap only)",
            ),
        },
        key=f"relations_{loaded}",
    )
    relations = [
        r for r in rows if r.get("a") and r.get("b") and r.get("output")
    ]
    with st.expander("Relate and bundle jobs (SLURM)", icon=":material/dns:"):
        info = cluster_info()
        c1, c2, c3, c4, c5 = st.columns(5)
        relate = {
            "partition": slurm_choice(
                c1,
                "relate partition",
                "relate_partition",
                sorted((info or {}).get("partitions", {})),
            ),
            "mem": c2.text_input("relate mem", **_init("relate_mem", value="")),
            "cpus": int(
                c3.number_input(
                    "relate cpus (0 = default)",
                    min_value=0,
                    **_init("relate_cpus", value=0),
                )
            )
            or None,
            "time": int(
                c4.number_input(
                    "relate minutes (0 = default)",
                    min_value=0,
                    **_init("relate_time", value=0),
                )
            )
            or None,
            "qos": slurm_choice(
                c5,
                "relate qos",
                "relate_qos",
                core.usable_qos(
                    info or {}, st.session_state.get("relate_partition") or None
                ),
            ),
        }
        fmt = st.selectbox(
            "bundle the finished store",
            ["zip", "iso", "none"],
            **_init("bundle_format"),
        )
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
            st.session_state.get("loaded_rules", []),
            num_rows="dynamic",
            column_config={
                "parent": st.column_config.SelectboxColumn(
                    "parent",
                    options=labels,
                    help="The objects that hold others (say the cells).",
                ),
                "child": st.column_config.SelectboxColumn(
                    "child",
                    options=labels,
                    help="What they should hold (say the nuclei).",
                ),
                "min": st.column_config.NumberColumn(
                    "min",
                    min_value=0,
                    step=1,
                    help="The fewest a parent should hold; fewer is flagged."
                    "\n\nExample: 1 nucleus per cell",
                ),
                "max": st.column_config.NumberColumn(
                    "max",
                    min_value=0,
                    step=1,
                    help="The most; blank: the same as min.\n\n"
                    "Example: 2 cilia per cell",
                ),
            },
            key=f"review_rules_{loaded}",
        )
        min_overlap = st.number_input(
            "min_overlap",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            **_init("min_overlap", value=0.5),
        )
        st.markdown("**Position** (apical / basal / lateral / central)")
        c1, c2, c3 = st.columns(3)
        pos_child = c1.selectbox(
            "classify", ["(none)"] + labels, **_init("pos_child")
        )
        pos_parent = c2.selectbox(
            "in", labels, **_init("pos_parent", index=min(1, len(labels) - 1))
        )
        pos_apical = c3.selectbox(
            "apical is",
            [f"away from {n}" for n in labels] + ["+z", "-z"],
            **_init("pos_apical"),
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
# Check, plan and launch
# ---------------------------------------------------------------------------


def upload(client, files: dict) -> None:
    """Write the run's config files (remote path -> config) to the cluster."""
    for path in files:
        parent = path.rsplit("/", 1)[0]
        run(client, f"mkdir -p {shlex.quote(parent)}", login=False)
    for path, content in files.items():
        write_remote(client, path, yaml.safe_dump(content, sort_keys=False))


def run_config_path(files: dict) -> str:
    """The file a run is started from: its multi.yaml, or its one config."""
    return next(
        (p for p in files if p.endswith("/multi.yaml")), next(iter(files))
    )


def check_on_cluster(client, wf_dir, setup, files) -> dict:
    """Upload the configs and run ``scripts/check.py`` on them there.

    On the cluster, in the run's environment: the installed Cellpose
    models, the plugins and the converted image are all there to check
    against, as the run itself will.
    """
    upload(client, files)
    rc, out, err = run(
        client,
        core.inner_command(
            wf_dir, setup, core.check_command(run_config_path(files))
        ),
        timeout=600,
    )
    report = core.parse_json_output(out)
    if report is None:
        return {
            "error": core.explain_remote_error(err or out) or f"exit {rc}",
            "output": (out + err)[-20000:],
        }
    return report


def show_previews(report: dict) -> None:
    """The tile each segmentation was tried on, its objects outlined."""
    for seg in report.get("segmentations", []):
        plan = seg.get("plan") or {}
        if not plan.get("preview"):
            continue
        try:
            with get_sftp().open(plan["preview"]) as fh:
                picture = fh.read()
        except OSError:
            st.caption(f"{seg['label_name']}: no picture at {plan['preview']}")
            continue
        plane = plan.get("preview_plane")
        st.image(
            picture,
            caption=f"{seg['label_name']}: tile {plan['preview_tile']}"
            + ("" if plane is None else f", plane {plane} of the tile")
            + f" · {plan['preview_objects']} objects · "
            f"{plan['seconds_per_tile']:.1f} s per tile",
        )


def show_check(report: dict) -> None:
    """Problems first, then the plan of every segmentation."""
    if "error" in report:
        st.error(f"check could not run: {report['error']}")
        with st.expander("output", icon=":material/terminal:"):
            st.code(report.get("output") or "(none)", language="text")
        return
    for problem in report["shared_problems"]:
        st.error(problem, icon=":material/error:")
    for seg in report["segmentations"]:
        for problem in seg["problems"]:
            st.error(f"**{seg['label_name']}**: {problem}")
        if seg.get("plan_error"):
            st.error(f"**{seg['label_name']}**: {seg['plan_error']}")
    rows = [
        core.plan_row(seg["label_name"], seg["plan"])
        for seg in report["segmentations"]
        if seg.get("plan")
    ]
    if report["valid"]:
        st.success("Every config is valid.", icon=":material/check_circle:")
    if rows:
        st.dataframe(rows, hide_index=True)
    elif report["valid"] and not report["converted"]:
        st.info(
            "Not converted yet, so no plan: launch **Convert only** below, "
            "and check again once that job is DONE (Jobs tab).",
            icon=":material/hourglass_empty:",
        )


def start_job(
    client, *, wf_dir, setup, inner, name, mode, slurm, record
) -> dict | None:
    """Start *inner* as a SLURM job or detached on the login node, and
    record it on the cluster so the Jobs tab finds it again."""
    log_dir = f"{wf_dir}/{core.LOG_DIR}"
    log = f"{log_dir}/{name}.log"
    rc, _, err = run(client, f"mkdir -p {shlex.quote(log_dir)}", login=False)
    if rc:
        st.error(f"could not create {log_dir}: {err}")
        return None
    job = {
        **record,
        "name": name,
        "log": log,
        "started": name.rsplit("_", 1)[-1],
        "mode": mode,
        # What Relaunch and Unlock need, later and from any browser.
        "setup": setup,
        "inner": inner,
        "slurm": slurm,
    }
    if mode == core.SLURM_JOB:
        script = f"{log_dir}/{name}.sbatch"
        write_remote(client, script, core.controller_job_script(inner, **slurm))
        label = name.rsplit("_", 1)[0]
        rc, out, err = run(
            client,
            core.inner_command(
                wf_dir,
                setup,
                core.sbatch_command(script, name=f"pw-{label}", log=log),
            ),
        )
        job["slurm_job"] = core.parse_sbatch(out)
        if rc or not job["slurm_job"]:
            st.error(f"sbatch failed ({rc}):\n{out}\n{err}")
            why = core.explain_sbatch_error(
                f"{out}\n{err}", cluster_info(), slurm
            )
            if why:
                st.info(why, icon=":material/lightbulb:")
            return None
        st.success(f"submitted as SLURM job {job['slurm_job']}")
    else:
        rc, out, err = run(client, core.detached_command(inner, log))
        job["pid"] = core.parse_tag(out, "PID")
        if not job["pid"]:
            st.error(f"failed to start:\n{out}\n{err}")
            return None
        st.success(f"started on the login node (PID {job['pid']})")
    write_remote(
        client,
        f"{wf_dir}/{core.REGISTRY}",
        core.registry_line(job),
        append=True,
    )
    st.caption(f"log: {log} — follow it in the Jobs tab")
    return job


def launch(
    client,
    *,
    wf_dir,
    setup_cmd,
    files,
    name,
    mode,
    profile,
    slurm,
    convert_only=False,
):
    """Upload *files* (remote path -> config dict) and start the run.

    A single config runs ``snakemake``; a multi run (whose files include a
    ``multi.yaml``) runs ``scripts/run_multi.py``, which segments, relates
    and bundles. With *convert_only*, only the conversion runs (from the
    first segmentation's config), as run_multi's first phase would.
    """
    if convert_only:
        name = f"convert_{name.rsplit('_', 1)[-1]}"
    upload(client, files)
    multi = next((p for p in files if p.endswith("/multi.yaml")), None)
    first = next(p for p in files if not p.endswith("/multi.yaml"))
    work_dir = files[first]["work_dir"]
    if convert_only:
        state = f"{work_dir.rstrip('/')}/.snakemake_convert"
        run(client, f"mkdir -p {shlex.quote(state)}", login=False)
        command = core.convert_command(
            first, files[first], mode, workflow_dir=wf_dir, profile=profile
        )
        multi = None
    elif multi:
        command = core.multi_command(multi, mode, profile=profile)
    else:
        command = core.snakemake_command(
            first,
            mode,
            profile=profile,
            label=files[first].get("label_name", "labels"),
        )
    inner = core.inner_command(wf_dir, setup_cmd, command)

    if mode != core.DRY_RUN and not convert_only:
        # Here, on the login node: the controller job and the GPU nodes may
        # have no internet access, and the first segment job would fail.
        with st.spinner("fetching the models on the login node…"):
            rc, out, err = run(
                client,
                core.inner_command(
                    wf_dir, setup_cmd, core.fetch_command(multi or first)
                ),
                timeout=1800,
            )
        if rc:
            st.warning(
                "Could not fetch the models here: "
                f"{core.explain_remote_error(err or out)}. The run fetches them "
                "itself, which fails if its nodes have no internet access.",
                icon=":material/cloud_off:",
            )
        else:
            st.caption(
                ":material/cloud_done: "
                + "; ".join(
                    line.removeprefix("[fetch] ")
                    for line in out.splitlines()
                    if line.startswith("[fetch]")
                )
            )
    if mode == core.DRY_RUN:
        with st.spinner("dry run…"):
            rc, out, err = run(client, inner, timeout=900)
        (st.success if rc == 0 else st.error)(f"dry run exited {rc}")
        st.code(
            (out + err)[-20000:] or "(no output)", language="text", height=400
        )
        return
    start_job(
        client,
        wf_dir=wf_dir,
        setup=setup_cmd,
        inner=inner,
        name=name,
        mode=mode,
        slurm=slurm,
        record={
            "config": multi or first,
            "work_dir": work_dir,
            "kind": "convert"
            if convert_only
            else "multi"
            if multi
            else "single",
        },
    )


def submit_timing(client, *, wf_dir, setup, files, name, segment, account):
    """A short GPU job trying the method on real tiles (check --sample).

    It runs where the segment jobs run -- their partition, QOS and GPU --
    so the seconds per tile are the ones the run will see, and draws one
    segmented tile per segmentation, to judge the settings by.
    """
    upload(client, files)
    first = next(c for p, c in files.items() if not p.endswith("multi.yaml"))
    stamp = name.rsplit("_", 1)[-1]
    previews = f"{first['work_dir'].rstrip('/')}/previews/{stamp}"
    inner = core.inner_command(
        wf_dir,
        setup,
        core.check_command(run_config_path(files), sample=3, preview=previews),
    )
    mem = core.first_attempt(segment.get("mem_mb")) or 16000
    return start_job(
        client,
        wf_dir=wf_dir,
        setup=setup,
        inner=inner,
        name=f"timing_{name.rsplit('_', 1)[-1]}",
        mode=core.SLURM_JOB,
        slurm={
            "time": "01:00:00",
            "mem": f"{mem}M",
            "partition": segment.get("slurm_partition") or "",
            "qos": segment.get("qos") or "",
            "account": account,
            "gres": segment.get("gres") or "",
            "cpus": int(segment.get("cpus_per_task") or 4),
        },
        record={
            "config": run_config_path(files),
            "kind": "check",
            "previews": previews,
        },
    )


def cluster_info(refresh: bool = False) -> dict | None:
    """Partitions, QOS and accounts (sinfo, sacctmgr), once per connection.

    None when the cluster answers neither (not SLURM, or no such commands
    on the login shell's PATH): the fields are then typed in.
    """
    if refresh or "cluster_info" not in st.session_state:
        _, out, _ = run(get_client(), core.CLUSTER_INFO_COMMAND, timeout=60)
        info = core.parse_cluster_info(out)
        st.session_state["cluster_info"] = (
            info if info["partitions"] or info["assoc"] else None
        )
    return st.session_state["cluster_info"]


def controller_qos_default(preset: dict, time: str) -> str:
    """The controller's QOS: the preset's, else one that allows *time*
    when the default QOS may not ("" for the default).

    Follows *time* and the partition while the QOS shown is still the
    one chosen here, not one the user picked.
    """
    if preset.get("controller_qos"):
        return preset["controller_qos"]
    suggested = core.qos_for(
        cluster_info() or {},
        st.session_state.get("ctrl_partition") or None,
        core.slurm_minutes(time),
    )
    if (
        "ctrl_qos" in st.session_state
        and st.session_state["ctrl_qos"]
        == st.session_state.get("ctrl_qos_auto")
        and suggested != st.session_state["ctrl_qos"]
    ):
        st.session_state["ctrl_qos"] = suggested
    st.session_state["ctrl_qos_auto"] = (
        suggested
        if st.session_state.get("ctrl_qos", suggested) == suggested
        else st.session_state.get("ctrl_qos_auto")
    )
    return suggested


def _set_qos(qos: str) -> None:
    st.session_state["ctrl_qos"] = qos


def slurm_choice(
    container,
    label: str,
    key: str,
    options: list[str],
    default: str = "",
    help: str | None = None,
) -> str:
    """A dropdown of what the cluster offers; anything else can be typed.

    Blank means SLURM's default (the default partition, the account's
    default QOS).
    """
    if not options:
        return container.text_input(
            label, **_init(key, help=help, value=default)
        )
    choices = ["", *options]
    if default and default not in choices:
        choices.append(default)
    if key in st.session_state and st.session_state[key] not in choices:
        choices.append(st.session_state[key])
    return (
        container.selectbox(
            label,
            choices,
            format_func=lambda o: o or "(default)",
            accept_new_options=True,
            **_init(key, help=help, index=choices.index(default)),
        )
        or ""
    )


def resources_editor(resources: dict, info: dict | None = None) -> dict:
    """The profile's resources per rule, editable; returns the overrides.

    With the cluster's *info*, partition, QOS and GPU are picked from what
    it has and the account may use.
    """
    rules = [
        r for r in ("segment", "convert", "prepare", "merge") if r in resources
    ]
    rows = [
        {
            "rule": rule,
            "partition": resources[rule].get("slurm_partition") or "",
            "qos": resources[rule].get("qos") or "",
            "gres": resources[rule].get("gres") or "",
            "runtime (min)": core.first_attempt(resources[rule].get("runtime")),
            "memory (MB, 1st try)": core.first_attempt(
                resources[rule].get("mem_mb")
            ),
        }
        for rule in rules
    ]
    pick: dict = {}
    if info:

        def column(name, values, help):
            current = {r[name] for r in rows if r[name]}
            return st.column_config.SelectboxColumn(
                name, options=sorted(set(values) | current), help=help
            )

        pick = {
            "partition": column(
                "partition", info["partitions"], "from sinfo: see Partitions"
            ),
            "qos": column(
                "qos", core.usable_qos(info), "the QOS your account may use"
            ),
            "gres": column(
                "gres",
                {g for p in info["partitions"].values() for g in p["gres"]},
                "a GPU of any type (gpu:1) or of one type",
            ),
        }
    edited = st.data_editor(
        rows,
        hide_index=True,
        disabled=["rule"],
        key=f"resources_{hash(str(rows))}",
        column_config={
            **pick,
            "runtime (min)": st.column_config.NumberColumn(min_value=1),
            "memory (MB, 1st try)": st.column_config.NumberColumn(
                min_value=1000, help="each retry asks for this much more"
            ),
        },
    )
    overrides: dict = {}
    names = {
        "partition": "slurm_partition",
        "qos": "qos",
        "gres": "gres",
        "runtime (min)": "runtime",
        "memory (MB, 1st try)": "mem_mb",
    }
    for before, after in zip(rows, edited):
        for column, key in names.items():
            if after[column] != before[column]:
                overrides.setdefault(after["rule"], {})[key] = after[column]
    return overrides


STATE_COLORS = {"RUNNING": "green", "PENDING": "orange", "DONE": "gray"}


def job_actions(
    job: dict,
    state: str,
    alive: bool,
    setup: str,
    auto: bool = False,
    run_jobs: list[str] | None = None,
) -> None:
    """The job's state, and Stop / Cancel all / Unlock / Relaunch.

    *run_jobs*: this run's SLURM jobs still in the queue (its controller
    included), for "Cancel all its jobs".
    """
    ident = job.get("slurm_job") or job.get("pid")
    kind = "SLURM job" if job.get("slurm_job") else "PID"
    color = STATE_COLORS.get(state, "red")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"**state**  \n:{color}[**{state}**]")
        c2.markdown(f"**{kind}**  \n{ident}")
        c3.markdown(f"**started**  \n{job['started']}")
        c4.button("Refresh", icon=":material/refresh:", key="refresh_job")
        c4.caption(
            f":material/schedule: updated {time.strftime('%H:%M:%S')}"
            + (" · every 15 s" if auto and alive else "")
        )
        st.caption(f"config: `{job.get('config')}`")
        c1, c2, c3 = st.columns(3)
        if alive:
            with c1.popover("Stop", icon=":material/stop_circle:"):
                st.markdown(
                    "Stops the run as Ctrl-C would: Snakemake cancels the "
                    "jobs it submitted, then exits. Finished steps stay "
                    "done; launching again resumes."
                )
                if st.button("Stop it", type="primary", key="stop"):
                    rc, out, err = run(
                        client,
                        core.inner_command(
                            wf_dir, setup_cmd, core.stop_command(job)
                        )
                        if job.get("slurm_job")
                        else core.stop_command(job),
                        login=bool(job.get("slurm_job")),
                        timeout=60,
                    )
                    (st.success if rc == 0 else st.error)(
                        "asked to stop" if rc == 0 else f"{out}{err}"
                    )
                if st.button("Force stop", key="kill"):
                    rc, out, err = run(
                        client,
                        core.inner_command(
                            wf_dir,
                            setup_cmd,
                            core.stop_command(job, force=True),
                        )
                        if job.get("slurm_job")
                        else core.stop_command(job, force=True),
                        login=bool(job.get("slurm_job")),
                        timeout=60,
                    )
                    (st.success if rc == 0 else st.error)(
                        "killed" if rc == 0 else f"{out}{err}"
                    )
                    st.caption(
                        "Jobs it had submitted may keep running: "
                        "**Cancel all its jobs** ends them."
                    )
        if run_jobs:
            with st.popover(
                f"Cancel all its jobs ({len(run_jobs)})",
                icon=":material/cancel:",
            ):
                st.markdown(
                    "Cancels every SLURM job of this run still in the queue "
                    "-- the steps it submitted and its controller -- at "
                    "once: running steps are lost, finished ones stay. "
                    "Then **Unlock** before launching again.\n\n"
                    f"Jobs: {', '.join(run_jobs)}"
                )
                if st.button("Cancel them", type="primary", key="cancel_all"):
                    rc, out, err = run(
                        client,
                        core.inner_command(
                            wf_dir, setup_cmd, core.cancel_command(run_jobs)
                        ),
                        timeout=60,
                    )
                    (st.success if rc == 0 else st.error)(
                        f"cancelled {len(run_jobs)} job(s)"
                        if rc == 0
                        else core.explain_remote_error(err or out)
                    )
        unlock = core.unlock_command(job, wf_dir)
        if not alive and unlock:
            if c2.button(
                "Unlock",
                icon=":material/lock_open:",
                help="after a run was killed: Snakemake then refuses to "
                "start again (directory cannot be locked) until unlocked",
            ):
                rc, out, err = run(
                    client,
                    core.inner_command(wf_dir, setup, unlock),
                    timeout=300,
                )
                (st.success if rc == 0 else st.error)(
                    "unlocked"
                    if rc == 0
                    else core.explain_remote_error(err or out)
                )
        if not alive and job.get("inner"):
            if c3.button(
                "Relaunch",
                icon=":material/replay:",
                help="the same command again: it resumes where this one "
                "stopped",
            ):
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                start_job(
                    client,
                    wf_dir=wf_dir,
                    setup=job.get("setup", setup_cmd),
                    inner=job["inner"],
                    name=f"{job['name'].rsplit('_', 1)[0]}_{stamp}",
                    mode=job["mode"],
                    slurm=job.get("slurm") or {},
                    record={
                        k: job[k]
                        for k in ("config", "work_dir", "kind")
                        if k in job
                    },
                )


def queue_view(rows: list[dict]) -> None:
    """Your patchworks jobs in the SLURM queue, and why each waits."""
    if not rows:
        return
    running = sum(r["state"] == "RUNNING" for r in rows)
    with section(
        ":material/queue: In the queue",
        f"{running} running, {len(rows) - running} waiting (all your "
        "patchworks jobs)",
    ):
        st.dataframe(rows, hide_index=True)
        for row in rows:
            if row["why it waits"].startswith("NEVER STARTS"):
                st.error(
                    f"job {row['job']} ({row['name']}): {row['why it waits']}",
                    icon=":material/block:",
                )


# The logs' last lines, fetched together once per refresh (fetch_job).
_TAILS: dict[str, str] = {}
TAIL_LINES = 300


def _fetch_tails(paths) -> None:
    """The last lines of each log in *paths*, in one command."""
    paths = [p for p in dict.fromkeys(paths) if p and p not in _TAILS]
    if not paths:
        return
    _, out, _ = run(
        client,
        core.tails_command(TAIL_LINES),
        login=False,
        timeout=60,
        stdin_text="".join(f"{p}\n" for p in paths),
    )
    _TAILS.update(core.parse_tails(out, paths))


def _tail(path: str, lines: int = 60) -> str:
    if path not in _TAILS:
        _fetch_tails([path])
    return "\n".join(_TAILS[path].splitlines()[-lines:])


def fetch_job(job: dict, setup: str) -> dict:
    """Everything the Jobs tab shows of *job*, fetched first and in
    parallel, so the page is redrawn at once rather than piece by piece."""
    _TAILS.clear()
    calls = {
        # squeue may live behind `module load`, so SLURM jobs are checked
        # through the setup line; a PID check and a tail need neither it
        # nor a login shell.
        "state": lambda: run(
            client,
            core.inner_command(wf_dir, setup_cmd, core.status_command(job))
            if job.get("slurm_job")
            else core.status_command(job),
            login=bool(job.get("slurm_job")),
            timeout=30,
        ),
        "log": lambda: _fetch_tails([job["log"]]),
    }
    if job.get("config") and job.get("kind") != "check":
        calls |= {
            "queue": lambda: run(client, core.QUEUE_COMMAND, timeout=30),
            "steps": lambda: run(
                client, core.steps_command(job["log"]), login=False, timeout=30
            ),
            "status": lambda: run(
                client,
                core.inner_command(
                    wf_dir, setup, core.run_status_command(job["config"])
                ),
                timeout=60,
            ),
        }
    # A few at once: the server caps a connection's open channels.
    with ThreadPoolExecutor(min(4, len(calls))) as pool:
        got = {
            k: f.result()
            for k, f in {k: pool.submit(c) for k, c in calls.items()}.items()
        }
    data = {
        "state": got["state"][1].strip() or "unknown",
        "queue": core.parse_queue(got["queue"][1]) if "queue" in got else [],
        "steps": core.parse_steps(got["steps"][1]) if "steps" in got else [],
        "status": got.get("status"),
    }
    # Then the step logs the page shows: failed steps, those not done.
    status = core.parse_json_output((data["status"] or (0, "", ""))[1]) or {}
    _fetch_tails(
        [
            f["log"]
            for f in list(status.get("failures") or [])
            + [
                f
                for seg in status.get("segmentations") or []
                for f in seg.get("failures") or []
            ]
        ]
        # The steps whose logs the page opens (the last few not done), and
        # those whose progress lines fill a bar -- not every running
        # step's: a multi run has hundreds.
        + [
            path
            for step in [s for s in data["steps"] if s["state"] != "done"][-5:]
            + [
                s
                for s in data["steps"]
                if s["state"] != "done"
                and s["rule"] in ("convert", "occupancy", "merge")
            ]
            for path in (step.get("log"), step.get("slurm_log"))
        ]
        # Why the earlier attempts failed: what SLURM wrote at their end.
        + [
            e["slurm_log"]
            for step in data["steps"][-40:]
            for e in step["earlier"]
        ]
    )
    return data


def step_progress(step: dict | None) -> tuple[float, str]:
    """How far a running step got, from the progress lines in its log."""
    if not step or not step.get("log"):
        return 0.0, "waiting to start"
    progress = core.last_progress(_tail(step["log"], TAIL_LINES))
    earlier = earlier_attempts(step)
    attempt = (
        f" (attempt {step['attempt']}; {'; '.join(earlier)})" if earlier else ""
    )
    if step["state"] == "failed":
        return 0.0, f"failed{attempt}"
    if not progress:
        return 0.0, f"started{attempt}, no progress line yet"
    return progress["fraction"], (
        f"{progress['label']}: {progress['fraction']:.0%}, "
        f"~{progress['left']} left{attempt}"
    )


def segmentation_progress(seg: dict, steps: list[dict]) -> tuple[float, str]:
    """One segmentation's bar: tiles segmented (any method: segment_tile.py
    checkpoints each), then the merge's own progress lines."""
    total, done = seg["batches"], seg["batches_done"]
    if seg["stage"] == "done":
        return 1.0, "merged"
    if seg["stage"] == "merging":
        merge = next(
            (
                s
                for s in reversed(steps)
                if s["rule"] == "merge"
                and (s.get("log") or "").endswith(
                    f"/{seg['label_name']}/logs/merge.log"
                )
            ),
            None,
        )
        frac, text = step_progress(merge)
        return frac, f"merging · {text}"
    if not total and seg["stage"] == "converted":
        prepare = next(
            (
                s
                for s in reversed(steps)
                if s["rule"] == "prepare"
                and (s.get("log") or "").endswith(
                    f"/{seg['label_name']}/logs/prepare.log"
                )
            ),
            None,
        )
        if prepare and prepare["state"] != "done":
            return 0.0, (
                "preparing the tiles"
                if prepare["state"] != "failed"
                else "preparing the tiles failed"
            )
        return 0.0, "converted, waiting to prepare the tiles"
    if total:
        tiles = seg.get("tiles")
        if tiles:
            return seg.get("tiles_done", 0) / tiles, (
                f"{seg.get('tiles_done', 0)}/{tiles} tiles segmented "
                f"({done}/{total} GPU jobs done)"
            )
        return done / total, f"{done}/{total} GPU jobs done"
    return 0.0, seg["stage"]


def _gb(mem_mb) -> str:
    return f"{mem_mb / 1000:g} GB" if mem_mb else "?"


def earlier_attempts(step: dict) -> list[str]:
    """Each earlier attempt of *step*: how it ended, with its memory."""
    return [
        f"{e['attempt']}: "
        + (
            (
                core.why_killed(_tail(e["slurm_log"]))
                if e.get("slurm_log")
                else None
            )
            or "failed"
        )
        + f" at {_gb(e['mem_mb'])}"
        for e in step.get("earlier", [])
    ]


def steps_view(steps: list[dict], queue: list[dict]) -> None:
    """The workflow's steps (from the controller's log): where each runs,
    and the log of each one not done."""
    if not steps:
        return
    states = {r["job"]: r["state"].lower() for r in queue}
    rows = [
        {
            "step": s["rule"]
            + (f" ({s['wildcards']})" if s["wildcards"] else ""),
            "state": states.get(s["slurm_job"] or "", s["state"]),
            "attempt": s["attempt"],
            "memory": _gb(s.get("mem_mb")) if s.get("mem_mb") else "",
            "SLURM job": s["slurm_job"] or "",
            "earlier attempts": "; ".join(earlier_attempts(s)),
        }
        for s in steps[-40:]
    ]
    short = sorted(
        {
            s["rule"]
            for s in steps[-40:]
            if any("out of memory" in a for a in earlier_attempts(s))
        }
    )
    with section(
        ":material/account_tree: Steps",
        f"{sum(s['state'] == 'done' for s in steps)} of {len(steps)} done; "
        "the latest attempt of each",
    ):
        st.dataframe(rows, hide_index=True)
        if short:
            st.warning(
                f"{', '.join(short)} ran out of memory and was retried with "
                "more: each retry costs the time the failed attempt ran. For "
                "the next run, start higher: that step's memory in Plan & "
                "launch → Cluster resources, or its mem_mb in the profile.",
                icon=":material/memory:",
            )
        open_steps = [s for s in steps if s["state"] != "done"][-5:]
        for s in open_steps:
            state = states.get(s["slurm_job"] or "", s["state"])
            name = s["rule"] + (
                f" ({s['wildcards']})" if s["wildcards"] else ""
            )
            with st.expander(
                f"{name} · {state} · its log",
                icon=":material/terminal:",
                expanded=state in ("running", "failed"),
            ):
                for label, path in (
                    ("step log", s.get("log")),
                    ("SLURM log", s.get("slurm_log")),
                ):
                    if not path:
                        continue
                    st.caption(f"{label}: `{path}`")
                    st.code(
                        _tail(path) or "(empty so far)",
                        language="text",
                        height=250,
                    )


def job_progress(job: dict, steps: list[dict], result) -> None:
    """How far the run got (scripts/status.py), and what it made."""
    rc, out, err = result
    status = core.parse_json_output(out)
    if not status or "segmentations" not in status:
        st.caption(
            f"progress unavailable: {core.explain_remote_error(err or out)}"
        )
        return
    failed = list(status.get("failures") or []) + [
        {**f, "step": f"{seg['label_name']}: {f['step']}"}
        for seg in status["segmentations"]
        for f in seg.get("failures") or []
    ]
    if failed:
        with section(":material/error: Failed", "and what to do about it"):
            for f in failed:
                st.error(f"**{f['step']}**: `{f['error']}`")
                if f.get("hint"):
                    st.markdown(f":material/lightbulb: {f['hint']}")
                with st.expander(
                    f"log: {f['log']}", icon=":material/terminal:"
                ):
                    st.code(_tail(f["log"], 80) or "(empty)", language="text")
    convert_only = job.get("kind") == "convert"
    with section(":material/data_usage: Progress", f"`{status['work_dir']}`"):
        # The shared steps first, while they matter: convert, occupancy.
        latest = {s["rule"]: s for s in steps}
        for rule, done, what in (
            ("convert", status.get("converted"), "converted"),
            ("occupancy", status.get("occupancy"), "occupancy map built"),
        ):
            if done is None or (done and not convert_only):
                continue
            frac, text = (
                (1.0, what) if done else step_progress(latest.get(rule))
            )
            st.progress(frac, text=f"**{rule}** · {text}")
        if convert_only:
            return
        for seg in status["segmentations"]:
            frac, what = segmentation_progress(seg, steps)
            st.progress(frac, text=f"**{seg['label_name']}** · {what}")
        # The relations (a multi run), once every segmentation is merged:
        # relate.py says what it is at in the controller's log.
        pending = [r for r in status.get("relations") or [] if not r["files"]]
        if pending and all(
            s["stage"] == "done" for s in status["segmentations"]
        ):
            said = [
                line
                for line in _tail(job["log"], TAIL_LINES).splitlines()
                if line.startswith("[relate]")
            ]
            for rel in pending:
                st.progress(
                    0.0,
                    text=f"**{rel['a']} → {rel['b']}** · relating"
                    + (
                        f": {said[-1].removeprefix('[relate] ')}"
                        if said
                        else ""
                    ),
                )
    done = [s for s in status["segmentations"] if s["stage"] == "done"]
    if not done and not status.get("relations"):
        return
    with section(":material/inventory_2: Results"):
        rows = []
        for seg in done:
            row = {"segmentation": seg["label_name"], "objects": seg["objects"]}
            for ax, rates in (seg.get("seams") or {}).items():
                row[f"seam {ax} (%)"] = round(
                    100 * (rates["seam_rate"] or 0), 1
                )
                inner = rates["interior_rate"]
                row[f"inside {ax} (%)"] = (
                    None if inner is None else round(100 * inner, 1)
                )
            rows.append(row)
        if rows:
            st.dataframe(rows, hide_index=True)
            st.caption(
                "seam: objects ending on a tile seam, against inside: on an "
                "arbitrary plane inside the tiles. Far more at the seams "
                "means the stitching cut objects."
            )
        for rel in status.get("relations", []):
            name = f"{rel['a']} → {rel['b']}"
            if not rel["files"]:
                st.caption(f"{name}: not yet")
            for path in rel["files"]:
                download_button(path, f"{name}: {path.rsplit('/', 1)[-1]}")
        store = f"{status['work_dir'].rstrip('/')}/image.zarr"
        bundle = status.get("bundle")
        st.markdown("**Look at it**")
        st.code(
            f"# on the cluster (needs a display)\n"
            f"cd {wf_dir} && pixi run -e viewer napari {store}\n"
            f"cd {wf_dir} && pixi run -e viewer review {store}\n"
            + (
                f"# or on your computer: copy the bundle, then\n"
                f"scp {st.session_state['ssh_target'].rsplit(':', 1)[0]}:"
                f"{bundle} .\n"
                f"cd patchworks/workflow/viewer && pixi run napari "
                f"{bundle.rsplit('/', 1)[-1]}"
                if bundle
                else ""
            ),
            language="bash",
        )


MAX_DOWNLOAD = 200 * 1024**2


def _fetch(path: str, key: str) -> None:
    sftp = get_sftp()
    with sftp.open(path) as fh:
        st.session_state[key] = fh.read()


def download_button(path: str, label: str) -> None:
    """Fetch a result file from the cluster on demand, then download it."""
    key = f"dl_{path}"
    data = st.session_state.get(key)
    if data is None:
        try:
            size = get_sftp().stat(path).st_size or 0
        except OSError:
            st.caption(f"{label}: missing")
            return
        if size > MAX_DOWNLOAD:
            st.caption(f"{label}: {size / 1024**2:.0f} MB, copy it with scp")
            return
        st.button(
            f"{label} ({size / 1024**2:.1f} MB)",
            icon=":material/cloud_download:",
            key=f"get_{path}",
            on_click=_fetch,
            args=(path, key),
        )
        return
    st.download_button(
        f"Save {label}",
        data,
        file_name=path.rsplit("/", 1)[-1],
        icon=":material/download:",
        key=f"save_{path}",
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### :material/dns: Cluster")
    if get_client() is None:
        if st.session_state.get("connection_lost"):
            st.warning(
                "The connection to the cluster was lost: log in again. "
                "Jobs keep running.",
                icon=":material/link_off:",
            )
        # Last time's choices (this computer's settings), else the preset.
        saved = core.load_settings()
        names = ["Custom"] + sorted(CLUSTERS)
        name = st.selectbox(
            "preset",
            names,
            index=names.index(saved["preset"])
            if saved.get("preset") in names
            else 0,
            help=HELP["preset"],
        )
        preset = CLUSTERS.get(name, {})
        same = saved.get("preset") == name
        fps = preset.get("host_key_fingerprints", [])
        host = st.text_input(
            "host",
            value=(same and saved.get("host")) or preset.get("host", ""),
            help=HELP["host"],
        )
        port = st.number_input(
            "port",
            help=HELP["port"],
            value=int((same and saved.get("port")) or preset.get("port", 22)),
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
                "yourself)",
                help=HELP["allow_unverified"],
            )
        auth = (
            st.segmented_control(
                "log in with",
                ["Password", "SSH key"],
                default="SSH key" if saved.get("auth") == "key" else "Password",
                help=HELP["auth"],
            )
            or "Password"
        )
        use_keys = auth == "SSH key"
        username_default = (same and saved.get("username")) or ""
        # With an SSH key and last time's user: connect by itself, once a
        # session (and once after a dropped connection) -- not after
        # Disconnect. Never to an unverified host: its key must be pinned
        # or in ~/.ssh/known_hosts (as after any ssh to it), else this
        # fails and the form below is there.
        if (
            use_keys
            and host
            and username_default
            and not st.session_state.get("auto_tried")
            and not st.session_state.get("disconnected")
        ):
            st.session_state["auto_tried"] = True
            try:
                with st.spinner(
                    f"connecting as {username_default} with your SSH key…"
                ):
                    connect(
                        host,
                        int(port),
                        username_default,
                        None,
                        fps,
                        False,
                        use_keys=True,
                    )
                st.session_state["preset"] = preset
                st.rerun()
            except Exception as exc:  # shown, then the form below
                st.info(
                    f"Could not log in with your SSH key ({exc}): log in "
                    "below.",
                    icon=":material/key_off:",
                )
        # A form that clears on submit: the password does not linger in
        # the session's widget state after the connection is made.
        with st.form("login", clear_on_submit=True):
            username = st.text_input(
                "username", value=username_default, help=HELP["username"]
            )
            password = st.text_input(
                "key passphrase (if any)" if use_keys else "password",
                type="password",
                help=HELP["key_passphrase" if use_keys else "password"],
            )
            submitted = st.form_submit_button("Connect", type="primary")
        if submitted and host and username:
            try:
                connect(
                    host,
                    int(port),
                    username,
                    password,
                    fps,
                    allow_unverified,
                    use_keys=use_keys,
                )
                st.session_state["preset"] = preset
                st.session_state.pop("disconnected", None)
                core.save_settings(
                    {
                        "preset": name,
                        "host": host,
                        "port": int(port),
                        "username": username,
                        "auth": "key" if use_keys else "password",
                    }
                )
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
            # Not straight back in with the SSH key: you asked to leave.
            st.session_state["disconnected"] = True
            for key in ("ssh_client", "sftp", "preset", "pixi_envs", "plans"):
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
                if core.load_settings().get("wf_dir") != wf_dir:
                    core.save_settings({"wf_dir": wf_dir})
                workflow_version_check(wf_dir)
            else:
                st.warning(
                    "No Snakefile in this folder: pick the clone's "
                    "`workflow/` folder."
                )
    setup_cmd = st.text_area(
        "environment setup (run before snakemake)",
        value=preset.get("setup_cmd", 'eval "$(pixi shell-hook)"'),
        help="puts snakemake, patchworks and sbatch on PATH. Which pixi "
        "environment is chosen in *Plan & launch*, from the config.",
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
    with st.expander(
        "Start from a template or a past run", icon=":material/history:"
    ):
        start_from_form(
            wf_dir,
            core.parse_registry(
                read_remote(client, f"{wf_dir}/{core.REGISTRY}") or ""
            ),
        )
    with section(":material/alt_route: Run"):
        kind = (
            st.segmented_control(
                "how many label images",
                core.RUN_KINDS,
                **_init("run_kind", default=core.RUN_KINDS[0]),
            )
            or core.RUN_KINDS[0]
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
        problems += sorted(
            {p for c in seg_cfgs for p in core.config_problems(c)}
        )
        if missing:
            problems.insert(0, f"still to fill in: {', '.join(missing)}")
    except Exception as exc:
        problems = [f"the config is not valid yet: {exc}"]

    with section(":material/fact_check: Check"):
        draft = files  # what the form holds, even with problems left
        if problems:
            files = {}
            for problem in problems:
                st.warning(problem, icon=":material/edit_note:")
        else:
            st.success(
                "The config is complete: go to **2 · Plan & launch**.",
                icon=":material/check_circle:",
            )
            for path, content in files.items():
                title = path.rsplit("/", 1)[-1]
                with st.expander(
                    f"Config to upload: {title}", icon=":material/code:"
                ):
                    st.caption(f"`{path}`")
                    st.code(yaml.safe_dump(content, sort_keys=False), "yaml")
        if draft:
            save_template_form(draft)
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
            ":material/deployed_code: Environment",
            "The pixi environment of the workflow the run starts from; its "
            "jobs inherit it. PlantSeg, CAREamics and GPU (cupy) steps "
            "each need one that has them.",
        ):
            envs = st.session_state.setdefault("pixi_envs", {})
            if wf_dir not in envs:
                envs[wf_dir] = core.pixi_environments(
                    read_remote(client, f"{wf_dir}/pixi.toml")
                )
            available = envs[wf_dir]
            suggested, reasons = core.suggest_environment(seg_cfgs, available)
            env = st.selectbox(
                "pixi environment",
                available,
                index=available.index(suggested) if suggested else 0,
                # A new suggestion (the config changed) resets the choice.
                key=f"pixi_env_{suggested}",
                help=HELP["pixi_env"],
            )
            for reason in reasons:
                st.caption(f":material/info: {reason}")
            if suggested is None:
                st.warning(
                    "No environment of the workflow's pixi.toml has every "
                    "package this run needs; split it, or add one."
                )
            elif env != suggested:
                st.warning(
                    f"`{suggested}` is the one with what this run needs."
                )
            run_setup = core.setup_with_environment(setup_cmd, env)
            if run_setup is None:
                run_setup = setup_cmd
                st.warning(
                    "The environment setup line (sidebar) does not use "
                    "`pixi shell-hook`, so the environment is not switched: "
                    f"make sure it activates `{env}`."
                )
            st.caption("runs first, in the workflow folder:")
            st.code(run_setup or "(nothing)", language="bash")

        with section(
            ":material/dns: Cluster resources",
            "What each step's jobs ask SLURM for, from the workflow "
            "profile. Change a cell to change it for this run only.",
        ):
            profile = st.text_input(
                "workflow profile",
                preset.get("profile", "profile/slurm"),
                help="A folder of the workflow holding the SLURM settings "
                "(`config.yaml`): partitions, GPUs, memory and time per step. "
                "A preset may name its own.\n\nExample: `profile/slurm`",
            )
            profile_text = read_remote(
                client, f"{wf_dir}/{profile.rstrip('/')}/config.yaml"
            )
            resources: dict = {}
            overrides: dict = {}
            if profile_text is None:
                st.warning(
                    f"No `{profile}/config.yaml` in the workflow folder."
                )
            else:
                resources = core.profile_resources(profile_text)
                overrides = resources_editor(resources, cluster_info())
                for rule, values in overrides.items():
                    resources[rule] = {**resources[rule], **values}
                # Cellpose's "auto" tiles, sized for the GPU the segment
                # jobs get rather than an 8 GB guess.
                if any(core.needs_gpu_memory(c) for c in seg_cfgs):
                    gb, gpu = core.gpu_memory_gb(
                        resources.get("segment", {}), cluster_info()
                    )
                    if gb:
                        files = core.with_gpu_memory(files, gb)
                        seg_cfgs = [
                            c
                            for p, c in files.items()
                            if not p.endswith("multi.yaml")
                        ]
                        st.caption(
                            f":material/memory: tiles sized for the segment "
                            f"GPU, {gpu}: gpu_memory_gb {gb} (set it in "
                            "Tiling to override)"
                        )
                    else:
                        st.warning(
                            f'tile_shape "auto" will be sized for an 8 GB GPU '
                            f"({gpu}): set gpu_memory_gb in Tiling to the "
                            "segment GPU's memory for larger tiles.",
                            icon=":material/memory:",
                        )
                if "mem_mb" in overrides.get("segment", {}):
                    st.caption(
                        "prepare gets the segment memory too: it sizes "
                        '`tile_shape: "auto"` from its own allocation.'
                    )
                # Checked as you change it, against what the cluster has
                # and the account may use.
                info = cluster_info()
                if info is None:
                    st.caption(
                        "sinfo/sacctmgr gave nothing here, so partitions and "
                        "QOS are typed in and not checked."
                    )
                else:
                    problems = core.resource_problems(resources, info)
                    for problem in problems:
                        st.error(problem, icon=":material/block:")
                    if not problems:
                        st.success(
                            "Every partition, QOS, GPU and runtime fits the "
                            "cluster and your account.",
                            icon=":material/check_circle:",
                        )
            if cluster_info() is not None:
                with st.expander("Partitions", icon=":material/view_list:"):
                    st.dataframe(
                        core.partition_rows(cluster_info()), hide_index=True
                    )
                    st.button(
                        "Refresh",
                        icon=":material/refresh:",
                        key="refresh_cluster",
                        on_click=cluster_info,
                        args=(True,),
                        help="idle CPUs change all the time",
                    )

        with section(
            ":material/fact_check: Check and plan",
            "Checks the configs on the cluster, in the environment above, "
            "as the run will. Once the image is converted it also plans "
            "each segmentation: tiles, how many hold signal, GPU jobs, "
            "memory per tile, size of the result.",
        ):
            c1, c2 = st.columns(2)
            if c1.button("Check and plan", icon=":material/play_arrow:"):
                with st.spinner("checking on the cluster…"):
                    st.session_state["report"] = check_on_cluster(
                        client, wf_dir, run_setup, files
                    )
            if c2.button(
                "Try it on a GPU",
                icon=":material/science:",
                help="a short SLURM job on the segment partition runs the "
                "method on 3 real tiles: a picture of one with its objects "
                "outlined, to judge the settings by, and the hours a job and "
                "the run take. It shows in the Jobs tab.",
                disabled="segment" not in resources,
            ):
                submit_timing(
                    client,
                    wf_dir=wf_dir,
                    setup=run_setup,
                    files=files,
                    name=st.session_state["run_name"],
                    segment=resources.get("segment", {}),
                    account=st.session_state.get(
                        "ctrl_account", preset.get("controller_account", "")
                    ),
                )
            if st.session_state.get("report"):
                show_check(st.session_state["report"])

        with section(
            ":material/rocket_launch: Launch",
            "Uploads the config"
            + (f"s of all {len(seg_cfgs)} segmentations" if many else "")
            + " and starts the run. Launching the same settings again "
            "resumes it.",
        ):
            what = (
                st.segmented_control(
                    "what to run",
                    ["Everything", "Convert only"],
                    default="Everything",
                    help=HELP["what_to_run"],
                )
                or "Everything"
            )
            convert_only = what == "Convert only"
            mode = st.radio("run mode", core.RUN_MODES, help=HELP["run_mode"])
            slurm: dict = {}
            if mode == core.SLURM_JOB:
                # What the controller is, in one line, filled in below once
                # its settings (defaults, or changed in the panel) are known.
                summary = st.container()
                panel = st.expander(
                    "Change the controller job",
                    icon=":material/settings:",
                    expanded=bool(st.session_state.get("ctrl_problems")),
                )
                panel.caption(
                    "Only for the controller. Each step's own CPUs, memory "
                    "and GPU are under Cluster resources above."
                )
                c1, c2, c3, c4, c5 = panel.columns(5)
                ctrl_time = c1.text_input(
                    "time",
                    preset.get("controller_time", "3-00:00:00"),
                    help=HELP["ctrl_time"],
                )
                slurm = {
                    "time": ctrl_time,
                    "mem": c2.text_input(
                        "mem",
                        preset.get("controller_mem", "4G"),
                        help=HELP["ctrl_mem"],
                    ),
                    "partition": slurm_choice(
                        c3,
                        "partition",
                        "ctrl_partition",
                        sorted((cluster_info() or {}).get("partitions", {})),
                        preset.get("controller_partition", ""),
                    ),
                    "qos": slurm_choice(
                        c4,
                        "qos",
                        "ctrl_qos",
                        core.usable_qos(
                            cluster_info() or {},
                            st.session_state.get("ctrl_partition") or None,
                        ),
                        controller_qos_default(preset, ctrl_time),
                    ),
                    "account": slurm_choice(
                        c5,
                        "account",
                        "ctrl_account",
                        (cluster_info() or {}).get("accounts", []),
                        preset.get("controller_account", ""),
                    ),
                }
                summary.markdown(
                    ":material/smart_toy: A small **controller** job (1 CPU, "
                    f"{slurm['mem']}) submits each step's jobs and waits for "
                    "them; it ends as soon as the run ends. It may run up to "
                    f"**{slurm['time']}**"
                    + f" · QOS **{slurm['qos'] or 'your default'}**"
                    + f" · partition **{slurm['partition'] or 'the default'}**"
                    + "."
                )
                summary.caption(
                    "The defaults suit a whole run. If the time runs out "
                    "first, launch again: the run resumes where it stopped."
                )
                problems = (
                    core.resource_problems(
                        {
                            "controller": {
                                "slurm_partition": slurm["partition"],
                                "qos": slurm["qos"],
                                "runtime": core.slurm_minutes(slurm["time"]),
                            }
                        },
                        cluster_info(),
                    )
                    if cluster_info() is not None
                    else []
                )
                st.session_state["ctrl_problems"] = problems
                if cluster_info() is not None:
                    for problem in problems:
                        st.error(problem, icon=":material/block:")
                    better = core.better_qos(
                        cluster_info(),
                        slurm["partition"] or None,
                        slurm["qos"],
                        core.slurm_minutes(slurm["time"]),
                    )
                    if better:
                        st.button(
                            f"Use QOS {better}",
                            icon=":material/build:",
                            on_click=_set_qos,
                            args=(better,),
                        )
            elif mode == core.LOGIN_NODE:
                st.warning(
                    "The controller runs for the whole workflow on the login "
                    "node, where a reboot or a process reaper ends it. Prefer "
                    "the SLURM-job mode unless your cluster forbids it."
                )
            if overrides and mode != core.DRY_RUN:
                st.caption(
                    ":material/tune: with your resource changes: "
                    + "; ".join(
                        f"{r}: "
                        + ", ".join(f"{k}={v}" for k, v in vals.items())
                        for r, vals in overrides.items()
                    )
                )
            with st.expander("Files to upload", icon=":material/upload:"):
                st.code("\n".join(files), language=None)
            if st.button(
                "Dry run" if mode == core.DRY_RUN else "Launch",
                type="primary",
                icon=":material/rocket_launch:",
            ):
                run_profile = profile
                if overrides and mode != core.DRY_RUN:
                    # This run's own copy of the profile, with the changes.
                    run_profile = (
                        f"{wf_dir}/config/launcher_"
                        f"{st.session_state['run_name']}_profile"
                    )
                    run(
                        client,
                        f"mkdir -p {shlex.quote(run_profile)}",
                        login=False,
                    )
                    write_remote(
                        client,
                        f"{run_profile}/config.yaml",
                        core.profile_with_overrides(profile_text, overrides),
                    )
                launch(
                    client,
                    wf_dir=wf_dir,
                    setup_cmd=run_setup,
                    files=files,
                    name=st.session_state["run_name"],
                    mode=mode,
                    profile=run_profile,
                    slurm=slurm,
                    convert_only=convert_only,
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
                format_func=lambda i: (
                    f"{jobs[i]['name']} · {jobs[i].get('kind', '')} · "
                    f"{jobs[i]['mode']}"
                ),
            )
            auto = c2.toggle("auto-refresh every 15 s", value=True)

        # Refreshed while the job runs; once it has ended, nothing changes.
        alive_key = f"alive_{jobs[pick]['name']}"

        @st.fragment(
            run_every=15
            if auto and st.session_state.get(alive_key, True)
            else None
        )
        def monitor(job=jobs[pick]):
            setup = job.get("setup", setup_cmd)
            try:
                data = fetch_job(job, setup)
            except ConnectionLost:
                # The whole page again: the sidebar asks to log in, and
                # says the jobs keep running.
                _forget_connection()
                st.rerun()
            state = data["state"]
            alive = state in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING")
            was_alive = st.session_state.get(alive_key, True)
            st.session_state[alive_key] = alive
            if was_alive != alive:  # started or ended: (un)set auto-refresh
                st.rerun()
            job_actions(
                job,
                state,
                alive,
                setup,
                auto,
                run_jobs=core.run_jobs_in_queue(
                    job, data["steps"], data["queue"]
                ),
            )
            if job.get("kind") == "check":
                if not alive:
                    report = core.parse_json_output(
                        _tail(job["log"], TAIL_LINES)
                    )
                    if report:
                        with section(":material/science: Tried on real tiles"):
                            show_check(report)
                            show_previews(report)
            elif job.get("config"):
                if alive:
                    queue_view(data["queue"])
                job_progress(job, data["steps"], data["status"])
                steps_view(data["steps"], data["queue"])
            with st.expander(
                "Snakemake log (the controller)",
                icon=":material/terminal:",
                # Open while running if nothing above shows the run's steps.
                expanded=alive
                and (job.get("kind") == "check" or not data["steps"]),
            ):
                st.caption(f"`{job['log']}`")
                tail = _tail(job["log"], 300)
                st.code(
                    tail or "(log is empty so far)",
                    language="text",
                    height=500,
                )

        monitor()
