"""Best-effort email notification.

Used for the events SLURM cannot report on its own: the end of a whole
multi-config run, and failures in a local (non-SLURM) run. Per-job
start/finish mail on a cluster is left to SLURM's own ``--mail-type``, which
is delivered by the controller and does not depend on a compute node being
able to reach an MTA.

Every function here is best-effort by design: a notification that cannot be
delivered must never fail a pipeline that otherwise succeeded, and must never
turn a real error into a confusing one about email.
"""

from __future__ import annotations

import logging
import re
import shutil
import socket
import subprocess
from email.message import EmailMessage
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

# Lines of a failing step's log to quote in an error mail. Enough to carry a
# traceback, short enough that the mail stays readable on a phone.
LOG_TAIL_LINES = 40


def log_tail(path: Union[str, Path], lines: int = LOG_TAIL_LINES) -> str:
    """Return the last *lines* of a log file, or a note if unreadable."""
    try:
        content = Path(path).read_text(errors="replace").splitlines()
    except OSError as exc:
        return f"(could not read {path}: {exc})"
    if not content:
        return f"({path} is empty)"
    return "\n".join(content[-lines:])


def send(
    to: Union[str, None],
    subject: str,
    body: str,
    *,
    sender: Union[str, None] = None,
) -> bool:
    """Send *body* to *to*, returning whether it went out.

    Tries a local ``sendmail`` first (what a cluster node normally has), then
    an SMTP server on localhost. Never raises.

    Parameters
    ----------
    to : str or None
        Recipient. ``None`` or empty disables the notification entirely, which
        is the default state -- no address configured, no mail, no error.
    subject : str
        Subject line.
    body : str
        Plain-text body.
    sender : str, optional
        From address. Defaults to ``patchworks@<hostname>``.

    Returns
    -------
    bool
        True when the message was handed to a transport.
    """
    if not to:
        return False

    msg = EmailMessage()
    msg["To"] = to
    msg["From"] = sender or f"patchworks@{socket.getfqdn()}"
    msg["Subject"] = subject
    msg.set_content(body)

    sendmail = shutil.which("sendmail") or shutil.which(
        "sendmail", path="/usr/sbin:/usr/lib"
    )
    if sendmail:
        try:
            subprocess.run(
                [sendmail, "-t", "-oi"],
                input=msg.as_bytes(),
                check=True,
                capture_output=True,
                timeout=30,
            )
            return True
        except (subprocess.SubprocessError, OSError) as exc:
            logger.warning(
                "sendmail failed (%s); trying SMTP on localhost", exc
            )

    try:
        import smtplib

        with smtplib.SMTP("localhost", timeout=30) as smtp:
            smtp.send_message(msg)
        return True
    except (OSError, Exception) as exc:  # noqa: BLE001 - never fail the run
        logger.warning(
            "could not send notification to %s (%s). The run itself is "
            "unaffected; set notify_email to '' to silence this.",
            to,
            exc,
        )
        return False


def slurm_mail_extra(
    email: Union[str, None], events: Union[list, tuple, None] = None
) -> str:
    """Build the ``slurm_extra`` fragment for per-job mail.

    SLURM's own ``--mail-type`` is used rather than sending from inside the
    job: the controller delivers it, so it still arrives when the job is
    killed by the OOM reaper or the scheduler -- exactly the cases worth
    hearing about, and exactly the ones an in-job notification misses.

    Parameters
    ----------
    email : str or None
        Recipient; empty/None yields an empty string (no mail configured).
    events : sequence of str, optional
        Any of ``"start"``, ``"finish"``, ``"error"``. Defaults to finish and
        error -- a BEGIN mail per job is rarely worth the inbox.

    Returns
    -------
    str
        Something like ``--mail-type=END,FAIL --mail-user=me@example.org``,
        or ``""`` when no address is configured.

    Examples
    --------
    >>> slurm_mail_extra("me@example.org", ["error"])
    '--mail-type=FAIL --mail-user=me@example.org'
    >>> slurm_mail_extra(None)
    ''
    """
    if not email:
        return ""
    mapping = {"start": "BEGIN", "finish": "END", "error": "FAIL"}
    chosen = list(events) if events else ["finish", "error"]
    unknown = sorted(set(chosen) - set(mapping))
    if unknown:
        raise ValueError(
            f"unknown notify_events {unknown}; use any of "
            f"{sorted(mapping)} (they map to SLURM's BEGIN/END/FAIL)"
        )
    # Keep SLURM's own order, not the config's, so the string is stable.
    types = [mapping[k] for k in ("start", "finish", "error") if k in chosen]
    return f"--mail-type={','.join(types)} --mail-user={email}"


def failing_step(
    snakemake_log: Union[str, Path, None],
) -> "tuple[Union[str, None], Union[str, None]]":
    """Find which rule failed, and its log, from Snakemake's own log file.

    Guessing from step-log timestamps does not work: a `segment` failure
    leaves `prepare.log` as the most recently written of the sequential
    steps, so a mail built that way quotes a log that *succeeded* and names
    the wrong step. Snakemake records the failing rule and the exact log path
    it used, so read that instead.

    Parameters
    ----------
    snakemake_log : str or Path or None
        Path to Snakemake's own log (the ``log`` variable inside an
        ``onerror`` handler).

    Returns
    -------
    tuple
        ``(rule_name, log_path)``, either of which may be None when the log
        is unreadable or records no rule error.
    """
    try:
        text = Path(snakemake_log).read_text(errors="replace")
    except (OSError, TypeError, ValueError):
        return None, None

    blocks = text.split("Error in rule ")
    if len(blocks) < 2:
        return None, None
    last = blocks[-1]
    rule = re.match(r"(\S+?):", last)
    # The log: line inside that error block points at the step's own log.
    path = re.search(r"^\s*log:\s*(\S+?)(?:,|\s|$)", last, re.M)
    return (
        rule.group(1) if rule else None,
        path.group(1) if path else None,
    )
