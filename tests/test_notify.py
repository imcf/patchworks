"""Tests for email notification.

The governing rule: a notification must never be able to fail a run. Delivery
is best-effort on someone else's mail infrastructure, and a pipeline that
segmented correctly must not report failure because an SMTP host was down.
"""

import pytest

from patchworks._notify import log_tail, send, slurm_mail_extra


def test_no_address_means_no_mail_and_no_error():
    """The default state is "unconfigured", which must be completely silent."""
    assert send(None, "subject", "body") is False
    assert send("", "subject", "body") is False
    assert slurm_mail_extra(None) == ""
    assert slurm_mail_extra("") == ""


def test_slurm_mail_extra_maps_events_to_slurm_types():
    """Config words map to SLURM's BEGIN/END/FAIL, in a stable order."""
    assert (
        slurm_mail_extra("me@x.org")
        == "--mail-type=END,FAIL --mail-user=me@x.org"
    )
    assert (
        slurm_mail_extra("me@x.org", ["error"])
        == "--mail-type=FAIL --mail-user=me@x.org"
    )
    # Order comes from SLURM's lifecycle, not from however the config listed it.
    assert slurm_mail_extra(
        "me@x.org", ["error", "start", "finish"]
    ) == slurm_mail_extra("me@x.org", ["start", "finish", "error"])


def test_a_misspelled_event_is_rejected_up_front():
    """Better a named error before submission than silently no mail.

    The value ends up in an sbatch argument, where a typo would otherwise
    either be ignored or fail every job at submission time.
    """
    with pytest.raises(ValueError, match="notify_events"):
        slurm_mail_extra("me@x.org", ["finished"])  # not "finish"


def test_delivery_failure_is_swallowed(monkeypatch):
    """An unreachable transport returns False rather than raising.

    This is the property that keeps a mail problem from turning a successful
    six-hour run into a failed one.
    """
    import patchworks._notify as notify

    monkeypatch.setattr(notify.shutil, "which", lambda *a, **k: None)

    def _boom(*args, **kwargs):
        raise OSError("no route to host")

    monkeypatch.setattr("smtplib.SMTP", _boom)
    assert send("me@x.org", "subject", "body") is False


def test_log_tail_quotes_the_end_and_survives_a_missing_file(tmp_path):
    """The tail is what makes a failure mail actionable."""
    p = tmp_path / "step.log"
    p.write_text("\n".join(f"line {i}" for i in range(200)))
    tail = log_tail(p, lines=5)
    assert tail.splitlines() == [f"line {i}" for i in range(195, 200)]

    # A missing or empty log must produce a note, not an exception -- this
    # runs inside an error handler, where raising would mask the real failure.
    assert "could not read" in log_tail(tmp_path / "nope.log")
    (tmp_path / "empty.log").write_text("")
    assert "empty" in log_tail(tmp_path / "empty.log")
