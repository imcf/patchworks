"""Tests for log-friendly progress reporting."""

import logging

from patchworks import _progress


def test_track_yields_everything_unchanged():
    """Progress reporting must never alter or drop the work it wraps."""
    items = list(range(50))
    assert list(_progress.track(iter(items), "x", len(items))) == items
    # ...including when disabled, which is the passthrough path.
    assert (
        list(_progress.track(iter(items), "x", len(items), enabled=False))
        == items
    )


def test_track_logs_on_a_non_tty(monkeypatch, caplog):
    """A batch job gets log records, not a redrawing bar.

    The whole point: a carriage-returning bar collapses a SLURM log into one
    unreadable line, which is why convert ran silent rather than use one.
    """
    monkeypatch.setattr(_progress, "is_interactive", lambda: False)
    # Force every item to report, instead of waiting out the real interval.
    monkeypatch.setattr(_progress, "PROGRESS_INTERVAL_S", -1.0)
    with caplog.at_level(logging.INFO, logger=_progress.logger.name):
        list(_progress.track(iter(range(3)), "merge chunks", 3))

    messages = [r.getMessage() for r in caplog.records]
    assert any("starting" in m for m in messages)
    assert any("merge chunks" in m and "%" in m for m in messages)
    # No carriage returns: that is what makes it readable in a file.
    assert not any("\r" in m for m in messages)


def test_track_reports_the_final_count_even_when_quiet():
    """A run shorter than one interval must still say it finished."""
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    _progress.logger.addHandler(handler)
    _progress.logger.setLevel(logging.INFO)
    try:
        list(_progress.track(iter(range(2)), "quick", 2))
    finally:
        _progress.logger.removeHandler(handler)
    assert any("2/2" in r.getMessage() for r in records)


def test_eta_is_sane_and_degrades_gracefully():
    """An ETA is a linear extrapolation, and refuses to invent one."""
    assert _progress.format_eta(0, 100, 10.0) == "?"  # nothing measured yet
    assert _progress.format_eta(100, 100, 10.0) == "?"  # already done
    # Half done in 60 s → roughly another 60 s.
    assert _progress.format_eta(50, 100, 60.0) == "60s"
    # Units scale so a six-hour job does not report "21600s".
    assert _progress.format_eta(1, 100, 60.0).endswith(("m", "h"))
    assert _progress.format_eta(1, 1000, 600.0).endswith("h")
