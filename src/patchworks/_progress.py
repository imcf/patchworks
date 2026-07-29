"""Progress reporting that survives being written to a log file.

Every long step in the workflow runs unattended in a batch job, where the
output is read hours later out of a file. A redrawing bar (dask's
``ProgressBar``, ``tqdm``) collapses into one enormous unreadable line there,
so the default here is periodic log records instead -- and a bar only when
someone is actually watching a terminal.

The interval matters more than the precision: the job runs for hours, and the
question being answered is "is this working or hung?", not "exactly how far".
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import nullcontext
from typing import Iterable, Iterator, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Often enough to tell progress from a hang, rare enough that a six-hour job
# leaves a log you can still read.
PROGRESS_INTERVAL_S = 60.0


def is_interactive() -> bool:
    """True when someone is plausibly watching a terminal."""
    return bool(getattr(sys.stderr, "isatty", lambda: False)())


def format_eta(done: int, total: int, elapsed: float) -> str:
    """Rough remaining time from a linear extrapolation.

    Parameters
    ----------
    done, total : int
        Units finished and expected.
    elapsed : float
        Seconds spent so far.

    Returns
    -------
    str
        A short human-readable duration, or ``"?"`` when it cannot be
        estimated yet.
    """
    if done <= 0 or done >= total:
        return "?"
    left = elapsed / done * (total - done)
    if left < 90:
        return f"{left:.0f}s"
    if left < 5400:
        return f"{left / 60:.0f}m"
    return f"{left / 3600:.1f}h"


def log_progress(label: str, done: int, total: int, started: float) -> None:
    """Emit one progress line.

    Parameters
    ----------
    label : str
        What is being worked on, e.g. ``"image.zarr/0"``.
    done, total : int
        Units finished and expected.
    started : float
        ``time.monotonic()`` when the work began.
    """
    elapsed = time.monotonic() - started
    logger.info(
        "%s: %s/%s (%.0f%%) after %.0fm, ~%s left",
        label,
        f"{done:,}",
        f"{total:,}",
        100.0 * done / max(1, total),
        elapsed / 60,
        format_eta(done, total, elapsed),
    )


def track(
    iterable: Iterable[T],
    label: str,
    total: int,
    *,
    enabled: bool = True,
) -> Iterator[T]:
    """Yield from *iterable*, reporting progress as it goes.

    Uses ``tqdm`` when attached to a terminal and periodic log lines
    otherwise, so the same call is right in a notebook and in a SLURM job.

    Parameters
    ----------
    iterable : iterable
        The work to iterate. Consumed lazily, so this is safe over
        ``imap_unordered``.
    label : str
        Description of the work.
    total : int
        Expected number of items, used for the percentage and the ETA.
    enabled : bool, optional
        Set ``False`` to pass items straight through. Default ``True``.

    Yields
    ------
    object
        The items of *iterable*, unchanged.
    """
    if not enabled:
        yield from iterable
        return

    if is_interactive():
        try:
            from tqdm.auto import tqdm

            yield from tqdm(iterable, total=total, desc=label)
            return
        except ImportError:
            pass

    started = time.monotonic()
    last = started
    done = 0
    logger.info("%s: starting (%s items)", label, f"{total:,}")
    for item in iterable:
        yield item
        done += 1
        now = time.monotonic()
        if now - last >= PROGRESS_INTERVAL_S:
            last = now
            log_progress(label, done, total, started)
    log_progress(label, done, total, started)


def dask_progress(label: str, enabled: bool = True):
    """Progress context manager for a dask computation.

    Returns
    -------
    contextmanager
        ``ProgressBar`` on a terminal, a periodically-logging callback
        otherwise, or a no-op when *enabled* is false.
    """
    if not enabled:
        return nullcontext()

    logger.info("writing %s …", label)
    if is_interactive():
        from dask.diagnostics import ProgressBar

        return ProgressBar()

    from dask.callbacks import Callback

    class _LogProgress(Callback):
        """Count finished dask tasks and log every PROGRESS_INTERVAL_S."""

        def _start_state(self, dsk, state):
            self._total = sum(
                len(state[k])
                for k in ("ready", "waiting", "running", "finished")
            )
            self._done = 0
            self._t0 = time.monotonic()
            self._last = self._t0

        def _posttask(self, key, result, dsk, state, worker_id):
            self._done += 1
            now = time.monotonic()
            if now - self._last >= PROGRESS_INTERVAL_S:
                self._last = now
                log_progress(label, self._done, self._total, self._t0)

        def _finish(self, dsk, state, errored):
            if not errored:
                log_progress(label, self._total, self._total, self._t0)

    return _LogProgress()
