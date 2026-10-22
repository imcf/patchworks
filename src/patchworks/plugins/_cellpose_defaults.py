"""What the Cellpose plugin fills in, decidable without importing Cellpose.

Importing ``cellpose.models`` pulls in torch: seconds to a minute from a
shared filesystem, which the merge step paid only to record these values
in the labels' provenance.
"""

from __future__ import annotations

import importlib.metadata
from typing import Any

# The first Cellpose whose ``eval`` takes flow3D_smooth (checked against the
# release wheels: absent in 3.1.0, present from 3.1.1 through 4.x).
_FLOW3D_SMOOTH_SINCE = (3, 1, 1)


def _installed_version() -> tuple[int, ...] | None:
    """The installed Cellpose's version as numbers, None if not installed."""
    try:
        raw = importlib.metadata.version("cellpose")
    except importlib.metadata.PackageNotFoundError:
        return None
    parts = []
    for segment in raw.split("."):
        digits = ""
        for c in segment:
            if not c.isdigit():
                break
            digits += c
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def takes_flow3d_smooth() -> bool:
    """Whether the installed Cellpose's ``eval`` takes ``flow3D_smooth``.

    Examples
    --------
    >>> isinstance(takes_flow3d_smooth(), bool)
    True
    """
    version = _installed_version()
    return version is not None and version >= _FLOW3D_SMOOTH_SINCE


def applied_defaults(
    do_3D: bool, settings: dict[str, Any], takes_smooth: bool | None = None
) -> dict[str, Any]:
    """What :func:`patchworks.plugins.cellpose.cellpose_fn` fills in that
    *settings* left open.

    ``fragments`` (``"auto"`` or absent: 0.1 in 3-D with a ``diameter``,
    else off: without one, a cell's size is a guess) and, in 3-D where the
    installed Cellpose takes it, ``flow3D_smooth=1``. *takes_smooth*
    defaults to :func:`takes_flow3d_smooth`.

    Examples
    --------
    >>> applied_defaults(True, {"diameter": 30}, takes_smooth=True)
    {'fragments': 0.1, 'flow3D_smooth': 1}
    >>> applied_defaults(False, {})
    {'fragments': None}
    """
    out: dict[str, Any] = {}
    fragments = settings.get("fragments", "auto")
    if fragments == "auto":
        fragments = 0.1 if do_3D and settings.get("diameter") else None
    out["fragments"] = fragments
    if do_3D and "flow3D_smooth" not in settings:
        if takes_smooth is None:
            takes_smooth = takes_flow3d_smooth()
        if takes_smooth:
            out["flow3D_smooth"] = 1
    return out
