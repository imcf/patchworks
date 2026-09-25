"""Seam quality report: do tile boundaries leave marks in the labels?

A well-stitched segmentation is indifferent to where the tiles were. One that
is not shows it at the seams: objects that end abruptly on a tile boundary,
because the tile on the other side decided differently (too little overlap,
or a model unstable near tile edges), or because two tiles' fragments were
not joined.

That is measurable without any ground truth. At a seam, count the labels on
one side that have **no** label continuing on the other side -- "orphans".
Objects do end somewhere, so some orphans are expected; the question is
whether seams produce more of them than planes that are not seams. The
report therefore measures the same thing on a control plane inside each
tile, where nothing was stitched, and compares the two rates.
"""

from __future__ import annotations

import logging
from itertools import product as _iproduct
from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np
import zarr

logger = logging.getLogger(__name__)


def _orphans(a: np.ndarray, b: np.ndarray, min_voxels: int) -> tuple[int, int]:
    """Labels on slice *a* (with >= *min_voxels* voxels there) and how many
    of them have no non-zero voxel of *b* over their footprint."""
    ids, counts = np.unique(a[a > 0], return_counts=True)
    ids = ids[counts >= min_voxels]
    if ids.size == 0:
        return 0, 0
    partnered = np.unique(a[(a > 0) & (b > 0)])
    return int(ids.size), int(np.setdiff1d(ids, partnered).size)


def seam_report(
    labels: Union[str, Path, "zarr.Array"],
    tile_shape: Sequence[int],
    *,
    component: str = "0",
    min_voxels: int = 4,
    max_faces: int = 64,
    warn_ratio: float = 2.0,
) -> dict[str, Any]:
    """Measure whether tile seams leave artifacts in a merged label image.

    For every axis, compares the **orphan rate** at tile seams -- the fraction
    of labels touching one side of a seam with nothing continuing on the
    other -- against the same rate on control planes halfway through each
    tile. A seam rate well above the interior rate means the tiling shows:
    raise ``overlap`` (it should cover about one object), or try
    ``stitch="iou"``.

    Parameters
    ----------
    labels : str, Path or zarr.Array
        The merged labels: a zarr array, or a group path plus *component*
        (default ``"0"``, a label group's full resolution).
    tile_shape : sequence of int
        The tile shape the segmentation ran with.
    component : str, optional
        Array inside *labels* when it is a path.
    min_voxels : int, optional
        Ignore labels with fewer voxels than this on the slice (edge grazes).
    max_faces : int, optional
        Cap on seam faces (and as many control faces) read per axis, spread
        evenly over the image (default 64). Each face is a two-voxel slab of
        one tile's cross-section, so this bounds the I/O on a huge store:
        64 z-faces of a 1024 x 1024 int32 tile read ~1 GB.
    warn_ratio : float, optional
        Log a warning when an axis' seam rate exceeds its interior rate by
        this factor.

    Returns
    -------
    dict
        ``{"axes": {axis: {...}}, "worst_seams": [...]}``. Per axis:
        ``seam_labels``, ``seam_orphans``, ``seam_rate``, ``interior_rate``
        (None when tiles are one voxel thick there, leaving no interior
        plane), and ``ratio``. ``worst_seams`` lists the faces with the most
        orphans, with their position, for a look in the viewer.

    Examples
    --------
    >>> seam_report("scan.zarr/labels/cells", (16, 1024, 1024))  # doctest: +SKIP
    {'axes': {0: {'seam_rate': 0.11, 'interior_rate': 0.10, ...}, ...}, ...}
    """
    arr = (
        zarr.open_group(str(labels), mode="r")[component]
        if isinstance(labels, (str, Path))
        else labels
    )
    shape = arr.shape
    tile = tuple(int(t) for t in tile_shape)
    if len(tile) != len(shape):
        raise ValueError(
            f"tile_shape {tile} has {len(tile)} axes; the labels have "
            f"{len(shape)}"
        )

    axes: dict[int, dict[str, Any]] = {}
    worst: list[dict[str, Any]] = []
    for ax, (n, t) in enumerate(zip(shape, tile)):
        positions = list(range(t, n, t))
        if not positions:
            continue
        other = [a for a in range(len(shape)) if a != ax]
        columns = list(_iproduct(*[range(0, shape[a], tile[a]) for a in other]))
        faces = [(p, c) for p in positions for c in columns]
        if len(faces) > max_faces:
            pick = np.linspace(0, len(faces) - 1, max_faces).astype(int)
            faces = [faces[i] for i in pick]

        def slab(pos: int, col: tuple[int, ...]) -> np.ndarray:
            sl: list[slice] = [slice(None)] * len(shape)
            sl[ax] = slice(pos - 1, pos + 1)
            for a, off in zip(other, col):
                sl[a] = slice(off, min(off + tile[a], shape[a]))
            return np.moveaxis(np.asarray(arr[tuple(sl)]), ax, 0)

        seam_n = seam_o = inner_n = inner_o = 0
        for pos, col in faces:
            s = slab(pos, col)
            # Both directions: an orphan on either side is a seam mark.
            n1, o1 = _orphans(s[0], s[1], min_voxels)
            n2, o2 = _orphans(s[1], s[0], min_voxels)
            seam_n += n1 + n2
            seam_o += o1 + o2
            if o1 + o2:
                worst.append(
                    {
                        "axis": ax,
                        "position": pos,
                        "offset": list(col),
                        "orphans": o1 + o2,
                        "labels": n1 + n2,
                    }
                )
            if t >= 2:
                c = slab(pos - t // 2, col)  # halfway into the tile below
                m1, p1 = _orphans(c[0], c[1], min_voxels)
                m2, p2 = _orphans(c[1], c[0], min_voxels)
                inner_n += m1 + m2
                inner_o += p1 + p2

        seam_rate = seam_o / seam_n if seam_n else 0.0
        interior = (inner_o / inner_n if inner_n else 0.0) if t >= 2 else None
        ratio = (
            seam_rate / interior
            if interior
            else (float("inf") if seam_rate and interior == 0.0 else None)
        )
        axes[ax] = {
            "seams": len(positions),
            "faces_read": len(faces),
            "seam_labels": seam_n,
            "seam_orphans": seam_o,
            "seam_rate": seam_rate,
            "interior_rate": interior,
            "ratio": ratio,
        }
        if ratio is not None and ratio > warn_ratio and seam_o:
            logger.warning(
                "seam report: axis %d seams orphan %.1f%% of labels vs "
                "%.1f%% inside tiles (%.1fx) -- the tiling shows. Raise "
                "overlap to about one object, or try stitch='iou'.",
                ax,
                100 * seam_rate,
                100 * (interior or 0.0),
                ratio,
            )
        else:
            logger.info(
                "seam report: axis %d seam orphan rate %.1f%%, interior %s",
                ax,
                100 * seam_rate,
                "n/a" if interior is None else f"{100 * interior:.1f}%",
            )

    worst.sort(key=lambda f: (-f["orphans"], f["axis"], f["position"]))
    return {"axes": axes, "worst_seams": worst[:20]}
