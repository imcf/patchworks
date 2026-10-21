"""A picture of each channel of a converted image, to pick channels by.

    pixi run channels config/local/my_run.yaml
    pixi run channels config/local/my_run.yaml --json     # for the launcher

Each channel's maximum over z, from a pyramid level about 1024 pixels
across, stretched between its 1st and 99.5th percentiles, as
``<work_dir>/previews/channels/channel_<i>.png``. Also prints the image's
shape and voxel size (a Cellpose diameter, a max_radius_um ...). Light
enough for the login node; the image must be converted first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check import load  # noqa: E402
from run_multi import _config_bases, _resolve  # noqa: E402
from status import converted  # noqa: E402

# The finest pyramid level no wider than this is drawn.
MAX_SIDE = 1024


def channel_pictures(store: str, out: Path, region=None) -> dict:
    """A picture per channel: each one's maximum over z.

    The whole image, from the finest pyramid level about MAX_SIDE wide; or
    with *region* ``(y0, y1, x0, x1)``, fractions of the image's height and
    width, that region alone, from the finest level that draws it about
    MAX_SIDE wide -- a zoom with the detail of the finer levels, reading
    only that region.
    """
    import numpy as np
    import zarr
    from patchworks import load_ome_zarr
    from patchworks.plugins.ome_zarr import read_pixel_size
    from PIL import Image

    attrs = dict(zarr.open_group(store, mode="r").attrs)
    ms = (attrs.get("multiscales") or attrs.get("ome", {}).get("multiscales"))[
        0
    ]
    axes = [a["name"] if isinstance(a, dict) else a for a in ms["axes"]]
    n_levels = len(ms["datasets"])
    full = load_ome_zarr(store, channel=None, level=0)
    span = (
        (1.0, 1.0)
        if region is None
        else (region[1] - region[0], region[3] - region[2])
    )
    level = n_levels - 1
    for lvl in range(n_levels):
        arr = load_ome_zarr(store, channel=None, level=lvl)
        drawn = [n * f for n, f in zip(arr.shape[-2:], span)]
        if max(drawn) <= MAX_SIDE:
            level = lvl
            break
    arr = load_ome_zarr(store, channel=None, level=level)
    if "c" not in axes:
        arr = arr[None]
    if region is not None:
        h, w = arr.shape[-2:]
        y0, y1, x0, x1 = region
        arr = arr[
            ...,
            int(y0 * h) : max(int(y0 * h) + 1, int(y1 * h)),
            int(x0 * w) : max(int(x0 * w) + 1, int(x1 * w)),
        ]
    out.mkdir(parents=True, exist_ok=True)
    pictures = []
    for c in range(arr.shape[0]):
        plane = np.asarray(arr[c].max(axis=0) if arr.ndim == 4 else arr[c])
        plane = plane.astype(np.float32)
        lo, hi = np.percentile(plane, (1, 99.5))
        gray = np.clip((plane - lo) / max(hi - lo, 1e-6), 0, 1)
        path = out / f"channel_{c}{'' if region is None else '_zoom'}.png"
        Image.fromarray((gray * 255).astype(np.uint8)).save(path)
        pictures.append(str(path))
    return {
        "store": store,
        "axes": "".join(axes),
        "shape": [int(n) for n in full.shape],
        "channels": int(arr.shape[0]),
        "voxel_um": read_pixel_size(store, level=0),
        "level": level,
        "region": None if region is None else [float(f) for f in region],
        "pictures": pictures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="a run config, or a multi config")
    parser.add_argument("--json", action="store_true", help="print JSON")
    parser.add_argument(
        "--region",
        metavar="Y0,Y1,X0,X1",
        help="only this region, as fractions of the height and width "
        "(e.g. 0.4,0.5,0.6,0.7): a zoom, from a finer level",
    )
    args = parser.parse_args()
    workflow_dir = Path(__file__).resolve().parent.parent
    path = _resolve(args.config, *_config_bases(workflow_dir))
    if not path.is_file():
        sys.exit(f"[channels] ERROR: {args.config} not found")
    _, cfgs, _ = load(path)
    work = Path(cfgs[0]["work_dir"])
    store = work / "image.zarr"
    if not converted(store):
        sys.exit(
            f"[channels] ERROR: {store} is not converted yet: convert first "
            "(the run's first step), or wait for it to finish"
        )
    region = (
        tuple(float(f) for f in args.region.split(",")) if args.region else None
    )
    if region is not None and (
        len(region) != 4
        or not (
            0 <= region[0] < region[1] <= 1 and 0 <= region[2] < region[3] <= 1
        )
    ):
        sys.exit(
            "[channels] ERROR: --region is Y0,Y1,X0,X1, fractions in [0, 1]"
        )
    info = channel_pictures(str(store), work / "previews" / "channels", region)
    if args.json:
        print(json.dumps(info, default=str))
        return
    voxel = (
        ", ".join(f"{k} {v:g}" for k, v in (info["voxel_um"] or {}).items())
        or "unknown"
    )
    print(
        f"{info['store']}: {info['axes']} {info['shape']}; voxel (µm): {voxel}"
    )
    for c, picture in enumerate(info["pictures"]):
        print(f"  channel {c}: {picture}")


if __name__ == "__main__":
    main()
