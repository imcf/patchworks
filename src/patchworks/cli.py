"""``patchworks`` command line: convert, segment, inspect and view.

The same building blocks as the Python API and the Snakemake workflow, for a
one-off run on a workstation without writing a script or a config::

    patchworks convert scan.czi scan.zarr
    patchworks segment scan.zarr --method cellpose --model cyto3 --diameter 30
    patchworks info scan.zarr
    patchworks seams scan.zarr/labels/labels --tile-shape 16,1024,1024
    patchworks view scan.zarr
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from functools import partial
from typing import Any, Callable, Sequence

import numpy as np

logger = logging.getLogger("patchworks.cli")


def _ints(text: str) -> tuple[int, ...]:
    try:
        return tuple(int(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers, got {text!r}"
        ) from None


def _floats(text: str) -> tuple[float, ...]:
    try:
        return tuple(float(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated numbers, got {text!r}"
        ) from None


def _channel(text: str) -> int | None:
    if text.lower() == "none":
        return None
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f'expected a channel index or "none", got {text!r}'
        ) from None


def _overlap(text: str) -> int | tuple[int, ...]:
    values = _ints(text)
    return values[0] if len(values) == 1 else values


def _threshold_label(tile: np.ndarray, threshold: float) -> np.ndarray:
    """Label connected regions above a fixed *threshold*."""
    from scipy.ndimage import label

    return label(tile > threshold)[0].astype("int32")


def _build_fn(args: argparse.Namespace, image: Any) -> Callable:
    """The per-tile segmentation function the flags describe."""
    from .plugins.ome_zarr import read_pixel_size

    voxel = read_pixel_size(args.image, level=args.level)
    if args.method == "threshold":
        threshold = args.threshold
        if threshold is None:
            # One threshold for the whole image, not Otsu per tile: a
            # per-tile threshold moves with each tile's content, so the
            # same object comes out different sizes in different tiles.
            from ._io import auto_empty_threshold

            threshold = auto_empty_threshold(image, args.channel, args.level)
        logger.info("threshold: %s", threshold)
        return partial(_threshold_label, threshold=float(threshold))
    if args.method == "cellpose":
        from .plugins.cellpose import cellpose_fn

        return cellpose_fn(
            args.model,
            gpu=args.gpu,
            diameter=args.diameter,
            do_3D=args.do_3d,
            voxel_size=voxel or None,
        )
    if args.method == "dog":
        from .plugins.dog import dog_label_fn

        if args.threshold is None:
            raise SystemExit("--method dog needs --threshold")
        return dog_label_fn(
            args.low_sigma,
            args.high_sigma,
            args.threshold,
            use_gpu=args.gpu,
            voxel_size=voxel or None,
            sigma_units=args.sigma_units,
        )
    # custom: "module:function"
    module, _, name = args.fn.partition(":")
    fn = getattr(importlib.import_module(module), name or "segment")
    kwargs = json.loads(args.fn_kwargs) if args.fn_kwargs else {}
    return partial(fn, **kwargs) if kwargs else fn


def _cmd_convert(args: argparse.Namespace) -> int:
    from .plugins.ome_zarr import to_ome_zarr

    pixel_size = None
    if args.pixel_size:
        pixel_size = dict(zip("zyx"[-len(args.pixel_size) :], args.pixel_size))
    out = to_ome_zarr(
        args.input,
        args.output,
        axes=args.axes,
        pixel_size=pixel_size,
        sequence_pattern=args.sequence_pattern,
        n_levels=args.levels,
        shard=args.shard,
        overwrite=args.overwrite,
        compression=args.compression,
    )
    print(out)
    return 0


def _cmd_segment(args: argparse.Namespace) -> int:
    from ._core import tile_process
    from ._io import compression, load_ome_zarr

    if args.method == "custom" and not args.fn:
        raise SystemExit("--method custom needs --fn module:function")
    image = load_ome_zarr(args.image, channel=args.channel, level=args.level)
    fn = _build_fn(args, image)
    tile_shape: Any = args.tile_shape
    if tile_shape not in (None, "auto"):
        tile_shape = _ints(tile_shape)
    with compression(args.compression or "zstd"):
        tile_process(
            args.image,
            fn,
            tile_shape=tile_shape,
            overlap=args.overlap,
            channel=args.channel,
            level=args.level,
            use_gpu=args.gpu,
            max_workers=args.workers,
            write_to=args.output,
            output_component=args.name,
            sequential_labels=not args.no_sequential,
            skip_empty=args.skip_empty,
            stitch=args.stitch,
            iou_threshold=args.iou_threshold,
            resume=args.resume,
            gpus=args.gpus,
        )
    print(args.output or f"{args.image}/labels/{args.name}")
    return 0


def _cmd_seams(args: argparse.Namespace) -> int:
    from ._seams import seam_report

    report = seam_report(
        args.labels,
        args.tile_shape,
        component=args.component,
        max_faces=args.max_faces,
    )
    print(json.dumps(report, indent=2, default=str))
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    from ._io import open_group_any
    from .plugins.ome_zarr import read_ngff_attr, read_pixel_size

    def describe(path: str, indent: str = "") -> None:
        grp = open_group_any(path)
        ms = read_ngff_attr(grp.attrs, "multiscales") or []
        if not ms:
            print(f"{indent}(no multiscales metadata)")
            return
        axes = "".join(a["name"] for a in ms[0]["axes"])
        print(f"{indent}axes: {axes}  calibration: {read_pixel_size(path)}")
        for i, ds in enumerate(ms[0]["datasets"]):
            arr = grp[ds["path"]]
            codec = ", ".join(type(c).__name__ for c in arr.compressors)
            print(
                f"{indent}  level {i}: shape={arr.shape} chunks={arr.chunks} "
                f"dtype={arr.dtype} codec={codec or 'none'}"
            )

    print(args.store)
    describe(args.store)
    try:
        labels = read_ngff_attr(
            open_group_any(f"{args.store}/labels").attrs, "labels", []
        )
    except (KeyError, FileNotFoundError, ValueError):
        labels = []
    for name in labels or []:
        path = f"{args.store}/labels/{name}"
        n = dict(open_group_any(path).attrs).get("n_objects")
        print(f"labels/{name}" + (f"  ({n} objects)" if n is not None else ""))
        describe(path, "  ")
    return 0


def _cmd_view(args: argparse.Namespace) -> int:
    from .plugins.napari import view_in_napari

    view_in_napari(args.image, labels=args.labels, channel=args.channel)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The ``patchworks`` argument parser (exposed for tests and docs)."""
    parser = argparse.ArgumentParser(
        prog="patchworks",
        description="Tiled processing of arbitrarily large images.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="log at DEBUG level"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("convert", help="convert an image to OME-ZARR")
    p.add_argument("input", help="image file, folder, array store or glob")
    p.add_argument("output", help="destination .zarr")
    p.add_argument("--axes", help='axis letters, e.g. "czyx" (default: auto)')
    p.add_argument(
        "--pixel-size",
        type=_floats,
        help="voxel size in um, z,y,x (default: read from the file)",
    )
    p.add_argument("--sequence-pattern", help="glob over a TIFF sequence")
    p.add_argument("--levels", type=int, default=5, help="pyramid levels")
    p.add_argument("--shard", action="store_true", help="write sharded")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--compression", help='"zstd" (default), "blosc", ...')
    p.set_defaults(func=_cmd_convert)

    p = sub.add_parser("segment", help="segment an OME-ZARR tile by tile")
    p.add_argument("image", help="OME-ZARR store")
    p.add_argument(
        "--method",
        choices=("threshold", "cellpose", "dog", "custom"),
        default="threshold",
    )
    p.add_argument(
        "--channel",
        type=_channel,
        default=0,
        help='channel index, or "none" to keep every channel (default 0)',
    )
    p.add_argument("--level", type=int, default=0, help="pyramid level")
    p.add_argument(
        "--tile-shape",
        default="auto",
        help='z,y,x or "auto" (default); "none" keeps the store chunks',
    )
    p.add_argument(
        "--overlap", type=_overlap, default=16, help="halo, N or z,y,x"
    )
    p.add_argument("--gpu", action="store_true")
    p.add_argument(
        "--gpus",
        type=int,
        help="segment on the first N GPUs at once (one process each)",
    )
    p.add_argument("--workers", type=int, help="staging/merge workers")
    p.add_argument(
        "--output",
        help="write here instead of into the store's labels/ group",
    )
    p.add_argument("--name", default="labels", help="label image name")
    p.add_argument("--skip-empty", action="store_true")
    p.add_argument(
        "--no-sequential",
        action="store_true",
        help="keep gappy ids instead of renumbering to 1..N",
    )
    p.add_argument("--stitch", choices=("touch", "iou"), default="touch")
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument(
        "--resume", action="store_true", help="resumable if interrupted"
    )
    p.add_argument("--compression", help='"zstd" (default), "blosc", ...')
    g = p.add_argument_group("threshold / dog")
    g.add_argument(
        "--threshold",
        type=float,
        help="intensity (threshold) or DoG cutoff (dog); threshold "
        "defaults to Otsu over the whole image",
    )
    g.add_argument("--low-sigma", type=_floats, default=(1.0,))
    g.add_argument("--high-sigma", type=_floats, default=(3.0,))
    g.add_argument(
        "--sigma-units",
        choices=("px", "um"),
        default="px",
        help="sigmas in pixels, or micrometres (per axis, from the store)",
    )
    g = p.add_argument_group("cellpose")
    g.add_argument("--model", default="cyto3")
    g.add_argument("--diameter", type=float)
    g.add_argument("--do-3d", action="store_true")
    g = p.add_argument_group("custom")
    g.add_argument("--fn", help="module:function returning labels")
    g.add_argument("--fn-kwargs", help="JSON object of keyword arguments")
    p.set_defaults(func=_cmd_segment)

    p = sub.add_parser("seams", help="does the tiling show in the labels?")
    p.add_argument("labels", help="label group (or any zarr group)")
    p.add_argument("--tile-shape", type=_ints, required=True)
    p.add_argument("--component", default="0")
    p.add_argument("--max-faces", type=int, default=64)
    p.set_defaults(func=_cmd_seams)

    p = sub.add_parser("info", help="levels, chunks, codecs and labels")
    p.add_argument("store")
    p.set_defaults(func=_cmd_info)

    p = sub.add_parser("view", help="open in napari")
    p.add_argument("image")
    p.add_argument("--labels", help="label store (default: all in labels/)")
    p.add_argument("--channel", type=_channel, help="index (default: all)")
    p.set_defaults(func=_cmd_view)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``patchworks`` command."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if getattr(args, "tile_shape", None) == "none":
        args.tile_shape = None
    for key in ("low_sigma", "high_sigma"):
        value = getattr(args, key, None)
        if value is not None and len(value) == 1:
            setattr(args, key, value[0])
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
