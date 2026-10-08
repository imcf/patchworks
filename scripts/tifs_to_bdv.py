# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "h5py", "tifffile"]
# ///
"""Convert a folder of single-plane TIFFs into a BigDataViewer H5/XML pair.

The input is one 2D TIFF per plane, named like::

    A1--W00001--P00082--Z00009--T00000--405nm.tif
    |   |       |       |       |       '-- channel (wavelength)
    |   |       |       |       '---------- timepoint
    |   |       |       '------------------ z plane in the stack
    |   |       '-------------------------- position (one tile)
    |   '---------------------------------- field / well index
    '-------------------------------------- well

Each (position, channel) becomes one BDV view setup -- a tile in one
channel -- so the result opens directly in BigDataViewer and BigStitcher.
Every T found becomes a timepoint; a position/channel missing at some T is
written as a MissingView rather than as a black stack. Well and field are
expected to be constant; if they are not, they become part of the tile
name so no two stacks collide.

Each stack gets a multiresolution pyramid (2x mean downsampling, z only
once the voxels are close to isotropic), which BigDataViewer needs to be
responsive on large data and BigStitcher uses for pairwise shifts.

Voxel size is read from the TIFF tags (ImageJ or resolution tags) when it
is there; pass ``--voxel-size X Y Z`` (in µm) to set or override it -- the
z step is rarely stored in single-plane files.

The file names carry no stage coordinates, so by default every tile sits
at the origin. Either arrange them afterwards in BigStitcher (Arrange
Views > Move Tiles to Regular Grid), or give the layout here with
``--grid-columns`` / ``--overlap`` / ``--snake``.

Usage
-----
    uv run scripts/tifs_to_bdv.py /data/scan --dry-run
    uv run scripts/tifs_to_bdv.py /data/scan --voxel-size 0.325 0.325 2
    uv run scripts/tifs_to_bdv.py /data/scan -o /out/scan.xml \\
        --voxel-size 0.325 0.325 2 --grid-columns 14 --overlap 10 --snake

Without uv: ``pip install numpy h5py tifffile`` and run it with python.
Only uint8/uint16 images are supported (BDV HDF5 stores uint16).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import tifffile

NAME_RE = re.compile(
    r"^(?P<well>.+?)--W(?P<field>\d+)--P(?P<pos>\d+)--Z(?P<z>\d+)"
    r"--T(?P<t>\d+)--(?P<chan>.+?)\.tiff?$",
    re.IGNORECASE,
)

# Largest chunk edge in x/y; z chunks are scaled by the anisotropy.
CHUNK_XY = 64
# Stop adding pyramid levels once every dimension is at most this.
MIN_LEVEL_SIZE = 64
MAX_LEVELS = 8


@dataclass
class Stack:
    """All planes of one (tile, channel, timepoint)."""

    planes: dict[int, Path] = field(default_factory=dict)  # z -> file


@dataclass
class Setup:
    id: int
    tile: int
    channel: int
    name: str
    size_xyz: tuple[int, int, int] = (0, 0, 0)
    z_min: int = 0
    factors: list[tuple[int, int, int]] = field(default_factory=list)
    chunks: list[tuple[int, int, int]] = field(default_factory=list)


# --------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------


def scan(folder: Path):
    """Group the TIFFs by tile, channel and timepoint."""
    stacks: dict[tuple, Stack] = defaultdict(Stack)
    skipped = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in (".tif", ".tiff"):
            continue
        m = NAME_RE.match(path.name)
        if not m:
            skipped.append(path.name)
            continue
        tile = (m["well"], int(m["field"]), int(m["pos"]))
        key = (tile, m["chan"], int(m["t"]))
        z = int(m["z"])
        if z in stacks[key].planes:
            sys.exit(f"Duplicate plane: {path.name}")
        stacks[key].planes[z] = path
    if skipped:
        print(
            f"Skipped {len(skipped)} TIFF(s) not matching the pattern, "
            f"e.g. {skipped[0]}"
        )
    if not stacks:
        sys.exit(f"No matching TIFFs in {folder}")
    return stacks


def channel_sort_key(name: str):
    """Sort channels by wavelength when there is a number in the name."""
    m = re.search(r"\d+(\.\d+)?", name)
    return (0, float(m.group()), name) if m else (1, 0.0, name)


def tile_names(tiles):
    """P00082, or A1-W00001-P00082 when well/field are not constant."""
    wells = {t[0] for t in tiles}
    fields = {t[1] for t in tiles}
    names = {}
    for well, fld, pos in tiles:
        parts = []
        if len(wells) > 1:
            parts.append(well)
        if len(fields) > 1:
            parts.append(f"W{fld:05d}")
        parts.append(f"P{pos:05d}")
        names[(well, fld, pos)] = "-".join(parts)
    return names


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------


def read_plane_info(path: Path):
    """Shape, dtype and (if stored) pixel size in µm of one TIFF."""
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        shape, dtype = page.shape, page.dtype
        pixel = None
        unit_scale = {1: None, 2: 25400.0, 3: 10000.0}
        res_unit = page.tags.get("ResolutionUnit")
        res_unit = int(res_unit.value) if res_unit else 1
        ij = tif.imagej_metadata or {}
        ij_unit = str(ij.get("unit", "")).lower()
        xres = page.tags.get("XResolution")
        yres = page.tags.get("YResolution")
        if xres and yres:
            rx = xres.value[0] / xres.value[1] if xres.value[1] else 0
            ry = yres.value[0] / yres.value[1] if yres.value[1] else 0
            if rx > 0 and ry > 0:
                scale = unit_scale.get(res_unit)
                if scale is None and ij_unit in ("micron", "um", "µm"):
                    scale = 1.0
                if scale is not None and not (rx == 1 and ry == 1):
                    pixel = (scale / rx, scale / ry)
        z_step = ij.get("spacing")
    if len(shape) != 2:
        sys.exit(f"{path.name}: expected a single 2D plane, got {shape}")
    return shape, dtype, pixel, z_step


# --------------------------------------------------------------------------
# Pyramid layout
# --------------------------------------------------------------------------


def plan_pyramid(size_xyz, voxel_xyz):
    """Per-level downsampling factors and chunk sizes (both xyz)."""
    factors = [(1, 1, 1)]
    while len(factors) < MAX_LEVELS:
        prev = factors[-1]
        dims = [max(s // f, 1) for s, f in zip(size_xyz, prev, strict=True)]
        if max(dims) <= MIN_LEVEL_SIZE:
            break
        phys = [v * f for v, f in zip(voxel_xyz, prev, strict=True)]
        smallest = min(phys)
        # Double a dimension only if it is not already much coarser than
        # the finest one and there is something left to halve.
        new = tuple(
            f * 2 if (p < 2 * smallest and d >= 2) else f
            for f, p, d in zip(prev, phys, dims, strict=True)
        )
        if new == prev:
            break
        factors.append(new)

    chunks = []
    for f in factors:
        dims = [max(s // fi, 1) for s, fi in zip(size_xyz, f, strict=True)]
        phys = [v * fi for v, fi in zip(voxel_xyz, f, strict=True)]
        cz = round(CHUNK_XY * min(phys[0], phys[1]) / phys[2])
        cz = max(1, min(CHUNK_XY, cz))
        chunks.append(
            (min(CHUNK_XY, dims[0]), min(CHUNK_XY, dims[1]), min(cz, dims[2]))
        )
    return factors, chunks


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def to_uint16(plane: np.ndarray) -> np.ndarray:
    if plane.dtype == np.uint16:
        return plane
    if plane.dtype == np.uint8:
        return plane.astype(np.uint16)
    sys.exit(f"Unsupported pixel type {plane.dtype}; only uint8/uint16.")


def write_level0(ds, stack: Stack, setup: Setup, pool, chunk_z: int):
    """Read the planes in batches of one z chunk and write them."""
    nx, ny, nz = setup.size_xyz
    zs = list(range(nz))
    blank = np.zeros((ny, nx), np.uint16)

    def load(z):
        path = stack.planes.get(z + setup.z_min)
        if path is None:
            return blank
        plane = tifffile.imread(path)
        if plane.shape != (ny, nx):
            sys.exit(f"{path.name}: shape {plane.shape} != {(ny, nx)}")
        return to_uint16(plane)

    for start in range(0, nz, chunk_z):
        batch = zs[start : start + chunk_z]
        planes = list(pool.map(load, batch))
        ds[start : start + len(batch)] = np.stack(planes).view(np.int16)


def downsample(src, dst, rel_zyx):
    """Mean-downsample one level into the next, a z slab at a time."""
    rz, ry, rx = rel_zyx
    nz, ny, nx = dst.shape
    slab = max(1, dst.chunks[0])
    for z0 in range(0, nz, slab):
        z1 = min(nz, z0 + slab)
        block = src[z0 * rz : z1 * rz, : ny * ry, : nx * rx]
        block = block.view(np.uint16).astype(np.float32)
        block = block.reshape(z1 - z0, rz, ny, ry, nx, rx).mean(axis=(1, 3, 5))
        dst[z0:z1] = np.round(block).astype(np.uint16).view(np.int16)


def write_setup_meta(h5, setup: Setup):
    grp = h5.require_group(f"s{setup.id:02d}")
    for name in ("resolutions", "subdivisions"):
        if name in grp:
            del grp[name]
    grp.create_dataset(
        "resolutions", data=np.array(setup.factors, dtype=np.float64)
    )
    grp.create_dataset(
        "subdivisions", data=np.array(setup.chunks, dtype=np.int32)
    )


def write_stack(h5, stack, setup, tp, pool, compression):
    grp = h5.require_group(f"t{tp:05d}/s{setup.id:02d}")
    prev = None
    prev_f = None
    for level, (f, c) in enumerate(
        zip(setup.factors, setup.chunks, strict=True)
    ):
        shape = tuple(
            max(s // fi, 1)
            for s, fi in zip(setup.size_xyz[::-1], f[::-1], strict=True)
        )
        ds = grp.create_dataset(
            f"{level}/cells",
            shape=shape,
            dtype=np.int16,
            chunks=tuple(c[::-1]),
            compression=compression,
        )
        if level == 0:
            write_level0(ds, stack, setup, pool, c[2])
        else:
            rel = tuple(
                a // b for a, b in zip(f[::-1], prev_f[::-1], strict=True)
            )
            downsample(prev, ds, rel)
        prev, prev_f = ds, f


# --------------------------------------------------------------------------
# XML
# --------------------------------------------------------------------------


def sub(parent, tag, text=None, **attrib):
    el = ET.SubElement(parent, tag, attrib)
    if text is not None:
        el.text = str(text)
    return el


def fmt(v: float) -> str:
    return repr(float(v))


def write_xml(
    xml_path,
    h5_path,
    setups,
    tiles,
    channels,
    timepoints,
    missing,
    voxel,
    tile_offsets,
):
    root = ET.Element("SpimData", version="0.2")
    sub(root, "BasePath", ".", type="relative")
    seq = sub(root, "SequenceDescription")
    loader = sub(seq, "ImageLoader", format="bdv.hdf5")
    rel = Path(h5_path).resolve().relative_to(Path(xml_path).resolve().parent)
    sub(loader, "hdf5", rel.as_posix(), type="relative")

    vs = sub(seq, "ViewSetups")
    for s in setups:
        el = sub(vs, "ViewSetup")
        sub(el, "id", s.id)
        sub(el, "name", s.name)
        sub(el, "size", " ".join(map(str, s.size_xyz)))
        vox = sub(el, "voxelSize")
        sub(vox, "unit", "µm")
        sub(vox, "size", " ".join(fmt(v) for v in voxel))
        attrs = sub(el, "attributes")
        sub(attrs, "illumination", 0)
        sub(attrs, "channel", s.channel)
        sub(attrs, "tile", s.tile)
        sub(attrs, "angle", 0)

    att = sub(vs, "Attributes", name="illumination")
    ill = sub(att, "Illumination")
    sub(ill, "id", 0)
    sub(ill, "name", 0)
    att = sub(vs, "Attributes", name="channel")
    for i, name in enumerate(channels):
        ch = sub(att, "Channel")
        sub(ch, "id", i)
        sub(ch, "name", name)
    att = sub(vs, "Attributes", name="tile")
    for i, name in enumerate(tiles):
        t = sub(att, "Tile")
        sub(t, "id", i)
        sub(t, "name", name)
        if tile_offsets is not None:
            loc = tile_offsets[i]
            sub(t, "location", " ".join(fmt(v) for v in loc))
    att = sub(vs, "Attributes", name="angle")
    ang = sub(att, "Angle")
    sub(ang, "id", 0)
    sub(ang, "name", 0)

    if timepoints == list(range(timepoints[0], timepoints[-1] + 1)):
        tps = sub(seq, "Timepoints", type="range")
        sub(tps, "first", timepoints[0])
        sub(tps, "last", timepoints[-1])
    else:
        tps = sub(seq, "Timepoints", type="pattern")
        sub(tps, "integerpattern", ", ".join(map(str, timepoints)))
    if missing:
        mv = sub(seq, "MissingViews")
        for tp, sid in missing:
            sub(mv, "MissingView", timepoint=str(tp), setup=str(sid))

    # Calibration is relative to the finest axis, as BigStitcher does it,
    # so world coordinates are in units of the smallest voxel edge.
    smallest = min(voxel)
    cal = [v / smallest for v in voxel]
    regs = sub(root, "ViewRegistrations")
    missing_set = set(missing)
    for tp in timepoints:
        for s in setups:
            if (tp, s.id) in missing_set:
                continue
            reg = sub(
                regs, "ViewRegistration", timepoint=str(tp), setup=str(s.id)
            )
            tx, ty = (0.0, 0.0)
            if tile_offsets is not None:
                tx, ty = (o / smallest for o in tile_offsets[s.tile][:2])
            tz = s.z_min * cal[2]
            if tx or ty or tz:
                tr = sub(reg, "ViewTransform", type="affine")
                sub(tr, "Name", "Translation to Regular Grid")
                sub(
                    tr,
                    "affine",
                    " ".join(
                        fmt(v) for v in (1, 0, 0, tx, 0, 1, 0, ty, 0, 0, 1, tz)
                    ),
                )
            tr = sub(reg, "ViewTransform", type="affine")
            sub(tr, "Name", "calibration")
            sub(
                tr,
                "affine",
                " ".join(
                    fmt(v)
                    for v in (cal[0], 0, 0, 0, 0, cal[1], 0, 0, 0, 0, cal[2], 0)
                ),
            )

    for tag in (
        "ViewInterestPoints",
        "BoundingBoxes",
        "PointSpreadFunctions",
        "StitchingResults",
        "IntensityAdjustments",
    ):
        sub(root, tag)

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)


def grid_offsets(n_tiles, cols, overlap, snake, column_major, tile_xy, voxel):
    """Physical (µm) xyz location of each tile on a regular grid."""
    step_x = tile_xy[0] * voxel[0] * (1 - overlap / 100)
    step_y = tile_xy[1] * voxel[1] * (1 - overlap / 100)
    rows = -(-n_tiles // cols)
    out = []
    for i in range(n_tiles):
        if column_major:
            c, r = divmod(i, rows)
            if snake and c % 2:
                r = rows - 1 - r
        else:
            r, c = divmod(i, cols)
            if snake and r % 2:
                c = cols - 1 - c
        out.append((c * step_x, r * step_y, 0.0))
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("folder", type=Path, help="folder with the TIFFs")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output .xml (default: <folder>/dataset.xml); the "
        ".h5 is written next to it with the same name",
    )
    p.add_argument(
        "--voxel-size",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="voxel size in µm (default: from the TIFF tags, "
        "z step 1 if absent)",
    )
    p.add_argument(
        "--compression",
        choices=("none", "gzip"),
        default="none",
        help="HDF5 compression (gzip: smaller, slower)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="threads reading TIFFs (default 8)",
    )
    p.add_argument(
        "--grid-columns",
        type=int,
        help="place the tiles on a grid this many columns wide, "
        "in position order",
    )
    p.add_argument(
        "--grid-rows", type=int, help="rows of the grid, with --column-major"
    )
    p.add_argument(
        "--overlap",
        type=float,
        default=10.0,
        help="tile overlap in %% for --grid-columns (default 10)",
    )
    p.add_argument(
        "--snake",
        action="store_true",
        help="serpentine grid: every other row runs backwards",
    )
    p.add_argument(
        "--column-major",
        action="store_true",
        help="positions fill columns first (needs --grid-rows)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="report what was found, write nothing",
    )
    p.add_argument(
        "--overwrite", action="store_true", help="replace an existing output"
    )
    args = p.parse_args(argv)

    folder = args.folder
    if not folder.is_dir():
        sys.exit(f"Not a folder: {folder}")
    xml_path = args.output or folder / "dataset.xml"
    if xml_path.suffix.lower() != ".xml":
        xml_path = xml_path.with_suffix(".xml")
    h5_path = xml_path.with_suffix(".h5")

    stacks = scan(folder)
    tiles = sorted({k[0] for k in stacks})
    channels = sorted({k[1] for k in stacks}, key=channel_sort_key)
    timepoints = sorted({k[2] for k in stacks})
    names = tile_names(tiles)
    tile_idx = {t: i for i, t in enumerate(tiles)}
    chan_idx = {c: i for i, c in enumerate(channels)}

    first = next(iter(stacks.values())).planes
    shape, dtype, pixel, z_step = read_plane_info(first[min(first)])
    if args.voxel_size:
        voxel = tuple(args.voxel_size)
    else:
        if pixel is None:
            print(
                "WARNING: no pixel size in the TIFF tags; using 1 µm. "
                "Pass --voxel-size X Y Z."
            )
            pixel = (1.0, 1.0)
        if z_step is None:
            print(
                "WARNING: no z step in the TIFF tags; using 1 µm. "
                "Pass --voxel-size X Y Z."
            )
            z_step = 1.0
        voxel = (float(pixel[0]), float(pixel[1]), float(z_step))

    # One setup per (tile, channel). The z range is the union over
    # timepoints, so a setup has one size as BDV requires.
    setups: list[Setup] = []
    setup_of = {}
    for tile in tiles:
        for chan in channels:
            zs = [
                z
                for (t, c, _), st in stacks.items()
                if t == tile and c == chan
                for z in st.planes
            ]
            if not zs:
                continue
            z_min, z_max = min(zs), max(zs)
            s = Setup(
                id=len(setups),
                tile=tile_idx[tile],
                channel=chan_idx[chan],
                name=f"{names[tile]}_{chan}",
                size_xyz=(shape[1], shape[0], z_max - z_min + 1),
                z_min=z_min,
            )
            s.factors, s.chunks = plan_pyramid(s.size_xyz, voxel)
            setups.append(s)
            setup_of[(tile, chan)] = s

    missing = []
    gaps = 0
    for tp in timepoints:
        for (tile, chan), s in setup_of.items():
            st = stacks.get((tile, chan, tp))
            if st is None:
                missing.append((tp, s.id))
            else:
                gaps += s.size_xyz[2] - len(st.planes)

    tile_offsets = None
    if args.grid_columns or args.grid_rows:
        if args.column_major:
            if not args.grid_rows:
                sys.exit("--column-major needs --grid-rows")
            cols = -(-len(tiles) // args.grid_rows)
        else:
            if not args.grid_columns:
                sys.exit(
                    "--grid-rows needs --column-major; otherwise give "
                    "--grid-columns"
                )
            cols = args.grid_columns
        tile_offsets = grid_offsets(
            len(tiles),
            cols,
            args.overlap,
            args.snake,
            args.column_major,
            (shape[1], shape[0]),
            voxel,
        )

    n_files = sum(len(st.planes) for st in stacks.values())
    zsizes = sorted({s.size_xyz[2] for s in setups})
    print(f"Found {n_files} planes in {folder}")
    print(
        f"  tiles:      {len(tiles)} ({names[tiles[0]]} .. {names[tiles[-1]]})"
    )
    print(f"  channels:   {len(channels)} ({', '.join(channels)})")
    print(
        f"  timepoints: {len(timepoints)} ({', '.join(map(str, timepoints))})"
    )
    print(f"  plane:      {shape[1]} x {shape[0]} px, {dtype}")
    print(f"  z planes:   {zsizes[0]}..{zsizes[-1]} per stack")
    print(f"  voxel:      {voxel[0]:g} x {voxel[1]:g} x {voxel[2]:g} µm")
    print(
        f"  setups:     {len(setups)}, pyramid levels "
        f"{min(len(s.factors) for s in setups)}.."
        f"{max(len(s.factors) for s in setups)}"
    )
    if missing:
        print(
            f"  missing:    {len(missing)} view(s) absent at some "
            "timepoint (written as MissingViews)"
        )
    if gaps:
        print(
            f"  WARNING: {gaps} plane(s) missing inside stacks, "
            "filled with zeros"
        )
    size_gb = n_files * shape[0] * shape[1] * 2 / 1e9
    print(
        f"  output:     {xml_path} + {h5_path.name} "
        f"(~{size_gb * 1.15:.1f} GB uncompressed)"
    )
    if args.dry_run:
        return

    for path in (xml_path, h5_path):
        if path.exists() and not args.overwrite:
            sys.exit(f"{path} exists; pass --overwrite to replace it")
    xml_path.parent.mkdir(parents=True, exist_ok=True)

    compression = None if args.compression == "none" else "gzip"
    todo = [
        (tp, tile, chan)
        for tp in timepoints
        for (tile, chan) in setup_of
        if (tile, chan, tp) in stacks
    ]
    t0 = time.time()
    with (
        h5py.File(h5_path, "w") as h5,
        ThreadPoolExecutor(args.workers) as pool,
    ):
        for s in setups:
            write_setup_meta(h5, s)
        for i, (tp, tile, chan) in enumerate(todo, 1):
            s = setup_of[(tile, chan)]
            write_stack(h5, stacks[(tile, chan, tp)], s, tp, pool, compression)
            elapsed = time.time() - t0
            eta = elapsed / i * (len(todo) - i)
            print(
                f"[{i}/{len(todo)}] t{tp} {s.name}  "
                f"{elapsed / 60:.1f} min, ~{eta / 60:.1f} min left",
                flush=True,
            )

    write_xml(
        xml_path,
        h5_path,
        setups,
        [names[t] for t in tiles],
        channels,
        timepoints,
        missing,
        voxel,
        tile_offsets,
    )
    print(f"Done: {xml_path}")


if __name__ == "__main__":
    main()
