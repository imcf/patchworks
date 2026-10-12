# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Make a MoBIE collection table from the BDV H5/XML files below a folder.

Every BigDataViewer ``.xml`` found (recursively) becomes one *view*, named
after the folder that holds it -- e.g. ``rescan_001/dataset.xml`` from
``tifs_to_bdv.py`` gives the view ``rescan_001``. ``--name-level 2`` names
it after the folder above instead, and so on. Selecting the view in
MoBIE shows all its tiles and channels together, each tile at the
position stored in the XML.

MoBIE loads one single-channel image per table row, and for a BDV XML its
``channel`` column picks a *view setup* (a tile in one channel). So the
table has one row per setup: with 168 tiles and 4 channels, 672 rows per
XML. Tiles of the same channel share one ``display``, so a single contrast
and color setting controls the whole channel.

**2D in 3D.** Datasets are grouped by *sample*: the folder above their own
(``ScanR/Acquisition`` and ``ScanR/Scan`` are both in sample ``ScanR``).
Within a sample, every 2D dataset (one z plane) is shifted along z to the
middle of the sample's 3D dataset, through the ``affine`` column, so it
sits inside the stack instead of at its top or bottom. If a sample has
several 3D datasets, the one with "scan" in its name is the reference
(``--reference`` picks another).

**Spots.** A ``positions.xml`` alone in a folder (no BDV XML next to it)
becomes a spots dataset: its coordinates are written to a
``positions_spots.tsv`` next to it (``spot_id``, ``x``, ``y``, ``z``) and
added with ``type`` ``spots`` and a ``spot_radius`` of 10. Spots without z
are put at the middle of the sample's 3D dataset, like 2D images.
Coordinates are taken as they are, so they must be in the same space as
the images; the dry run prints both ranges to check, and
``--spot-scale`` converts (e.g. µm to pixels).

Columns written (see https://mobie.github.io/tutorials/mobie_collection_table.html):

- ``uri``: the XML or spots table, relative to the collection table (or
  absolute, ``--absolute``)
- ``type``: ``intensities`` or ``spots``
- ``channel``: setup index in the XML
- ``name``: ``<view>_<setup name>``, unique per row
- ``view``: the view name -- all rows of one XML are shown together
- ``display``: ``<view>_<channel>``, shared by all tiles of a channel
- ``color``: from the channel's wavelength (405 Blue, 488 Green, ...)
- ``contrast_limits``: empty (MoBIE's default) unless ``--contrast-limits``
- ``spot_radius``: for spots
- ``affine``: the z shift of 2D datasets, empty otherwise
- ``channel_name``, ``pixel_unit``, ``pixel_size_x/y/z``,
  ``num_pixels_x/y/z``, ``sample``, ``tile``: annotations for browsing,
  the first ones as Fiji's own "Create MoBIE Collection Table" writes them

Usage
-----
    uv run scripts/bdv_to_mobie_table.py /data/project --dry-run
    uv run scripts/bdv_to_mobie_table.py /data/project
    uv run scripts/bdv_to_mobie_table.py /data/project -o /data/project/mobie.tsv

Then in Fiji: Plugins > MoBIE > Open > Open Collection Table..., pick the
table, ``Data Root``: ``UseTableFolder`` (``PathsInTableAreAbsolute`` with
``--absolute``), ``Viewing mode``: ``ThreeDimensional``.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

# Order and annotation columns as Fiji's MoBIE > Create > Create MoBIE
# Collection Table writes them.
COLUMNS = [
    "uri",
    "type",
    "name",
    "channel_name",
    "pixel_unit",
    "pixel_size_x",
    "pixel_size_y",
    "pixel_size_z",
    "num_pixels_x",
    "num_pixels_y",
    "num_pixels_z",
    "channel",
    "view",
    "display",
    "color",
    "contrast_limits",
    "spot_radius",
    "affine",
    "sample",
    "tile",
]

# (upper wavelength bound in nm, MoBIE color)
WAVELENGTH_COLORS = [
    (460, "Blue"),
    (530, "Green"),
    (600, "Red"),
    (10_000, "Magenta"),
]

IDENTITY = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

# Spot coordinate keys, compared lowercased without separators:
# x, posx, positionx, xpos, centerx, stagex, xum, positionxum, ...
COORD_RE = {
    axis: re.compile(
        rf"^(?:pos(?:ition)?|center|centre|stage)?{axis}"
        rf"(?:pos(?:ition)?)?(?:um|µm|micron|px|pixel)?$"
    )
    for axis in "xyz"
}
ID_KEYS = ("id", "spotid", "name", "label", "index")


@dataclass
class SetupInfo:
    """One view setup of a BDV XML.

    Attributes
    ----------
    index : int
        Position in the id-sorted setup list; MoBIE's ``channel`` value.
    name : str
        Setup name, e.g. ``P00082_405nm``.
    tile : str
        Name of its tile, empty if the XML has no tile attribute.
    channel : str
        Name of its channel, empty if the XML has no channel attribute.
    unit : str
        Unit of the voxel size, e.g. ``µm``.
    voxel : tuple of str
        Voxel size x, y, z, as written in the XML.
    size : tuple of str
        Image size in pixels x, y, z, as written in the XML.
    z_range : tuple of float
        Lowest and highest z the setup covers in world coordinates,
        through its registration at the first timepoint.
    """

    index: int
    name: str
    tile: str
    channel: str
    unit: str = ""
    voxel: tuple[str, str, str] = ("", "", "")
    size: tuple[str, str, str] = ("", "", "")
    z_range: tuple[float, float] = (0.0, 0.0)


@dataclass
class Source:
    """One dataset of the table: a BDV XML or a spots file.

    Attributes
    ----------
    path : Path
        The BDV XML, or the ``positions.xml`` of a spots dataset.
    kind : str
        ``intensities`` or ``spots``.
    view : str
        View name in MoBIE.
    sample : Path
        Folder that groups datasets for the 2D-in-3D placement.
    setups : list of SetupInfo
        Setups of an XML; empty for spots.
    spots : list of dict
        ``spot_id``, ``x``, ``y``, ``z`` (None if absent) of a spots set.
    dz : float
        z shift applied to put a 2D dataset (or z-less spots) in the
        middle of the sample's 3D reference.
    """

    path: Path
    kind: str
    view: str = ""
    sample: Path = Path()
    setups: list[SetupInfo] = field(default_factory=list)
    spots: list[dict] = field(default_factory=list)
    dz: float = 0.0

    @property
    def is_3d(self):
        """Whether any setup has more than one z plane."""
        return any(s.size[2] not in ("", "1") for s in self.setups)

    @property
    def z_range(self):
        """World z range covered by all setups together."""
        return (
            min(s.z_range[0] for s in self.setups),
            max(s.z_range[1] for s in self.setups),
        )


# --------------------------------------------------------------------------
# BDV XML
# --------------------------------------------------------------------------


def compose(a, b):
    """Compose two row-packed 3x4 affines, ``a`` after ``b``.

    Parameters
    ----------
    a, b : sequence of float
        12 values each, row-packed.

    Returns
    -------
    tuple of float
        The 12 values of ``a @ b``.
    """
    out = []
    for r in range(3):
        row = a[4 * r : 4 * r + 4]
        for c in range(4):
            v = sum(row[k] * b[4 * k + c] for k in range(3))
            out.append(v + (row[3] if c == 3 else 0.0))
    return tuple(out)


def registrations(root):
    """World transform of each setup at its first timepoint.

    BDV lists a view's transforms outermost first, so the world
    transform is their product in document order.

    Parameters
    ----------
    root : xml.etree.ElementTree.Element
        The ``SpimData`` element.

    Returns
    -------
    dict of int to tuple of float
        Setup id to its row-packed 3x4 affine.
    """
    found: dict[int, tuple[int, tuple]] = {}
    for reg in root.iter("ViewRegistration"):
        sid, tp = int(reg.get("setup")), int(reg.get("timepoint"))
        if sid in found and found[sid][0] <= tp:
            continue
        m = IDENTITY
        for vt in reg.findall("ViewTransform"):
            values = [float(v) for v in (vt.findtext("affine") or "").split()]
            if len(values) == 12:
                m = compose(m, values)
        found[sid] = (tp, m)
    return {sid: m for sid, (_, m) in found.items()}


def world_z_range(m, size):
    """Lowest and highest world z of an image box under an affine.

    Parameters
    ----------
    m : sequence of float
        Row-packed 3x4 affine.
    size : tuple of str
        Image size x, y, z.

    Returns
    -------
    tuple of float
    """
    try:
        dims = [max(int(v) - 1, 0) for v in size]
    except ValueError:
        dims = [0, 0, 0]
    zs = [
        m[8] * x + m[9] * y + m[10] * z + m[11]
        for x in (0, dims[0])
        for y in (0, dims[1])
        for z in (0, dims[2])
    ]
    return min(zs), max(zs)


def read_setups(xml_path: Path):
    """Read the view setups of a BDV XML.

    Parameters
    ----------
    xml_path : Path
        A BigDataViewer / BigStitcher XML.

    Returns
    -------
    list of SetupInfo or None
        The setups in id order, as MoBIE indexes them; None if the file
        is not a BDV XML.
    """
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None
    if root.tag != "SpimData":
        return None
    vs = root.find("SequenceDescription/ViewSetups")
    if vs is None:
        return None
    regs = registrations(root)

    # Attribute id -> name, per attribute kind (tile, channel...).
    attr_names: dict[str, dict[str, str]] = {}
    for attrs in vs.findall("Attributes"):
        kind = attrs.get("name", "")
        attr_names[kind] = {
            (el.findtext("id") or "").strip(): (
                el.findtext("name") or el.findtext("id") or ""
            ).strip()
            for el in attrs
        }

    setups = []
    for el in vs.findall("ViewSetup"):
        sid = int(el.findtext("id"))
        attrs = el.find("attributes")

        def attr(kind, attrs=attrs):
            if attrs is None:
                return ""
            value = (attrs.findtext(kind) or "").strip()
            return attr_names.get(kind, {}).get(value, value)

        size = (el.findtext("size") or "").split()
        voxel = (el.findtext("voxelSize/size") or "").split()
        size = tuple(size) if len(size) == 3 else ("", "", "")
        setups.append(
            (
                sid,
                SetupInfo(
                    index=0,
                    name=(el.findtext("name") or "").strip() or f"setup{sid}",
                    tile=attr("tile"),
                    channel=attr("channel"),
                    unit=(el.findtext("voxelSize/unit") or "").strip(),
                    voxel=tuple(voxel) if len(voxel) == 3 else ("", "", ""),
                    size=size,
                    z_range=world_z_range(regs.get(sid, IDENTITY), size),
                ),
            )
        )
    setups.sort(key=lambda t: t[0])
    for i, (_, s) in enumerate(setups):
        s.index = i
    return [s for _, s in setups]


def check_image_file(xml_path: Path):
    """Check that the data file a BDV XML points to exists.

    MoBIE opens the file named in the XML's ``ImageLoader`` (``<hdf5>``
    or ``<n5>``), not the one that happens to sit next to the XML, so a
    data file renamed after the XML was written fails only in MoBIE.

    Parameters
    ----------
    xml_path : Path
        A BDV XML.

    Returns
    -------
    str or None
        A description of the problem, None if the file exists or the
        loader is not file based.
    """
    root = ET.parse(xml_path).getroot()
    loader = root.find("SequenceDescription/ImageLoader")
    if loader is None:
        return None
    el = next((e for e in loader if e.tag in ("hdf5", "n5", "zarr")), None)
    if el is None or not (el.text or "").strip():
        return None
    ref = Path(el.text.strip().replace("\\", "/"))
    if el.get("type", "relative") == "relative":
        base = xml_path.parent
        bp = root.find("BasePath")
        if bp is not None and bp.get("type", "relative") == "relative":
            base = base / (bp.text or ".").strip()
        target = base / ref
    else:
        target = ref
    if target.exists():
        return None
    others = sorted(p.name for p in xml_path.parent.glob(f"*{ref.suffix}"))
    msg = (
        f"{xml_path.name} points to <{el.tag}> {el.text.strip()}, which "
        f"does not exist ({target})."
    )
    if others:
        msg += (
            f" {ref.suffix} files next to it: {', '.join(others)}. "
            f"Edit the <{el.tag}> line of the XML to the right name."
        )
    return msg


# --------------------------------------------------------------------------
# Spots
# --------------------------------------------------------------------------


def _key(name: str) -> str:
    return re.sub(r"[\s_\-.()\[\]]", "", name.rsplit("}", 1)[-1].lower())


def read_positions(path: Path):
    """Read spot coordinates from a ``positions.xml``.

    The format is not fixed, so every element is checked for x and y
    values, as attributes (``<Spot x="1" y="2"/>``) or as child elements
    (``<Position><X>1</X><Y>2</Y></Position>``), under the usual names
    (``x``, ``PosX``, ``PositionX``, ``X_um``, ...). The element type
    with the most such matches holds the spots.

    Parameters
    ----------
    path : Path
        The ``positions.xml``.

    Returns
    -------
    list of dict
        ``spot_id``, ``x``, ``y`` and ``z`` (None if absent) per spot.
    """
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        sys.exit(f"{path}: not readable as XML ({e})")

    def values(el):
        found = {}
        items = [(k, v) for k, v in el.attrib.items()]
        items += [(c.tag, c.text or "") for c in el if len(c) == 0]
        for k, v in items:
            k = _key(k)
            for axis, rx in COORD_RE.items():
                if axis not in found and rx.match(k):
                    try:
                        found[axis] = float(v.strip().replace(",", "."))
                    except ValueError:
                        pass
            if k in ID_KEYS and "spot_id" not in found:
                found["spot_id"] = v.strip()
        return found

    by_tag: dict[str, list[dict]] = {}
    for el in root.iter():
        v = values(el)
        if "x" in v and "y" in v:
            by_tag.setdefault(el.tag, []).append(v)
    if not by_tag:
        return []
    spots = max(by_tag.values(), key=len)
    for i, s in enumerate(spots, 1):
        s.setdefault("z", None)
        # MoBIE needs numeric, unique spot ids.
        sid = s.get("spot_id", "")
        s["spot_id"] = sid if sid.isdigit() else str(i)
    if len({s["spot_id"] for s in spots}) < len(spots):
        for i, s in enumerate(spots, 1):
            s["spot_id"] = str(i)
    return spots


# --------------------------------------------------------------------------
# Table
# --------------------------------------------------------------------------


def channel_color(channel: str) -> str:
    """MoBIE color for a channel, from the wavelength in its name.

    Parameters
    ----------
    channel : str
        Channel name, e.g. ``488nm``.

    Returns
    -------
    str
        ``Blue``, ``Green``, ``Red`` or ``Magenta`` by wavelength;
        ``White`` if the name holds no plausible wavelength.
    """
    m = re.search(r"\d{3}", channel)
    if not m or not 300 <= int(m.group()) <= 1000:
        return "White"
    nm = int(m.group())
    return next(color for limit, color in WAVELENGTH_COLORS if nm < limit)


def view_names(paths: list[Path], level: int = 1):
    """Name each dataset after a folder above it, adding detail if needed.

    Parameters
    ----------
    paths : list of Path
        The BDV XMLs and positions files.
    level : int, optional
        Which folder names the view: 1 is the folder holding the file,
        2 its parent, and so on.

    Returns
    -------
    dict of Path to str
        The folder name. Where two views would share it, the folders
        from there down to the file (``SampleA_Acquisition``), and the
        file name only if that still clashes.
    """

    def folder(x, lvl):
        parents = x.parents
        return parents[min(lvl, len(parents)) - 1].name

    def dedupe(names, longer):
        counts: dict[str, int] = {}
        for n in names.values():
            counts[n] = counts.get(n, 0) + 1
        return {x: longer(x) if counts[n] > 1 else n for x, n in names.items()}

    names = {x: folder(x, level) for x in paths}
    # Clashes: spell out the folders from the naming one down to the file
    # (SampleA_Acquisition), then, if still needed, add the file name.
    below = {
        x: "_".join(x.parent.parts[len(x.parent.parts) - level :])
        for x in paths
    }
    names = dedupe(names, lambda x: below[x])
    return dedupe(names, lambda x: f"{below[x]}_{x.stem}")


def pick_reference(sources, wanted: str | None):
    """The 3D dataset 2D data of a sample is centred on.

    Parameters
    ----------
    sources : list of Source
        Datasets of one sample.
    wanted : str or None
        Case-insensitive part of the reference's view name; by default
        "scan", and the only 3D dataset if none matches.

    Returns
    -------
    Source or None
        None if the sample has no 3D dataset.
    """
    stacks = [s for s in sources if s.kind == "intensities" and s.is_3d]
    if not stacks:
        return None
    key = (wanted or "scan").lower()
    named = [s for s in stacks if key in s.view.lower()]
    if named:
        return named[0]
    if wanted:
        print(
            f"WARNING: no 3D dataset matching '{wanted}' in {stacks[0].sample}"
        )
    return stacks[0]


def fmt(v: float) -> str:
    """Compact number for the table."""
    return f"{v:.6g}"


def main(argv=None):
    """Run from the command line.

    Parameters
    ----------
    argv : list of str, optional
        Arguments; defaults to ``sys.argv[1:]``.
    """
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("root", type=Path, help="folder to search for BDV XMLs")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        help="table to write, .tsv or .csv (default: <root>/mobie_collection.tsv)",
    )
    p.add_argument(
        "--name-level",
        type=int,
        default=1,
        help="folder that names a view: 1 = the file's folder (default), "
        "2 = its parent, ...",
    )
    p.add_argument(
        "--absolute",
        action="store_true",
        help="write absolute paths instead of paths relative to the table",
    )
    p.add_argument(
        "--contrast-limits",
        metavar="MIN,MAX",
        help="initial contrast for every channel, e.g. 100,5000, or "
        "'auto' (default: empty, MoBIE's own default)",
    )
    p.add_argument(
        "--reference",
        metavar="NAME",
        help="3D dataset that 2D datasets are centred on, by part of its "
        "view name (default: the one with 'scan' in it, else the only one)",
    )
    p.add_argument(
        "--spot-radius",
        type=float,
        default=10.0,
        help="radius of spots from positions.xml (default 10)",
    )
    p.add_argument(
        "--spot-scale",
        type=float,
        default=1.0,
        help="multiply spot coordinates by this, to match the images' "
        "space (default 1)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="list the views and preview the table, write nothing",
    )
    args = p.parse_args(argv)

    root = args.root.resolve()
    if not root.is_dir():
        sys.exit(f"Not a folder: {root}")
    out = (args.output or root / "mobie_collection.tsv").resolve()
    delimiter = "," if out.suffix.lower() == ".csv" else "\t"

    # Collect the BDV XMLs, then positions.xml files alone in a folder.
    sources: list[Source] = []
    for x in sorted(root.rglob("*.xml")):
        setups = read_setups(x)
        if setups:
            sources.append(Source(path=x, kind="intensities", setups=setups))
    bdv_folders = {s.path.parent for s in sources}
    for x in sorted(root.rglob("*")):
        if x.name.lower() == "positions.xml" and x.parent not in bdv_folders:
            spots = read_positions(x)
            if not spots:
                print(f"WARNING: no x/y coordinates recognised in {x}")
                continue
            for s in spots:
                for axis in "xyz":
                    if s[axis] is not None:
                        s[axis] *= args.spot_scale
            sources.append(Source(path=x, kind="spots", spots=spots))
    if not sources:
        sys.exit(f"No BDV XML or positions.xml below {root}")

    names = view_names([s.path for s in sources], args.name_level)
    for s in sources:
        s.view = names[s.path]
        s.sample = s.path.parent.parent

    # 2D datasets and z-less spots go to the middle of the 3D reference.
    by_sample: dict[Path, list[Source]] = {}
    for s in sources:
        by_sample.setdefault(s.sample, []).append(s)
    notes = []
    for members in by_sample.values():
        ref = pick_reference(members, args.reference)
        if ref is None:
            continue
        lo, hi = ref.z_range
        mid = (lo + hi) / 2
        for s in members:
            if s.kind == "intensities" and not s.is_3d:
                s.dz = mid - s.z_range[0]
                notes.append(
                    f"{s.view}: 2D, moved by z {fmt(s.dz)} to the middle "
                    f"of {ref.view} (z {fmt(lo)}..{fmt(hi)})"
                )
            elif s.kind == "spots" and any(p["z"] is None for p in s.spots):
                for p in s.spots:
                    if p["z"] is None:
                        p["z"] = mid
                notes.append(
                    f"{s.view}: spots without z put at z {fmt(mid)}, the "
                    f"middle of {ref.view}"
                )

    contrast = ""
    if args.contrast_limits == "auto":
        contrast = "auto"
    elif args.contrast_limits:
        lo, hi = (float(v) for v in args.contrast_limits.split(","))
        contrast = f"({lo:g},{hi:g})"

    def rel(path):
        if args.absolute:
            return path.as_posix()
        try:
            return path.relative_to(out.parent).as_posix()
        except ValueError:
            sys.exit(
                f"{path} is not below the table's folder {out.parent}; "
                "pass --absolute or write the table higher up."
            )

    rows = []
    problems = []
    spot_tables = {}
    for src in sources:
        print(f"  {src.view}  <- {src.path.relative_to(root)}")
        if src.kind == "spots":
            table = src.path.with_name("positions_spots.tsv")
            spot_tables[table] = src.spots
            xs = [p["x"] for p in src.spots]
            ys = [p["y"] for p in src.spots]
            print(
                f"      {len(src.spots)} spots, x {fmt(min(xs))}..{fmt(max(xs))}"
                f", y {fmt(min(ys))}..{fmt(max(ys))}, radius "
                f"{fmt(args.spot_radius)} -> {table.name}"
            )
            rows.append(
                {
                    "uri": rel(table),
                    "type": "spots",
                    "name": src.view,
                    "view": src.view,
                    "display": src.view,
                    "spot_radius": fmt(args.spot_radius),
                    "sample": src.sample.name,
                }
            )
            continue

        problem = check_image_file(src.path)
        if problem:
            problems.append(problem)
        affine = ""
        if src.dz:
            affine = "(" + ",".join(fmt(v) for v in IDENTITY[:11]) + ","
            affine += fmt(src.dz) + ")"
        for s in src.setups:
            rows.append(
                {
                    "uri": rel(src.path),
                    "type": "intensities",
                    "name": f"{src.view}_{s.name}",
                    "channel_name": s.channel,
                    # MoBIE reads "um"; BDV XMLs often say "µm".
                    "pixel_unit": "um" if s.unit in ("µm", "μm") else s.unit,
                    "pixel_size_x": s.voxel[0],
                    "pixel_size_y": s.voxel[1],
                    "pixel_size_z": s.voxel[2],
                    "num_pixels_x": s.size[0],
                    "num_pixels_y": s.size[1],
                    "num_pixels_z": s.size[2],
                    "channel": s.index,
                    "view": src.view,
                    "display": f"{src.view}_{s.channel}"
                    if s.channel
                    else src.view,
                    "color": channel_color(s.channel),
                    "contrast_limits": contrast,
                    "affine": affine,
                    "sample": src.sample.name,
                    "tile": s.tile,
                }
            )
        tiles = {st.tile for st in src.setups}
        chans = dict.fromkeys(st.channel for st in src.setups)
        lo, hi = src.z_range
        print(
            f"      {len(src.setups)} rows: {len(tiles)} tile(s) x "
            f"{len(chans)} channel(s): "
            + ", ".join(f"{c or 'unnamed'} ({channel_color(c)})" for c in chans)
            + f"; {'3D' if src.is_3d else '2D'}, z {fmt(lo)}..{fmt(hi)}"
        )

    if notes:
        print()
    for note in notes:
        print(f"  {note}")
    for problem in problems:
        print(f"\nWARNING: {problem}")
    if problems:
        print("MoBIE will fail to open these views until that is fixed.")

    n_views = len(sources)
    if args.dry_run:
        print(
            f"\nDry run: would write {out} ({n_views} views, {len(rows)} rows)"
            + (", replacing the existing file" if out.exists() else "")
            + (
                f", and {len(spot_tables)} spots table(s)"
                if spot_tables
                else ""
            )
            + "."
        )
        preview = 5
        print(f"First {min(preview, len(rows))} rows:")
        writer = csv.DictWriter(sys.stdout, COLUMNS, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows[:preview])
        return

    for table, spots in spot_tables.items():
        with open(table, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["spot_id", "x", "y", "z"])
            for s in spots:
                writer.writerow(
                    [s["spot_id"], fmt(s["x"]), fmt(s["y"]), fmt(s["z"] or 0)]
                )
        print(f"Wrote {table} ({len(spots)} spots)")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, COLUMNS, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)
    data_root = "PathsInTableAreAbsolute" if args.absolute else "UseTableFolder"
    print(
        f"Wrote {out} ({n_views} views, {len(rows)} rows).\n"
        "Open in Fiji: Plugins > MoBIE > Open > Open Collection Table..., "
        f"Data Root: {data_root}, Viewing mode: ThreeDimensional."
    )


if __name__ == "__main__":
    main()
