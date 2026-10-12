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

Columns written (see https://mobie.github.io/tutorials/mobie_collection_table.html):

- ``uri``: the XML, relative to the table (or absolute, ``--absolute``)
- ``channel``: setup index in the XML
- ``name``: ``<folder>_<setup name>``, unique per row
- ``view``: the folder name -- all rows of one XML are shown together
- ``display``: ``<folder>_<channel>``, shared by all tiles of a channel
- ``color``: from the channel's wavelength (405 blue, 488 green, ...)
- ``contrast_limits``: ``auto`` unless given with ``--contrast-limits``
- ``exclusive``: ``true``, so opening a view replaces the previous one
- ``sample``, ``tile``, ``channel_name``: annotations for browsing

Usage
-----
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
from dataclasses import dataclass
from pathlib import Path

COLUMNS = [
    "uri",
    "channel",
    "name",
    "view",
    "display",
    "color",
    "contrast_limits",
    "exclusive",
    "sample",
    "tile",
    "channel_name",
]

# (upper wavelength bound in nm, MoBIE color)
WAVELENGTH_COLORS = [
    (460, "blue"),
    (530, "green"),
    (600, "red"),
    (10_000, "magenta"),
]


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
    """

    index: int
    name: str
    tile: str
    channel: str


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

        name = (el.findtext("name") or "").strip()
        setups.append((sid, name, attr("tile"), attr("channel")))
    setups.sort()
    return [
        SetupInfo(index=i, name=name or f"setup{sid}", tile=tile, channel=ch)
        for i, (sid, name, tile, ch) in enumerate(setups)
    ]


def channel_color(channel: str) -> str:
    """MoBIE color for a channel, from the wavelength in its name.

    Parameters
    ----------
    channel : str
        Channel name, e.g. ``488nm``.

    Returns
    -------
    str
        ``blue``, ``green``, ``red`` or ``magenta`` by wavelength;
        ``white`` if the name holds no plausible wavelength.
    """
    m = re.search(r"\d{3}", channel)
    if not m or not 300 <= int(m.group()) <= 1000:
        return "white"
    nm = int(m.group())
    return next(color for limit, color in WAVELENGTH_COLORS if nm < limit)


def view_names(xmls: list[Path], level: int = 1):
    """Name each XML after a folder above it, adding detail if needed.

    Parameters
    ----------
    xmls : list of Path
        The BDV XMLs.
    level : int, optional
        Which folder names the view: 1 is the folder holding the XML,
        2 its parent, and so on.

    Returns
    -------
    dict of Path to str
        The folder name. Where two views would share it, the folders
        from there down to the XML (``SampleA_Acquisition``), and the
        XML's file name only if that still clashes.
    """

    def folder(x, lvl):
        parents = x.parents
        return parents[min(lvl, len(parents)) - 1].name

    def dedupe(names, longer):
        counts: dict[str, int] = {}
        for n in names.values():
            counts[n] = counts.get(n, 0) + 1
        return {x: longer(x) if counts[n] > 1 else n for x, n in names.items()}

    names = {x: folder(x, level) for x in xmls}
    # Clashes: spell out the folders from the naming one down to the XML
    # (SampleA_Acquisition), then, if still needed, add the file name.
    below = {
        x: "_".join(x.parent.parts[len(x.parent.parts) - level :]) for x in xmls
    }
    names = dedupe(names, lambda x: below[x])
    return dedupe(names, lambda x: f"{below[x]}_{x.stem}")


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
        help="folder that names a view: 1 = the XML's folder (default), "
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
        help="initial contrast for every channel, e.g. 100,5000 "
        "(default: auto)",
    )
    p.add_argument(
        "--not-exclusive",
        action="store_true",
        help="opening a view adds to what is shown instead of replacing it",
    )
    args = p.parse_args(argv)

    root = args.root.resolve()
    if not root.is_dir():
        sys.exit(f"Not a folder: {root}")
    out = (args.output or root / "mobie_collection.tsv").resolve()
    delimiter = "," if out.suffix.lower() == ".csv" else "\t"

    xmls = []
    for x in sorted(root.rglob("*.xml")):
        setups = read_setups(x)
        if setups:
            xmls.append((x, setups))
    if not xmls:
        sys.exit(f"No BDV XML below {root}")

    contrast = "auto"
    if args.contrast_limits:
        lo, hi = (float(v) for v in args.contrast_limits.split(","))
        contrast = f"({lo:g},{hi:g})"

    names = view_names([x for x, _ in xmls], args.name_level)
    rows = []
    for x, setups in xmls:
        view = names[x]
        if args.absolute:
            uri = x.as_posix()
        else:
            try:
                uri = x.relative_to(out.parent).as_posix()
            except ValueError:
                sys.exit(
                    f"{x} is not below the table's folder {out.parent}; "
                    "pass --absolute or write the table higher up."
                )
        h5 = list(x.parent.glob(f"{x.stem}*.h5"))
        if not h5:
            print(f"WARNING: no .h5 next to {x}")
        for s in setups:
            rows.append(
                {
                    "uri": uri,
                    "channel": s.index,
                    "name": f"{view}_{s.name}",
                    "view": view,
                    "display": f"{view}_{s.channel}" if s.channel else view,
                    "color": channel_color(s.channel),
                    "contrast_limits": contrast,
                    "exclusive": "false" if args.not_exclusive else "true",
                    "sample": view,
                    "tile": s.tile,
                    "channel_name": s.channel,
                }
            )
        print(f"  {view}: {len(setups)} setups ({x.relative_to(root)})")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, COLUMNS, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)
    data_root = "PathsInTableAreAbsolute" if args.absolute else "UseTableFolder"
    print(
        f"Wrote {out} ({len(xmls)} views, {len(rows)} rows).\n"
        "Open in Fiji: Plugins > MoBIE > Open > Open Collection Table..., "
        f"Data Root: {data_root}, Viewing mode: ThreeDimensional."
    )


if __name__ == "__main__":
    main()
