"""Compare two benchmark runs as a Markdown table.

    python benchmarks/compare.py base.json head.json [--threshold 1.25]

Prints the table (append it to $GITHUB_STEP_SUMMARY in CI) and a GitHub
warning annotation per case that got slower than *threshold*. It never fails:
shared CI runners are too noisy for a hard gate, so this is for a human to
read on every pull request.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("head")
    ap.add_argument("--threshold", type=float, default=1.25)
    args = ap.parse_args()
    base = json.loads(Path(args.base).read_text())
    head = json.loads(Path(args.head).read_text())

    print("| case | base (s) | head (s) | head / base |")
    print("| --- | ---: | ---: | ---: |")
    slow = []
    for name in sorted(set(base) | set(head)):
        b, h = base.get(name), head.get(name)
        ratio = h / b if b and h else None
        mark = ""
        if ratio is not None and ratio > args.threshold:
            mark = " ⚠️"
            slow.append((name, ratio))
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
        r = "n/a" if ratio is None else f"{ratio:.2f}x{mark}"
        print(f"| {name} | {fmt(b)} | {fmt(h)} | {r} |")
    for name, ratio in slow:
        print(f"::warning title=benchmark::{name} is {ratio:.2f}x slower")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
