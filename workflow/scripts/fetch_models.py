"""Download every model a run segments with, here and now.

    pixi run fetch config/local/my_run.yaml           # one segmentation
    pixi run -e plantseg fetch my_multi.yaml          # several

Run it on the login node (it has internet access; GPU nodes often have
none) before a run whose Snakemake controller runs as a job of its own.
Cellpose's weights and PlantSeg's U-Nets are cached on the shared home,
where the segment jobs read them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check import load  # noqa: E402
from run_multi import _config_bases, _resolve  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="a run config, or a multi config")
    args = parser.parse_args()
    from _pw import fetch_models

    workflow_dir = Path(__file__).resolve().parent.parent
    path = _resolve(args.config, *_config_bases(workflow_dir))
    if not path.is_file():
        sys.exit(f"[fetch] ERROR: {args.config} not found")
    _, cfgs, _ = load(path)
    failed = False
    for cfg in cfgs:
        label = cfg.get("label_name", "labels")
        try:
            print(f"[fetch] {label}: {fetch_models(cfg)}", flush=True)
        except Exception as exc:  # reported, the other configs go on
            print(f"[fetch] {label}: FAILED: {exc}", file=sys.stderr)
            failed = True
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
