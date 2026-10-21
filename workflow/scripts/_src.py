"""Import patchworks from this clone's ``src/``, not the installed copy.

The workflow's pixi environment installs this clone's patchworks as an
editable package whose version comes from git, so every ``git pull`` makes
the next pixi call reinstall it. For that moment nothing can import
patchworks: a cluster job or Snakemake starting then fails at once with
"No module named 'patchworks'". The source itself never goes away, so
reading it straight from ``src/`` closes that gap.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"


def use_clone_src() -> None:
    """Put ``src/`` first on this process's path and its children's.

    Children get it through ``PYTHONPATH``: Snakemake's jobs, the scripts
    its rules run, and the SLURM jobs it submits (sbatch passes the
    environment on).
    """
    if not (SRC / "patchworks").is_dir():
        return
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    paths = (os.environ.get("PYTHONPATH") or "").split(os.pathsep)
    if str(SRC) not in paths:
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [str(SRC), *(p for p in paths if p)]
        )
