"""Tests for the multi-config driver's SLURM-facing behaviour."""

import re
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "workflow" / "scripts")
)

from run_multi import slurm_jobname_prefix  # noqa: E402

# The SLURM executor's own rule (snakemake_executor_plugin_slurm): it raises a
# WorkflowError and aborts the whole run if the prefix does not match.
_EXECUTOR_RULE = re.compile(r"^[A-Za-z0-9_-]{1,50}$")


def test_jobname_prefix_satisfies_the_executor():
    """Whatever a label_name contains, the prefix must stay submittable.

    The executor names jobs after a UUID and refuses a --job-name override, so
    this prefix is the only thing that makes squeue readable -- and an invalid
    one fails the run rather than degrading.
    """
    for label in ("nuclei_labels", "cyto_labels", "convert", "a"):
        assert _EXECUTOR_RULE.match(slurm_jobname_prefix(label))

    # Characters a label might plausibly pick up are sanitised, not passed on.
    assert _EXECUTOR_RULE.match(slurm_jobname_prefix("cilia/v2 (test)"))
    assert _EXECUTOR_RULE.match(slurm_jobname_prefix("run 1: nuclei"))
    # And an over-long label is truncated to the executor's 50-char limit.
    assert _EXECUTOR_RULE.match(slurm_jobname_prefix("x" * 200))


def test_jobname_prefix_keeps_the_label_readable():
    """The label must lead, since that is what a queue listing truncates to."""
    assert slurm_jobname_prefix("nuclei_labels") == "pw-nuclei_labels"
    assert slurm_jobname_prefix("convert") == "pw-convert"
