"""Tests for workflow/scripts/status.py (how far a run has got)."""

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workflow" / "scripts"))

import status  # noqa: E402


def _markers(work, label, *, batches=4, done=0, merged=False, objects=None):
    run = work / label
    (run / "seg").mkdir(parents=True)
    (run / "tiles.json").write_text(json.dumps({"batches": [[]] * batches}))
    for i in range(done):
        (run / "seg" / f"{i}.done").touch()
    if merged:
        (run / "labels.done").touch()
        level0 = work / "image.zarr" / "labels" / label / "0"
        level0.mkdir(parents=True)
        (level0 / "zarr.json").write_text(
            json.dumps({"attributes": {"patchworks_n_objects": objects}})
        )
        (run / "seams.json").write_text(
            json.dumps(
                {"axes": {"2": {"seam_rate": 0.1, "interior_rate": 0.05}}}
            )
        )


def test_a_multi_run_part_way(tmp_path, capsys, monkeypatch):
    work = tmp_path / "results"
    (work / "image.zarr").mkdir(parents=True)
    (work / "image.zarr" / "zarr.json").write_text(
        json.dumps({"attributes": {"ome": {"multiscales": [{}]}}})
    )
    _markers(work, "nuclei", batches=4, done=4, merged=True, objects=1234)
    _markers(work, "cells", batches=10, done=3)
    (work / "n_c.xlsx").touch()
    common = tmp_path / "common.yaml"
    common.write_text(yaml.safe_dump({"work_dir": str(work)}))
    paths = []
    for name in ("nuclei", "cells", "cilia"):
        paths.append(tmp_path / f"{name}.yaml")
        paths[-1].write_text(yaml.safe_dump({"label_name": name}))
    multi = tmp_path / "multi.yaml"
    multi.write_text(
        yaml.safe_dump(
            {
                "common": str(common),
                "segmentations": [str(p) for p in paths],
                "relations": [
                    {"a": "nuclei", "b": "cells", "output": "n_c.xlsx"},
                    {"a": "cilia", "b": "cells", "output": "ci_c.xlsx"},
                ],
            }
        )
    )
    got = status.run_status(*status._configs(multi))
    stages = {s["label_name"]: s for s in got["segmentations"]}
    assert stages["nuclei"]["stage"] == "done"
    assert stages["nuclei"]["objects"] == 1234
    assert stages["nuclei"]["seams"] == {
        "x": {"seam_rate": 0.1, "interior_rate": 0.05}
    }
    assert stages["cells"]["stage"] == "segmenting"
    assert (stages["cells"]["batches_done"], stages["cells"]["batches"]) == (
        3,
        10,
    )
    assert stages["cilia"]["stage"] == "converted"
    assert got["converted"] is True and got["occupancy"] is False
    assert [r["files"] != [] for r in got["relations"]] == [True, False]
    assert got["bundle"] is None

    monkeypatch.setattr(sys, "argv", ["status.py", str(multi)])
    status.main()
    text = capsys.readouterr().out
    assert "3/10 batches" in text and "1,234 objects" in text
    assert (
        "seams x: 10.0%" in text and "relation cilia -> cells: not yet" in text
    )

    monkeypatch.setattr(sys, "argv", ["status.py", str(multi), "--json"])
    status.main()
    assert json.loads(capsys.readouterr().out)["work_dir"] == str(work)


def test_tiles_count_within_running_batches(tmp_path):
    """Tiles segmented so far: every tile of a finished batch, and those a
    running batch checkpointed (any method: segment_tile.py writes it)."""
    run = tmp_path / "cells"
    (run / "seg").mkdir(parents=True)
    stage = tmp_path / "stage.zarr"
    stage.mkdir()
    batches = [[0, 1, 2], [3, 4, 5], [6, 7]]
    (run / "tiles.json").write_text(
        json.dumps({"batches": batches, "target_path": str(stage)})
    )
    (run / "seg" / "0.done").touch()
    (stage / ".patchworks_segment_1.json").write_text(
        json.dumps({"indices": [3, 4, 5], "counts": {"3": 12, "4": 7}})
    )
    # A stale checkpoint (other tiles) does not count.
    (stage / ".patchworks_segment_2.json").write_text(
        json.dumps({"indices": [9], "counts": {"9": 1}})
    )
    got = status.segmentation_status(
        {"work_dir": str(tmp_path), "label_name": "cells"}
    )
    assert (got["tiles_done"], got["tiles"]) == (5, 8)
    assert got["stage"] == "segmenting"
    for i in (1, 2):
        (run / "seg" / f"{i}.done").touch()
    got = status.segmentation_status(
        {"work_dir": str(tmp_path), "label_name": "cells"}
    )
    assert got["stage"] == "merging" and got["tiles_done"] == 8


def test_one_config_not_started(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"work_dir": str(tmp_path / "w")}))
    got = status.run_status(*status._configs(cfg))
    assert got["segmentations"][0]["stage"] == "not started"
    assert got["converted"] is False
    assert "relations" not in got


def test_failures_say_what_went_wrong_and_what_to_do(tmp_path):
    work = tmp_path / "results"
    (work / "image.zarr").mkdir(parents=True)
    (work / "image.zarr" / "zarr.json").write_text(
        json.dumps({"attributes": {"ome": {"multiscales": [{}]}}})
    )
    _markers(work, "cells", batches=3, done=1)
    logs = work / "cells" / "logs" / "segment"
    logs.mkdir(parents=True)
    # Batch 0 failed once but is done since: not a failure any more.
    (logs / "0.log").write_text("Traceback ...\nRuntimeError: boom\n")
    (logs / "1.log").write_text(
        "[patchworks] tile 4\nTraceback (most recent call last):\n"
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB\n"
    )
    (logs / "2.log").write_text("[patchworks] tile 7 (1/4)\n")  # running
    # SLURM ended a job of batch 2 for its time limit.
    slurm = work / "cells" / ".snakemake" / ".snakemake" / "slurm_logs"
    killed = slurm / "rule_segment" / "batch_2" / "123.log"
    killed.parent.mkdir(parents=True)
    killed.write_text(
        "slurmstepd: error: *** JOB 123 CANCELLED AT 2026-10-13T03:00:00 "
        "DUE TO TIME LIMIT ***\n"
    )
    cfg = {"work_dir": str(work), "label_name": "cells"}
    found = {f["step"]: f for f in status.failures(cfg)}
    assert set(found) == {"segment job 1", "segment job 2 (SLURM)"}
    assert "CUDA out of memory" in found["segment job 1"]["error"]
    assert "GPU ran out of memory" in found["segment job 1"]["hint"]
    assert "time limit" in found["segment job 2 (SLURM)"]["hint"]

    (work / "logs").mkdir()
    (work / "logs" / "convert.log").write_text(
        "OSError: No space left on device\n"
    )
    assert status.shared_failures(cfg) == []  # converted, so not failed
    # Started, not finished: the root is there, the multiscales not yet.
    (work / "image.zarr" / "zarr.json").write_text("{}")
    (shared,) = status.shared_failures(cfg)
    assert "disk is full" in shared["hint"]

    text = status.format_status(
        {
            "work_dir": str(work),
            "segmentations": [
                {
                    **status.segmentation_status(cfg),
                    "failures": [found["segment job 1"]],
                }
            ],
            "failures": [shared],
        }
    )
    assert "FAILED convert" in text and "FAILED cells: segment job 1" in text
    assert "-> the GPU ran out of memory" in text


def test_explain_without_a_known_cause():
    line, hint = status.explain("step 1\nValueError: overlap too wide\n\n")
    assert line == "ValueError: overlap too wide" and hint is None


def test_failures_of_an_earlier_run_are_not_this_runs(tmp_path, monkeypatch):
    """A step log older than the run shown is an earlier run's failure,
    whose step has not run again yet: left out."""
    import os

    work = tmp_path / "results"
    _markers(work, "cells", batches=2, done=0)
    logs = work / "cells" / "logs" / "segment"
    logs.mkdir(parents=True)
    old, new = logs / "0.log", logs / "1.log"
    for log in (old, new):
        log.write_text("Traceback (most recent call last):\nRuntimeError: x\n")
    os.utime(old, (1_000_000, 1_000_000))
    cfg = {"work_dir": str(work), "label_name": "cells"}
    assert len(status.failures(cfg)) == 2
    monkeypatch.setenv("PATCHWORKS_STATUS_SINCE", "2000000")
    (only,) = status.failures(cfg)
    assert only["step"] == "segment job 1"


def test_a_seeded_segmentation_says_what_it_waits_for(tmp_path):
    """Grown from another's labels: it waits for their merge, and says so."""
    work = tmp_path / "w"
    (work / "image.zarr").mkdir(parents=True)
    cfg = {"work_dir": str(work), "label_name": "cyto", "seed_labels": "nuclei"}
    assert status.segmentation_status(cfg)["waits_for"] == "nuclei"
    nuclei = work / "image.zarr" / "labels" / "nuclei"
    nuclei.mkdir(parents=True)
    (nuclei / "zarr.json").write_text(
        json.dumps({"attributes": {"ome": {"multiscales": [{}]}}})
    )
    assert "waits_for" not in status.segmentation_status(cfg)


def test_a_multi_runs_snakemake_that_stopped(tmp_path, monkeypatch):
    """run_multi's record says a segmentation's Snakemake ended, and why."""
    from run_multi import RUN_STATE, SNAKEMAKE_OUT, write_run_state

    work = tmp_path / "w"
    run = work / "nuclei"
    run.mkdir(parents=True)
    write_run_state(run, state="running", pid=1, started=100.0)
    (run / SNAKEMAKE_OUT).write_text(
        "Building DAG of jobs...\n\nLockException:\n"
        "Error: Directory cannot be locked.\n"
    )
    cfg = {"work_dir": str(work), "label_name": "nuclei"}
    got = status.segmentation_status(cfg)["snakemake"]
    assert got["state"] == "running" and got["pid"] == 1
    assert got["last"][-1] == "Error: Directory cannot be locked."

    write_run_state(run, state="failed", exit=1, ended=200.0)
    got = status.segmentation_status(cfg)["snakemake"]
    assert got["state"] == "failed" and got["pid"] == 1 and got["exit"] == 1
    assert "cannot be locked" in got["error"] and "unlock" in got["hint"]
    assert "its Snakemake stopped (exit 1)" in status.format_status(
        {
            "work_dir": str(work),
            "segmentations": [status.segmentation_status(cfg)],
        }
    )

    # An earlier run's record is not this run's.
    monkeypatch.setenv(
        "PATCHWORKS_STATUS_SINCE",
        str(json.loads((run / RUN_STATE).read_text())["updated"] + 10),
    )
    assert status.segmentation_status(cfg)["snakemake"] is None


def test_a_killed_snakemake_says_so(tmp_path):
    """Killed (exit -9), its output ends mid-startup with no error line."""
    from run_multi import SNAKEMAKE_OUT, write_run_state

    run = tmp_path / "w" / "nuclei"
    run.mkdir(parents=True)
    write_run_state(run, state="running", pid=1, started=100.0)
    (run / SNAKEMAKE_OUT).write_text("Using workflow specific profile x\n")
    write_run_state(run, state="failed", exit=-9, ended=200.0)
    got = status.segmentation_status(
        {"work_dir": str(tmp_path / "w"), "label_name": "nuclei"}
    )["snakemake"]
    assert got["error"] == "killed by SIGKILL"
    assert "controller job ran out of memory" in got["hint"]


def test_objects_come_from_the_label_group(tmp_path):
    """The label group's n_objects is the final count (after the size
    filter); the launcher showed None where only that one was written."""
    work = tmp_path
    _markers(work, "cells", merged=True, objects=None)
    group = work / "image.zarr" / "labels" / "cells"
    (group / "zarr.json").write_text(
        json.dumps({"attributes": {"n_objects": 384076}})
    )
    cfg = {"work_dir": str(work), "label_name": "cells"}
    assert status.segmentation_status(cfg)["objects"] == 384076


def test_a_running_cellpose_job_is_not_failed(tmp_path):
    """Every Cellpose job logs PyTorch's "Memory errors (e.g. SEGFAULT)
    will occur ..." warning: any "error" counted, so every running job
    showed as failed."""
    work = tmp_path / "results"
    _markers(work, "cells", batches=2, done=0)
    logs = work / "cells" / "logs" / "segment"
    logs.mkdir(parents=True)
    running = (
        "2026-10-27 12:56:03,311 WARNING cellpose.vit: Could not import "
        "CPDINO, run `pip install ...` to use CPDINO model\n"
        "/x/cellpose/dynamics.py:541: UserWarning: Sparse invariant checks "
        "are implicitly disabled. Memory errors (e.g. SEGFAULT) will occur "
        "when operating on a sparse tensor ...\n"
        "  coo = torch.sparse_coo_tensor(pt, torch.ones(pt.shape[1]))\n"
        "2026-10-27 12:57:33,354 INFO cellpose.models: masks created\n"
        "[patchworks] tile 1948 (1/4): 547 label(s) in 255s\n"
    )
    (logs / "0.log").write_text(running)
    (logs / "1.log").write_text(
        running + "Traceback (most recent call last):\n"
        '  File "x.py", line 1\nValueError: overlap too wide\n'
    )
    cfg = {"work_dir": str(work), "label_name": "cells"}
    (failed,) = status.failures(cfg)
    assert failed["step"] == "segment job 1"
    assert failed["error"] == "ValueError: overlap too wide"


def test_a_relation_from_before_the_labels_changed_is_to_redo(tmp_path):
    """A re-segmented label image leaves its relations' workbooks from the
    earlier run in place: shown as done, they hid the relations being
    recomputed. Current only once written after both labels were merged."""
    import os

    work = tmp_path / "results"
    (work / "image.zarr").mkdir(parents=True)
    _markers(work, "nuclei", batches=1, done=1, merged=True, objects=5)
    _markers(work, "cells", batches=1, done=1, merged=True, objects=5)
    book = work / "n_c.xlsx"
    book.touch()
    os.utime(book, (1, 1))  # written long before the merges
    common = tmp_path / "common.yaml"
    common.write_text(yaml.safe_dump({"work_dir": str(work)}))
    paths = []
    for name in ("nuclei", "cells"):
        paths.append(tmp_path / f"{name}.yaml")
        paths[-1].write_text(yaml.safe_dump({"label_name": name}))
    multi = tmp_path / "multi.yaml"
    multi.write_text(
        yaml.safe_dump(
            {
                "common": str(common),
                "segmentations": [str(p) for p in paths],
                "relations": [
                    {"a": "nuclei", "b": "cells", "output": "n_c.xlsx"}
                ],
            }
        )
    )
    got = status.run_status(*status._configs(multi))
    assert got["relations"][0]["current"] is False
    assert "to redo" in status.format_status(got)
    book.touch()  # rewritten now
    got = status.run_status(*status._configs(multi))
    assert got["relations"][0]["current"] is True
    assert got["bundle_current"] is False  # no bundle yet
