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
