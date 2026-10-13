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
    (work / "image.zarr" / "zarr.json").write_text("{}")
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


def test_one_config_not_started(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"work_dir": str(tmp_path / "w")}))
    got = status.run_status(*status._configs(cfg))
    assert got["segmentations"][0]["stage"] == "not started"
    assert "relations" not in got
