"""`sync` copies the pipeline's latest dated run, and that run's date names the outputs."""
from __future__ import annotations

import json

from multiaxis_eci import config
from multiaxis_eci.sync import latest_run, sync


def _pipeline(root, dates):
    for date in dates:
        for name, folder in config.INPUT_FILES.items():
            d = root / folder / date
            d.mkdir(parents=True, exist_ok=True)
            manifest = {"git_commit": f"c{date}", "built_at": f"{date}T0900+0200"}
            body = json.dumps(manifest) if name.endswith(".json") else f"run,{date}\n"
            (d / name).write_text(body)
    (root / "2_database" / "notes").mkdir()   # not a run
    return root


def test_sync_takes_the_latest_run_and_records_its_date(tmp_path):
    pipe = _pipeline(tmp_path / "pipe", ["20260908", "20261001", "20260930"])
    assert latest_run(pipe) == "20261001"
    prov = sync(pipe, tmp_path / "in")
    assert prov["pipeline_run_date"] == "20261001" and prov["pipeline_commit"] == "c20261001"
    assert (tmp_path / "in" / "models.csv").read_text() == "run,20261001\n"
    saved = json.loads((tmp_path / "in" / config.PROVENANCE_FILE).read_text())
    assert saved["pipeline_run_date"] == "20261001"


def test_data_tag_is_the_synced_run_date():
    prov = json.loads((config.INPUT_DIR / config.PROVENANCE_FILE).read_text())
    assert config.DATA_TAG == prov.get("pipeline_run_date", "")
