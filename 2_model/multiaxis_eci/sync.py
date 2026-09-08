"""Copy the tables of benchmark-data-pipeline into 0_input/, with their provenance."""
from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from multiaxis_eci.config import INPUT_DIR, INPUT_FILES, PIPELINE_DIR, PROVENANCE_FILE


def sync(pipeline_dir: Path = PIPELINE_DIR, input_dir: Path = INPUT_DIR) -> dict:
    """Copy the views and tables listed in INPUT_FILES, write provenance.json, return it."""
    pipeline_dir = Path(pipeline_dir)
    input_dir.mkdir(exist_ok=True)
    for name, rel in INPUT_FILES.items():
        src = pipeline_dir / rel
        if not src.exists():
            raise FileNotFoundError(
                f"{src}: run `python -m benchmark_data build` in the pipeline first")
        shutil.copyfile(src, input_dir / name)
        print(f"copied {src} ({src.stat().st_size} bytes)")
    manifest = json.loads((input_dir / "build_manifest.json").read_text())
    provenance = {
        "pipeline_dir": str(pipeline_dir),
        "pipeline_commit": manifest.get("git_commit", ""),
        "pipeline_built_at": manifest.get("built_at", ""),
        "schema_version": manifest.get("schema_version", ""),
        "synced_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": list(INPUT_FILES),
    }
    (input_dir / PROVENANCE_FILE).write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance
