"""Command line: `python -m multiaxis_eci sync [--pipeline PATH]`."""
from __future__ import annotations

import argparse
from pathlib import Path

from multiaxis_eci import config
from multiaxis_eci.sync import sync


def main() -> None:
    parser = argparse.ArgumentParser(prog="multiaxis_eci")
    sub = parser.add_subparsers(dest="command", required=True)
    p_sync = sub.add_parser("sync", help="copy the pipeline's views and tables into 0_input/")
    p_sync.add_argument("--pipeline", type=Path, default=config.PIPELINE_DIR,
                        help="benchmark-data-pipeline checkout (default: sibling directory)")
    args = parser.parse_args()
    if args.command == "sync":
        provenance = sync(args.pipeline)
        print(f"0_input/ now mirrors pipeline commit {provenance['pipeline_commit'][:8]} "
              f"built {provenance['pipeline_built_at']}")


if __name__ == "__main__":
    main()
