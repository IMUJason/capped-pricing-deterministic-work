from __future__ import annotations

import argparse
from pathlib import Path

from .provenance import index_local_miplib, prepare_vrptw_assets
from .reporting import make_core_figures
from .study import run_study_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan 2 VRPTW reproducibility CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare-data")
    prepare_parser.add_argument("--project-root", type=Path, required=True)

    run_parser = subparsers.add_parser("run-study")
    run_parser.add_argument("--config", type=Path, required=True)

    figure_parser = subparsers.add_parser("make-figures")
    figure_parser.add_argument("--summary", type=Path, required=True)
    figure_parser.add_argument("--iterations", type=Path, required=True)
    figure_parser.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "prepare-data":
        prepare_vrptw_assets(args.project_root)
        index_local_miplib(args.project_root)
        return

    if args.command == "run-study":
        run_study_from_config(args.config)
        return

    if args.command == "make-figures":
        make_core_figures(args.summary, args.iterations, args.output_dir)


if __name__ == "__main__":
    main()
