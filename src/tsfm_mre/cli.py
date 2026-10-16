"""Command-line interface: ``tsfm-mre run`` and ``tsfm-mre validate``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tsfm_mre._version import __version__


def _run(args: argparse.Namespace) -> int:
    from tsfm_mre.config import load_config
    from tsfm_mre.pipeline import run

    config = load_config(args.config)
    out_dir = Path(args.out) if args.out else Path("runs") / config.name
    run(
        config,
        out_dir,
        extra_predictions=args.predictions,
        log=lambda m: None if args.quiet else print(m),
    )
    return 0


def _validate(args: argparse.Namespace) -> int:
    from tsfm_mre.schema import read_predictions

    table = read_predictions(args.path)
    models = ", ".join(sorted(table["model"].unique()))
    print(
        f"{Path(args.path).name}: {len(table):,} valid forecasts "
        f"(models: {models}; horizons: {sorted(table['horizon'].unique().tolist())})"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Argument parser with the ``run`` and ``validate`` subcommands."""
    parser = argparse.ArgumentParser(
        prog="tsfm-mre",
        description="Reproducible evaluation of probabilistic forecasts for financial time series.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    run_cmd = commands.add_parser("run", help="run an experiment from a YAML configuration")
    run_cmd.add_argument("config", help="path to the experiment configuration")
    run_cmd.add_argument("--out", help="output directory (default: runs/<name>)")
    run_cmd.add_argument(
        "--predictions",
        nargs="+",
        default=[],
        metavar="FILE",
        help="prediction tables of further models to evaluate alongside the configured ones",
    )
    run_cmd.add_argument("--quiet", action="store_true", help="suppress progress messages")
    run_cmd.set_defaults(handler=_run)

    validate_cmd = commands.add_parser(
        "validate", help="check a prediction table against the schema"
    )
    validate_cmd.add_argument("path", help="prediction table (.csv, .csv.gz or .parquet)")
    validate_cmd.set_defaults(handler=_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns 0 on success and 2 on invalid input or I/O errors."""
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
