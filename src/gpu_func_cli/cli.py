"""Public CLI entry point and top-level dispatch for gpu_func_cli."""

from __future__ import annotations

import sys

from .client import RestClient
from .commands import _cmd_custom, _cmd_exercise, _cmd_exercise_mode, _cmd_workers
from .constants import RC_SETUP
from .errors import CliError
from .parser import EXERCISE_MODES, build_parser as _build_parser
from .reports import _cmd_report


def main(argv: list[str] | None = None) -> int:
    # Single funnel for the whole CLI: any CliError raised deep in a journey
    # surfaces here as a message + its exit code (constants.RC_*).
    try:
        return _main(argv)
    except CliError as exc:
        print(f"gpu_func_cli: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("gpu_func_cli: interrupted", file=sys.stderr)
        return 130


def _main(argv: list[str] | None = None) -> int:
    parser = _build_parser()                       # parser.py: the argparse grammar
    args = parser.parse_args(argv)
    # Dispatch to one of four journeys (see flow.md). The first three submit a
    # job to a remote GPU worker; `report` stays local on an existing .ncu-rep.
    if args.command_name == "workers":
        client = RestClient.from_args(args)
        return _cmd_workers(client)                # read-only: list live workers
    if args.command_name == "exercise":
        return _cmd_exercise(args)                 # course exercise (vendored or --course-root)
    if args.command_name in EXERCISE_MODES:
        return _cmd_exercise_mode(args)            # top-level shortcut: cwd-detected exercise
    if args.command_name == "custom":
        return _cmd_custom(args)                   # arbitrary kernel (+ optional harness)
    if args.command_name == "report":
        return _cmd_report(args)                   # local: summary / feedback on a .ncu-rep
    parser.print_help()
    return RC_SETUP
