import argparse
import sys
from typing import Dict, Type, TYPE_CHECKING

from . import commands
from .reporter import TerminalReporter, Reporter

if TYPE_CHECKING:
    from .exercise import Exercise


def prepare_parser():
    parser = argparse.ArgumentParser(
        prog='grading',
        usage='./grading [-h|--help] [options] command [tests ...]',
        description='PPC grading tool',
        allow_abbrev=False,
    )

    parser.add_argument(
        "--file",
        default=None,
        help="specify a non-default file to use"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="print a line for every test, not just failures"
    )

    parser.add_argument(
        "--json",
        default=None,
        metavar="PATH",
        help="also write a JSON results file to PATH"
    )

    parser.add_argument(
        "--arch",
        default=None,
        metavar="SM",
        help="CUDA SM target (e.g. 90, sm_90, 9.0); "
    )

    subparsers = parser.add_subparsers(
        title='commands',
        required=False,
        dest='command')

    for command in COMMANDS:
        sub = subparsers.add_parser(command.name, help=command.help)
        if command.has_tests:
            sub.add_argument("tests", nargs="*", help="Test files to process")

    return parser


def command_from_name(command: str, exercise: "Exercise", reporter: Reporter,
                      options: "commands.Options") -> commands.Command:
    cmd_list: Dict[str, Type[commands.Command]] = {
        c.name: c for c in COMMANDS
    }

    if command not in cmd_list:
        raise argparse.ArgumentError(None, f'Unknown command `{command}`')

    return cmd_list[command](exercise, reporter, options)


def cli(exercise: "Exercise"):
    command_parser = prepare_parser()
    args = command_parser.parse_args()

    if args.command is None:
        command_parser.print_help()
        return

    if args.file:
        exercise.source = args.file

    options = commands.Options(
        verbose=args.verbose,
        json_path=args.json,
        sm_target=args.arch if args.arch is not None else exercise.arch,
    )

    reporter = TerminalReporter(color=True)
    command = command_from_name(args.command, exercise, reporter, options)
    tests = getattr(args, 'tests', None)
    passed = command.exec(
        tests=tests,
    )
    sys.exit(0 if passed else 1)

COMMANDS = [
    commands.PTXCommand,
    commands.SASSCommand,
    commands.CompileCommand,
    commands.TestCommand,
    commands.BenchmarkCommand,
    commands.SanitizerCommand,
    commands.ProfileCommand,
    commands.GradeCommand,
]
