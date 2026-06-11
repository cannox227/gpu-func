"""
Reporter is responsible for gathering and reporting the results of a run
"""
import shlex
import sys
from pathlib import Path

from .profile import aggregate_profiling_info
from .compiler import CompileResult
from .results import ProfileReport, ProfileFeedback


def file_link(path, label=None):
    label = label or str(path)
    uri = Path(path).resolve().as_uri()
    return f"\033]8;;{uri}\a{label}\033]8;;\a"


class Reporter:
    def log(self, msg: str, kind=None):
        raise NotImplementedError()

    def log_sep(self):
        raise NotImplementedError()

    def finalize(self):
        raise NotImplementedError()

    def log_output_file(self, message: str, path: str) -> None:
        raise NotImplementedError()

    def log_compile(self, result: CompileResult):
        raise NotImplementedError()

    def log_profile(self, result: ProfileReport, report_path: str):
        raise NotImplementedError()

    def log_profile_feedback(self, feedback: list[ProfileFeedback]):
        raise NotImplementedError()

    def record_ptx(self, path: str) -> None:
        self.log_output_file("Generated PTX at", path)

    def record_sass(self, path: str) -> None:
        self.log_output_file("Generated SASS at", path)

    def log_command(self, args: list[str]):
        raise NotImplementedError()


class TerminalReporter(Reporter):
    def __init__(self, color=True):
        self.color: bool = color or sys.stdout.isatty()
        self.sep_printed = False

    def log_sep(self):
        if not self.sep_printed:
            print()
            self.sep_printed = True

    def _color_text(self, text: str, *codes: str) -> str:
        reset = '\033[0m'
        begin = "".join(codes)
        if not self.color or len(begin) == 0:
            return text
        return begin + text + reset

    def log(self, msg: str, kind=None):
        msg = msg.rstrip()
        code = ''
        if kind is not None:
            code = {
                'title': '\033[34;1m',
                'heading': '\033[1m',
                'error': '\033[31;1m',
                'pass': '\033[34m',
                'command': '\033[34m',
                'output': '\033[34m',
            }.get(kind, '')
        print(self._color_text(msg, code))
        self.sep_printed = False

    def log_output_file(self, message: str, path: str) -> None:
        if self.color:
            print(f"{message} {file_link(path)}", flush=True)
        else:
            print(f"{message} {path}")

    def log_compile(self, result: CompileResult):
        if not result.is_success:
            self.log_command(result.cmd.args)
            self.log(result.stderr, kind='error')

    def log_command(self, args: list[str]):
        msg = ">> " + " ".join(shlex.quote(a) for a in args)
        print(self._color_text(msg, "\033[34m"), flush=True)

    def log_profile(self, result: ProfileReport, report_path: str):
        self.log_sep()
        self.log("Profiling results summary", "heading")

        aggregate = aggregate_profiling_info(result.metrics)

        total_bytes_read = aggregate["dram_read_bytes"] / 1024 / 1024
        total_bytes_written = aggregate["dram_write_bytes"] / 1024 / 1024

        self.log(f" Duration:            {aggregate['duration']/1000:.1f} µs")
        self.log(f" Cycles:              {int(aggregate['cycles']):,}")
        self.log(f" Instructions:        {int(aggregate['instructions']):,}")
        self.log(f" Total bytes read:    {total_bytes_read:.1f} MiB")
        self.log(f" Total bytes written: {total_bytes_written:.1f} MiB")

        self.log_sep()
        self.log_output_file("For the full report, see", report_path)
        self.log_sep()

    def log_profile_feedback(self, feedback: list[ProfileFeedback]):
        if len(feedback) == 0:
            return

        self.log_sep()
        self.log("Profiling Feedback", "heading")
        for f in feedback:
            prefix, kind = " ", ""
            if f.kind == "warning":
                kind = "error"
            elif f.kind == "praise":
                kind = "pass"

            self.log(prefix + f.text, kind=kind)

        self.log_sep()
