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


import re
import unicodedata

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _display_width(s: str) -> int:
    """Visible column width of a string in a monospace terminal.

    Strips ANSI SGR escapes, counts East-Asian wide/fullwidth chars and
    emoji as 2 columns, zero-width/combining marks as 0, everything else
    as 1.
    """
    s = _ANSI_RE.sub("", s)
    width = 0
    for ch in s:
        cp = ord(ch)
        if unicodedata.combining(ch) or cp == 0xFE0F:
            continue
        if (
                unicodedata.east_asian_width(ch) in ("W", "F")
                or 0x1F300 <= cp <= 0x1FAFF   # misc symbols, emoji, supplemental
                or 0x2600 <= cp <= 0x27BF     # misc symbols + dingbats (🚀 lives at 1F680, ⚠ at 26A0)
                or cp == 0xFE0F               # variation selector-16 (emoji presentation)
        ):
            width += 2
        else:
            width += 1
    return width


def gutter_indent(text: str, marker: str = "", gutter: int = 6) -> str:
    """Indent `text` to a fixed display column, placing `marker` in the gutter.

    The first line gets `marker` left-justified in a `gutter`-wide column
    (measured in display columns, so ANSI codes and double-width emoji
    don't throw off alignment); continuation lines are indented with plain
    spaces to the same column. Returns the assembled multi-line string.

        gutter_indent("line one\nline two", "🚀", gutter=6)
        ->  "🚀    line one\n      line two"

    With marker="" it degenerates to a plain fixed indent, so it also
    works for feedback kinds that have no prefix.
    """
    pad = max(0, gutter - _display_width(marker))
    body_indent = " " * gutter

    head, *rest = text.split("\n")
    lines = [marker + (" " * pad) + head]
    lines += [body_indent + r if r else r for r in rest]
    return "\n".join(lines)


def _bar(pct: float, width: int = 20) -> str:
    """Unicode meter for a 0–100(+) percentage, color-ramped by level.
    Clamps the fill to `width`; values >100 still color green but can't
    overflow the bar."""
    frac = max(0.0, min(pct, 100.0)) / 100.0
    filled = round(frac * width)
    return f"{'█' * filled}{'░' * (width - filled)}"


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
        dram_throughput = aggregate["dram_throughput"]
        sm_throughput = aggregate["sm_throughput"]
        occupancy = aggregate["occupancy"]

        self.log(f" Duration:            {aggregate['duration']/1000:.1f} µs")
        self.log(f" Cycles:              {int(aggregate['cycles']):,}")
        self.log(f" Instructions:        {int(aggregate['instructions']):,}")
        self.log(f" Total bytes read:    {total_bytes_read:.1f} MiB")
        self.log(f" Total bytes written: {total_bytes_written:.1f} MiB")
        self.log(f" DRAM throughput:     {_bar(dram_throughput)} {dram_throughput:.1f}%")
        self.log(f" SM throughput:       {_bar(sm_throughput)} {sm_throughput:.1f}%")
        self.log(f" Occupancy:           {_bar(occupancy)} {occupancy:.1f}%")

        self.log_sep()
        self.log_output_file("For the full report, see", report_path)
        self.log_sep()

    def log_profile_feedback(self, feedback: list[ProfileFeedback]):
        if len(feedback) == 0:
            return

        self.log_sep()
        self.log("Profiling Feedback", "heading")
        for f in feedback:
            prefix = ""
            if f.kind == "warning":
                prefix = " \033[31;1m⚠️\033[0m"
            elif f.kind == "praise":
                prefix = " 🚀"
            elif f.kind == "technique":
                prefix = " 🔧"
            self.log(gutter_indent(f.text, prefix, gutter=4))

        self.log_sep()
