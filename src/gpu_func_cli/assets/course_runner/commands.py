import glob
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from . import testing
from .compiler import nvcc, run_compile, CompilerCommand
from .profile import extract_curated_metrics
from .testing import parse_test_file, TestConfig, TestResult
from .results import (
    RunResult, write_json,
    parse_report, Report, TestReport, BenchmarkReport, ProfileReport,
)
from .reporter import Reporter

if TYPE_CHECKING:
    from .exercise import Exercise


_COMPILE_TIMEOUT = 20

# Exit codes
RC_OK       = 0
RC_COMPILE  = 1
RC_CRASH    = 2
RC_WRONG    = 3
RC_TIMEOUT  = 4
RC_SETUP    = 5


def _as_text(s) -> str:
    """TimeoutExpired.stdout/stderr may be bytes or str depending on encoding."""
    if s is None:
        return ""
    if isinstance(s, bytes):
        return s.decode("utf-8", "replace")
    return s


@dataclass
class Options:
    """Run-time options from the CLI, threaded into every command."""
    verbose: bool = False          # print per-test detail even on pass
    json_path: Optional[str] = None  # if set, write a JSON results file here


class Command:
    # specifies the name of this command for the CLI
    name: str
    # specifies the help text for the command
    help: str
    # does the command have a tests argument
    has_tests: bool = False
    # title message
    title: str

    def __init__(self, exercise: "Exercise", reporter: Reporter,
                 options: Optional[Options] = None):
        self.exercise = exercise
        self.reporter = reporter
        self.options = options or Options()

    def log(self, *args, **kwargs):
        self.reporter.log(*args, **kwargs)

    def exec(self, tests: List[str]) -> bool:
        self.log(self.title, 'title')
        return self._exec(tests)

    def _exec(self, tests: List[str]) -> bool:
        raise NotImplementedError()


def run_compile_with_reporter(command: CompilerCommand, reporter: Reporter):
    reporter.log_command(command.args)
    output = run_compile(command, _COMPILE_TIMEOUT)
    reporter.log_compile(output)
    return output


class CompileCommandBase(Command):
    """Base for commands that compile a source file and act on the result."""
    extension: str = ''

    def _out(self) -> str:
        return f"{self.exercise.name}.{self.extension}" if self.extension else self.exercise.name

    def _build_command(self, out: str) -> CompilerCommand:
        raise NotImplementedError()

    def _on_success(self, out: str, output) -> None:
        pass

    def _exec(self, tests: List[str]) -> bool:
        out = self._out()
        cmd = self._build_command(out)
        output = run_compile_with_reporter(cmd, self.reporter)
        if not output.is_success:
            return False
        self._on_success(out, output)
        return True


class PTXCommand(CompileCommandBase):
    name = 'ptx'
    help = 'Compile to PTX'
    title = 'Compiling to PTX'
    extension = 'ptx'

    def _build_command(self, out: str) -> CompilerCommand:
        return nvcc(self.exercise.source, out=out).with_flags('-ptx', '-src-in-ptx')

    def _on_success(self, out: str, output) -> None:
        self.reporter.record_ptx(out)


class SASSCommand(CompileCommandBase):
    name = 'sass'
    help = 'generate SASS assembly'
    title = 'Compiling to SASS assembly'
    extension = 'cubin'

    def _build_command(self, out: str) -> CompilerCommand:
        return nvcc(self.exercise.source, out=out).with_flags('-cubin')

    def _on_success(self, out: str, output) -> None:
        sass_path = f"{self.exercise.name}.sass"
        Path(sass_path).write_text(self._extract_assembly(out))
        self.reporter.record_sass(sass_path)

    def _extract_assembly(self, cubin: str) -> str:
        # --no-vliw: Conventional mode; disassemble paired instructions in normal syntax
        args = ["nvdisasm", "--life-range-mode", "count", "--no-vliw", cubin]
        return subprocess.check_output(args, encoding="utf-8", timeout=_COMPILE_TIMEOUT)


class CompileCommand(CompileCommandBase):
    help = 'Compile to executable'
    name = 'compile'
    title = 'Compiling executable'

    def _build_command(self, out: str) -> CompilerCommand:
        return (
            nvcc(self.exercise.source, *self.exercise.tester, out=out)
            .with_include_dirs(self.exercise.include_path()).with_flags("-Xcompiler", "-Og")
        )

    def _on_success(self, out: str, output) -> None:
        self.reporter.log_output_file("Created executable at", out)


class RunCommandBase(Command):
    has_tests = True
    run_flag: str
    launcher: List[str] = []
    compile_flags: List[str] = []
    default_tests: List[str] = []

    # returncode -> (failure label, RunResult status, exit code)
    _FAILURES = {
        RC_SETUP: ("invalid test spec", "setup-error", RC_SETUP),
    }

    def _crash_label(self) -> str:
        """What a nonzero exit means for this command (overridden by sanitizer)."""
        return "crashed"

    def _launcher(self, config: TestConfig) -> List[str]:
        """Command prefix wrapping the binary. Per-test so a command can inject
        run-specific args (e.g. a profiler's per-test --export path)."""
        return self.launcher

    def _after_run(self, test: Path) -> None:
        """Hook after a successful (returncode==0, passed) run. Default no-op;
        used by the profiler to record the exported .ncu-rep artifact."""
        pass

    def _report_failure(self, name, label, stdout="", stderr="") -> None:
        self.reporter.log(f"{name}: {label}", kind='error')
        if stdout:
            self.reporter.log(stdout)
        if stderr:
            self.reporter.log(stderr, kind='error')

    def exec(self, tests: List[str]) -> bool:
        tests = testing.resolve_tests(tests, self.default_tests)

        bin_path = self._compile()
        if bin_path is None:
            return False

        runs: List[RunResult] = []
        for test in tests:
            self._run_one(Path(test), bin_path, runs)

        if self.options.json_path:
            write_json(Path(self.options.json_path), self.name, bin_path, runs)
        return True

    def _compile(self) -> Optional[Path]:
        self.log('Compiling', 'title')
        bin_path = Path(self.exercise.name)
        command = (
            nvcc(self.exercise.source, *self.exercise.tester, out=str(bin_path))
            .with_include_dirs(self.exercise.include_path())
            .with_flags(*self.compile_flags)
        )
        output = run_compile_with_reporter(command, self.reporter)
        return bin_path if output.is_success else None

    def _run_one(self, test: Path, bin_path: Path, runs: List[RunResult]) -> None:
        """Run one test, append its RunResult, and exit on any failure.

        Owns the report file: a fresh temp path per test so a passing run never
        reads a previous run's stale payload. The child fopens it itself, so we
        read back by path rather than through a pre-opened handle.
        """
        config = parse_test_file(test)
        fd, report_path = tempfile.mkstemp(suffix=".report", prefix="report_")
        os.close(fd)
        try:
            env = dict(os.environ)
            env["REPORT_PATH"] = report_path
            args = self._launcher(config) + [f"./{bin_path}", self.run_flag, str(test)]
            self.reporter.log_command(args)

            try:
                result = subprocess.run(
                    args, timeout=config.timeout, encoding="utf-8",
                    capture_output=True, env=env,
                )
            except subprocess.TimeoutExpired as e:
                self._report_failure(
                    config.name, "timed out",
                    stdout=e.stdout or "", stderr=e.stderr or "",
                )
                self._flush_and_exit([RunResult.for_config(
                    test, status="timeout",
                    stdout=_as_text(e.stdout), stderr=_as_text(e.stderr),
                )], RC_TIMEOUT)

            with open(report_path, "r") as rf:
                report_text = rf.read()
        finally:
            try:
                os.unlink(report_path)
            except OSError:
                pass

        report = parse_report(report_text)

        if result.returncode != RC_OK:
            label, status, code = self._FAILURES.get(
                result.returncode, (self._crash_label(), "error", RC_CRASH))
            self._report_failure(config.name, label,
                                 stdout=result.stdout, stderr=result.stderr)
            runs.append(RunResult.for_config(
                test, status=status,
                stdout=result.stdout or "", stderr=result.stderr or "",
            ))
            self._flush_and_exit(runs, code)

        runs.append(RunResult.from_report(test, report, stdout=result.stdout or ""))
        self.report_result(config.name, config, report)
        if not report.passed:
            self._flush_and_exit(runs, RC_WRONG)
        self._after_run(test)

    def _flush_and_exit(self, runs: List[RunResult], code: int):
        if self.options.json_path:
            write_json(Path(self.options.json_path), self.name,
                       Path(self.exercise.name), runs)
        sys.exit(code)

    def report_result(self, name: str, config: TestConfig, report: Report) -> None:
        raise NotImplementedError()


class TestCommand(RunCommandBase):
    help = 'Run tests'
    name = 'test'
    run_flag = "--test"
    default_tests = ['tests/*']
    compile_flags = ["-Xcompiler", "-Og"]

    def report_result(self, name, config: TestConfig, report: Report) -> None:
        if report.passed:
            if self.options.verbose:
                self.reporter.log(f"{name}: passed")
        else:
            self.reporter.log(f"{name}: wrong answer", kind='error')
            assert isinstance(report, TestReport)
            detail = self.exercise.format_failure(report, config)
            if detail:
                self.reporter.log(detail, kind='error')


class BenchmarkCommand(RunCommandBase):
    help = 'Run benchmarks'
    name = 'benchmark'
    run_flag = "--benchmark"
    default_tests = ['benchmarks/*']
    compile_flags = ["-O3"]

    def report_result(self, name, config: TestConfig, report: Report) -> None:
        if report.passed:
            assert isinstance(report, BenchmarkReport)
            detail = self.exercise.format_benchmark(report, config)
            if detail:
                self.reporter.log(detail)
        else:
            self.reporter.log(f"{name}: wrong answer", kind='error')
            assert isinstance(report, TestReport)
            detail = self.exercise.format_failure(report, config)
            if detail:
                self.reporter.log(detail, kind='error')


class SanitizerCommand(RunCommandBase):
    help = 'Run compute-sanitizer'
    name = 'sanitizer'
    # --error-exitcode makes compute-sanitizer return nonzero when it detects
    # a violation. Without it, the tool prints the error but propagates the
    # child's exit code (0 in Test mode regardless of correctness), so a
    # memory error would be silently reported as a pass.
    launcher = ['compute-sanitizer', '--tool=memcheck', '--error-exitcode=2']
    run_flag = "--test"
    default_tests = ['tests/*']
    compile_flags = ["-Xcompiler", "-Og"]

    def _crash_label(self) -> str:
        return "sanitizer violation"

    def report_result(self, name, config: TestConfig, report: Report) -> None:
        if report.passed:
            if self.options.verbose:
                self.reporter.log(f"{name}: passed")
        else:
            self.reporter.log(f"{name}: wrong answer", kind='error')
            assert isinstance(report, TestReport)
            detail = self.exercise.format_failure(report, config)
            if detail:
                self.reporter.log(detail, kind='error')


class ProfileCommand(RunCommandBase):
    help = 'Run profiler'
    name = 'profile'
    launcher = [
        'ncu', '--set=full',
        '--nvtx', '--nvtx-include', 'profile_kernel/',
        '--force-overwrite',
    ]
    run_flag = "--profile"
    default_tests = ['benchmarks/*']
    compile_flags = ["-Xcompiler", "-Og"]

    def _export_base(self, name: str) -> str:
        """Path passed to ncu --export. ncu appends .ncu-rep itself, so this is
        the name without that suffix."""
        return f"{self.exercise.name}.{name}"

    def _launcher(self, config: TestConfig) -> List[str]:
        return self.launcher + ['--export', self._export_base(config.name)]

    def _after_run(self, test: Path) -> None:
        pass
        #rep = f"{self._export_base(test)}.ncu-rep"
        #if Path(rep).exists():
        #    self.reporter.log_output_file("Profile written to", rep)

    def report_result(self, name, config: TestConfig, report: Report) -> None:
        assert isinstance(report, ProfileReport)
        report_path = f"{self._export_base(config.name)}.ncu-rep"
        report.metrics, report.device = extract_curated_metrics(report_path)
        self.reporter.log_profile(report, report_path)
        self.reporter.log_profile_feedback(self.exercise.format_profiling(report, config))


class GradeCommand(Command):
    name  = "grade"
    help  = "Test, sanitize, and benchmark"
    title = "Grading"
    has_tests = False

    _steps = [TestCommand, SanitizerCommand, BenchmarkCommand]

    def _exec(self, tests: List[str]) -> bool:
        for cls in self._steps:
            cmd = cls(self.exercise, self.reporter, self.options)
            if not cmd.exec(tests):
                return False
        return True
