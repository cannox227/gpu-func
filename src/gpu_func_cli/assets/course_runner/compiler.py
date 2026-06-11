import copy
import math
import subprocess
from dataclasses import dataclass

MAX_COMPILER_OUTPUT = 30_000
DEFAULT_FLAGS = ['-std=c++20', '-Xptxas', '--warn-on-spills', "-lineinfo"]


@dataclass(slots=True)
class CompilerCommand:
    program: str
    sources: list[str]
    flags:   list[str]
    libs:    list[str]
    out:     str

    @property
    def args(self) -> list[str]:
        return [self.program] + self.flags + self.sources + ['-o', self.out] + self.libs

    def with_sources(self, *sources) -> 'CompilerCommand':
        me = copy.deepcopy(self)
        me.sources.extend(sources)
        return me

    def with_flags(self, *flags) -> 'CompilerCommand':
        me = copy.deepcopy(self)
        me.flags.extend(flags)
        return me

    def with_libs(self, *libs) -> 'CompilerCommand':
        me = copy.deepcopy(self)
        me.libs.extend(libs)
        return me

    def with_include_dirs(self, *dirs) -> 'CompilerCommand':
        me = copy.deepcopy(self)
        for d in dirs:
            me.flags.append("-I" + d)
        return me


@dataclass(slots=True)
class CompileResult:
    cmd: CompilerCommand
    stdout: str
    stderr: str
    exit_code: int

    @property
    def is_success(self) -> bool:
        return self.exit_code == 0


@dataclass(slots=True)
class CompileTimeout(Exception):
    cmd: CompilerCommand
    timeout: float


def run_compile(cmd: CompilerCommand, timeout: float) -> CompileResult:
    args = cmd.args
    # subprocess.run cannot handle infinite timeouts, needs explicit None
    if not math.isfinite(timeout):
        timeout = None

    try:
        result = subprocess.run(args,
                                timeout=timeout,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                encoding='utf-8',
                                errors='utf-8')
        output = CompileResult(
            cmd,
            result.stdout[:MAX_COMPILER_OUTPUT],
            result.stderr[:MAX_COMPILER_OUTPUT],
            result.returncode)
    except subprocess.TimeoutExpired:
        raise CompileTimeout(cmd, timeout)

    return output


def nvcc(*sources: str, out: str = "a.out"):
    return CompilerCommand("nvcc", list(sources), DEFAULT_FLAGS, [], out)
