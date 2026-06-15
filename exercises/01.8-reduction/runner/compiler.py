import copy
import math
import re
import subprocess
from dataclasses import dataclass

MAX_COMPILER_OUTPUT = 30_000
DEFAULT_FLAGS = ['-std=c++20', '-Xptxas', '--warn-on-spills', "-lineinfo"]


@dataclass(frozen=True, slots=True)
class SmTarget:
    """A CUDA compute capability target.

    Accepts numeric (9.0, 90) or string ("90", "sm_90", "90a") forms,
    all normalized to an integer `version` == major*10 + minor.
    """
    version: int  # e.g. 90 for sm_90, 100 for sm_100, 121 for sm_121

    @property
    def needs_accelerated(self) -> bool:
        """Use "accelerated" features set for everything starting with hopper"""
        return self.version >= 90

    @property
    def gencode_flags(self) -> list[str]:
        """The -arch / -gencode flags for this target."""
        if self.needs_accelerated:
            v = f"{self.version}a"
            return ['-gencode', f'arch=compute_{v},code=sm_{v}']
        return [f'-arch=sm_{self.version}']

    @classmethod
    def parse(cls, spec: str) -> 'SmTarget':
        s = spec.strip().lower()
        s = re.sub(r'^(sm_)', '', s)
        # trailing 'a' accelerated marker is implied by family, ignore it
        s = s.rstrip('a')
        if '.' in s:
            major, _, minor = s.partition('.')
            return cls(int(major) * 10 + int(minor))
        if not s.isdigit():
            raise ValueError(f"unrecognized SM target: {spec!r}")
        return cls(int(s))


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

    def with_target(self, target: str) -> 'CompilerCommand':
        me = copy.deepcopy(self)
        me.flags.extend(SmTarget.parse(target).gencode_flags)
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
                                errors='replace')
        output = CompileResult(
            cmd,
            result.stdout[:MAX_COMPILER_OUTPUT],
            result.stderr[:MAX_COMPILER_OUTPUT],
            result.returncode)
    except subprocess.TimeoutExpired:
        raise CompileTimeout(cmd, timeout)

    return output


def nvcc(*sources: str, out: str = "a.out"):
    cmd = CompilerCommand("nvcc", list(sources), list(DEFAULT_FLAGS), [], out)
    return cmd
