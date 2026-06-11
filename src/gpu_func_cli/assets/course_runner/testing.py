"""Config file discovery and conventions.

Conventions (fixed across all exercises):
  - Tests live in <exercise_root>/tests/*.txt
  - Benchmarks live in <exercise_root>/benchmarks/*.txt
  - A test is "slow" iff '-slow' appears in its stem (e.g. big-slow.txt)
"""
import glob
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


# Subdirectory names used by every exercise.
TESTS_SUBDIR     = "tests"
BENCH_SUBDIR     = "benchmarks"
SLOW_MARKER      = "-slow"
TEST_EXT         = ".txt"


def is_slow(cfg: Path) -> bool:
    """Whether a config is marked slow (skipped by sanitizers)."""
    return SLOW_MARKER in cfg.stem


def resolve_tests(globs: List[str], default: List[str]) -> List[str]:
    """Expand glob of tests, defaulting to default if no globs were provided."""
    if not globs:
        globs = default
    tests = []
    for pattern in globs:
        if os.path.exists(pattern):
            tests.append(pattern)
        else:
            tests.extend(sorted(glob.glob(pattern)))
    return tests


@dataclass
class TestConfig:
    path: Path
    name: str
    timeout: float
    slow: bool
    args: dict[str, list]


@dataclass
class TestResult:
    passed: bool
    info: dict[str, str]


def parse_test_file(path: Path) -> TestConfig:
    """Parse a test spec file (plain key=value text)."""
    cfg = TestConfig(path, "", 0.0, is_slow(path), {})
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            continue
        name, _, val = line.partition("=")
        name = name.strip()
        val = val.split()
        if name == "timeout":
            cfg.timeout = float(val[0])
        elif name == "name":
            cfg.name = val[0]
        else:
            cfg.args[name] = val[0] if len(val) == 1 else val
    return cfg
