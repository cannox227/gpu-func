"""Argparse construction for the public gpu_func_cli command surface."""

from __future__ import annotations

import argparse
import os

# The exercise actions, usable both as `exercise <id> <mode>` and as a
# top-level `gpu_func_cli <mode>` that auto-detects the exercise from the cwd.
EXERCISE_MODES = ["compile", "test", "benchmark", "sanitizer", "profile", "grade"]


def _add_common_exercise_opts(p: argparse.ArgumentParser) -> None:
    """Options shared by `exercise` and every top-level mode command.

    Kept identical between the two surfaces so they behave the same; the only
    difference is how the exercise is located (positional id vs. cwd auto-detect).
    """
    p.add_argument("specs", nargs="*")
    p.add_argument("--file", dest="source_file")
    p.add_argument(
        "--course-root",
        default=os.environ.get("CUDA_COURSE_REPO"),
        help="path to a cuda-course checkout (or set CUDA_COURSE_REPO). "
        "Default: auto-detect from --file / the cwd.",
    )
    p.add_argument(
        "--exercise-dir",
        help="path to a flat exercise dir (run.py + runner/ side by side, e.g. an "
        "unzipped exercise). Runs it directly, bypassing the cuda-course layout.",
    )
    p.add_argument("--gpu", default="B200")
    p.add_argument("--gpu-type")
    p.add_argument("--image", default="cuda-nvcc")
    p.add_argument("--arch")
    p.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="wall-clock budget (seconds) for the WHOLE remote run -- compile plus "
        "every test/benchmark/profile run together. Separate from the per-spec "
        "`timeout=` inside each benchmark/test file, which bounds a single binary "
        "run. Default: 600.",
    )
    p.add_argument("--wait-timeout", type=float)
    p.add_argument("--json", dest="json_path")
    p.add_argument("--artifact-dir")
    p.add_argument("--ncu-args", default="--set basic")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--report-max-mismatches", type=int, default=20)
    p.add_argument("--keep-going", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpu_func_cli")
    parser.add_argument("--api-base", default=os.environ.get("GFAAS_API_BASE"))
    parser.add_argument("--api-key", default=os.environ.get("GFAAS_API_KEY"))
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)

    sub = parser.add_subparsers(dest="command_name")

    sub.add_parser("workers", help="List live GFAAS workers")

    exercise = sub.add_parser("exercise", help="Run a course exercise action")
    exercise.add_argument("exercise_id")
    exercise.add_argument("exercise_command", choices=EXERCISE_MODES)
    _add_common_exercise_opts(exercise)

    # Top-level shortcuts: `gpu_func_cli benchmark [specs...]` auto-detects the
    # exercise from the cwd (an unzipped exercise: run.py + runner/ siblings), so
    # the `exercise <id>` prefix and `--exercise-dir` become optional. Passing
    # --exercise-dir still works from anywhere. With no specs, the runner runs
    # every test/benchmark for that mode.
    for mode in EXERCISE_MODES:
        mp = sub.add_parser(
            mode,
            help=f"Run the {mode} action on the exercise in the cwd "
            "(or --exercise-dir)",
        )
        mp.add_argument(
            "--exercise-id",
            help="exercise id for reporting (default: the exercise dir name)",
        )
        _add_common_exercise_opts(mp)
        mp.set_defaults(exercise_command=mode)

    custom = sub.add_parser("custom", help="Compile, run, or profile a custom CUDA program")
    custom.add_argument("custom_command", choices=["compile", "run", "profile"])
    custom.add_argument("source", help="CUDA source file containing the kernel or host wrapper")
    custom.add_argument("--harness", help="Optional CUDA/C++ source file containing main()")
    custom.add_argument("--output", default="custom_kernel")
    custom.add_argument("--arg", action="append", default=[], help="Program argument, repeatable")
    custom.add_argument("--nvcc-flags", default="-std=c++20 -O3 -lineinfo")
    custom.add_argument("--gpu", default="B200")
    custom.add_argument("--gpu-type")
    custom.add_argument("--image", default="cuda-nvcc")
    custom.add_argument("--arch")
    custom.add_argument("--timeout", type=int, default=600)
    custom.add_argument("--wait-timeout", type=float)
    custom.add_argument("--json", dest="json_path")
    custom.add_argument("--artifact-dir")
    custom.add_argument("--ncu-args", default="--set basic")
    custom.add_argument("--nvtx-range", default="profile_kernel")
    custom.add_argument("--no-nvtx-filter", action="store_true")
    custom.add_argument(
        "--report-name",
        help="base name for the profile .ncu-rep (default: source file stem)",
    )
    custom.add_argument("--verbose", action="store_true")

    report = sub.add_parser("report", help="Inspect local Nsight Compute reports")
    report_sub = report.add_subparsers(dest="report_command", required=True)
    summary = report_sub.add_parser(
        "summary",
        help="Print a generic metric summary from a local .ncu-rep",
    )
    summary.add_argument("report", help="Path to a local .ncu-rep file")
    summary.add_argument("--per-kernel", action="store_true")
    summary.add_argument("--json", dest="json_path")

    feedback = report_sub.add_parser(
        "feedback",
        help="Run CUDA course feedback rules against a local .ncu-rep",
    )
    feedback.add_argument("report", help="Path to a local .ncu-rep file")
    feedback.add_argument(
        "--course-dir",
        default=os.environ.get("CUDA_COURSE_DIR"),
        help="CUDA course checkout containing runner/ and exercises/ "
        "(or set CUDA_COURSE_DIR)",
    )
    feedback.add_argument("--exercise", default="01-haxpy")
    feedback.add_argument("--benchmark", default="benchmarks/01_aligned_small.txt")
    feedback.add_argument("--json", dest="json_path")
    feedback.add_argument("--verbose", action="store_true")
    return parser
