"""Argparse construction for the public gpu_func_cli command surface."""

from __future__ import annotations

import argparse
import os


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
    exercise.add_argument(
        "exercise_command",
        choices=["compile", "test", "benchmark", "sanitizer", "profile", "grade"],
    )
    exercise.add_argument("specs", nargs="*")
    exercise.add_argument("--file", dest="source_file")
    exercise.add_argument(
        "--course-root",
        default=os.environ.get("CUDA_COURSE_REPO"),
        help="path to a cuda-course checkout (or set CUDA_COURSE_REPO); required "
        "for exercises. Default: auto-detect from --file / the cwd.",
    )
    exercise.add_argument("--gpu", default="B200")
    exercise.add_argument("--gpu-type")
    exercise.add_argument("--image", default="cuda-nvcc")
    exercise.add_argument("--arch")
    exercise.add_argument("--timeout", type=int, default=600)
    exercise.add_argument("--wait-timeout", type=float)
    exercise.add_argument("--json", dest="json_path")
    exercise.add_argument("--artifact-dir")
    exercise.add_argument("--ncu-args", default="--set basic")
    exercise.add_argument("--verbose", action="store_true")
    exercise.add_argument("--report-max-mismatches", type=int, default=20)
    exercise.add_argument("--keep-going", action="store_true")

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
