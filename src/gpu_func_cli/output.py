"""Terminal rendering for remote checkout-exercise and custom-kernel results."""

from __future__ import annotations

import argparse
import base64
import shlex
import sys
from pathlib import Path
from typing import Any

from .constants import RC_COMPILE, RC_CRASH, RC_OK, RC_SETUP, RC_TIMEOUT


def _print_course_runner_result(result: dict[str, Any], args: argparse.Namespace) -> int:
    """Render a checkout-exercise result: print the course runner's stdout/stderr,
    save any returned ``.ncu-rep`` artifacts, and map its return code to an exit code."""
    runner = result.get("course_runner") or {}
    stdout = runner.get("stdout") or ""
    stderr = runner.get("stderr") or ""
    returncode = runner.get("returncode")

    if stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n")
    if returncode not in (0, None) and stderr:
        print(stderr, file=sys.stderr, end="" if stderr.endswith("\n") else "\n")

    artifacts = result.get("artifacts") or {}
    ncu_reps = artifacts.get("ncu_reps") or {}
    for filename, item in ncu_reps.items():
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if args.artifact_dir and isinstance(content, str):
            out_dir = Path(args.artifact_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / filename
            out_path.write_bytes(base64.b64decode(content.encode("ascii")))
            print(f"gpu_func_cli: profile report -> {out_path}", file=sys.stderr)
        else:
            print(f"gpu_func_cli: profile report -> {filename}", file=sys.stderr)

    if runner.get("timed_out"):
        return RC_TIMEOUT
    if returncode is None:
        return RC_SETUP
    if 0 <= int(returncode) <= 5:
        return int(returncode)
    return RC_SETUP


def _print_custom_result(result: dict[str, Any], args: argparse.Namespace) -> int:
    """Render a custom-kernel result: compile/run stdout+stderr, save the returned
    ``.ncu-rep`` when ``--artifact-dir`` is set, and map status to an exit code."""
    compile_result = result.get("compile") or {}
    if compile_result:
        print("Compiling")
        if args.verbose:
            print(">> " + " ".join(shlex.quote(str(x)) for x in compile_result.get("args", [])))
        if compile_result.get("stdout"):
            print(compile_result["stdout"], end="" if compile_result["stdout"].endswith("\n") else "\n")
        if compile_result.get("returncode") != 0:
            if compile_result.get("stderr"):
                print(compile_result["stderr"], file=sys.stderr, end="" if compile_result["stderr"].endswith("\n") else "\n")
            return RC_COMPILE

    run_result = result.get("run") or {}
    if run_result:
        if args.verbose:
            print(">> " + " ".join(shlex.quote(str(x)) for x in run_result.get("args", [])))
        if run_result.get("stdout"):
            print(run_result["stdout"], end="" if run_result["stdout"].endswith("\n") else "\n")
        if run_result.get("stderr"):
            print(run_result["stderr"], file=sys.stderr, end="" if run_result["stderr"].endswith("\n") else "\n")
        if run_result.get("timed_out"):
            return RC_TIMEOUT
        if run_result.get("returncode") not in (0, None):
            return RC_CRASH

    report = (result.get("artifacts") or {}).get("ncu_report")
    if isinstance(report, dict):
        filename = report.get("filename") or "custom_profile.ncu-rep"
        content = report.get("content")
        if args.artifact_dir and isinstance(content, str):
            out_dir = Path(args.artifact_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / filename
            out_path.write_bytes(base64.b64decode(content.encode("ascii")))
            print(f"gpu_func_cli: profile report -> {out_path}", file=sys.stderr)
        else:
            print(f"gpu_func_cli: profile report -> {filename}", file=sys.stderr)

    if result.get("status") == "passed":
        print(f"Custom {args.custom_command} passed")
        return RC_OK
    if result.get("status") == "compile_failed":
        return RC_COMPILE
    return RC_SETUP
