"""Shared constants for CLI defaults, exit codes, and remote payload policy."""

from __future__ import annotations

USER_AGENT = "gpu_func_cli/0.1 curl-compatible"
TERMINAL_STATES = {"completed", "failed", "timed_out", "cancelled"}

# Remote payload policy: every job bundle is a single module named WORKER_MODULE
# whose WORKER_QUALNAME function is the entry point, uploaded as OCTET_STREAM.
WORKER_MODULE = "gpu_func_job"
WORKER_QUALNAME = "run"
OCTET_STREAM = "application/octet-stream"
DEFAULT_COMPILE_FLAGS = ["-std=c++20", "-Xptxas", "--warn-on-spills", "-lineinfo"]
MI = 1024 * 1024
GPU_DEFAULTS = {
    "B200": ("b200", "sm_100a"),
    "B300": ("b300", "sm_100a"),
    "H200": ("h200", "sm_90a"),
    "H100": ("h100", "sm_90a"),
    "A100": ("a100", "sm_80"),
    "RTX6000": ("rtx6000", "sm_89"),
}

RC_OK = 0
RC_COMPILE = 1
RC_CRASH = 2
RC_WRONG = 3
RC_TIMEOUT = 4
RC_SETUP = 5

_CHECKOUT_SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
