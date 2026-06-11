"""Payload builders for checkout-exercise and custom CUDA remote jobs."""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
from pathlib import Path
from typing import Any

from .constants import GPU_DEFAULTS, _CHECKOUT_SKIP_DIRS
from .errors import CliError


def _resolve_course_root(args: argparse.Namespace) -> Path | None:
    """Locate a cuda-course checkout for the requested exercise, or ``None``.

    Tries ``--course-root`` (and its parents), then walks up from ``--file`` and
    the cwd. Returns ``None`` when no checkout is found -- exercises require one,
    so the caller raises a clear error in that case.
    """
    exercise_id = args.exercise_id

    def is_root(d: Path) -> bool:
        return (d / "runner" / "cli.py").is_file() and (d / "exercises" / exercise_id / "run.py").is_file()

    explicit = getattr(args, "course_root", None)
    if explicit:
        p = Path(explicit).expanduser()
        for d in [p, *p.resolve().parents]:
            if is_root(d):
                return d
        return None

    starts: list[Path] = []
    if args.source_file:
        starts.append(Path(args.source_file).expanduser())
    starts.append(Path.cwd())
    seen: set[str] = set()
    for start in starts:
        for d in [start, *start.resolve().parents]:
            key = str(d)
            if key in seen:
                continue
            seen.add(key)
            if is_root(d):
                return d

    return None


def _norm_spec(spec: str, exercise_id: str) -> str:
    """Normalise a spec arg to a checkout-relative path (drop the ``exercises/<id>/`` prefix)."""
    return spec.replace("\\", "/").removeprefix(f"exercises/{exercise_id}/")


def _walk_checkout(files: dict[str, str], hashes: dict[str, str], root: Path, base: Path, extra_skip: set[str] = frozenset()) -> None:
    """Collect every UTF-8 text file under *base* into *files*/*hashes* (keyed
    relative to *root*), skipping VCS/cache dirs and any *extra_skip* names."""
    skip = _CHECKOUT_SKIP_DIRS | set(extra_skip)
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for fn in filenames:
            full = Path(dirpath) / fn
            data = full.read_bytes()
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                continue
            rel = _clean_payload_path(full.relative_to(root).as_posix())
            files[rel] = text
            hashes[rel] = hashlib.sha256(data).hexdigest()


def _build_checkout_payload(
    *,
    course_root: Path,
    exercise_id: str,
    mode: str,
    source_file: Path | None,
    specs: list[str],
    gpu: str,
    gpu_type: str,
    arch: str,
    image: str,
    timeout_s: int,
    verbose: bool,
) -> dict[str, Any]:
    """Build a job that ships the live ``runner/`` + the chosen exercise and runs
    the exercise's own ``run.py`` on the worker. The exercise's ``solutions/`` dir
    is never shipped."""
    ex_dir = course_root / "exercises" / exercise_id
    runner_dir = course_root / "runner"
    if not (ex_dir / "run.py").is_file():
        raise CliError(f"exercise {exercise_id!r} not found under {course_root}")
    if not (runner_dir / "cli.py").is_file():
        raise CliError(f"no course runner under {runner_dir}")

    files: dict[str, str] = {}
    hashes: dict[str, str] = {}
    _walk_checkout(files, hashes, course_root, runner_dir)
    _walk_checkout(files, hashes, course_root, ex_dir, extra_skip={"solutions"})

    inc = runner_dir / "include"
    if inc.is_dir():
        for header in sorted(inc.iterdir()):
            if header.is_file():
                data = header.read_bytes()
                rel = _clean_payload_path(f"exercises/{exercise_id}/runner/include/{header.name}")
                files[rel] = data.decode("utf-8")
                hashes[rel] = hashlib.sha256(data).hexdigest()

    file_arg: str | None = None
    if source_file is not None:
        if not source_file.is_file():
            raise CliError(f"--file not found: {source_file}")
        data = source_file.read_bytes()
        abs_src = source_file.resolve()
        try:
            file_arg = abs_src.relative_to(ex_dir.resolve()).as_posix()
        except ValueError:
            file_arg = "__submitted__.cu"
        rel = _clean_payload_path(f"exercises/{exercise_id}/{file_arg}")
        files[rel] = data.decode("utf-8")
        hashes[rel] = hashlib.sha256(data).hexdigest()

    json_out = "_gpu_func_cli.json"
    command = ["python3", "run.py", "--json", json_out]
    if file_arg:
        command += ["--file", file_arg]
    if verbose:
        command.append("-v")
    if arch:
        command += ["--arch", arch]
    command.append(mode)
    command.extend(_norm_spec(s, exercise_id) for s in specs)

    return {
        "schema_version": 1,
        "asset_version": "checkout",
        "target": {"kind": "exercise", "exercise_id": exercise_id, "source": "checkout"},
        "remote": {"gpu": gpu, "gpu_type": gpu_type, "arch": arch, "image": image, "timeout_s": timeout_s},
        "command": {"mode": mode},
        "course_runner": {
            "enabled": True,
            "cwd": f"exercises/{exercise_id}",
            "command": command,
            "json_out": json_out,
            "artifact_globs": [json_out, "*.ncu-rep"],
            "timeout_s": timeout_s,
        },
        "files": files,
        "hashes": hashes,
    }


def _resolve_gpu(gpu: str, gpu_type: str | None, arch: str | None) -> tuple[str, str]:
    """Resolve a GPU label to ``(gpu_type, arch)`` via GPU_DEFAULTS, honouring overrides."""
    default_type, default_arch = GPU_DEFAULTS.get(gpu.upper(), (gpu.lower(), ""))
    return gpu_type or default_type, arch if arch is not None else default_arch


def _build_custom_payload(args: argparse.Namespace, gpu_type: str, arch: str) -> dict[str, Any]:
    """Build a custom-kernel job: source -> ``kernel.cu`` (+ optional ``harness.cu``),
    compile flags, ncu args, and a sanitised ``report_name`` (default: source stem)."""
    source_path = Path(args.source)
    if not source_path.is_file():
        raise CliError(f"source file not found: {source_path}")
    raw_report_name = args.report_name or source_path.stem
    report_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in raw_report_name) or "custom_profile"
    harness_path = Path(args.harness) if args.harness else None
    if harness_path and not harness_path.is_file():
        raise CliError(f"harness file not found: {harness_path}")

    files: dict[str, str] = {}
    hashes: dict[str, str] = {}

    def add_text(path: str, text: str) -> None:
        clean = _clean_payload_path(path)
        files[clean] = text
        hashes[clean] = hashlib.sha256(text.encode("utf-8")).hexdigest()

    add_text("kernel.cu", source_path.read_text(encoding="utf-8"))
    sources = ["kernel.cu"]
    if harness_path:
        add_text("harness.cu", harness_path.read_text(encoding="utf-8"))
        sources.append("harness.cu")

    flags = shlex.split(args.nvcc_flags) if args.nvcc_flags else []
    if arch:
        flags.append(f"-arch={arch}")

    return {
        "schema_version": 1,
        "target": {
            "kind": "custom",
            "source": str(source_path),
            "harness": str(harness_path) if harness_path else None,
        },
        "remote": {
            "gpu": args.gpu,
            "gpu_type": gpu_type,
            "image": args.image,
            "timeout_s": args.timeout,
        },
        "command": {
            "mode": f"custom-{args.custom_command}",
        },
        "custom": {
            "command": args.custom_command,
            "sources": sources,
            "flags": flags,
            "output": args.output,
            "report_name": report_name,
            "program_args": list(args.arg),
            "ncu_args": shlex.split(args.ncu_args) if args.ncu_args else ["--set", "basic"],
            "nvtx_range": "" if args.no_nvtx_filter else args.nvtx_range,
            "timeout_s": args.timeout,
        },
        "files": files,
        "hashes": hashes,
    }


def _clean_payload_path(path: str) -> str:
    """Return a normalised relative payload path, rejecting absolute or ``..`` paths."""
    clean = Path(path)
    if clean.is_absolute() or ".." in clean.parts:
        raise CliError(f"unsafe payload path {path!r}")
    return clean.as_posix()
