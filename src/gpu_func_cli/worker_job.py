"""Remote worker module source embedded into submitted GFAAS bundles."""

from __future__ import annotations




WORKER_TEMPLATE = r'''
import base64
import glob
import hashlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

_JOB_B64 = "__JOB_B64_PLACEHOLDER__"


def run():
    try:
        job = json.loads(base64.b64decode(_JOB_B64).decode("utf-8"))
        return run_course_job(job)
    except Exception as exc:
        return {
            "schema_version": 1,
            "status": "setup_error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def run_course_job(job):
    workdir = tempfile.mkdtemp(prefix="gpu_func_cli_", dir=workdir_root())
    file_hashes = {}
    for rel, text in job["files"].items():
        path = safe_path(workdir, rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        file_hashes[rel] = hashlib.sha256(text.encode("utf-8")).hexdigest()

    expected = job.get("hashes", {})
    bad_hashes = {
        key: {"expected": expected[key], "actual": file_hashes.get(key)}
        for key in expected
        if file_hashes.get(key) != expected[key]
    }
    if bad_hashes:
        return {
            "schema_version": 1,
            "status": "setup_error",
            "error": "payload hash mismatch",
            "bad_hashes": bad_hashes,
        }

    # Remote dispatch -- mirrors the client's payload kinds (see flow.md):
    #   custom         -> nvcc, then run or ncu-profile the binary
    #   course_runner  -> run the exercise's own run.py (checkout exercises)
    #   else           -> vendored: compile with the tester, run each spec
    if job.get("target", {}).get("kind") == "custom":
        return run_custom_job(workdir, job, file_hashes)

    if job.get("course_runner", {}).get("enabled"):
        return run_course_runner(workdir, job, file_hashes)

    compile_result = compile_executable(workdir, job)
    if compile_result["timed_out"]:
        return {
            "schema_version": 1,
            "status": "setup_error",
            "compile": compile_result,
            "error": "nvcc timed out",
        }
    if compile_result["returncode"] != 0:
        return {
            "schema_version": 1,
            "status": "compile_failed",
            "compile": compile_result,
            "runs": [],
            "worker": worker_info(),
            "file_hashes": file_hashes,
        }

    if job["command"]["mode"] == "compile":
        return {
            "schema_version": 1,
            "status": "passed",
            "compile": compile_result,
            "runs": [],
            "worker": worker_info(),
            "file_hashes": file_hashes,
        }

    runs = []
    for spec in job["command"]["selected_specs"]:
        run_result = run_one_spec(workdir, job, spec)
        runs.append(run_result)
        if run_result["status"] != "passed" and not job["command"].get("keep_going", False):
            break

    status = "passed"
    for run_result in runs:
        if run_result["status"] != "passed":
            status = run_result["status"]
            break
    return {
        "schema_version": 1,
        "status": status,
        "compile": compile_result,
        "runs": runs,
        "worker": worker_info(),
        "file_hashes": file_hashes,
    }


def run_custom_job(workdir, job, file_hashes):
    custom = job["custom"]
    compile_args = (
        [which("nvcc")]
        + host_cxx_flags()
        + list(custom.get("flags", []))
        + list(custom["sources"])
        + ["-o", custom["output"]]
    )
    compile_result = run_process(compile_args, workdir, min(custom.get("timeout_s", 600), 300), env=None)
    if compile_result.get("timed_out"):
        return {
            "schema_version": 1,
            "status": "setup_error",
            "compile": compile_result,
            "error": "nvcc timed out",
            "file_hashes": file_hashes,
        }
    if compile_result.get("returncode") != 0:
        return {
            "schema_version": 1,
            "status": "compile_failed",
            "compile": compile_result,
            "run": None,
            "artifacts": {},
            "worker": worker_info(),
            "file_hashes": file_hashes,
        }

    command = custom["command"]
    if command == "compile":
        return {
            "schema_version": 1,
            "status": "passed",
            "compile": compile_result,
            "run": None,
            "artifacts": {},
            "worker": worker_info(),
            "file_hashes": file_hashes,
        }

    program = ["./" + custom["output"], *custom.get("program_args", [])]
    artifacts = {}
    if command == "profile":
        export_base = Path(custom.get("report_name") or "custom_profile").name or "custom_profile"
        run_args = [which("ncu"), *custom.get("ncu_args", ["--set", "basic"])]
        nvtx_range = custom.get("nvtx_range", "")
        if nvtx_range:
            run_args += ["--nvtx", "--nvtx-include", nvtx_range + ("/" if not nvtx_range.endswith("/") else "")]
        run_args += ["--force-overwrite", "--export", export_base, *program]
    else:
        run_args = program

    run_result = run_process(run_args, workdir, custom.get("timeout_s", 600), env=None)
    if command == "profile":
        report_file = Path(workdir, export_base + ".ncu-rep")
        if report_file.exists():
            data = report_file.read_bytes()
            artifacts["ncu_report"] = {
                "filename": report_file.name,
                "encoding": "base64",
                "content": base64.b64encode(data).decode("ascii"),
                "size_bytes": len(data),
            }

    status = "passed"
    if run_result.get("timed_out"):
        status = "timeout"
    elif run_result.get("returncode") != 0:
        status = "error"

    return {
        "schema_version": 1,
        "status": status,
        "compile": compile_result,
        "run": run_result,
        "artifacts": artifacts,
        "worker": worker_info(),
        "file_hashes": file_hashes,
    }


def run_course_runner(workdir, job, file_hashes):
    spec = job["course_runner"]
    cwd = safe_path(workdir, spec.get("cwd", "."))
    env = {"PYTHONPATH": workdir}
    result = run_process(
        list(spec["command"]),
        str(cwd),
        spec.get("timeout_s") or job.get("remote", {}).get("timeout_s") or 600,
        env=env,
    )

    artifacts = {"ncu_reps": {}}
    report_json = None
    json_name = spec.get("json_out", "_gpu_func_cli.json")
    json_path = Path(cwd, json_name)
    if json_path.exists():
        json_text = json_path.read_text(encoding="utf-8")
        artifacts["json_report"] = {"filename": json_name, "content": json_text}
        try:
            report_json = json.loads(json_text)
        except Exception:
            report_json = None

    for pattern in spec.get("artifact_globs", []):
        if pattern == json_name:
            continue
        for path in glob.glob(str(Path(cwd, pattern))):
            p = Path(path)
            if not p.is_file() or not p.name.endswith(".ncu-rep"):
                continue
            data = p.read_bytes()
            artifacts["ncu_reps"][p.name] = {
                "filename": p.name,
                "encoding": "base64",
                "content": base64.b64encode(data).decode("ascii"),
                "size_bytes": len(data),
            }

    status = "passed"
    if result.get("timed_out"):
        status = "timeout"
    elif result.get("returncode") != 0:
        status = "error"

    return {
        "schema_version": 1,
        "status": status,
        "course_runner": result,
        "report_json": report_json,
        "artifacts": artifacts,
        "worker": worker_info(),
        "file_hashes": file_hashes,
    }


def compile_executable(workdir, job):
    args = (
        [which("nvcc")]
        + host_cxx_flags()
        + list(job["compile"]["flags"])
        + list(job["compile"]["sources"])
        + ["-o", job["compile"]["output"]]
    )
    return run_process(args, workdir, job["compile"].get("timeout_s", 120), env=None)


def run_one_spec(workdir, job, spec):
    report_path = os.path.join(workdir, "report_" + spec["name"] + ".txt")
    env = {}
    env["REPORT_PATH"] = report_path
    env["REPORT_MAX_MISMATCHES"] = str(job["runtime"].get("report_max_mismatches", 20))
    mode = job["command"]["mode"]
    program = ["./" + job["compile"]["output"], job["runtime"]["run_flag"], spec["path"]]
    artifacts = {}
    if mode == "profile":
        ncu = which("ncu")
        export_base = job["compile"]["output"] + "." + spec["name"]
        args = [
            ncu,
            *job["runtime"].get("ncu_args", ["--set", "basic"]),
            "--nvtx",
            "--nvtx-include",
            "profile_kernel/",
            "--force-overwrite",
            "--export",
            export_base,
            *program,
        ]
    else:
        args = list(job["runtime"].get("launcher", [])) + program
    result = run_process(args, workdir, spec.get("timeout") or 30.0, env=env)
    report_text = ""
    try:
        report_text = Path(report_path).read_text(encoding="utf-8")
    except FileNotFoundError:
        pass
    if mode == "profile":
        report_file = Path(workdir, job["compile"]["output"] + "." + spec["name"] + ".ncu-rep")
        if report_file.exists():
            data = report_file.read_bytes()
            artifacts["ncu_report"] = {
                "filename": report_file.name,
                "encoding": "base64",
                "content": base64.b64encode(data).decode("ascii"),
                "size_bytes": len(data),
            }
    result["spec"] = spec["path"]
    result["name"] = spec["name"]
    result["report_text"] = report_text
    result["artifacts"] = artifacts

    if result["timed_out"]:
        result["status"] = "timeout"
    elif result["returncode"] == 5:
        result["status"] = "setup-error"
    elif result["returncode"] != 0:
        result["status"] = "error"
    elif "status=failed" in report_text:
        result["status"] = "failed"
    elif "status=passed" in report_text:
        result["status"] = "passed"
    else:
        result["status"] = "error"
    return result


def run_process(args, cwd, timeout_s, env):
    start = time.monotonic()
    tool_env = subprocess_env(cwd)
    if env:
        tool_env.update(env)
    try:
        cp = subprocess.run(
            args,
            cwd=cwd,
            timeout=timeout_s,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            env=tool_env,
        )
        return {
            "args": args,
            "returncode": cp.returncode,
            "stdout": cp.stdout,
            "stderr": cp.stderr,
            "ms": int((time.monotonic() - start) * 1000),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "args": args,
            "returncode": None,
            "stdout": as_text(exc.stdout),
            "stderr": as_text(exc.stderr),
            "ms": int((time.monotonic() - start) * 1000),
            "timed_out": True,
        }
    except FileNotFoundError as exc:
        return {
            "args": args,
            "returncode": 127,
            "stdout": "",
            "stderr": str(exc),
            "ms": int((time.monotonic() - start) * 1000),
            "timed_out": False,
        }


def safe_path(root, rel):
    path = Path(root, rel).resolve()
    root_path = Path(root).resolve()
    if not str(path).startswith(str(root_path) + os.sep):
        raise ValueError("unsafe path: " + rel)
    return path


def which(cmd):
    found = shutil.which(cmd)
    if found:
        return found
    candidates = ["/usr/local/cuda/bin/" + cmd]
    if cmd == "ncu":
        candidates.extend(sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"), reverse=True))
    for candidate in candidates:
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError("required binary not found in PATH: " + cmd)


def host_cxx_flags():
    override = os.environ.get("GFAAS_NVCC_CCBIN") or os.environ.get("CXX")
    candidates = [override, "/usr/bin/g++", shutil.which("g++"), shutil.which("c++")]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return ["-ccbin", candidate]
    return []


def subprocess_env(workdir):
    env = os.environ.copy()
    base_path = "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    current_path = env.get("PATH", "")
    env["PATH"] = base_path + (":" + current_path if current_path else "")

    home = env.get("HOME", "")
    if not home or not os.path.isdir(home) or not os.access(home, os.W_OK):
        env["HOME"] = workdir

    xdg = env.get("XDG_CONFIG_HOME", "")
    if not xdg or not os.path.isdir(xdg) or not os.access(xdg, os.W_OK):
        env["XDG_CONFIG_HOME"] = os.path.join(workdir, ".config")

    os.makedirs(env["HOME"], exist_ok=True)
    os.makedirs(env["XDG_CONFIG_HOME"], exist_ok=True)
    return env


def workdir_root():
    root = os.environ.get("FC_IO_ROOT")
    if root and os.path.isdir(root) and os.access(root, os.W_OK | os.X_OK):
        return root
    return None


def worker_info():
    return {
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def as_text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)
'''
