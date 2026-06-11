"""Command handlers that coordinate payload creation, REST submission, and output."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
from pathlib import Path
from typing import Any

from .bundles import _build_bundle
from .client import RestClient
from .constants import OCTET_STREAM, RC_OK, RC_SETUP, WORKER_MODULE, WORKER_QUALNAME
from .errors import CliError
from .output import _print_course_runner_result, _print_custom_result
from .payloads import _build_checkout_payload, _build_custom_payload, _resolve_course_root, _resolve_gpu


def _cmd_workers(client: "RestClient") -> int:
    # Read-only journey: no bundle, no job — just GET the live worker list.
    payload = client.get_json("/v1/workers")
    workers = payload.get("workers", [])
    if not workers:
        print("No live workers")
        return RC_OK
    for worker in workers:
        gpu_type = worker.get("gpu_type", "?")
        gpu = worker.get("gpu", worker.get("gpu_name", "?"))
        images = ", ".join(worker.get("images", []) or [])
        active = worker.get("active_jobs", "?")
        max_jobs = worker.get("max_jobs", "?")
        print(f"{gpu_type}: {gpu} active={active}/{max_jobs} images=[{images}]")
    return RC_OK


def _cmd_custom(args: argparse.Namespace) -> int:
    # Custom-kernel journey: build a {target: "custom"} job spec, then run it
    # through the shared remote spine and render the result.
    client = RestClient.from_args(args)
    gpu_type, arch = _resolve_gpu(args.gpu, args.gpu_type, args.arch)   # GPU label -> (gpu_type, arch)
    payload = _build_custom_payload(args, gpu_type, arch)              # args + source/harness -> JSON job
    result, job_id = _submit_payload(
        client, args, payload, gpu_type,
        app_name="gpu-func-cli-custom",
        label=f"custom {args.custom_command}",
        arch=arch or "default",
    )
    if result is None:
        return RC_SETUP

    exit_code = _print_custom_result(result, args)
    if args.json_path:
        out = {
            "mode": f"custom-{args.custom_command}",
            "remote": {
                "job_id": job_id,
                "gpu": args.gpu,
                "gpu_type": gpu_type,
                "image": args.image,
            },
            "result": result,
        }
        Path(args.json_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Results written to {args.json_path}")
    return exit_code


def _cmd_exercise(args: argparse.Namespace) -> int:
    # Every exercise runs from a cuda-course checkout: the CLI ships that exercise
    # plus the course runner/ and runs the exercise's own run.py, so output
    # (correctness, GiB/s, feedback) is exact for any exercise and any mode.
    client = RestClient.from_args(args)
    course_root = _resolve_course_root(args)
    if course_root is None or not (course_root / "exercises" / args.exercise_id / "run.py").is_file():
        raise CliError(
            f"no cuda-course checkout with exercises/{args.exercise_id}/run.py found. "
            "Pass --course-root <cuda-course>, set CUDA_COURSE_REPO, or run from "
            "inside a checkout."
        )
    gpu_type, _arch = _resolve_gpu(args.gpu, args.gpu_type, args.arch)
    return _run_checkout_exercise(args, client, course_root, gpu_type)


def _submit_payload(
    client: RestClient,
    args: argparse.Namespace,
    payload: dict[str, Any],
    gpu_type: str,
    app_name: str,
    label: str,
    arch: str | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """The shared remote spine: build bundle -> upload -> submit -> poll -> fetch.

    Used by ``_cmd_custom`` and the checkout-exercise path (``_run_checkout_exercise``).
    Returns ``(result, job_id)``; ``result`` is ``None`` if the job did not complete.
    """
    #   1 build bundle  2 upload  3 submit  4 poll  5 fetch result_json
    bundle = _build_bundle(payload)                             # 1: embed payload in worker_job + tar.gz
    banner = f"Remote {label} on {args.gpu} ({gpu_type}), image={args.image}"
    if arch is not None:
        banner += f", arch={arch}"
    print(banner)
    bundle_resp = client.post_bytes(                           # 2: upload (server dedupes by sha256)
        f"/v1/bundles?{urllib.parse.urlencode({'sha256': bundle['sha256']})}",
        bundle["data"],
        content_type=OCTET_STREAM,
    )
    submit_req = {
        "app_name": app_name,
        "image_name": args.image,
        "image_spec": None,
        "bundle_id": bundle_resp["bundle_id"],
        "function": {
            "module": WORKER_MODULE,
            "qualname": WORKER_QUALNAME,
            "gpu": args.gpu,
            "timeout_s": args.timeout,
            "env": {},
            "cwd": None,
            "host_network": False,
        },
        "gpu_type": gpu_type,
        "input_blob_id": None,
        "input_codec": None,
    }
    submit_resp = client.post_json("/v1/submit", submit_req)   # 3: submit (worker entry = WORKER_MODULE.run)
    job_id = submit_resp["job_id"]
    print(f"job: {job_id}")
    row = client.wait_job(job_id, timeout_s=args.wait_timeout)  # 4: poll until a terminal state
    if row.get("status") != "completed":
        msg = row.get("error") or row.get("traceback") or f"job ended as {row.get('status')}"
        print(msg, file=sys.stderr)
        return None, job_id
    result_payload = client.get_json(f"/v1/jobs/{job_id}/result_json")   # 5: fetch the worker's result
    result = result_payload.get("result")
    if not isinstance(result, dict):
        raise CliError(f"unexpected result_json payload: {result_payload!r}")
    return result, job_id


def _run_checkout_exercise(args: argparse.Namespace, client, course_root: Path, gpu_type: str) -> int:
    # Branch A: ship the live cuda-course runner/ + the chosen exercise and run
    # that exercise's own run.py on the worker (exact output, any exercise).
    mode = args.exercise_command
    payload = _build_checkout_payload(
        course_root=course_root,
        exercise_id=args.exercise_id,
        mode=mode,
        source_file=Path(args.source_file) if args.source_file else None,
        specs=list(args.specs),
        gpu=args.gpu,
        gpu_type=gpu_type,
        image=args.image,
        timeout_s=args.timeout,
        verbose=args.verbose,
    )
    result, job_id = _submit_payload(client, args, payload, gpu_type, "cuda-course", mode)
    if result is None:
        return RC_SETUP
    code = _print_course_runner_result(result, args)
    if args.json_path:
        out = {
            "mode": mode,
            "exercise": args.exercise_id,
            "remote": {
                "job_id": job_id,
                "gpu": payload["remote"]["gpu"],
                "gpu_type": payload["remote"]["gpu_type"],
                "image": payload["remote"]["image"],
            },
            "status": result.get("status"),
            "course_runner": result.get("course_runner"),
            "report_json": result.get("report_json"),
        }
        Path(args.json_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Results written to {args.json_path}")
    return code
