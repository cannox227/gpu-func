"""Tar bundle assembly for shipping encoded worker jobs to GFAAS."""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import tarfile
from typing import Any

from .constants import WORKER_MODULE
from .worker_job import WORKER_TEMPLATE


def _build_bundle(payload: dict[str, Any]) -> dict[str, Any]:
    """Build the upload bundle: embed *payload* in the worker module, tar.gz it.

    Returns ``{"data": <gzipped tar bytes>, "sha256": <hex digest>}``.
    """
    # Bake the job spec INTO the worker module so the whole job ships as one
    # self-contained file (gpu_func_job.py) with no separate data blob or deps.
    payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    job_b64 = base64.b64encode(payload_json).decode("ascii")
    module_source = WORKER_TEMPLATE.replace("__JOB_B64_PLACEHOLDER__", job_b64).encode("utf-8")

    # Deterministic tar.gz (fixed mtime/uid/gid/name) so identical jobs produce
    # identical bytes -> the server can dedupe bundles by sha256.
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6, mtime=0) as gzip_file:
        with tarfile.open(fileobj=gzip_file, mode="w", format=tarfile.PAX_FORMAT) as tar:
            info = tarfile.TarInfo(f"{WORKER_MODULE}.py")
            info.size = len(module_source)
            info.mode = 0o644
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            tar.addfile(info, io.BytesIO(module_source))
    data = buf.getvalue()
    return {"data": data, "sha256": hashlib.sha256(data).hexdigest()}
