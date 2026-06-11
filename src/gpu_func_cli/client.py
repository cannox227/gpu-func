"""HTTP client for GFAAS bundle submission, job polling, and result retrieval."""

from __future__ import annotations

import argparse
import http.client
import json
import socket
import time
import urllib.error
import urllib.request
from typing import Any

from .constants import RC_TIMEOUT, TERMINAL_STATES, USER_AGENT
from .errors import CliError

_RETRYABLE_TRANSPORT_ERRORS = (
    ConnectionResetError,
    TimeoutError,
    socket.timeout,
    http.client.RemoteDisconnected,
)


class RestClient:
    def __init__(
        self,
        *,
        api_base: str,
        api_key: str | None,
        request_timeout: float,
        poll_interval: float,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.request_timeout = request_timeout
        self.poll_interval = poll_interval

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "RestClient":
        if not args.api_base:
            raise CliError("GFAAS_API_BASE is not set; pass --api-base or export it")
        return cls(
            api_base=args.api_base,
            api_key=args.api_key,
            request_timeout=args.request_timeout,
            poll_interval=args.poll_interval,
        )

    # --- remote-spine primitives (commands.py steps 2-5 are built from these) ---
    def get_json(self, path: str) -> dict[str, Any]:
        return self._request_json("GET", path, retry_transport=True)

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        return self._request_json("POST", path, data=data, content_type="application/json")

    def post_bytes(self, path: str, payload: bytes, *, content_type: str) -> dict[str, Any]:
        return self._request_json(
            "POST",
            path,
            data=payload,
            content_type=content_type,
            retry_transport=path.startswith("/v1/bundles"),
        )

    def wait_job(self, job_id: str, *, timeout_s: float | None) -> dict[str, Any]:
        deadline = time.time() + timeout_s if timeout_s else None
        last_status = None
        while True:
            row = self.get_json(f"/v1/jobs/{job_id}")
            status = row.get("status", "queued")
            if status != last_status:
                if status in {"running", "completed", "failed", "timed_out", "cancelled"}:
                    print(f"status: {status}")
                last_status = status
            if status in TERMINAL_STATES:
                return row
            if deadline and time.time() > deadline:
                raise CliError(f"job {job_id} did not finish within {timeout_s}s", RC_TIMEOUT)
            time.sleep(self.poll_interval)

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        data: bytes | None = None,
        content_type: str | None = None,
        retry_transport: bool = False,
    ) -> dict[str, Any]:
        url = self.api_base + path
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if content_type:
            headers["Content-Type"] = content_type
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        attempts = 3 if retry_transport else 1
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                    body = resp.read()
                    break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")
                raise CliError(f"HTTP {exc.code} for {method} {path}: {body}") from exc
            except urllib.error.URLError as exc:
                if attempt + 1 < attempts and _is_retryable_transport_error(exc):
                    time.sleep(min(0.25 * (2**attempt), 1.0))
                    continue
                raise CliError(f"request failed for {method} {path}: {exc}") from exc
            except _RETRYABLE_TRANSPORT_ERRORS as exc:
                if attempt + 1 < attempts:
                    time.sleep(min(0.25 * (2**attempt), 1.0))
                    continue
                raise CliError(f"request failed for {method} {path}: {exc}") from exc
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise CliError(f"non-JSON response for {method} {path}: {body[:500]!r}") from exc


def _is_retryable_transport_error(exc: urllib.error.URLError) -> bool:
    reason = getattr(exc, "reason", exc)
    if isinstance(reason, _RETRYABLE_TRANSPORT_ERRORS):
        return True
    message = str(reason).lower()
    return "connection reset" in message or "remote end closed connection" in message or "timed out" in message
