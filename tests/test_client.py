import io
import unittest
import urllib.error
import urllib.request
from unittest import mock

from gpu_func_cli.client import RestClient
from gpu_func_cli.errors import CliError


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self.body


class ClientTests(unittest.TestCase):
    def test_rest_client_posts_json_with_headers(self):
        seen = {}

        def fake_urlopen(req, timeout):
            seen["req"] = req
            seen["timeout"] = timeout
            return FakeResponse(b'{"ok": true}')

        client = RestClient(api_base="https://example.test/api/", api_key="secret", request_timeout=3, poll_interval=0)
        with mock.patch.object(urllib.request, "urlopen", fake_urlopen):
            self.assertEqual(client.post_json("/v1/submit", {"a": 1}), {"ok": True})

        self.assertEqual(seen["timeout"], 3)
        self.assertEqual(seen["req"].full_url, "https://example.test/api/v1/submit")
        self.assertEqual(seen["req"].get_method(), "POST")
        self.assertEqual(seen["req"].get_header("X-api-key"), "secret")
        self.assertEqual(seen["req"].get_header("Content-type"), "application/json")
        self.assertEqual(seen["req"].data, b'{"a": 1}')

    def test_rest_client_wraps_http_errors(self):
        def fake_urlopen(req, timeout):
            raise urllib.error.HTTPError(
                req.full_url,
                500,
                "server error",
                hdrs={},
                fp=io.BytesIO(b"broken"),
            )

        client = RestClient(api_base="https://example.test/api", api_key=None, request_timeout=3, poll_interval=0)
        with mock.patch.object(urllib.request, "urlopen", fake_urlopen):
            with self.assertRaisesRegex(CliError, "HTTP 500"):
                client.get_json("/v1/workers")

    def test_bundle_upload_retries_connection_reset(self):
        calls = []

        def fake_urlopen(req, timeout):
            calls.append(req.full_url)
            if len(calls) == 1:
                raise urllib.error.URLError(ConnectionResetError(104, "Connection reset by peer"))
            return FakeResponse(b'{"bundle_id": "bundle-1"}')

        client = RestClient(api_base="https://example.test/api", api_key=None, request_timeout=3, poll_interval=0)
        with mock.patch.object(urllib.request, "urlopen", fake_urlopen):
            result = client.post_bytes(
                "/v1/bundles?sha256=abc",
                b"payload",
                content_type="application/octet-stream",
            )

        self.assertEqual(result, {"bundle_id": "bundle-1"})
        self.assertEqual(len(calls), 2)

    def test_submit_does_not_retry_connection_reset(self):
        calls = []

        def fake_urlopen(req, timeout):
            calls.append(req.full_url)
            raise urllib.error.URLError(ConnectionResetError(104, "Connection reset by peer"))

        client = RestClient(api_base="https://example.test/api", api_key=None, request_timeout=3, poll_interval=0)
        with mock.patch.object(urllib.request, "urlopen", fake_urlopen):
            with self.assertRaisesRegex(CliError, "Connection reset"):
                client.post_json("/v1/submit", {"bundle_id": "bundle-1"})

        self.assertEqual(len(calls), 1)

    def test_wait_job_timeout_uses_cli_timeout_exit_code(self):
        client = RestClient(api_base="https://example.test/api", api_key=None, request_timeout=3, poll_interval=0)
        with mock.patch.object(client, "get_json", return_value={"status": "queued"}):
            with self.assertRaisesRegex(CliError, "did not finish") as ctx:
                client.wait_job("job-1", timeout_s=0.001)

        self.assertEqual(ctx.exception.exit_code, 4)


if __name__ == "__main__":
    unittest.main()
