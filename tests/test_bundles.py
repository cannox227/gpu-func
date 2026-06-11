import base64
import json
import re
import tarfile
import unittest
from io import BytesIO
from unittest import mock

from gpu_func_cli.bundles import _build_bundle


class BundleTests(unittest.TestCase):
    def test_build_bundle_embeds_encoded_job_payload(self):
        payload = {"schema_version": 1, "files": {"kernel.cu": "int main() { return 0; }\n"}}

        bundle = _build_bundle(payload)

        self.assertEqual(set(bundle), {"data", "sha256"})
        with tarfile.open(fileobj=BytesIO(bundle["data"]), mode="r:gz") as tar:
            names = tar.getnames()
            self.assertEqual(names, ["gpu_func_job.py"])
            source = tar.extractfile("gpu_func_job.py").read().decode("utf-8")
        self.assertNotIn("__JOB_B64_PLACEHOLDER__", source)
        match = re.search(r'_JOB_B64 = "([^"]+)"', source)
        self.assertIsNotNone(match)
        self.assertEqual(json.loads(base64.b64decode(match.group(1)).decode("utf-8")), payload)

    def test_build_bundle_digest_is_stable_without_gzip_clock(self):
        payload = {"schema_version": 1, "files": {"a": "b"}}

        with mock.patch("gzip.time.time", side_effect=AssertionError("gzip mtime must be explicit")):
            first = _build_bundle(payload)
            second = _build_bundle(payload)

        self.assertEqual(first["sha256"], second["sha256"])
        self.assertEqual(first["data"], second["data"])


if __name__ == "__main__":
    unittest.main()
