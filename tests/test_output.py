import argparse
import base64
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from gpu_func_cli.constants import RC_OK
from gpu_func_cli.output import _print_custom_result


class OutputTests(unittest.TestCase):
    def test_print_custom_result_writes_profile_artifact(self):
        payload = base64.b64encode(b"report bytes").decode("ascii")
        result = {
            "status": "passed",
            "compile": {"returncode": 0, "args": ["nvcc"], "stdout": ""},
            "run": {"returncode": 0, "args": ["ncu"], "stdout": "ok\n", "stderr": "", "timed_out": False},
            "artifacts": {"ncu_report": {"filename": "custom.ncu-rep", "content": payload}},
        }
        stdout = io.StringIO()
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(verbose=False, artifact_dir=tmp, custom_command="profile")
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = _print_custom_result(result, args)
            report_path = Path(tmp) / "custom.ncu-rep"
            self.assertEqual(report_path.read_bytes(), b"report bytes")

        self.assertEqual(code, RC_OK)
        self.assertIn("Custom profile passed", stdout.getvalue())
        self.assertIn("profile report", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
