import inspect
import os
from typing import List

from .results import ProfileFeedback
from .testing import TestConfig


class Exercise:
    name: str         = None
    tester: list[str] = []
    source: str       = None
    key_benchmarks    = []
    arch:str          = "80"
    _base_dir: str    = None

    @property
    def base_dir(self):
        if not self._base_dir:
            self._base_dir = os.path.dirname(os.path.abspath(inspect.getfile(type(self))))
        return self._base_dir

    def include_path(self):
        return os.path.abspath(os.path.join(self.base_dir, "runner", "include"))

    def format_failure(self, payload, config: TestConfig) -> str:
        """Render a structured failure payload into human-readable lines.

        `payload` is a testing.FailurePayload; `config` is the TestConfig
        for the failing test. Exercises override this to interpret their own
        mismatch fields (e.g. "row 3 col 12: window dropped its halo"). The
        default renders the raw records generically.
        """
        if not payload.mismatches_list and payload.count == 0:
            return ""
        lines = [f"{payload.count} mismatch(es)"]
        shown = len(payload.mismatches_list)
        for m in payload.mismatches_list:
            idx = m.get("index", "?")
            rest = " ".join(f"{k}={v}" for k, v in m.items() if k != "index")
            lines.append(f"  [{idx}] {rest}" if rest else f"  [{idx}]")
        if payload.count > shown:
            lines.append(f"  … and {payload.count - shown} more (capped)")
        return "\n".join(lines)

    def format_benchmark(self, payload, config) -> str:
        return f"benchmark result: {payload}"

    def format_profiling(self, payload, config) -> List[ProfileFeedback]:
        return []

