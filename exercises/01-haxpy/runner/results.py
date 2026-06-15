"""Result and report processing — generic across all exercises.

Two related stages live here, in the order data flows through them:

  1. Report hierarchy: the structured payload the tester binary emits
     (see tester.h Reporter), parsed back out of REPORT_PATH via
     parse_report().

         Report          base: status only
         ├── TestReport      test / sanitizer runs: mismatch records
         ├── BenchmarkReport benchmark runs: timing metrics
         └── ProfileReport   profile runs: ncu metric collection over the
                             profile_kernel/ NVTX range

     The tester emits a `mode=` line on every run, so parse_report dispatches
     on it directly — each subclass builds itself from the parsed fields.

  2. RunResult: one durable record per (config, mode) invocation, written
     to JSON. It wraps the runner's own framing (config path, captured
     stdout/stderr, final status) around an optional embedded Report. The
     report is absent for runs that never produced one — timeouts, crashes,
     setup errors.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Structured report hierarchy (see tester.h Reporter). The tester emits a
# `mode=` line on every run; parse_report dispatches on it to the matching
# subclass. The reports themselves are plain data holders.
# ---------------------------------------------------------------------------

@dataclass
class Report:
    """Base: fields guaranteed by every tester run."""
    status: str  # "passed" | "failed"

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TestReport(Report):
    """Produced by --test and --sanitizer runs."""
    count: int = 0                             # total mismatches (may exceed len(mismatches_list))
    mismatches_list: list[dict[str, str]] = field(default_factory=list)


@dataclass
class ProfileReport(Report):
    """Produced by --profile runs.
    """
    device: str = ""
    metrics: dict[str, dict] = field(default_factory=dict)

@dataclass
class BenchmarkReport(Report):
    """Produced by --benchmark runs."""
    iters: int = 0
    avg_ms_wall: float = 0.0
    avg_ms_gpu: float = 0.0
    peak_bw: float = 0.0


@dataclass
class ProfileFeedback:
    KINDS = ("warning", "info", "praise", "technique")
    kind: str
    text: str


def parse_report(text: str) -> Report:
    """Parse the REPORT_PATH key=value block into the Report for its mode.

    The repeated `mismatch=` key is collected into a list; every other key
    lands flat in `fields`. Dispatch is on the `mode=` line the tester always
    emits, defaulting to a TestReport when it's missing or unrecognized.
    """
    mismatches: list[dict[str, str]] = []
    fields: dict[str, str] = {}

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key == "mismatch":
            mismatches.append(_parse_mismatch_fields(value))
        else:
            fields[key] = value

    status = fields.get("status", "")
    mode = fields.get("mode", "")

    if mode == "benchmark":
        return BenchmarkReport(
            status=status,
            iters=int(fields.get("iters", 0)),
            avg_ms_wall=float(fields.get("avg_ms_wall", 0.0)),
            avg_ms_gpu=float(fields.get("avg_ms_gpu", 0.0)),
            peak_bw=float(fields.get("peak_bw", 0.0)),
        )
    if mode == "profile":
        return ProfileReport(status=status)

    try:
        count = int(fields.get("mismatches", 0))
    except ValueError:
        count = 0
    return TestReport(status=status, count=count, mismatches_list=mismatches)


def _parse_mismatch_fields(s: str) -> dict[str, str]:
    """Parse `<index> k=v k=v ...`; leading bare token stored under 'index'."""
    fields: dict[str, str] = {}
    for i, tok in enumerate(s.split()):
        k, sep, v = tok.partition("=")
        if sep:
            fields[k] = v
        elif i == 0:
            fields["index"] = tok
    return fields


# ---------------------------------------------------------------------------
# RunResult — the per-invocation record serialized to JSON.
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    config: str
    name:   str
    status: str                       # "passed" | "failed" | "timeout" | "error" | "setup-error"
    stdout: str = ""
    stderr: str = ""
    # The parsed tester payload, when one was produced. Absent for runs that
    # never got that far (timeout, crash, setup error).
    report: Optional[Report] = None

    @classmethod
    def for_config(cls, cfg: Path, status: str, **kw: Any) -> "RunResult":
        """Build a record for a run with no usable report (timeout/crash/setup)."""
        return cls(config=str(cfg), name=cfg.stem, status=status, **kw)

    @classmethod
    def from_report(cls, cfg: Path, report: Report, **kw: Any) -> "RunResult":
        """Build a record for a completed run, embedding its parsed report."""
        status = "passed" if report.passed else "failed"
        return cls(config=str(cfg), name=cfg.stem, status=status,
                   report=report, **kw)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "config": self.config,
            "name":   self.name,
            "status": self.status,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }
        if self.report is not None:
            d["report"] = self.report.to_dict()
        return d


def summarize(results: list[RunResult]) -> dict[str, int]:
    """Tally results by status."""
    counts = {"passed": 0, "failed": 0, "skipped": 0,
              "error": 0, "timeout": 0, "setup-error": 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    return counts


def write_json(path: Path, mode: str, binary: Path, results: list[RunResult]) -> None:
    payload = {
        "mode":    mode,
        "binary":  str(binary),
        "runs":    [r.to_dict() for r in results],
        "summary": summarize(results),
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"\n  Results written to {path}")
