"""Core NCU utilities: report I/O + raw extraction + pure interpreters.

This module needs the NVIDIA ``ncu_report`` library and reads ``.ncu-rep`` files,
but it does NOT cache. The canonical metric dict shape it produces/consumes is:

    { action_name: { metric_name: scalar | {instance: value} } }

Caching lives in ``ncu_cache`` (which imports this module). Interactive / CLI
code should import only this module, so the caching code is never loaded and
"no caching when used interactively" holds structurally — not by convention.
"""

import os
import sys
import glob
import re


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class NCUError(Exception):
    """Base class for NCU-related errors."""
    pass


class NCUReportNotFoundError(NCUError):
    """Raised when the NCU report file is not found."""
    pass


class NCUMetricNotFoundError(NCUError):
    """Raised when a requested metric is not found in the report."""
    pass


class NCULibraryNotFoundError(NCUError):
    """Raised when ncu_report library is not available."""
    pass


# ---------------------------------------------------------------------------
# Library discovery + report I/O
# ---------------------------------------------------------------------------

def find_ncu_python_path():
    """Try to find the path to ncu_report.py in common NVIDIA installation locations."""
    search_paths = [
        '/opt/nvidia/nsight-compute/*/extras/python/',
        '/usr/local/cuda/nsight-compute-*/extras/python/',
    ]

    found_paths = []
    for pattern in search_paths:
        found_paths.extend(glob.glob(pattern))

    if not found_paths:
        return None

    # Sort numerically on the version components in the path
    def version_key(p):
        return [int(x) for x in re.findall(r"\d+", p)]
    found_paths.sort(key=version_key, reverse=True)
    return found_paths[0]


# Add NCU Python path to sys.path
ncu_path = find_ncu_python_path()
if ncu_path:
    if ncu_path not in sys.path:
        sys.path.append(ncu_path)
    try:
        import ncu_report
    except ImportError:
        ncu_report = None
else:
    ncu_report = None


def simplify_gpu_name(full_name):
    """Simplify a full GPU name, e.g., 'NVIDIA GeForce RTX 3080' -> 'rtx3080'."""
    if not full_name:
        return None

    full_name = full_name.lower()

    # Mapping for common datacentre GPUs.
    if "rtx" not in full_name:
        if "h100" in full_name: return "h100"
        if "a100" in full_name: return "a100"
        if "v100" in full_name: return "v100"
        if "t4" in full_name: return "t4"
        if "b200" in full_name: return "b200"
        if "b300" in full_name: return "b300"

    # RTX Pro blackwell
    if "rtx pro blackwell" in full_name: return "rtxproblackwell"
    pro_match = re.search(r"rtx\s*pro\s*(\d+)\s*blackwell", full_name)
    if pro_match:
        return f"b{pro_match.group(1)}"

    # Handle gaming RTX cards (e.g., "nvidia geforce rtx 5060 ti" -> "rtx5060ti")
    rtx_match = re.search(r"rtx\s*(\d+\w*)", full_name)
    if rtx_match:
        return f"rtx{rtx_match.group(1)}"

    # Fallback: remove spaces and non-alphanumeric
    simplified = re.sub(r'[^a-z0-9]', '', full_name.replace("nvidia", "").replace("geforce", "").replace("tesla", "").replace("quadro", ""))
    return simplified if simplified else "gpu"


def load_report(report_path):
    if ncu_report is None:
        raise NCULibraryNotFoundError("ncu_report library not found")

    if not os.path.exists(report_path):
        raise NCUReportNotFoundError(f"Report file {report_path} not found")

    return ncu_report.load_report(str(report_path))


def extract_all_metrics(action) -> dict:
    """
    Extracts all metrics that are relevant for our purposes from an action.
    """
    metrics = {}
    _blocked = ["breakdown:", "group:", "warpsampling:", "pmsampling:", "TriageCompute.", "nvlink__", "profiler__", "numa__", "launch__occupancy_per"]
    for metric_name in action.metric_names():
        log_metric = True
        for blocked in _blocked:
            if metric_name.startswith(blocked):
                log_metric = False
                break
        if not log_metric:
            continue

        metric = action.metric_by_name(metric_name)
        if not metric.has_correlation_ids():
            metrics[metric_name] = metric.value()
        elif metric.num_instances() > 1:
            cids = metric.correlation_ids()
            if cids.as_string(0) is None:
                continue
            value_dict = {}
            for k in range(cids.num_instances()):
                value_dict[cids.as_string(k)] = metric.value(k)
            metrics[metric_name] = value_dict
        else:
            metrics[metric_name] = metric.value()
    return metrics


def report_to_dict(report) -> dict:
    """Exhaustive producer: dump every relevant metric for every action.

    Iterates all ranges (NCU reports are virtually always single-range, but
    looping is free and keeps the per-kernel set complete regardless). Actions
    are keyed by name; if the same name recurred across ranges, the later wins.
    """
    as_dict = {}
    for i in range(report.num_ranges()):
        range_obj = report.range_by_idx(i)
        for j in range(range_obj.num_actions()):
            action = range_obj.action_by_idx(j)
            as_dict[action.name()] = extract_all_metrics(action)
    return as_dict


def get_gpu_name_from_report(report_path):
    """Read the GPU display name directly from the report (no cache written).

    Used at build time before a cache exists, to derive the report's filename
    suffix, so it reads in-memory and does not persist a JSON cache for what may
    be a temporary report file.
    """
    data = report_to_dict(load_report(report_path))
    return get_ncu_metric_from_dict(data, "", "device__attribute_display_name")


# ---------------------------------------------------------------------------
# Pure interpreters: metric dict in, value out.
# No I/O, no library, no cache — shared by the cached (markdown) path and the
# live (CLI) path alike.
# ---------------------------------------------------------------------------

def _format_metric_value(value) -> str:
    """Format a raw metric value as a display string.

    Integers render without a decimal point; non-integral floats are rounded to
    two decimals; strings pass through unchanged.
    """
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:.2f}"
    return str(value)


def _as_number(value):
    """Coerce a (possibly string) metric value to int/float; 0.0 on failure."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    try:
        f = float(value)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return 0.0


def _find_action_metrics(data: dict, kernel_name_pattern: str) -> dict:
    """Return the metric dict for the first action whose name contains the
    pattern (an empty pattern matches the first action)."""
    for action_name, metrics in data.items():
        if kernel_name_pattern in action_name:
            return metrics
    raise NCUMetricNotFoundError(
        f"No kernel matching `{kernel_name_pattern}` found in report"
    )


def get_ncu_metric_from_dict(data: dict, kernel_name_pattern: str, metric_name: str) -> str:
    """Extract a single formatted metric value from an already-loaded dict.

    ``metric_name`` may be a plain name or carry an instance selector, e.g.
    ``sass__inst_executed_per_opcode[FFMA]``.
    """
    base_metric_name = metric_name
    instance_name = None
    if metric_name.endswith("]") and "[" in metric_name:
        base_metric_name, instance_name = metric_name[:-1].split("[", 1)

    metrics = _find_action_metrics(data, kernel_name_pattern)

    if base_metric_name not in metrics:
        raise NCUMetricNotFoundError(
            f"Metric `{metric_name}` not found for kernel `{kernel_name_pattern}`"
        )
    value = metrics[base_metric_name]

    if instance_name is not None:
        if not isinstance(value, dict):
            raise NCUMetricNotFoundError(
                f"Metric `{base_metric_name}` does not have instances"
            )
        if instance_name not in value:
            raise NCUMetricNotFoundError(
                f"Instance `{instance_name}` not found in metric `{base_metric_name}`"
            )
        return _format_metric_value(value[instance_name])

    if isinstance(value, dict):
        raise NCUMetricNotFoundError(
            f"Metric `{base_metric_name}` has instances; request a specific one "
            f"as `{base_metric_name}[instance]`"
        )
    return _format_metric_value(value)


def get_ncu_opcodes_from_dict(data: dict, kernel_name_pattern: str,
                              metric_name: str = "sass__inst_executed_per_opcode") -> dict:
    """Extract all instances (opcodes) and their numeric values for a metric."""
    metrics = _find_action_metrics(data, kernel_name_pattern)
    value = metrics.get(metric_name)
    if not isinstance(value, dict):
        raise NCUMetricNotFoundError(
            f"Opcode metric `{metric_name}` not found (or has no instances) "
            f"for kernel `{kernel_name_pattern}`"
        )
    return {opcode: _as_number(v) for opcode, v in value.items()}


def get_ncu_stall_reasons_from_dict(data: dict, kernel_name_pattern: str) -> dict:
    """Extract warp stall reasons (issue-stall sampling) from a metric dict.

    Returns ``{reason: value}`` for every ``smsp__average_warps_issue_stalled_*per_issue_active.ratio``
    metric with a positive value (excluding the ``*_not_issued`` variants).
    """
    metrics = _find_action_metrics(data, kernel_name_pattern)

    prefix = "smsp__average_warps_issue_stalled"
    suffix = "per_issue_active.ratio"
    results = {}
    for metric_name, value in metrics.items():
        if not metric_name.startswith(prefix) or not metric_name.endswith(suffix):
            continue
        if isinstance(value, dict):
            continue
        num = _as_number(value)
        if num > 0:
            reason = metric_name[len(prefix):-len(suffix)].replace("_", " ").title()
            results[reason] = num
    if not results:
        raise NCUMetricNotFoundError(
            f"Stall metrics not found for kernel `{kernel_name_pattern}`"
        )
    return results
