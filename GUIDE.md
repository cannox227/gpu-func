# gpu_func_cli — Guide

Run CUDA **course exercises** and **custom kernels** on a remote GPU through the
GFAAS REST API. Your local machine needs no CUDA, `nvcc`, Nsight Compute, or
GPU — the CLI ships a self-contained job to a GFAAS worker, and the worker does
the CUDA work. (Local `.ncu-rep` parsing with `report summary` / `report
feedback` additionally needs Nsight Compute's `ncu_report.py` — still no GPU.)

## Contents

1. [Install](#1-install)
2. [Configure GFAAS](#2-configure-gfaas)
3. [Quickstart](#3-quickstart)
4. [Hands-on walkthrough](#4-hands-on-walkthrough)
5. [Course exercises](#5-course-exercises)
6. [Custom kernels](#6-custom-kernels)
7. [Reports: summary vs feedback](#7-reports-summary-vs-feedback)
8. [Course website example reports](#8-course-website-example-reports)
9. [Command reference](#9-command-reference)
10. [What happens internally](#10-what-happens-internally)
11. [Exit codes](#11-exit-codes)
12. [Troubleshooting](#12-troubleshooting)

**New here?** Read 1–4 top to bottom. **Already have a `.cu`?** Jump to
[5 (exercises)](#5-course-exercises) or [6 (custom)](#6-custom-kernels).
**Need a flag?** [Section 9](#9-command-reference).

---

## 1. Install

```bash
# On PATH (editable, so local edits apply live):
uv tool install --editable /path/to/gpu_func_cli
# ...or into a venv:
#   cd /path/to/gpu_func_cli && python3 -m venv .venv && . .venv/bin/activate && pip install .

gpu_func_cli --help
```

The remote-run client uses Python standard-library modules only — no GFAAS SDK,
fast-containers, CUDA, Nsight Compute, or the cuda-course Python package.

## 2. Configure GFAAS

```bash
export GFAAS_API_BASE="https://<hub-host>/api"
export GFAAS_API_KEY="<your-api-key>"
gpu_func_cli workers      # expect a worker advertising gpu_type b200, image cuda-nvcc
```

Defaults: `--gpu B200`, `--gpu-type b200`, `--image cuda-nvcc`, `--arch sm_100a`.
If `gpu_func_cli workers` lists a B200 / `cuda-nvcc` worker, you are ready.

## 3. Quickstart

Exercises run against a **cuda-course checkout** (the CLI ships no exercise
content). Point `--course-root` at one — or set `CUDA_COURSE_REPO`, or run from
inside a checkout — and grade your `haxpy.cu` solution:

```bash
export CUDA_COURSE_REPO=/path/to/cuda-course
gpu_func_cli exercise 01-haxpy grade --file your_haxpy.cu --gpu B200
gpu_func_cli exercise 01-haxpy profile benchmarks/01_aligned_small.txt \
  --file your_haxpy.cu --gpu B200 --artifact-dir ./ncu-artifacts
```

`your_haxpy.cu` is **your completed solution** — the course starter
`exercises/01-haxpy/haxpy.cu` is an empty `// TODO` stub that fails `grade`. No
solution yet? The [walkthrough](#4-hands-on-walkthrough) builds a working one,
or copy a known-correct `solutions/correct/basic.cu` from the checkout.

(`custom` kernels need no checkout — see [Section 6](#6-custom-kernels).)

`grade` runs test + sanitizer + benchmark; `profile` prints exercise-specific
feedback and saves `haxpy.aligned_small.ncu-rep`. For any other exercise, see
[Section 5](#5-course-exercises). For your own kernels, [Section 6](#6-custom-kernels).

## 4. Hands-on walkthrough

A complete copy-paste journey for a fresh install with live GFAAS credentials.
It creates its own test files, so you do not need to bring a CUDA program.

### 4.1 Create the test files

```bash
mkdir -p /tmp/gpu-course-demo
```

**Self-contained custom program** (its own `main()`, so no harness):

```bash
cat > /tmp/gpu-course-demo/vecadd.cu <<'EOF'
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <cstdio>
#include <vector>

__global__ void vecadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int n = 1 << 20;
    std::vector<float> a(n, 1.0f), b(n, 2.0f), c(n);

    float *da, *db, *dc;
    cudaMalloc(&da, n * sizeof(float));
    cudaMalloc(&db, n * sizeof(float));
    cudaMalloc(&dc, n * sizeof(float));
    cudaMemcpy(da, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int block = 256, grid = (n + block - 1) / block;
    nvtxRangePush("profile_kernel");           // lets `custom profile` work without --no-nvtx-filter
    vecadd<<<grid, block>>>(da, db, dc, n);
    cudaDeviceSynchronize();
    nvtxRangePop();

    cudaMemcpy(c.data(), dc, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(da); cudaFree(db); cudaFree(dc);
    std::printf("vecadd c[0]=%.1f c[%d]=%.1f (expected 3.0)\n", c[0], n - 1, c[n - 1]);
    return 0;
}
EOF
```

**Kernel-only source + harness** (the kernel has no `main()`, so the harness
supplies one):

```bash
cat > /tmp/gpu-course-demo/scale_kernel.cu <<'EOF'
#include <cuda_runtime.h>

__global__ void scale_kernel(float* y, const float* x, int n, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = alpha * x[i];
}
void launch_scale(float* y, const float* x, int n, float alpha) {
    int block = 256, grid = (n + block - 1) / block;
    scale_kernel<<<grid, block>>>(y, x, n, alpha);
}
EOF

cat > /tmp/gpu-course-demo/scale_harness.cu <<'EOF'
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <cstdio>
#include <vector>

void launch_scale(float* y, const float* x, int n, float alpha);

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : (1 << 20);
    float alpha = argc > 2 ? std::atof(argv[2]) : 2.5f;
    std::vector<float> x(n), y(n);
    for (int i = 0; i < n; ++i) x[i] = (i % 1024) / 1024.0f;

    float *dx, *dy;
    cudaMalloc(&dx, n * sizeof(float)); cudaMalloc(&dy, n * sizeof(float));
    cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    nvtxRangePush("profile_kernel");
    launch_scale(dy, dx, n, alpha);
    cudaDeviceSynchronize();
    nvtxRangePop();

    cudaMemcpy(y.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(dy);
    std::printf("custom scale passed n=%d alpha=%f\n", n, alpha);
    return 0;
}
EOF
```

**An `01-haxpy` course solution:**

```bash
cat > /tmp/gpu-course-demo/haxpy.cu <<'EOF'
#include <cuda_bf16.h>
#include <cuda_runtime.h>

__global__ void haxpy_kernel(int n, float alpha, const nv_bfloat16* x, nv_bfloat16* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float xf = __bfloat162float(x[i]), yf = __bfloat162float(y[i]);
        y[i] = __float2bfloat16(alpha * xf + yf);
    }
}
void haxpy(int n, float alpha, const nv_bfloat16* x, nv_bfloat16* y) {
    int block = 256, grid = (n + block - 1) / block;
    haxpy_kernel<<<grid, block>>>(n, alpha, x, y);
}
EOF
```

### 4.2 Run a custom kernel

Self-contained source — no `--harness`:

```bash
gpu_func_cli custom run     /tmp/gpu-course-demo/vecadd.cu --gpu B200
gpu_func_cli custom profile /tmp/gpu-course-demo/vecadd.cu --gpu B200 --artifact-dir /tmp/gpu-course-demo/out
```

Kernel + harness:

```bash
gpu_func_cli custom run /tmp/gpu-course-demo/scale_kernel.cu \
  --harness /tmp/gpu-course-demo/scale_harness.cu --gpu B200
```

Expected (run) ends with:

```text
custom scale passed n=1048576 alpha=2.500000
Custom run passed
```

### 4.3 Run the course exercise

Exercises need a cuda-course checkout — point `CUDA_COURSE_REPO` at one (the
`haxpy.cu` you wrote above is your solution; `--file` can live anywhere):

```bash
export CUDA_COURSE_REPO=/path/to/cuda-course
gpu_func_cli exercise 01-haxpy compile --file /tmp/gpu-course-demo/haxpy.cu --gpu B200
gpu_func_cli exercise 01-haxpy grade   --file /tmp/gpu-course-demo/haxpy.cu --gpu B200
gpu_func_cli exercise 01-haxpy profile benchmarks/01_aligned_small.txt \
  --file /tmp/gpu-course-demo/haxpy.cu --gpu B200 --artifact-dir /tmp/gpu-course-demo/out
```

Expected (grade) ends with:

```text
All 7 test(s) passed
All 7 sanitizer run(s) passed
All 3 benchmark(s) passed
Grade passed
```

### 4.4 Inspect the reports (needs `ncu_report.py` locally)

```bash
gpu_func_cli report summary /tmp/gpu-course-demo/out/vecadd.ncu-rep --per-kernel
gpu_func_cli report feedback /tmp/gpu-course-demo/out/haxpy.aligned_small.ncu-rep \
  --course-dir /path/to/cuda-course \
  --exercise 01-haxpy --benchmark benchmarks/01_aligned_small.txt
```

## 5. Course exercises

Actions: `compile`, `test`, `benchmark`, `sanitizer`, `profile`, `grade`
(`grade` = test + sanitizer + benchmark). Add `--verbose` for a line per test;
`--keep-going` to continue after a failing spec.

### Point at a cuda-course checkout

Exercises are **not** vendored — the CLI ships no exercise content. Pass
`--course-root` (or set `CUDA_COURSE_REPO`, or run from inside a checkout) and
the CLI ships that exercise plus the live course `runner/` and runs the
exercise's own `run.py`, so output (correctness, GiB/s, feedback) is exact for
any exercise. The exercise's `solutions/` dir is never uploaded.

`--file` is **your completed solution**. For `01-haxpy` it must provide:

```cpp
void haxpy(int n, float alpha, const nv_bfloat16* d_x, nv_bfloat16* d_y);
```

```bash
export CUDA_COURSE_REPO=/path/to/cuda-course

gpu_func_cli exercise 01-haxpy grade --file your_haxpy.cu --gpu B200
gpu_func_cli exercise 01-haxpy test tests/01_corner_n1.txt --file your_haxpy.cu --gpu B200   # one test
gpu_func_cli exercise 01-haxpy profile benchmarks/01_aligned_small.txt \
  --file your_haxpy.cu --gpu B200 --artifact-dir ./ncu-artifacts

gpu_func_cli exercise 02-softmax128 grade \
  --file "$CUDA_COURSE_REPO/exercises/02-softmax128/solutions/correct/02-baseline.cu" --gpu B200

gpu_func_cli exercise 03-max-pool grade \
  --file "$CUDA_COURSE_REPO/exercises/03-max-pool/solutions/correct/naive.cu" --gpu B200
```

`profile` runs the real course runner (`ncu --set=full`), prints exercise
**Profiling Feedback**, and saves e.g. `haxpy.aligned_small.ncu-rep`.

### Exercise options

| Option | Meaning |
| --- | --- |
| `--file PATH` | Your CUDA solution to test. |
| `--course-root DIR` | A `cuda-course` checkout; enables any exercise. Auto-detected from `--file`/cwd. |
| `--gpu B200` | Target GPU label (`--gpu-type` / `--arch` derived from it). |
| `--image cuda-nvcc` | Worker image. |
| `--artifact-dir DIR` | Save returned `.ncu-rep` reports. |
| `--json PATH` | Write a machine-readable result JSON (job id, status, runner stdout/stderr, parsed `report_json`). |
| `--timeout SEC` / `--wait-timeout SEC` | Remote job / local poll timeouts. |
| `--verbose` / `--keep-going` | Line per test / continue after a failing spec. |
| `--report-max-mismatches N` | Cap mismatch lines (default 20). |

## 6. Custom kernels

```bash
gpu_func_cli custom compile SOURCE.cu [--harness HARNESS.cu] --gpu B200
gpu_func_cli custom run     SOURCE.cu [--harness HARNESS.cu] --gpu B200
gpu_func_cli custom profile SOURCE.cu [--harness HARNESS.cu] --gpu B200 --artifact-dir ./out
```

### Do you need a harness?

Not always. `custom` always links a real executable on the worker
(`nvcc <sources> -o custom_kernel`, then runs `./custom_kernel`), so it needs a
`main()` — but that `main()` can come from **either** the source or the harness:

- **Self-contained source** (already has `main()`, like `vecadd.cu` above): pass
  it alone, no `--harness`.
- **Kernel-only source** (just the `__global__` kernel + launcher): add a
  `--harness` that supplies `main()`, or the link fails with `undefined
  reference to main`.

So `--harness` is just a convenient place to add `main()` (allocation, init,
launch, optional correctness check) for a kernel that doesn't have one.

The harness/program must, for profiling, wrap the measured region in an NVTX
range named `profile_kernel`:

```cpp
nvtxRangePush("profile_kernel");
launch_my_kernel(...);
cudaDeviceSynchronize();
nvtxRangePop();
```

If neither source nor harness has that range, pass `--no-nvtx-filter` so
Nsight Compute profiles the whole binary instead of capturing nothing.

> Exercises differ: in `gpu_func_cli exercise …` your source is always
> kernel-only and the checkout's `tester.cu` is the harness — you never pass
> `--harness` to `exercise`.

### Custom options

| Option | Meaning |
| --- | --- |
| `SOURCE` | CUDA source. Sent as `kernel.cu` remotely. |
| `--harness PATH` | Optional file with `main()`. Sent as `harness.cu`. |
| `--arg VALUE` | Program argument (repeatable). |
| `--nvcc-flags STR` | Compile flags. Default `-std=c++20 -O3 -lineinfo`. |
| `--ncu-args STR` | Nsight Compute args. Default `--set basic`. Use `--set full` for warp-stall / SASS metrics. |
| `--nvtx-range NAME` | NVTX range to profile. Default `profile_kernel`. |
| `--no-nvtx-filter` | Profile the whole executable (no `profile_kernel` range needed). |
| `--report-name NAME` | Base name for the `.ncu-rep`. Default: source file stem (`vecadd.cu` → `vecadd.ncu-rep`). |
| `--output NAME` | Remote executable name. Default `custom_kernel`. |
| `--gpu` / `--image` / `--arch` / `--timeout` / `--json` / `--artifact-dir` / `--verbose` | As for exercises. |

## 7. Reports: summary vs feedback

```bash
gpu_func_cli report summary  REPORT.ncu-rep [--per-kernel] [--json PATH]
gpu_func_cli report feedback REPORT.ncu-rep \
  --course-dir /path/to/cuda-course --exercise 01-haxpy --benchmark benchmarks/01_aligned_small.txt [--json PATH]
```

- **`summary`** parses any `.ncu-rep` and prints duration, DRAM bytes/throughput,
  SM throughput, instructions, loads/stores, etc. No course checkout needed.
- **`feedback`** re-runs a specific exercise's `format_profiling()` rules against
  a report. It needs `--course-dir`, `--exercise`, and `--benchmark` because
  those define what "good" means (e.g. expected memory traffic comes from the
  benchmark's `n`). Only meaningful for a report that matches that exercise's
  contract, and the report must be metrics-rich (`--set full`).

Both report commands need `ncu_report.py` locally (ships with Nsight Compute, no
GPU required). If it is missing, the CLI says so; point Python at it:

```bash
export PYTHONPATH="/opt/nvidia/nsight-compute/<version>/extras/python:$PYTHONPATH"
```

For course kernels, prefer `exercise profile` over manual `report feedback` — it
profiles, parses, and applies the rules in one remote job.

### Metrics the course extracts

| Course key | Nsight Compute metric |
| --- | --- |
| `dram_read_bytes` / `dram_write_bytes` | `dram__bytes_read.sum` / `dram__bytes_write.sum` (or `..._op_read/write.sum`) |
| `dram_throughput` | `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` |
| `sm_throughput` | `sm__throughput.avg.pct_of_peak_sustained_elapsed` |
| `instructions` / `cycles` / `duration` | `smsp__inst_executed.sum` / `gpc__cycles_elapsed.max` / `gpu__time_duration.sum` |
| `loads` / `stores` / `ldgsts` | `sass__inst_executed_global_loads` / `..._global_stores` / `smsp__inst_executed_op_ldgsts.sum` |

## 8. Course website example reports

The `cuda-course` site (`content/*.md`) embeds Nsight Compute charts that read
`.ncu-rep` files under `examples/partN/reports/`. Pages map to `examples/partN/`
by their number prefix; every example `.cu` is self-contained, so no `--harness`.

Regenerate a report with `--set full` (charts need PC-sampling / SASS metrics),
naming it to match what the page expects:

```bash
cd /path/to/cuda-course

gpu_func_cli custom profile examples/part1/saxpy.cu --gpu B200 --no-nvtx-filter \
  --ncu-args "--set full" --report-name saxpy --artifact-dir examples/part1/reports

# Page 04's comparison table needs these two:
gpu_func_cli custom profile examples/part2/saxpy-v5.cu --gpu B200 --no-nvtx-filter \
  --ncu-args "--set full" --report-name saxpy-v5-b200 --artifact-dir examples/part2/reports
gpu_func_cli custom profile examples/part4/saxpy-v6.cu --gpu B200 --no-nvtx-filter \
  --ncu-args "--set full" --report-name saxpy-v6-b200 --artifact-dir examples/part4/reports
```

Per-page mapping:

| Page (→ dir) | Embedded exercise | Reports it reads |
| --- | --- | --- |
| `01-core-concepts.md` (part1) | — | `saxpy` (warp-stall chart) |
| `02-thread-coarsening.md` (part2) | `01-haxpy` | `saxpy-b4000`, `saxpy-b200`, `saxpy-v2-b4000`, `saxpy-v3a-b4000` |
| `03-warp-shuffles.md` (part3) | `02-softmax128` | none |
| `04-async-data-movement.md` (part4) | `03-max-pool` | `part2/saxpy-v5-b200`, `saxpy-v6-b200` |

### Rebuild the JSON cache

The site reads a `.json` cache next to each `.ncu-rep`, and it only regenerates
that cache when it is **missing** — replacing a `.ncu-rep` does not refresh it.
After (re)generating a report, rebuild its cache:

```bash
cd /path/to/cuda-course
export PYTHONPATH="/opt/nvidia/nsight-compute/<version>/extras/python:src:$PYTHONPATH"
python -c "from cuda_course import ncu_cache as c; \
c.generate_report_cache('examples/part1/reports/saxpy.ncu-rep'); print('ok')"
```

### Warp-stall chart on Blackwell / newer Nsight Compute

B200 + Nsight Compute 2025.x do not emit the PC-sampling
`smsp__pcsamp_warps_issue_stalled_*` metrics even with `--set full`; they emit
`smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio` instead.
`cuda-course`'s `runner/ncu_utils.get_ncu_stall_reasons_from_dict()` falls back
to those averages, so the chart works on both — just capture with `--set full`.

## 9. Command reference

```bash
gpu_func_cli workers
gpu_func_cli exercise <id> <compile|test|benchmark|sanitizer|profile|grade> [specs...] [options]
gpu_func_cli custom   <compile|run|profile> SOURCE.cu [--harness H.cu] [options]
gpu_func_cli report   summary  REPORT.ncu-rep [--per-kernel] [--json PATH]
gpu_func_cli report   feedback REPORT.ncu-rep --course-dir DIR --exercise ID --benchmark PATH [--json PATH]
```

Top-level options (before the subcommand): `--api-base` / `--api-key` (default to
`GFAAS_API_BASE` / `GFAAS_API_KEY`), `--request-timeout` (60s),
`--poll-interval` (1s). Per-command options are in
[Section 5](#exercise-options) and [Section 6](#custom-options).

## 10. What happens internally

For remote commands the CLI: validates local paths → builds a JSON payload →
embeds it in a small Python worker module → tars and uploads it
(`POST /v1/bundles`) → submits the job (`POST /v1/submit`) → polls
(`GET /v1/jobs/<id>`) → fetches the result (`GET /v1/jobs/<id>/result_json`) →
prints output and writes artifacts/JSON locally when requested.

On the worker, a custom job runs roughly:

```text
nvcc <flags> kernel.cu [harness.cu] -o custom_kernel
./custom_kernel
# profile adds: ncu --set basic --nvtx --nvtx-include profile_kernel/ \
#   --force-overwrite --export <source-stem> ./custom_kernel
```

Exercise jobs ship the live cuda-course `runner/` plus the chosen exercise and
run that exercise's own `run.py`, so output and feedback match the checkout
exactly. The CLI bundles no exercise content.

## 11. Exit codes

| Code | Meaning |
| --- | --- |
| `0` | success |
| `1` | compile failure |
| `2` | crash or sanitizer/runtime error |
| `3` | wrong answer |
| `4` | timeout |
| `5` | setup, API, worker, or report-parser issue |
| `130` | interrupted |

## 12. Troubleshooting

- **`GFAAS_API_BASE is not set`** — export `GFAAS_API_BASE` and `GFAAS_API_KEY`.
- **No live workers / `cuda-nvcc` missing** — run `gpu_func_cli workers`; if no
  B200 / `cuda-nvcc` worker appears, the backend is offline/busy or the GFAAS
  operator must prepare the image.
- **`nvcc`, `ncu`, or `compute-sanitizer` missing** — a worker-image issue; the
  local machine never installs CUDA for remote runs.
- **`ncu_report.py is not available`** — only needed for local `report`
  commands; set `PYTHONPATH` to Nsight Compute's `extras/python` (see Section 7).
- **Profile is slow** — `--set full` replays the kernel once per metric pass;
  that's expected. Course `profile` uses `--set=full` for feedback metrics.
- **An exercise won't start** — exercises need a cuda-course checkout; pass
  `--course-root <cuda-course>`, set `CUDA_COURSE_REPO`, or run from inside a
  checkout. See Section 5.
- **`report feedback` gives odd advice** — use it only for a report matching the
  selected exercise + benchmark; for arbitrary kernels use `report summary`.
- **Warp-stall chart empty on B200** — capture with `--set full`; see Section 8.
