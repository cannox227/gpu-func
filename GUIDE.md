# gpu_func_cli Guide

Run **custom CUDA kernels** on a remote GPU through the GFAAS REST API. Your
local machine needs no CUDA, `nvcc`, Nsight Compute, or GPU. The CLI sends a
self-contained job to a GFAAS worker, and the worker does the CUDA work. (Local
`.ncu-rep` parsing with `report summary` additionally needs Nsight Compute's
`ncu_report.py`; no GPU is needed for parsing.)

## Contents

1. [Install](#1-install)
2. [Configure GFAAS](#2-configure-gfaas)
3. [Quickstart](#3-quickstart)
4. [Hands-on walkthrough](#4-hands-on-walkthrough)
5. [Custom kernels](#5-custom-kernels)
6. [Reports and feedback](#6-reports-and-feedback)
7. [Command reference](#7-command-reference)
8. [What happens internally](#8-what-happens-internally)
9. [Exit codes](#9-exit-codes)
10. [Troubleshooting](#10-troubleshooting)

Recommended path: read Sections 1-4 first, then [5](#5-custom-kernels) for the
full custom-kernel reference and [6](#6-reports-and-feedback) for the
profile-and-read feedback loop.

---

## 1. Install

```bash
# On PATH (editable, so local edits apply live):
uv tool install --editable /path/to/gpu_func_cli
# ...or into a venv:
#   cd /path/to/gpu_func_cli && python3 -m venv .venv && . .venv/bin/activate && pip install .

gpu_func_cli --help
```

The remote-run client uses Python standard-library modules only; no GFAAS SDK,
fast-containers, CUDA, or Nsight Compute.

## 2. Configure GFAAS

```bash
export GFAAS_API_BASE="https://<hub-host>/api"
export GFAAS_API_KEY="<your-api-key>"
gpu_func_cli workers      # expect a worker advertising gpu_type b200, image cuda-nvcc
```

Defaults: `--gpu B200`, `--gpu-type b200`, `--image cuda-nvcc`, `--arch sm_100a`.
If `gpu_func_cli workers` lists a B200 / `cuda-nvcc` worker, you are ready.

## 3. Quickstart

`custom` runs any `.cu` on the remote GPU — no checkout, nothing to bring but
your source. The fastest check is a self-contained program that has its own
`main()`:

```bash
gpu_func_cli custom run /path/to/your_kernel.cu --gpu B200
```

If you don't have one handy, the [walkthrough](#4-hands-on-walkthrough) writes a
working `vecadd.cu` (no harness) and a kernel-plus-harness pair you can run
as-is. `custom` has three actions:

```bash
gpu_func_cli custom compile SOURCE.cu [--harness H.cu] --gpu B200   # just build
gpu_func_cli custom run     SOURCE.cu [--harness H.cu] --gpu B200   # build + run
gpu_func_cli custom profile SOURCE.cu [--harness H.cu] --gpu B200 --artifact-dir ./out
```

See [Section 5](#5-custom-kernels) for the harness rules and every flag, and
[Section 6](#6-reports-and-feedback) for reading the profile that
`custom profile` saves.

## 4. Hands-on walkthrough

This walkthrough creates temporary source files and runs them with live GFAAS
credentials. You do not need to bring a CUDA program.

### 4.1 Create the test files

```bash
mkdir -p /tmp/gpu-custom-demo
```

**Self-contained custom program** (its own `main()`, so no harness):

```bash
cat > /tmp/gpu-custom-demo/vecadd.cu <<'EOF'
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
cat > /tmp/gpu-custom-demo/scale_kernel.cu <<'EOF'
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

cat > /tmp/gpu-custom-demo/scale_harness.cu <<'EOF'
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

### 4.2 Run a custom kernel

Self-contained source, no `--harness`:

```bash
gpu_func_cli custom run     /tmp/gpu-custom-demo/vecadd.cu --gpu B200
gpu_func_cli custom profile /tmp/gpu-custom-demo/vecadd.cu --gpu B200 --artifact-dir /tmp/gpu-custom-demo/out
```

Kernel + harness:

```bash
gpu_func_cli custom run /tmp/gpu-custom-demo/scale_kernel.cu \
  --harness /tmp/gpu-custom-demo/scale_harness.cu --gpu B200
```

Expected (run) ends with:

```text
custom scale passed n=1048576 alpha=2.500000
Custom run passed
```

### 4.3 Inspect the report (needs `ncu_report.py` locally)

`custom profile` above saved `vecadd.ncu-rep`. Read it locally:

```bash
gpu_func_cli report summary /tmp/gpu-custom-demo/out/vecadd.ncu-rep --per-kernel
```

## 5. Custom kernels

```bash
gpu_func_cli custom compile SOURCE.cu [--harness HARNESS.cu] --gpu B200
gpu_func_cli custom run     SOURCE.cu [--harness HARNESS.cu] --gpu B200
gpu_func_cli custom profile SOURCE.cu [--harness HARNESS.cu] --gpu B200 --artifact-dir ./out
```

### Do you need a harness?

Not always. `custom` always links a real executable on the worker
(`nvcc <sources> -o custom_kernel`, then runs `./custom_kernel`), so it needs a
`main()`, but that `main()` can come from **either** the source or the harness:

- **Self-contained source** (already has `main()`, like `vecadd.cu` above): pass
  it alone, no `--harness`.
- **Kernel-only source** (just the `__global__` kernel + launcher): add a
  `--harness` that supplies `main()`, or the link fails with `undefined
  reference to main`.

So `--harness` is just a convenient place to add `main()` (allocation, init,
launch, optional correctness check) for a kernel that doesn't have one. One
harness can drive many kernels, and one kernel can be exercised by different
harnesses; pass run-time inputs with `--arg` (repeatable).

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
| `--gpu LABEL` | Target GPU label; `--gpu-type` / `--arch` are derived from it. Default `B200`. |
| `--image NAME` | Worker image. Default `cuda-nvcc`. |
| `--artifact-dir DIR` | Save returned `.ncu-rep` reports (for `profile`). |
| `--json PATH` | Write a machine-readable result JSON (job id, status, stdout/stderr). |
| `--timeout SEC` / `--wait-timeout SEC` | Remote job / local poll timeouts. |
| `--verbose` | Extra progress output. |

## 6. Reports and feedback

The feedback loop for a custom kernel is two commands: profile it on the worker,
then read the report locally.

```bash
gpu_func_cli custom profile mykernel.cu [--harness H.cu] --gpu B200 --artifact-dir ./out
gpu_func_cli report summary ./out/mykernel.ncu-rep [--per-kernel] [--json PATH]
```

- **`custom profile`** runs Nsight Compute on the worker and saves an
  `.ncu-rep` into `--artifact-dir`. By default it profiles only the
  `profile_kernel` NVTX range (override with `--nvtx-range`, or profile the
  whole binary with `--no-nvtx-filter`).
- **`report summary`** parses any `.ncu-rep` and prints duration, DRAM
  bytes/throughput, SM throughput, instructions, loads/stores, etc. No GPU and
  no checkout needed; `--per-kernel` breaks the numbers down per kernel launch.

For richer detail (warp-stall reasons, SASS-level metrics) capture with
`--ncu-args "--set full"`; note `--set full` replays the kernel once per metric
pass, so it is slower.

`report summary` needs `ncu_report.py` locally (ships with Nsight Compute, no
GPU required). If it is missing, the CLI says so; point Python at it:

```bash
export PYTHONPATH="/opt/nvidia/nsight-compute/<version>/extras/python:$PYTHONPATH"
```

## 7. Command reference

```bash
gpu_func_cli workers
gpu_func_cli custom <compile|run|profile> SOURCE.cu [--harness H.cu] [options]
gpu_func_cli report summary REPORT.ncu-rep [--per-kernel] [--json PATH]
```

Top-level options (before the subcommand): `--api-base` / `--api-key` (default to
`GFAAS_API_BASE` / `GFAAS_API_KEY`), `--request-timeout` (60s),
`--poll-interval` (1s). Per-command options are in [Section 5](#custom-options).

## 8. What happens internally

For remote commands the CLI validates local paths, builds a JSON payload,
embeds it in a small Python worker module, uploads it with `POST /v1/bundles`,
submits the job with `POST /v1/submit`, polls `GET /v1/jobs/<id>`, fetches
`GET /v1/jobs/<id>/result_json`, then prints output and writes artifacts/JSON
locally when requested.

On the worker, a custom job runs roughly:

```text
nvcc <flags> kernel.cu [harness.cu] -o custom_kernel
./custom_kernel
# profile adds: ncu --set basic --nvtx --nvtx-include profile_kernel/ \
#   --force-overwrite --export <source-stem> ./custom_kernel
```

The returned `.ncu-rep` is written to `--artifact-dir`; nothing CUDA-related
runs on your machine.

## 9. Exit codes

| Code | Meaning |
| --- | --- |
| `0` | success |
| `1` | compile failure |
| `2` | crash or runtime error |
| `3` | wrong answer (a harness correctness check returned non-zero) |
| `4` | timeout |
| `5` | setup, API, worker, or report-parser issue |
| `130` | interrupted |

## 10. Troubleshooting

- **`GFAAS_API_BASE is not set`**: export `GFAAS_API_BASE` and `GFAAS_API_KEY`.
- **No live workers / `cuda-nvcc` missing**: run `gpu_func_cli workers`; if no
  B200 / `cuda-nvcc` worker appears, the backend is offline/busy or the GFAAS
  operator must prepare the image.
- **`nvcc` or `ncu` missing**: a worker-image issue; the local machine never
  installs CUDA for remote runs.
- **`undefined reference to main`**: your source is kernel-only — add a
  `--harness` that supplies `main()` (see [Section 5](#do-you-need-a-harness)).
- **Profile captured nothing / empty report**: there is no `profile_kernel`
  NVTX range in the source or harness — add one, or pass `--no-nvtx-filter` to
  profile the whole binary.
- **`ncu_report.py is not available`**: only needed for local `report summary`;
  set `PYTHONPATH` to Nsight Compute's `extras/python` (see Section 6).
- **Profile is slow**: `--set full` replays the kernel once per metric pass;
  that's expected. `--set basic` (the default) is much faster.
