# gpu_func_cli

Standalone CLI for running **course exercises** and **custom CUDA kernels** on a
remote GPU through the GFAAS REST API. The local machine needs no CUDA, `nvcc`,
Nsight Compute, or GPU. The CLI sends a self-contained job to a GFAAS worker,
and the worker does the CUDA work.

Full documentation: [`GUIDE.md`](GUIDE.md). It covers install, configuration,
running a `starter.zip` exercise, custom kernels (with and without a harness),
reports and feedback, command reference, and troubleshooting.

## Quick start

```bash
uv tool install --editable /path/to/gpu_func_cli   # or: pip install .
export GFAAS_API_BASE="https://<hub-host>/api"
export GFAAS_API_KEY="<your-api-key>"
gpu_func_cli workers
```

### Run a course exercise (starter.zip)

Unzip the `starter.zip`, edit the starter `.cu`, then run an action from inside
the folder — the exercise is auto-detected from the cwd:

```bash
unzip 01-haxpy.zip -d 01-haxpy && cd 01-haxpy
# edit haxpy.cu (your solution), then:
gpu_func_cli test          # correctness tests
gpu_func_cli benchmark     # timing + GiB/s + % of peak
gpu_func_cli grade         # full suite: test + sanitizer + benchmark
# from elsewhere, point at the unzipped dir: --exercise-dir /path/to/01-haxpy
```

### Run a custom kernel

```bash
# any self-contained .cu (has its own main()) — nothing else to bring:
gpu_func_cli custom run /path/to/your_kernel.cu --gpu B200

# kernel-only source? add a --harness that supplies main():
gpu_func_cli custom run kernel.cu --harness harness.cu --gpu B200

# profile on the GPU, then read the report locally:
gpu_func_cli custom profile your_kernel.cu --gpu B200 --artifact-dir ./out
gpu_func_cli report summary ./out/your_kernel.ncu-rep --per-kernel
```

New to the tool? Section 3 of `GUIDE.md` walks the
[starter.zip flow](GUIDE.md#3-run-a-starterzip-exercise) end to end, and the
[custom walkthrough](GUIDE.md#5-hands-on-walkthrough-custom) creates its own
test files so you don't need to bring a CUDA program.
