# gpu_func_cli

Standalone CLI for running CUDA **course exercises** and **custom CUDA kernels**
on a remote GPU through the GFAAS REST API. The local machine needs no CUDA,
`nvcc`, Nsight Compute, or GPU. The CLI sends a self-contained job to a GFAAS
worker.

Full documentation: [`GUIDE.md`](GUIDE.md). It covers install, configuration,
the walkthrough, course exercises, custom kernels, reports, command reference,
and troubleshooting.

## Quick start

```bash
uv tool install --editable /path/to/gpu_func_cli   # or: pip install .
export GFAAS_API_BASE="https://<hub-host>/api"
export GFAAS_API_KEY="<your-api-key>"
gpu_func_cli workers

export CUDA_COURSE_REPO="set to your cuda-course repo"

# quick check: saxpy is self-contained, nothing to implement
gpu_func_cli custom run "$CUDA_COURSE_REPO/examples/part1/saxpy.cu" --gpu B200

# a graded exercise, using the solution the checkout ships:
gpu_func_cli exercise 01-haxpy grade \
  --file "$CUDA_COURSE_REPO/exercises/01-haxpy/solutions/correct/basic.cu" --gpu B200

# any other exercise (same pattern):
gpu_func_cli exercise 02-softmax128 grade \
  --file "$CUDA_COURSE_REPO/exercises/02-softmax128/solutions/correct/02-baseline.cu" --gpu B200
```

> Note: exercises need a cuda-course checkout (nothing is vendored). The commands
> above use the checkout's bundled `solutions/correct/*.cu` so they run without
> writing a kernel; swap `--file` for *your* solution to grade your own work. The
> course starter `exercises/01-haxpy/haxpy.cu` is a `// TODO` stub that fails
> `grade`.

For a first run, start with the
[hands-on walkthrough](GUIDE.md#4-hands-on-walkthrough) in `GUIDE.md`. It
creates its own test files, so you don't need to bring a CUDA program.
