# gpu_func_cli

Standalone CLI for running CUDA **course exercises** and **custom CUDA kernels**
on a remote GPU through the GFAAS REST API. The local machine needs no CUDA,
`nvcc`, Nsight Compute, or GPU — the CLI ships a self-contained job to a GFAAS
worker.

**→ Full documentation: [`GUIDE.md`](GUIDE.md)** — install, configure, a
hands-on walkthrough, course exercises, custom kernels, reports, command
reference, and troubleshooting.

## Quick start

```bash
uv tool install --editable /path/to/gpu_func_cli   # or: pip install .
export GFAAS_API_BASE="https://<hub-host>/api"
export GFAAS_API_KEY="<your-api-key>"
gpu_func_cli workers

# vendored 01-haxpy (no checkout needed); pass YOUR completed solution:
gpu_func_cli exercise 01-haxpy grade --file your_haxpy.cu --gpu B200

# any other exercise (via a cuda-course checkout):
gpu_func_cli exercise 02-softmax128 grade \
  --file solutions/correct/02-baseline.cu --course-root /path/to/cuda-course --gpu B200

# a custom kernel (a self-contained .cu needs no harness):
gpu_func_cli custom run my_kernel.cu --gpu B200
```

> Note: `--file haxpy.cu` is *your* completed solution — the course starter is a
> `// TODO` stub. `GUIDE.md` §4 builds a ready-to-run one if you just want to try
> the tool.

**New here?** Start with the
[hands-on walkthrough](GUIDE.md#4-hands-on-walkthrough) in `GUIDE.md` — it
creates its own test files, so you don't need to bring a CUDA program.
