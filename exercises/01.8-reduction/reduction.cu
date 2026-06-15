#include <cuda_runtime.h>

constexpr int BLOCK = 1024;
constexpr int ITEMS = 64;
constexpr int WARPS = BLOCK / 32;

__device__ __forceinline__ float warp_reduce_sum(float x)
{
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        x += __shfl_down_sync(0xffffffff, x, offset);
    }
    return x;
}

__global__ void reduce_kernel(const float *__restrict__ in, float *__restrict__ out, int n)
{
    __shared__ float warp_sums[WARPS];

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int base = blockIdx.x * (BLOCK * ITEMS) + tid;

    float x = 0.0f;
#pragma unroll
    for (int k = 0; k < ITEMS; ++k)
    {
        int i = base + k * BLOCK;
        if (i < n)
            x += in[i];
    }

    x = warp_reduce_sum(x);

    if (lane == 0)
        warp_sums[warp] = x;
    __syncthreads();

    if (warp == 0)
    {
        x = (lane < WARPS) ? warp_sums[lane] : 0.0f;
        x = warp_reduce_sum(x);
        if (lane == 0)
            out[blockIdx.x] = x;
    }
}

void reduce(int n, const float *in, float *out)
{
    if (n <= 0)
        return;

    int grid = (n + BLOCK * ITEMS - 1) / (BLOCK * ITEMS);
    reduce_kernel<<<grid, BLOCK>>>(in, out, n);
}
