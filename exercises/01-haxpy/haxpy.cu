#include <cuda_bf16.h>
#include <cstdint>

struct __align__(16) bf16x8 {
    nv_bfloat16 v[8];
};

__global__ void haxpy_kernel(int n, float alpha, const nv_bfloat16* __restrict__ x, nv_bfloat16* __restrict__ y) {
    constexpr int VEC = 8;

    uintptr_t addr = reinterpret_cast<uintptr_t>(x);
    int prefix = (16 - (int)(addr & 15)) & 15;
    prefix /= (int)sizeof(nv_bfloat16);
    if (prefix > n) prefix = n;

    int body = n - prefix;
    int vec_n = body / VEC;
    int tail = body % VEC;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < vec_n) {
        const bf16x8* x8 = reinterpret_cast<const bf16x8*>(x + prefix);
        bf16x8* y8 = reinterpret_cast<bf16x8*>(y + prefix);

        bf16x8 xv = x8[tid];
        bf16x8 yv = y8[tid];

#pragma unroll
        for (int j = 0; j < VEC; ++j) {
            float xf = __bfloat162float(xv.v[j]);
            float yf = __bfloat162float(yv.v[j]);
            yv.v[j] = __float2bfloat16(alpha * xf + yf);
        }

        y8[tid] = yv;
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i = 0; i < prefix; ++i) {
            float xf = __bfloat162float(x[i]);
            float yf = __bfloat162float(y[i]);
            y[i] = __float2bfloat16(alpha * xf + yf);
        }

        int tail_start = prefix + vec_n * VEC;
        for (int i = 0; i < tail; ++i) {
            int idx = tail_start + i;
            float xf = __bfloat162float(x[idx]);
            float yf = __bfloat162float(y[idx]);
            y[idx] = __float2bfloat16(alpha * xf + yf);
        }
    }
}

void haxpy(int n, float alpha, const nv_bfloat16* x, nv_bfloat16* y) {
    if (n <= 0) return;

    constexpr int BLOCK = 256;
    int vec_n = (n + 7) / 8;
    int grid = (vec_n + BLOCK - 1) / BLOCK;
    if (grid < 1) grid = 1;
    haxpy_kernel<<<grid, BLOCK>>>(n, alpha, x, y);
}
