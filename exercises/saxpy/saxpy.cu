#include <cuda_runtime.h>
#include <cstdio>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) assert((call) == cudaSuccess)

__global__ void saxpy(int n, float alpha, const float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = alpha * x[i] + y[i];
}

void launch_saxpy(int n, float a, float *x, float *y)
{
    constexpr int BLOCK = 256;
    saxpy<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(n, a, x, y);
}

int main()
{
    const int N = 1 << 25; // 32 M elements
    const float A = 2.0f;
    const size_t sz = N * sizeof(float);

    std::vector<float> hX(N), hY(N);
    for (int i = 0; i < N; i++)
    {
        hX[i] = 0.001f * i;
        hY[i] = 1.0f;
    }

    float *dX, *dY;
    CUDA_CHECK(cudaMalloc(&dX, sz));
    CUDA_CHECK(cudaMalloc(&dY, sz));
    CUDA_CHECK(cudaMemcpy(dX, hX.data(), sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dY, hY.data(), sz, cudaMemcpyHostToDevice));

    launch_saxpy(N, A, dX, dY);
    CUDA_CHECK(cudaGetLastError());      // errors from the launch itself
    CUDA_CHECK(cudaDeviceSynchronize()); // errors from kernel execution

    CUDA_CHECK(cudaMemcpy(hY.data(), dY, sz, cudaMemcpyDeviceToHost));
    for (int i : {0, 1, 12345, N - 1})
        std::printf("y[%8d] = %g\n", i, (double)hY[i]);

    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dY));
    return 0;
}
