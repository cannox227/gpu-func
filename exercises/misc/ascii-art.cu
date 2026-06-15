// ascii_art.cu
#include <cuda_runtime.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                 \
    do                                                                   \
    {                                                                    \
        cudaError_t err__ = (call);                                      \
        if (err__ != cudaSuccess)                                        \
        {                                                                \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",               \
                         __FILE__, __LINE__, cudaGetErrorString(err__)); \
            std::exit(1);                                                \
        }                                                                \
    } while (0)

static constexpr int kRampLen = 10; // "@%#*+=-:. "

__global__ void ascii_art_kernel(
    const unsigned char *gray,
    int width,
    int height,
    int cell_w,
    int cell_h,
    char *out,
    int out_w,
    int out_h)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= out_w || oy >= out_h)
        return;

    int x0 = ox * cell_w;
    int y0 = oy * cell_h;
    int x1 = min(x0 + cell_w, width);
    int y1 = min(y0 + cell_h, height);

    int sum = 0;
    int count = 0;
    for (int y = y0; y < y1; ++y)
    {
        int row = y * width;
        for (int x = x0; x < x1; ++x)
        {
            sum += gray[row + x];
            ++count;
        }
    }

    unsigned char avg = count ? static_cast<unsigned char>(sum / count) : 0;
    int idx = (avg * (kRampLen - 1)) / 255;

    // dark -> light
    out[oy * out_w + ox] = "@%#*+=-:. "[idx];
}

static std::vector<char> run_ascii_cuda(
    const std::vector<unsigned char> &gray,
    int width,
    int height,
    int cell_w,
    int cell_h)
{
    assert(width > 0 && height > 0);
    assert(cell_w > 0 && cell_h > 0);
    assert((int)gray.size() == width * height);

    int out_w = (width + cell_w - 1) / cell_w;
    int out_h = (height + cell_h - 1) / cell_h;

    unsigned char *d_gray = nullptr;
    char *d_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_gray, gray.size()));
    CUDA_CHECK(cudaMalloc(&d_out, out_w * out_h));

    CUDA_CHECK(cudaMemcpy(d_gray, gray.data(), gray.size(), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x,
              (out_h + block.y - 1) / block.y);

    ascii_art_kernel<<<grid, block>>>(
        d_gray, width, height, cell_w, cell_h, d_out, out_w, out_h);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<char> out(out_w * out_h);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, out.size(), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_gray));
    CUDA_CHECK(cudaFree(d_out));
    return out;
}

static void print_ascii(const std::vector<char> &out, int out_w, int out_h)
{
    for (int y = 0; y < out_h; ++y)
    {
        std::fwrite(out.data() + y * out_w, 1, out_w, stdout);
        std::fputc('\n', stdout);
    }
}

int main()
{
    const std::vector<std::string> art = {
        "@   @ ##### @@@@  @@@@   @@@ ",
        "@   @ @     @   @ @   @ @   @",
        "@   @ @     @   @ @   @ @   @",
        "@   @ ####  @@@@  @   @ @@@@@",
        " @ @  @     @ @   @   @ @   @",
        " @ @  @     @  @  @   @ @   @",
        "  @   ##### @   @ @@@@  @   @",
    };

    const int width = static_cast<int>(art.front().size());
    const int height = static_cast<int>(art.size());
    const int cell_w = 1;
    const int cell_h = 1;
    const int out_w = width;
    const int out_h = height;

    std::vector<unsigned char> gray(width * height, 255);
    std::vector<char> expected;
    expected.reserve(width * height);

    for (int y = 0; y < height; ++y)
    {
        assert(static_cast<int>(art[y].size()) == width);
        for (int x = 0; x < width; ++x)
        {
            char ch = art[y][x];
            gray[y * width + x] = (ch == ' ') ? 255 : 0;
            expected.push_back((ch == ' ') ? ' ' : '@');
        }
    }

    auto out = run_ascii_cuda(gray, width, height, cell_w, cell_h);
    assert(out == expected);

    print_ascii(out, out_w, out_h);
    return 0;
}
