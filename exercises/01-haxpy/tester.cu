// haxpy_test.cu
//
// Usage:
//   ./haxpy_test --test      <testspec.txt>   -- verify only
//   ./haxpy_test --benchmark <testspec.txt>   -- benchmark + verify
//   ./haxpy_test --profile   <testspec.txt>   -- NVTX-wrapped iterations, no verify
//
// The student must provide:
//   void haxpy(long n, float alpha, const nv_bfloat16* d_x, nv_bfloat16* d_y);

#include "tester.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// ---------------------------------------------------------------------------
// Student-supplied function (declared here, defined in a separate TU)
// ---------------------------------------------------------------------------
void haxpy(int n, float alpha, const nv_bfloat16* d_x, nv_bfloat16* d_y);

// ---------------------------------------------------------------------------
// Test specification
// ---------------------------------------------------------------------------
struct TestSpec {
    int      n         = 1 << 20;
    float    alpha     = 1.f;
    Dist     x_dist;
    Dist     y_dist;
    long     x_offset  = 0;
    long     y_offset  = 0;
    uint64_t seed      = 42;
};

static TestSpec loadSpec(const std::string& path) {
    TestSpec spec;
    parseSpecFile(path, [&](const std::string& key, const std::string& val) {
        if      (key == "n")        spec.n        = std::stol(val);
        else if (key == "alpha")    spec.alpha    = float(std::stod(val));
        else if (key == "x_dist")   spec.x_dist   = parseDist(val);
        else if (key == "y_dist")   spec.y_dist   = parseDist(val);
        else if (key == "x_offset") spec.x_offset = std::stol(val);
        else if (key == "y_offset") spec.y_offset = std::stol(val);
        else if (key == "seed")     spec.seed     = std::stoull(val);
        else return false;
        return true;
    });
    return spec;
}

// ---------------------------------------------------------------------------
// Reference: alpha*x + y in float32, rounded to BF16. Inputs are the
// BF16-rounded values, matching what the kernel sees.
// ---------------------------------------------------------------------------
static nv_bfloat16 refOp(float alpha, float x, float y) {
    return __float2bfloat16(alpha * x + y);
}

int main(int argc, char** argv) {
    ParsedArgs args = parseMode(argc, argv);

    TestSpec spec;
    try {
        spec = loadSpec(args.spec_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading spec from " << args.spec_path << ": " << e.what() << "\n";
        return RC_SETUP;
    }

    // -----------------------------------------------------------------------
    // Generate host data in float32, convert to BF16
    // -----------------------------------------------------------------------
    std::mt19937_64 rng(spec.seed);

    long nx = spec.n + spec.x_offset;
    long ny = spec.n + spec.y_offset;

    std::vector<float> hx_bf(nx), hy_bf(ny);
    fillHost(hx_bf, spec.x_dist, rng);
    fillHost(hy_bf, spec.y_dist, rng);

    // Reference uses BF16-rounded inputs
    std::vector<nv_bfloat16> href(spec.n);
    for (long i = 0; i < spec.n; ++i) {
        float xi = __bfloat162float(hx_bf[spec.x_offset + i]);
        float yi = __bfloat162float(hy_bf[spec.y_offset + i]);
        href[i]  = refOp(spec.alpha, xi, yi);
    }

    // -----------------------------------------------------------------------
    // Upload to device
    // -----------------------------------------------------------------------
    thrust::device_vector<nv_bfloat16> dx(hx_bf);
    thrust::device_vector<nv_bfloat16> dy(hy_bf);

    const nv_bfloat16* d_x = thrust::raw_pointer_cast(dx.data()) + spec.x_offset;
          nv_bfloat16* d_y = thrust::raw_pointer_cast(dy.data()) + spec.y_offset;

    // -----------------------------------------------------------------------
    // Lambdas for the harness
    // -----------------------------------------------------------------------
    auto run = [&] {
        haxpy(spec.n, spec.alpha, d_x, d_y);
    };

    auto reset = [&] {
        thrust::copy(hy_bf.begin(), hy_bf.end(), dy.begin());
    };

    auto verify = [&](Reporter& reporter) -> bool {
        thrust::host_vector<nv_bfloat16> result(spec.n);
        thrust::copy(dy.begin() + spec.y_offset,
                     dy.begin() + spec.y_offset + spec.n,
                     result.begin());

        long failures = 0;
        for (long i = 0; i < spec.n; ++i) {
            if (!within1ulp(result[i], href[i])) {
                if (failures < reporter.max_mismatches()) {
                    std::ostringstream rec;
                    rec << i
                        << " x="   << __bfloat162float(hx_bf[spec.x_offset + i])
                        << " y="   << __bfloat162float(hy_bf[spec.y_offset + i])
                        << " got=" << __bfloat162float(result[i])
                        << " exp=" << __bfloat162float(href[i]);
                    reporter.record_mismatch(rec.str());
                }
                ++failures;
            }
        }
        if (failures != 0) {
            for (int i = 0; i < std::min(spec.n, 10); ++i) {

            }
        }
        reporter.record("mismatches", failures);
        return failures == 0;
    };

    return runHarness(args.mode, run, reset, verify);
}
