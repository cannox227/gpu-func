// reduction_test.cu
//
// Usage:
//   ./reduction --test      <testspec.txt>   -- verify only
//   ./reduction --benchmark <testspec.txt>   -- benchmark + verify
//   ./reduction --profile   <testspec.txt>   -- NVTX-wrapped iterations, no verify
//
// The student must provide:
//   void reduce(int n, const float* d_in, float* d_out);
// Each block reduces its chunk of the input to one partial sum and writes it
// to d_out[blockIdx.x]; this harness sums the per-block partials on the host.

#include "tester.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cfloat>

// One output slot per block, sized for the smallest block we allow (one warp),
// so any block size that is a multiple of the warp size fits. Unused slots stay
// zero and do not affect the sum.
static constexpr int MIN_BLOCK = 32;

// ---------------------------------------------------------------------------
// Student-supplied function (declared here, defined in a separate TU)
// ---------------------------------------------------------------------------
void reduce(int n, const float* d_in, float* d_out);

// ---------------------------------------------------------------------------
// Test specification
// ---------------------------------------------------------------------------
struct TestSpec {
    int      n    = 1 << 20;
    Dist     dist;
    uint64_t seed = 42;
};

static TestSpec loadSpec(const std::string& path) {
    TestSpec spec;
    parseSpecFile(path, [&](const std::string& key, const std::string& val) {
        if      (key == "n")    spec.n    = std::stoi(val);
        else if (key == "dist") spec.dist = parseDist(val);
        else if (key == "seed") spec.seed = std::stoull(val);
        else return false;
        return true;
    });
    return spec;
}

// ---------------------------------------------------------------------------
// Reference: sum in float64. The L1 norm (sum of absolute values) is returned
// alongside it because it sets the scale of the unavoidable fp32 rounding
// error, and hence the comparison tolerance.
// ---------------------------------------------------------------------------
static double refSum(const std::vector<float>& in, double& abs_sum) {
    double sum = 0.0;
    abs_sum    = 0.0;
    for (float v : in) {
        sum     += (double)v;
        abs_sum += std::fabs((double)v);
    }
    return sum;
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
    // Generate host data in float32
    // -----------------------------------------------------------------------
    std::mt19937_64 rng(spec.seed);

    std::vector<float> h_in(spec.n);
    fillHost(h_in, spec.dist, rng);

    double abs_sum = 0.0;
    double ref     = refSum(h_in, abs_sum);

    // -----------------------------------------------------------------------
    // Upload to device. The output holds one partial per block; slots the
    // kernel never writes stay zero, so any block size >= MIN_BLOCK works.
    // -----------------------------------------------------------------------
    long n_slots = ((long)spec.n + MIN_BLOCK - 1) / MIN_BLOCK;
    if (n_slots < 1) n_slots = 1;

    thrust::device_vector<float> d_in(h_in);
    thrust::device_vector<float> d_out(n_slots, 0.0f);

    const float* d_in_ptr  = thrust::raw_pointer_cast(d_in.data());
          float* d_out_ptr = thrust::raw_pointer_cast(d_out.data());

    // -----------------------------------------------------------------------
    // Lambdas for the harness
    // -----------------------------------------------------------------------
    auto run = [&] {
        reduce(spec.n, d_in_ptr, d_out_ptr);
    };

    auto reset = [&] {
        thrust::fill(d_out.begin(), d_out.end(), 0.0f);
    };

    auto verify = [&](Reporter& reporter) -> bool {
        // Combine the per-block partials in float64.
        thrust::host_vector<float> partials(d_out);
        double result = 0.0;
        for (float p : partials) result += (double)p;

        // Probabilistic bound on fp32 summation error: it grows like sqrt(n)
        // and scales with the data's L1 norm. The constant floor handles sums
        // that cancel to near zero.
        double diff = std::fabs(result - ref);
        double tol  = 4.0 * std::sqrt((double)spec.n) * (double)FLT_EPSILON * abs_sum + 1e-4;
        bool   ok   = std::isfinite(result) && diff <= tol;

        long failures = ok ? 0 : 1;
        if (!ok) {
            std::ostringstream rec;
            rec << 0
                << " got="     << result
                << " exp="     << ref
                << " absdiff=" << diff
                << " tol="     << tol;
            reporter.record_mismatch(rec.str());
        }

        reporter.record("mismatches", failures);
        std::cout << "got=" << result << " exp=" << ref
                  << " absdiff=" << diff << " tol=" << tol << "\n";
        std::cout << "result=" << (ok ? "PASS" : "FAIL") << "\n";
        return ok;
    };

    return runHarness(args.mode, run, reset, verify);
}
