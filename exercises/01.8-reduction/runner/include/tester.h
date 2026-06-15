#pragma once
//
// Shared utilities for kernel test drivers.
//
// Each test driver provides:
//   - A TestSpec struct with kernel-specific fields
//   - A loadSpec() that uses parseSpecFile() with a key/value callback
//   - A kernel-invocation lambda and a verify lambda
//   - A main() that calls parseMode() and runHarness()
//
// Everything else lives here.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <nvtx3/nvToolsExt.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "reporter.h"

// ---------------------------------------------------------------------------
// Exit codes (shared contract with the Python runner and build script):
//   0 ok
//   1 compile error      (produced by nvcc, never by the tester binary)
//   2 crash              (runtime fault: illegal address, OOM, kernel assert)
//   3 wrong answer       (verify failed)
//   4 timeout            (enforced by the runner, never by the tester binary)
//   5 setup / bad spec   (the *exercise* is broken: unknown dist, missing or
//                          unparseable field, wrong invocation) — distinct
//                          from a user-code crash, because every solution will
//                          hit it and none of the results are meaningful.
// The tester binary itself only ever returns 0, 2, 3, or 5.
// ---------------------------------------------------------------------------
static constexpr int RC_OK     = 0;
static constexpr int RC_CRASH  = 2;
static constexpr int RC_WRONG  = 3;
static constexpr int RC_SETUP  = 5;


// ---------------------------------------------------------------------------
// CUDA error checking
// ---------------------------------------------------------------------------
static inline void checkCuda(cudaError_t result, const char* file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result)
                  << " at " << file << ":" << line << std::endl;
        std::exit(RC_CRASH);
    }
}
#define CHECK_CUDA(val) checkCuda((val), __FILE__, __LINE__)

// ---------------------------------------------------------------------------
// 1-ULP comparison in BF16 bit space
// ---------------------------------------------------------------------------
static inline bool within1ulp(nv_bfloat16 a, nv_bfloat16 b) {
    uint16_t ua, ub;
    std::memcpy(&ua, &a, 2);
    std::memcpy(&ub, &b, 2);
    int32_t diff = static_cast<int32_t>(ua) - static_cast<int32_t>(ub);
    return std::abs(diff) <= 1;
}

// ---------------------------------------------------------------------------
// Distribution descriptor used by spec files
// ---------------------------------------------------------------------------
struct Dist {
    enum Kind { CONSTANT, UNIFORM, NORMAL } kind = CONSTANT;
    float lo  = 0.f, hi  = 0.f;   // uniform: range; normal: lo unused, hi=stddev
    float val = 0.f;               // constant: value; normal: mean
};

static inline std::string trim(const std::string& s) {
    auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return {};
    auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

static inline Dist parseDist(const std::string& val) {
    std::istringstream ss(val);
    std::string kind;
    ss >> kind;
    Dist d;
    if (kind == "constant") {
        d.kind = Dist::CONSTANT;
        ss >> d.val;
    } else if (kind == "uniform") {
        d.kind = Dist::UNIFORM;
        ss >> d.lo >> d.hi;
    } else if (kind == "normal") {
        // normal <mean> <stddev>
        d.kind = Dist::NORMAL;
        ss >> d.val >> d.hi;   // val=mean, hi=stddev
    } else {
        throw std::runtime_error("Unknown distribution: " + kind);
    }
    return d;
}

static inline double get_peak_bw() {
    // Peak bandwidth from device attributes
    int dev = 0;
    int mem_clock_khz = 0, bus_width_bits = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&mem_clock_khz,  cudaDevAttrMemoryClockRate,      dev));
    CHECK_CUDA(cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, dev));
    double peak_bw = mem_clock_khz * 1e3 * (bus_width_bits / 8.0) * 2;
    return peak_bw;
}

// ---------------------------------------------------------------------------
// Spec file walker. Caller supplies a callback that handles each key/value
// pair. Unknown keys produce a warning, except for keys in IGNORED_SPEC_KEYS
// — those belong to the test orchestrator (e.g. timeout, enforced by the
// Python subprocess wrapper) and are silently skipped here. Caller is
// responsible for validating required fields after the walk completes.
// ---------------------------------------------------------------------------
using SpecHandler = std::function<bool(const std::string& key,
                                       const std::string& value)>;

static inline bool isOrchestratorKey(const std::string& key) {
    // Keys consumed by the Python test orchestrator, not the C++ harness.
    static const char* kIgnored[] = {
        "timeout", "name"
    };
    for (const char* k : kIgnored)
        if (key == k) return true;
    return false;
}

static inline void parseSpecFile(const std::string& path,
                                 const SpecHandler& handler) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open spec file: " + path);

    std::string line;
    while (std::getline(f, line)) {
        auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        line = trim(line);
        if (line.empty()) continue;

        auto eq = line.find('=');
        if (eq == std::string::npos)
            throw std::runtime_error("Bad line in spec: " + line);

        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));

        if (isOrchestratorKey(key)) continue;
        if (!handler(key, val))
            std::cerr << "WARN: unknown key: " << key << "\n";
    }
}

// ---------------------------------------------------------------------------
// Host-buffer fill from a Dist.
// T may be float or nv_bfloat16; values are always drawn as float and
// converted element-wise, so no temporary allocation is needed.
// ---------------------------------------------------------------------------
template <typename T>
static inline void fillHost(std::vector<T>& buf,
                            const Dist& d,
                            std::mt19937_64& rng) {
    auto store = [&](float v) {
        if constexpr (std::is_same_v<T, float>)
            return v;
        else
            return __float2bfloat16(v);
    };

    if (d.kind == Dist::CONSTANT) {
        std::fill(buf.begin(), buf.end(), store(d.val));
    } else if (d.kind == Dist::UNIFORM) {
        std::uniform_real_distribution<float> dist(d.lo, d.hi);
        for (auto& v : buf) v = store(dist(rng));
    } else if (d.kind == Dist::NORMAL) {
        std::normal_distribution<float> dist(d.val, d.hi);
        for (auto& v : buf) v = store(dist(rng));
    }
}

// ---------------------------------------------------------------------------
// Argument parsing: spec file is always the last positional argument.
// Exactly one of --test / --benchmark / --profile must appear before it.
// ---------------------------------------------------------------------------
enum class Mode { Test, Benchmark, Profile };

struct ParsedArgs {
    Mode        mode;
    std::string spec_path;
};

static inline ParsedArgs parseMode(int argc, char** argv) {
    auto usage = [&] {
        std::cerr << "Usage: " << argv[0]
                  << " --test|--benchmark|--profile <testspec.txt>\n";
        std::exit(RC_SETUP);
    };

    if (argc != 3) usage();

    std::string flag = argv[1];
    ParsedArgs a;
    if      (flag == "--test")      a.mode = Mode::Test;
    else if (flag == "--benchmark") a.mode = Mode::Benchmark;
    else if (flag == "--profile")   a.mode = Mode::Profile;
    else {
        std::cerr << "Unknown flag: " << flag << "\n";
        usage();
    }
    a.spec_path = argv[2];
    return a;
}

// ---------------------------------------------------------------------------
// Harness: mode-specific work, with verify for Test and Benchmark.
//
//   mode    : which path to take
//   run     : invokes the kernel-under-test against its already-prepared
//             device buffers
//   reset   : called before the final verify run to reset any in-place
//             outputs back to their initial state. Skipped in --profile.
//   verify  : called after the verify run with a Reporter&; returns
//             (pass, mismatches). The verify lambda records per-mismatch
//             detail via reporter.record_mismatch(...); the harness emits the
//             terminal status afterwards. Skipped in --profile.
//
// Human-readable lines (mismatches=, result=, timing) go to stdout. The
// structured payload goes to the Reporter's sink (REPORT_PATH); when that env
// var is unset the Reporter is inert. Returns the process exit code.
// ---------------------------------------------------------------------------
static inline int runHarness(Mode                mode,
                             const std::function<void()>& run,
                             const std::function<void()>& reset,
                             const std::function<bool(Reporter&)>& verify) {
    constexpr double BENCH_TARGET_SEC = 1.0;
    constexpr int    BENCH_MAX_ITERS  = 10000;

    Reporter reporter;
    reporter.record("mode",
        mode == Mode::Profile   ? "profile"   :
        mode == Mode::Benchmark ? "benchmark" : "test");

    if (mode == Mode::Profile) {
        nvtxRangePush("profile_kernel");
        run();
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        nvtxRangePop();
        reporter.finish(true);
        return EXIT_SUCCESS;
    }

    if (mode == Mode::Benchmark) {
        // Warmup, then adaptive timing loop. Both chrono and CUDA events
        // are reported; large disagreement between the two suggests host-
        // side noise or measurement bugs.
        run();
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t ev_start, ev_stop, ev_sync;
        CHECK_CUDA(cudaEventCreate(&ev_start));
        CHECK_CUDA(cudaEventCreate(&ev_sync, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreate(&ev_stop));

        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(ev_start));
        auto t0 = std::chrono::high_resolution_clock::now();

        int iters = 0;
        while (iters < BENCH_MAX_ITERS) {
            CHECK_CUDA(cudaEventRecord(ev_sync));
            run();
            CHECK_CUDA(cudaEventSynchronize(ev_sync));
            ++iters;
            double elapsed = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - t0).count();
            if (elapsed >= BENCH_TARGET_SEC) break;
        }

        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));
        auto t1 = std::chrono::high_resolution_clock::now();

        double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        float  gpu_ms_total = 0.f;
        CHECK_CUDA(cudaEventElapsedTime(&gpu_ms_total, ev_start, ev_stop));
        CHECK_CUDA(cudaEventDestroy(ev_start));
        CHECK_CUDA(cudaEventDestroy(ev_sync));
        CHECK_CUDA(cudaEventDestroy(ev_stop));

        reporter.record("iters", iters);
        reporter.record("avg_ms_wall", wall_ms / iters);
        reporter.record("avg_ms_gpu", gpu_ms_total / iters);
        reporter.record("peak_bw", get_peak_bw());
    }

    // Verify (Test and Benchmark). Test mode skips warmup; correctness
    // doesn't depend on cache state and the extra launch is wasted work.
    reset();
    run();
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    bool status = verify(reporter);
    reporter.finish(status);

    return EXIT_SUCCESS;
}
