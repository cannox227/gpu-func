// ---------------------------------------------------------------------------
// Structured failure reporting.
//
// On failure the tester can emit a small machine-readable payload that the
// Python runner parses and hands to an exercise-specific formatter. The
// payload is a series of `key=value` lines:
//
//   status=failed
//   mismatches=37
//   mismatch=12 got=2.5 exp=9.0
//   mismatch=88 got=1.0 exp=4.0
//   ...                            (at most REPORT_MAX_MISMATCHES `mismatch=` lines)
//
// `status` and the repeated `mismatch=` key are the only conventions the
// runner relies on; the fields *inside* a mismatch line are exercise-defined
// (the verify lambda writes whatever its record needs).
//
// Transport is chosen via environment variables set by the runner:
//   REPORT_PATH           - path to write the payload to.
//   REPORT_MAX_MISMATCHES - cap on recorded `mismatch=` lines (default 20).
//
// The payload stream is kept separate from the human-readable stdout lines
// (mismatches=, result=, timing) so a person reading the terminal is not
// shown the raw records unless stdout *is* the fallback sink.
// ---------------------------------------------------------------------------
#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>

class Reporter {
public:
    Reporter() {
        const char* path_s = std::getenv("REPORT_PATH");
        const char* max_s  = std::getenv("REPORT_MAX_MISMATCHES");
        if (max_s) {
            try { max_mismatches_ = std::stol(max_s); } catch (...) {}
        }
        if (path_s && path_s[0])
            out_ = std::fopen(path_s, "w");
    }

    ~Reporter() {
        if (out_) { std::fflush(out_); std::fclose(out_); }
    }

    long max_mismatches() const { return max_mismatches_; }

    // Append a raw `key=value ...` record line under the repeated `mismatch=`
    // key. `fields` is the text after `mismatch=` (e.g. "12 got=2.5 exp=9.0").
    // Silently drops records beyond the cap, but keeps counting (the total is
    // reported separately via finish()).
    void record_mismatch(const std::string& fields) {
        if (recorded_ < max_mismatches_)
            line("mismatch=" + fields);
        ++recorded_;
    }

    // Emit the mismatch total and terminal status line. Call exactly once,
    // after recording.
    void finish(bool pass) {
        line("mismatches=" + std::to_string(recorded_));
        line(std::string("status=") + (pass ? "passed" : "failed"));
    }

    template<typename T>
    void record(const std::string& key, const T& value) {
        line(key + "=" + std::to_string(value));
    }

    // String-valued record (e.g. mode=profile, kernel=...). Separate from the
    // numeric template since std::to_string doesn't accept strings.
    void record(const std::string& key, const std::string& value) {
        line(key + "=" + value);
    }

    void record(const std::string& key, const char* value) {
        line(key + "=" + value);
    }

private:
    void line(const std::string& s) {
        if (out_) { std::fputs(s.c_str(), out_); std::fputc('\n', out_); }
    }

    std::FILE* out_           = nullptr;
    long       max_mismatches_ = 20;
    long       recorded_       = 0;
};
