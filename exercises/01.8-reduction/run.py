from runner import cli, Exercise, TestConfig, BenchmarkReport, aggregate_profiling_info, ProfileFeedback
from runner.results import ProfileReport


MI = 1024 * 1024


class Reduction(Exercise):
    name   = "reduction"
    tester = ["tester.cu"]
    source = "reduction.cu"
    key_benchmarks = ["large"]

    def format_failure(self, payload, config: "TestConfig") -> str:
        if not payload.mismatches_list and payload.count == 0:
            return ""

        lines = [f" n={config.args['n']} — reduction result is wrong"]
        for m in payload.mismatches_list:
            got     = m.get("got",     "?")
            exp     = m.get("exp",     "?")
            absdiff = m.get("absdiff", "?")
            tol     = m.get("tol",     "?")
            lines.append(f"  got={got}  expected={exp}  |diff|={absdiff}  tolerance={tol}")
        return "\n".join(lines)

    def _expected_memory(self, config: TestConfig):
        n = int(config.args['n'])
        reads  = 4 * n / MI                 # every element read exactly once
        writes = 4 * ((n + 255) // 256) / MI  # one partial per block (assuming 256)
        return reads, writes

    def format_benchmark(self, payload: BenchmarkReport, config: "TestConfig") -> str:
        n = int(config.args['n'])
        reads, writes = self._expected_memory(config)
        duration  = payload.avg_ms_gpu
        peak_bw   = payload.peak_bw / MI
        bandwidth = (reads + writes) / duration * 1000
        lines  = [f"Reduced {n} elements in {duration:.3f} ms"]
        lines += [f"  reads={reads:.1f} MiB"]
        lines += [f"  => Memory bandwidth {bandwidth / 1024:.1f} GiB/s"]
        lines += [f"  Peak BW {peak_bw / 1024:.1f} GiB/s"]
        lines += [f"  => {bandwidth / peak_bw * 100:.1f}%"]

        if config.name in self.key_benchmarks:
            lines.append(f"  Key benchmark: {config.name}")

        return "\n".join(lines)

    def format_profiling(self, payload: ProfileReport, config) -> list[ProfileFeedback]:
        feedback = []
        aggregate = aggregate_profiling_info(payload.metrics)

        total_read = aggregate["dram_read_bytes"] / MI
        dram_throughput = aggregate["dram_throughput"]
        expect_read, _ = self._expected_memory(config)

        if total_read > 1.5 * expect_read:
            output = (f"For this task, we expect to read about {expect_read:.1f} MiB from global "
                      f"memory (every element exactly once).\n"
                      f"Your kernel read {total_read:.1f} MiB — are you passing over the input more than once?\n")
            feedback += [ProfileFeedback(kind="warning", text=output)]

        cc = aggregate['cc_major']
        baseline, good = 80, 101
        if cc >= 9:
            baseline, good = 25, 40

        if dram_throughput < baseline:
            output = (f"For this task, we generally expect a baseline kernel to reach about {baseline}% DRAM bandwidth.\n"
                      f"Your kernel achieved only {dram_throughput:.1f}%.\n")
            feedback += [ProfileFeedback(kind="warning", text=output)]
        elif dram_throughput >= good:
            output = (f"Your kernel achieved {dram_throughput:.1f}% DRAM bandwidth. Reduction is memory-bound, "
                      f"so this is close to the ceiling — excellent work!\n")
            feedback += [ProfileFeedback(kind="praise", text=output)]

        # Shared-memory traffic. The standard tree reduction leans heavily on
        # shared memory; warp-level primitives (Part 3) remove most of it.
        smem = aggregate.get('smem_loads', 0) + aggregate.get('smem_stores', 0)
        n = int(config.args['n'])
        if smem > 0 and n > 0:
            per_elem = smem * 32 / n  # shared instructions are warp-wide
            output = (f"Your kernel issued roughly {per_elem:.1f} shared-memory accesses per input element.\n"
                      f"Can you cut shared-memory traffic with warp shuffles (`__shfl_down_sync`)? "
                      f"You will revisit exactly this in Part 3.\n")
            feedback += [ProfileFeedback(kind="info", text=output)]

        return feedback


exercise = Reduction()


if __name__ == "__main__":
    cli(exercise)
