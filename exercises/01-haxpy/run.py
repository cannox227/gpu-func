from runner import cli, Exercise, TestConfig, BenchmarkReport, aggregate_profiling_info, ProfileFeedback
from runner.results import ProfileReport


MI = 1024 * 1024


class Haxpy(Exercise):
    name   = "haxpy"
    tester = ["tester.cu"]
    source = "haxpy.cu"
    key_benchmarks = ["misaligned_large"]

    def format_failure(self, payload, config: "TestConfig") -> str:
        if not payload.mismatches_list and payload.count == 0:
            return ""

        alpha = config.args['alpha']
        lines = [
            f" n={config.args['n']} α={alpha} — {payload.count} mismatch(es)",
        ]

        for m in payload.mismatches_list:
            idx = m.get("index", "?")
            x   = m.get("x",   "?")
            y   = m.get("y",   "?")
            got = m.get("got", "?")
            exp = m.get("exp", "?")

            lines.append(f"  [{idx}]  x={x}  y={y}  got={got}  exp={alpha}*{x}+{y} = {exp}")

        if payload.count > len(payload.mismatches_list):
            lines.append(f"  … and {payload.count - len(payload.mismatches_list)} more (capped)")

        return "\n".join(lines)

    def format_benchmark(self, payload: BenchmarkReport, config: "TestConfig") -> str:
        n = int(config.args['n'])
        reads = 4 * n / MI
        writes = 2 * n / MI
        duration = payload.avg_ms_gpu
        peak_bw = payload.peak_bw / MI
        bandwidth = (reads + writes) / duration * 1000
        lines = [f"Added {n} elements in {duration:.2f} ms"]
        lines += [f"  reads={reads:.1f} MiB writes={writes:.1f} MiB"]
        lines += [f"  => Memory bandwidth {bandwidth / 1024:.1f} GiB/s"]
        lines += [f"  Peak BW {peak_bw / 1024:.1f} GiB/s"]
        lines += [f"  => {bandwidth / peak_bw * 100:.1f}%"]

        if config.name in self.key_benchmarks:
            lines.append(f"  Key benchmark: {config.name}")

        return "\n".join(lines)

    def _expected_memory(self,  config: TestConfig):
        n = int(config.args['n'])
        reads = 4 * n / MI
        writes = 2 * n / MI
        return reads, writes

    def format_profiling(self, payload: ProfileReport, config) -> list[ProfileFeedback]:
        feedback = []
        aggregate = aggregate_profiling_info(payload.metrics)
        if aggregate['num_kernels'] > 1:
            output = "Detected more than one kernel launch.\n"
            output += "While not a hard requirement, we generally expect only a single kernel "
            output += "implementing axpy."
            feedback += [ProfileFeedback(kind="info", text=output)]


        total_bytes_read = aggregate["dram_read_bytes"] / MI
        total_bytes_written = aggregate["dram_write_bytes"] / MI
        dram_throughput = aggregate["dram_throughput"]

        expect_read, expect_write = self._expected_memory(config)
        if total_bytes_read > 1.1 * expect_read:
            output = (f"For this task, we expect a total amount of global memory reads of {expect_read:.1f} MiB.\n"
                       f"Your kernel read {total_bytes_read:.1f} MiB.\n")
            feedback += [ProfileFeedback(kind="warning", text=output)]
        if total_bytes_written > 1.1 * expect_write:
            output = (f"For this task, we expect a total amount of global memory writes of {expect_write:.1f} MiB.\n"
                       f"Your kernel wrote {total_bytes_written:.1f} MiB.\n")
            feedback += [ProfileFeedback(kind="warning", text=output)]

        cc = aggregate['cc_major']
        baseline, good, excellent = 80, 101, 101
        if cc == 10:
            baseline, good, excellent = 25, 40, 80
        elif cc == 9:
            baseline, good, excellent = 25, 40, 80

        if dram_throughput < baseline:
            output = f"For this task, we generally expect a baseline kernel to achieve about {baseline}% DRAM bandwidth.\n"
            output += f"Your kernel achieved only {dram_throughput:.1f}% DRAM bandwidth.\n"
            feedback += [ProfileFeedback(kind="warning", text=output)]
        elif dram_throughput >= excellent:
            output = f"Congratulations! Your kernel achieved {dram_throughput:.1f}% DRAM bandwidth. Excellent work!\n"
            feedback += [ProfileFeedback(kind="praise", text=output)]
        elif dram_throughput >= good:
            output = f"For this task, we generally expect an  optimized kernel to achieve about >={good}% DRAM bandwidth.\n"
            output += f"Your kernel achieved {dram_throughput:.1f}% DRAM bandwidth. Good work!\n"
            feedback += [ProfileFeedback(kind="praise", text=output)]

        # estimate vectorization
        loads = aggregate['loads'] + aggregate.get('ldgsts', 0)
        writes = aggregate['stores']
        bytes_per_load = aggregate["dram_read_bytes"] / loads / 32
        bytes_per_store = aggregate["dram_write_bytes"] / writes / 32
        bytes_per_inst = (aggregate["dram_read_bytes"] + aggregate["dram_write_bytes"]) / (loads + writes) / 32
        if bytes_per_inst > 15:
            output = f"Your kernel is well vectorized, reading an average of {bytes_per_load:.1f} bytes per load instruction "
            output += f"and writing an average of {bytes_per_store:.1f} bytes per store instruction\n"
            feedback += [ProfileFeedback(kind="praise", text=output)]
        elif bytes_per_inst > 3.9:
            output = f"Your kernel used some vectorized loads, but the overall degree of vectorization is still low\n"
            output +=f"On average, each load read {bytes_per_load:.1f} bytes "
            output +=f"and each store wrote {bytes_per_store:.1f} bytes\n"
            feedback += [ProfileFeedback(kind="info", text=output)]
        else:
            output = f"Your kernel used mostly scalar loads, reading an average of {bytes_per_load:.1f} bytes per load instruction "
            output += f"and writing an average of {bytes_per_store:.1f} bytes per store instruction\n"
            feedback += [ProfileFeedback(kind="warning", text=output)]

        return feedback




exercise = Haxpy()


if __name__ == "__main__":
    cli(exercise)
