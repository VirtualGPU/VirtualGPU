"""GB10 spill study: does an explicit maximum-L1 carveout remove the two states of the
1 KiB spill frames? Diagnostic (docs/results/analysis/gb10-spill-carveout-plan.md).

Runs `vgpu_bench --mode spill-latency` under a frozen schedule with the SM clock probe,
alternating the driver's carveout (`default`) and `--spill-carveout 0` (maximum L1,
`max_l1`), ARMS_PER_PROCESS times each, each run as its own process traced by
host_quiet.run_traced after a quiet-host gate. Run under the GPU lock.

    python microbench/spill_carveout.py BINARY SCHEDULE OUTPUT.json
    python microbench/spill_carveout.py --judge OUTPUT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import host_quiet  # noqa: E402

PROCESSES = 3
ARMS = (("default", []), ("max_l1", ["--spill-carveout", "0"]))
SERIAL_POINTS = (8, 32, 68, 128)  # 1 KiB frames, serial chain
BATCH_POINTS = (68, 128)  # 1 KiB frames, batched reloads
SLOW = 1.4  # a sample is slow above this multiple of its point's lower quartile


def frame_points(result, mode):
    return {
        p["reload_count"]: p["launch_ns_samples"]
        for p in result["reload_latency_by_count"]
        if p["frame_bytes_per_thread"] == 1024 and p["dependency_mode"] == mode
    }


def slow_count(samples):
    low = statistics.quantiles(samples, n=4, method="inclusive")[0]
    return sum(x > SLOW * low for x in samples)


def iqr_ratio(samples):
    q1, _, q3 = statistics.quantiles(samples, n=4, method="inclusive")
    return (q3 - q1) / statistics.median(samples)


def judge(record):
    """The frozen reading of the plan."""
    arms = {}
    for name, _ in ARMS:
        runs = [r for r in record["runs"] if r["arm"] == name]
        serial = [frame_points(r["result"], "serial_chain") for r in runs]
        batch = [frame_points(r["result"], "batch_before_dep") for r in runs]
        arms[name] = {
            "processes": len(runs),
            "host_checks": [r["host_check"]["status"] for r in runs],
            "serial_slow": {rc: [slow_count(s[rc]) for s in serial] for rc in SERIAL_POINTS},
            "serial_iqr": {
                rc: [round(iqr_ratio(s[rc]), 4) for s in serial] for rc in SERIAL_POINTS
            },
            "batch_iqr": {rc: [round(iqr_ratio(b[rc]), 4) for b in batch] for rc in BATCH_POINTS},
            "serial_median_ns": {
                rc: [statistics.median(s[rc]) for s in serial] for rc in SERIAL_POINTS
            },
        }
    control, treated = arms["default"], arms["max_l1"]
    reproduced = (
        sum(sum(n > 0 for n in control["serial_slow"][rc]) >= 2 for rc in SERIAL_POINTS) >= 3
    )
    gone = all(n == 0 for rc in SERIAL_POINTS for n in treated["serial_slow"][rc]) and all(
        r <= 0.10 for rc in SERIAL_POINTS for r in treated["serial_iqr"][rc]
    )
    quiet = all(s == "passed" for a in arms.values() for s in a["host_checks"])
    if not quiet:
        verdict = "void: host check"
    elif not reproduced:
        verdict = "record only: control did not show two states"
    elif gone:
        verdict = "two states gone with the maximum-L1 carveout"
    else:
        verdict = "two states remain with the maximum-L1 carveout"
    return {"verdict": verdict, "control_reproduced": reproduced, "arms": arms}


def capture(binary, schedule, output):
    gate = host_quiet.wait_for_quiet_host([os.getpid()])
    runs = []
    for i in range(PROCESSES):
        for name, extra in ARMS:
            cmd = [str(binary), "--mode", "spill-latency", "--samples", "15", "--warmup", "3"]
            cmd += ["--launch-schedule", str(schedule), "--clock-probe", *extra]
            run = host_quiet.run_traced(cmd)
            runs.append({"process": i, "arm": name, **run, "result": json.loads(run.pop("stdout"))})
    record = {
        "binary_sha256": hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
        "schedule_sha256": hashlib.sha256(Path(schedule).read_bytes()).hexdigest(),
        "host_quiet_gate": gate,
        "runs": runs,
    }
    output.write_text(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--judge", action="store_true")
    args = parser.parse_args()
    if args.judge:
        print(json.dumps(judge(json.loads(args.paths[0].read_text())), indent=1))
        return
    binary, schedule, output = args.paths
    if output.exists():
        raise FileExistsError(output)
    capture(binary, schedule, output)


if __name__ == "__main__":
    main()
