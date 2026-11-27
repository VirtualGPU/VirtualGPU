"""GB10 spill study: do the 4 KiB spill points move with the local memory allocation?
Diagnostic (docs/results/analysis/gb10-spill-alloc-plan.md).

Runs `vgpu_bench --mode spill-alloc-diag` in PROCESSES processes, each traced by
host_quiet.run_traced after a quiet-host gate, on the spill points listed in POINTS at
the steps of a frozen spill-latency freeze record. Run under the GPU lock.

    python microbench/spill_alloc.py BINARY FREEZE.json OUTPUT.json
    python microbench/spill_alloc.py --judge OUTPUT.json
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
MODES = ("serial_chain", "batch_before_dep")
RELOADS = (0, 8, 32, 68, 128)
# The 4 KiB points of the third failure class, and two 512 B L1 points as a control.
POINTS = [(m, 4096, rc) for m in MODES for rc in RELOADS] + [
    ("batch_before_dep", 512, 0),
    ("batch_before_dep", 512, 128),
]
CONTEXTS = ((0, 1, 2), (3, 4, 5))  # allocations in the first and the second context


def points_file(freeze):
    steps = {
        (
            p["point"]["dependency_mode"],
            p["point"]["frame_bytes_per_thread"],
            p["point"]["reload_count"],
        ): p["steps"]
        for p in freeze["points"]
    }
    return "".join(f"{m} {f} {rc} {steps[(m, f, rc)]}\n" for m, f, rc in POINTS)


def medians(result):
    """{(block, visit, point): median launch ns}."""
    out = {}
    for block in result["blocks"]:
        for v, visit in enumerate(block["visits"]):
            for p in visit:
                key = (p["dependency_mode"], p["frame_bytes_per_thread"], p["reload_count"])
                out[(block["block"], v, key)] = statistics.median(p["launch_ns_samples"])
    return out


def block_levels(result, select):
    """Per block: the median over the selected points of each point's median over that
    process's median of the point, averaged over the block's two visits; and the two
    visits' difference."""
    m = medians(result)
    keys = [k for k in POINTS if select(k)]
    ref = {k: statistics.median(v for (b, vi, kk), v in m.items() if kk == k) for k in keys}
    visit = {
        (b, v): statistics.median(m[(b, v, k)] / ref[k] for k in keys)
        for b in range(len(result["blocks"]))
        for v in range(2)
    }
    return {
        b: ((visit[(b, 0)] + visit[(b, 1)]) / 2, abs(visit[(b, 0)] - visit[(b, 1)]))
        for b in range(len(result["blocks"]))
    }


def span(values):
    return max(values) - min(values)


def judge(record):
    """The frozen reading of the plan."""
    procs = []
    for run in record["runs"]:
        res = run["result"]
        blocks = res["blocks"]
        first = {b["allocation"]: b["block"] for b in reversed(blocks) if b["action"] != "same"}
        again = {b["allocation"]: b["block"] for b in blocks if b["action"] == "same"}
        reload_ = block_levels(res, lambda k: k[1] == 4096 and k[2] > 0)
        zero = block_levels(res, lambda k: k[1] == 4096 and k[2] == 0)
        level = [reload_[first[a]][0] for a in range(6)]
        zero_level = [zero[first[a]][0] for a in range(6)]
        drift = [abs(reload_[again[a]][0] - reload_[first[a]][0]) for a in range(6)]
        ctx = [statistics.mean(level[a] for a in c) for c in CONTEXTS]
        procs.append(
            {
                "host_check": run["host_check"]["status"],
                "levels": [round(x, 4) for x in level],
                "zero_levels": [round(x, 4) for x in zero_level],
                "stable": [reload_[first[a]][1] <= 0.03 for a in range(6)],
                "same_allocation_drift": [round(x, 4) for x in drift],
                "span": round(span(level), 4),
                "zero_span": round(span(zero_level), 4),
                "context_spans": [round(span([level[a] for a in c]), 4) for c in CONTEXTS],
                "context_gap": round(abs(ctx[0] - ctx[1]), 4),
            }
        )
    raw = [medians(r["result"]) for r in record["runs"]]
    between = []
    for k in [k for k in POINTS if k[1] == 4096 and k[2] > 0]:
        per_proc = [statistics.median(v for (b, vi, kk), v in m.items() if kk == k) for m in raw]
        between.append(span(per_proc) / statistics.median(per_proc))
    steady = sum(d <= 0.03 for p in procs for d in p["same_allocation_drift"]) >= 15
    drifting = sum(sum(d > 0.03 for d in p["same_allocation_drift"]) >= 3 for p in procs) >= 2
    placement = (
        steady
        and sum(p["span"] >= 0.12 and p["zero_span"] <= 0.03 for p in procs) >= 2
        and sum(sum(p["stable"]) for p in procs) >= 15
    )
    context = sum(max(p["context_spans"]) <= 0.03 and p["context_gap"] >= 0.08 for p in procs) >= 2
    session = all(p["span"] <= 0.03 for p in procs) and sum(b > 0.10 for b in between) >= 5
    flat = all(p["span"] <= 0.05 for p in procs) and sum(b > 0.05 for b in between) == 0
    if any(p["host_check"] != "passed" for p in procs):
        verdict = "void: host check"
    elif flat:
        verdict = "not reproduced"
    elif placement:
        verdict = "placement: the level moves with the local memory allocation"
    elif drifting:
        verdict = "drift: the level moves within one allocation over time"
    elif context:
        verdict = "context: the level is set per context"
    elif session:
        verdict = "session: the level is set per process"
    else:
        verdict = "neither supported"
    return {
        "verdict": verdict,
        "same_allocation_steady": steady,
        "processes": procs,
        "between_process_spread": [round(b, 4) for b in between],
    }


def capture(binary, freeze, output):
    points = output.with_suffix(".points")
    points.write_text(points_file(json.loads(freeze.read_text())), newline="\n")
    gate = host_quiet.wait_for_quiet_host([os.getpid()])
    runs = []
    for i in range(PROCESSES):
        cmd = [str(binary), "--mode", "spill-alloc-diag", "--samples", "15", "--warmup", "3"]
        run = host_quiet.run_traced(cmd + ["--point-order", str(points)])
        runs.append({"process": i, **run, "result": json.loads(run.pop("stdout"))})
    record = {
        "binary_sha256": hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
        "freeze_sha256": hashlib.sha256(freeze.read_bytes()).hexdigest(),
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
    binary, freeze, output = args.paths
    if output.exists():
        raise FileExistsError(output)
    capture(binary, freeze, output)


if __name__ == "__main__":
    main()
