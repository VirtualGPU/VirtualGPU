"""GB10 sync-load-reuse: do synchronous loads reuse L1 between the CTAs of an SM?

Registration: docs/results/analysis/v0.3-sync-load-gemm-plan.md (as amended). Diagnostic;
not part of a profile. Run `capture` under the GPU lock.

    python microbench/sync_load_reuse.py capture BINARY RUN_DIR
    python microbench/sync_load_reuse.py judge RUN_DIR/capture.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import capture_posthoc  # noqa: E402
import host_quiet  # noqa: E402
import long_launch  # noqa: E402

MODE = "sync-load-reuse"
PROCESSES = 3
SHOWN = 0.05  # disjoint at least 5% slower than shared
SAME = 0.02  # differences under 2%


def bench(binary, schedule, samples, warmup, probe):
    cmd = [str(binary), "--mode", MODE, "--samples", str(samples), "--warmup", str(warmup)]
    return cmd + ["--launch-schedule", str(schedule)] + (["--clock-probe"] if probe else [])


def capture(binary, run_dir):
    run_dir.mkdir(parents=True, exist_ok=False)
    gate = host_quiet.wait_for_quiet_host([os.getpid()])
    pilot = host_quiet.run_traced(bench(binary, "pilot", 3, 1, False))
    pilot_raw = json.loads(pilot.pop("stdout"))
    (run_dir / "pilot.json").write_text(json.dumps(pilot_raw) + "\n")
    schedule, record = long_launch.freeze(MODE, pilot_raw)
    (run_dir / f"{MODE}.schedule").write_text(schedule, newline="\n")
    (run_dir / f"{MODE}.freeze.json").write_text(json.dumps(record) + "\n")
    runs, replaced = [], False
    while len([r for r in runs if r["valid"]]) < PROCESSES:
        run = host_quiet.run_traced(bench(binary, run_dir / f"{MODE}.schedule", 15, 3, True))
        raw = json.loads(run.pop("stdout"))
        long_launch.verify(record, raw)
        valid = run["host_check"]["status"] == "passed"
        runs.append(
            {"process": len(runs), "valid": valid, "host_check": run["host_check"], "raw": raw}
        )
        if not valid:
            if replaced:
                break
            replaced = True
    out = {
        "binary_sha256": hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
        "host_quiet_gate": gate,
        "pilot_host_check": pilot["host_check"],
        "runs": runs,
    }
    (run_dir / "capture.json").write_text(json.dumps(out) + "\n")
    return out


def key(p):
    return (p["level"], p["ctas_per_sm"], p["sharing"], p["load"])


def fails(points):
    """A rule on any process's launches, rule 3 across them, or CTAs not co-resident as asked."""
    meds = []
    for p in points:
        v = p["launch_ns_samples"]
        q = st.quantiles(v, n=4)
        if (q[2] - q[0]) / st.median(v) > capture_posthoc.IQR_RATIO:
            return "rule 1"
        if max(v) / min(v) > capture_posthoc.SPIKE_RATIO:
            return "rule 2"
        if p["ctas_per_sm_observed"] != [p["ctas_per_sm"], p["ctas_per_sm"]]:
            return "co-residency"
        meds.append(st.median(v))
    if (max(meds) - min(meds)) / st.median(meds) > capture_posthoc.REPRO_RANGE:
        return "rule 3"
    return None


def judge(record):
    runs = [r["raw"] for r in record["runs"] if r["valid"]]
    if len(runs) != PROCESSES:
        return {"status": "not decided", "reason": f"{len(runs)} valid processes"}
    by = {}
    for raw in runs:
        for p in raw["points"]:
            by.setdefault(key(p), []).append(p)

    def t(k, i):
        return st.median(by[k][i]["ns_per_step_samples"])

    levels = {}
    for level in ("l2", "dram"):
        ks = [k for k in by if k[0] == level]
        failing = {"/".join(map(str, k)): f for k in ks if (f := fails(by[k]))}
        two = {
            (s, ld): (level, 2, s, ld) for s in ("shared", "disjoint") for ld in ("default", "cg")
        }
        one = {ld: (level, 1, "disjoint", ld) for ld in ("default", "cg")}
        res = {"failing_points": failing}
        res["two_ctas_per_sm"] = {
            "default_disjoint_over_shared": [
                t(two["disjoint", "default"], i) / t(two["shared", "default"], i) - 1
                for i in range(PROCESSES)
            ],
            "cg_disjoint_over_shared": [
                t(two["disjoint", "cg"], i) / t(two["shared", "cg"], i) - 1
                for i in range(PROCESSES)
            ],
        }
        res["one_cta_per_sm"] = {
            "default_over_cg": [
                t(one["default"], i) / t(one["cg"], i) - 1 for i in range(PROCESSES)
            ]
        }
        if failing:
            res["l1_reuse"] = res["default_path"] = "not resolved: a point fails a rule"
        else:
            d = res["two_ctas_per_sm"]["default_disjoint_over_shared"]
            c = res["two_ctas_per_sm"]["cg_disjoint_over_shared"]
            if all(x >= SHOWN for x in d) and all(abs(x) < SAME for x in c):
                res["l1_reuse"] = "shown"
            elif all(abs(x) < SAME for x in d):
                res["l1_reuse"] = "not shown"
            else:
                res["l1_reuse"] = "not resolved"
            g = res["one_cta_per_sm"]["default_over_cg"]
            if all(abs(x) >= SHOWN for x in g):
                res["default_path"] = "differs"
            elif all(abs(x) < SAME for x in g):
                res["default_path"] = "does not differ"
            else:
                res["default_path"] = "not resolved"
        res["per_step_ns"] = {
            "/".join(map(str, k)): st.median(t(k, i) for i in range(PROCESSES)) for k in ks
        }
        levels[level] = res
    return {
        "status": "judged",
        "candidate_1_l1_reuse": levels["l2"]["l1_reuse"],
        "rules_1_2_per_process": [capture_posthoc.dispersion_check(r)["rules"] for r in runs],
        "rule_3": capture_posthoc.reproducibility_check(runs),
        "levels": levels,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="action", required=True)
    c = sub.add_parser("capture")
    c.add_argument("binary", type=Path)
    c.add_argument("run_dir", type=Path)
    j = sub.add_parser("judge")
    j.add_argument("capture", type=Path)
    args = ap.parse_args()
    if args.action == "capture":
        capture(args.binary, args.run_dir)
    else:
        print(json.dumps(judge(json.loads(args.capture.read_text())), indent=1))


if __name__ == "__main__":
    main()
