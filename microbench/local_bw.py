"""GB10 local-bw: local memory bandwidth per SM in the L1 and L2 tiers.

Registration: docs/results/analysis/v0.3-local-pipe-bw-plan.md. Diagnostic; not part of a
profile. Run `capture` under the GPU lock.

    python microbench/local_bw.py sass BINARY OUT.json          # before any capture
    python microbench/local_bw.py capture BINARY RUN_DIR
    python microbench/local_bw.py judge RUN_DIR/capture.json PROFILE.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics as st
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import capture_posthoc  # noqa: E402
import host_quiet  # noqa: E402
import long_launch  # noqa: E402

MODE = "local-bw"
PROCESSES = 3
INSTANCES = [(f, 4, False) for f in (32, 64, 128, 1024, 2048, 4096)]
INSTANCES += [(f, 4, True) for f in (32, 64, 128)] + [(128, 16, False)]
SATURATED = 0.03
FRAME_SPREAD = 0.05
TIER_RATIO = 0.8
ALLOCATION_SPAN = 0.05
STORE_LOAD = 0.10


# The assembler check ---------------------------------------------------------------


def functions(sass):
    out, name = {}, None
    for line in sass.splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            name = m.group(1)
            out[name] = []
            continue
        m = re.match(r"\s*/\*([0-9a-f]{4,})\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)(.*?);", line)
        if m and name:
            out[name].append((int(m.group(1), 16), m.group(2), m.group(3).strip()))
    return out


def loop_body(instructions):
    loops = []
    for addr, op, text in instructions:
        m = re.search(r"0x([0-9a-f]+)", text) if op.startswith("BRA") else None
        if m and int(m.group(1), 16) < addr:
            loops.append((int(m.group(1), 16), addr))
    if not loops:
        return None
    # The step loop is the last one: the frame's initialization loop, if not unrolled, comes
    # before it.
    start, end = loops[-1]
    return [op for addr, op, _ in instructions if start <= addr <= end]


def check_instance(instructions, width, store):
    body = loop_body(instructions)
    if body is None:
        return {"status": "failed", "reason": "no loop"}
    local = [op for op in body if op.startswith(("LDL", "STL"))]
    want = ("STL" if store else "LDL") + (".128" if width == 16 else "")
    ok = len(local) == 8 and all(op == want for op in local)
    return {"status": "passed" if ok else "failed", "local_ops": local, "expected": f"8 x {want}"}


def instance_name(frame, width, store):
    return f"_ZN8local_bw6kernelILi{frame}ELi{width}ELb{int(store)}EEEvPji"


def sass_check(sass):
    funcs = functions(sass)
    results = {}
    for frame, width, store in INSTANCES:
        name = instance_name(frame, width, store)
        key = f"{'store' if store else 'load'}{width}_F{frame}"
        results[key] = (
            check_instance(funcs[name], width, store)
            if name in funcs
            else {"status": "failed", "reason": "instance not found"}
        )
    status = "passed" if all(r["status"] == "passed" for r in results.values()) else "failed"
    return {"status": status, "instances": results}


# Capture ---------------------------------------------------------------------------


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
    (run_dir / "local-bw.schedule").write_text(schedule, newline="\n")
    (run_dir / "local-bw.freeze.json").write_text(json.dumps(record) + "\n")
    runs, replaced = [], False
    while len([r for r in runs if r["valid"]]) < PROCESSES:
        run = host_quiet.run_traced(bench(binary, run_dir / "local-bw.schedule", 15, 3, True))
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


# Reading ---------------------------------------------------------------------------


def key(p):
    return (p["op"], p["width_bytes"], p["tier"], p["num_warps"], p["frame_bytes_per_thread"])


def bytes_per_step(p):
    return p["num_warps"] * 32 * 8 * p["width_bytes"]


def estimate(p):
    """GB/s per SM from the mean of the per-allocation medians."""
    ns = st.mean(st.median(a) for a in p["allocation_launch_ns"])
    return bytes_per_step(p) * p["steps"] / ns


def span(p):
    meds = [st.median(a) for a in p["allocation_launch_ns"]]
    return (max(meds) - min(meds)) / st.median(meds)


def fails(points):
    """Rule 1 or 2 on any allocation's launches in any process, or rule 3 across processes."""
    for p in points:
        for a in p["allocation_launch_ns"]:
            q = st.quantiles(a, n=4)
            if (q[2] - q[0]) / st.median(a) > capture_posthoc.IQR_RATIO:
                return True
            if max(a) / min(a) > capture_posthoc.SPIKE_RATIO:
                return True
    values = [estimate(p) for p in points]
    return (max(values) - min(values)) / st.median(values) > capture_posthoc.REPRO_RANGE


def judge(record, profile):
    runs = [r["raw"] for r in record["runs"] if r["valid"]]
    if len(runs) != PROCESSES:
        return {"status": "not decided", "reason": f"{len(runs)} valid processes"}
    by = {}
    for raw in runs:
        for p in raw["points"]:
            by.setdefault(key(p), []).append(p)
    rows = {}
    for k, pts in by.items():
        rows[k] = {
            "value_gbps_per_sm": st.median(estimate(p) for p in pts),
            "allocation_spans": [span(p) for p in pts],
            "fails_a_rule": fails(pts),
        }
        spans = rows[k]["allocation_spans"]
        rows[k]["allocation"] = (
            "independent" if sum(s <= ALLOCATION_SPAN for s in spans) >= 2 else "dependent"
        )
        rows[k]["pooled_half_span"] = (
            span({"allocation_launch_ns": [a for p in pts for a in p["allocation_launch_ns"]]}) / 2
        )

    def ok(k):
        return k in rows and not rows[k]["fails_a_rule"]

    def value(k):
        return rows[k]["value_gbps_per_sm"]

    curves = {}
    for op, width in (("load", 4), ("store", 4), ("load", 16)):
        ks = [k for k in rows if k[:3] == (op, width, "l1") and not (k[3] == 8 and k[4] != 128)]
        ks.sort(key=lambda k: k[3])
        if not all(ok(k) for k in ks):
            curves[f"{op}{width}"] = {"reading": "not resolved: a point fails a rule"}
            continue
        vals = sorted((value(k) for k in ks), reverse=True)
        curves[f"{op}{width}"] = {
            "value_gbps_per_sm": vals[0],
            "saturated": (vals[0] - vals[1]) / vals[0] <= SATURATED,
            "by_warps": {k[3]: value(k) for k in ks},
        }
    frames = [("load", 4, "l1", 8, f) for f in (32, 64, 128)]
    if all(ok(k) for k in frames):
        fv = [value(k) for k in frames]
        frame = (
            "clean" if max(abs(v / st.median(fv) - 1) for v in fv) <= FRAME_SPREAD else "not clean"
        )
    else:
        frame = "not resolved: a point fails a rule"
    l2 = [k for k in rows if k[2] == "l2"]
    l1_load = curves["load4"].get("value_gbps_per_sm")
    if all(ok(k) for k in l2) and l1_load is not None:
        l2_value = max(value(k) for k in l2)
        tiers = {"l2_value_gbps_per_sm": l2_value, "separate": l2_value <= TIER_RATIO * l1_load}
    else:
        tiers = {"reading": "not resolved: a point fails a rule"}
    store = curves["store4"].get("value_gbps_per_sm")
    if store is not None and l1_load is not None:
        one_rate = abs(store / l1_load - 1) <= STORE_LOAD
    else:
        one_rate = "not resolved"
    spill = profile["measured"]["spill"]
    return {
        "status": "judged",
        "rules_1_2_per_process": [capture_posthoc.dispersion_check(r)["rules"] for r in runs],
        "rule_3": capture_posthoc.reproducibility_check(runs),
        "curves": curves,
        "frame_independence": frame,
        "tiers": tiers,
        "one_rate_for_loads_and_stores": one_rate,
        "points": {"/".join(map(str, k)): v for k, v in rows.items()},
        "estimates": {
            "local_mem_l1_hit_bw_per_sm_gbps": spill["local_mem_l1_hit_bw_per_sm_gbps"]["value"],
            "local_mem_l1_miss_bw_per_sm_gbps": spill["local_mem_l1_miss_bw_per_sm_gbps"]["value"],
            "smem_bw_per_sm_gbps": profile["measured"]["memory"]["smem_bw_per_sm_gbps"]["value"],
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="action", required=True)
    s = sub.add_parser("sass")
    s.add_argument("binary", type=Path)
    s.add_argument("out", type=Path)
    c = sub.add_parser("capture")
    c.add_argument("binary", type=Path)
    c.add_argument("run_dir", type=Path)
    j = sub.add_parser("judge")
    j.add_argument("capture", type=Path)
    j.add_argument("profile", type=Path)
    args = ap.parse_args()
    if args.action == "sass":
        tool = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
        result = sass_check(subprocess.check_output([tool, "-sass", str(args.binary)], text=True))
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        print(result["status"])
        sys.exit(0 if result["status"] == "passed" else 1)
    if args.action == "capture":
        capture(args.binary, args.run_dir)
    else:
        result = judge(json.loads(args.capture.read_text()), json.loads(args.profile.read_text()))
        print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
