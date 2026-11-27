"""GB10 barrier-step: what one more barrier per loop step costs at fixed work.

Registration: docs/results/analysis/v0.3-barrier-sync-plan.md. Diagnostic; not part of a
profile. Run `capture` under the GPU lock.

    python microbench/barrier_step.py sass BINARY OUT.json          # before any capture
    python microbench/barrier_step.py capture BINARY RUN_DIR
    python microbench/barrier_step.py judge RUN_DIR/capture.json PROFILE.json
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

MODE = "barrier-step"
BARRIERS = (0, 1, 2, 4, 8, 14)
FIT = (1, 2, 4, 8, 14)
WORK = (0, 64, 512)
PROCESSES = 3
SHARE = 0.05  # 5% of t(1)
HALF_SYNC = 0.5  # half of the profile's fixed_sync_ns
LINEAR = 0.05  # largest fit residual over t(B)
LOOP_EXTRA = (
    "IADD3",
    "UIADD3",
    "IADD",
    "VIADD",
    "ISETP",
    "UISETP",
    "BRA",
    "NOP",
)  # counter, compare, branch


# The assembler check ---------------------------------------------------------------


def functions(sass):
    """{function name: [(address, opcode, text)]} from `cuobjdump -sass` output."""
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
    """The instructions of the one backward branch's loop: from its target to the branch."""
    loops = []
    for addr, op, text in instructions:
        m = re.search(r"0x([0-9a-f]+)", text) if op.startswith("BRA") else None
        if m and int(m.group(1), 16) < addr:
            loops.append((int(m.group(1), 16), addr))
    if len(loops) != 1:
        return None
    start, end = loops[0]
    return [op for addr, op, _ in instructions if start <= addr <= end]


def check_instance(instructions, barriers, work):
    body = loop_body(instructions)
    if body is None:
        return {"status": "failed", "reason": "not exactly one loop"}
    bars = sum(op.startswith("BAR.SYNC") for op in body)
    ffma = sum(op == "FFMA" for op in body)
    other = sorted({op for op in body if op != "FFMA" and not op.startswith("BAR.SYNC")})
    stray = [op for op in other if not op.startswith(LOOP_EXTRA)]
    ok = bars == barriers and ffma == work and not stray
    return {
        "status": "passed" if ok else "failed",
        "bar_sync": bars,
        "ffma": ffma,
        "other": other,
        "loop_instructions": len(body),
    }


def sass_check(sass):
    funcs = functions(sass)
    results = {}
    for b in BARRIERS:
        for w in WORK:
            name = f"_Z19barrier_step_kernelILi{b}ELi{w}EEvPfi"
            key = f"B{b}_W{w}"
            results[key] = (
                check_instance(funcs[name], b, w)
                if name in funcs
                else {"status": "failed", "reason": "instance not found"}
            )
    status = "passed" if all(r["status"] == "passed" for r in results.values()) else "failed"
    return {"status": status, "instances": results}


def cuobjdump():
    found = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
    return found


# Capture ---------------------------------------------------------------------------


def bench(binary, schedule, samples, warmup, probe):
    cmd = [str(binary), "--mode", MODE, "--samples", str(samples), "--warmup", str(warmup)]
    cmd += ["--launch-schedule", str(schedule)] + (["--clock-probe"] if probe else [])
    return cmd


def capture(binary, run_dir):
    run_dir.mkdir(parents=True, exist_ok=False)
    gate = host_quiet.wait_for_quiet_host([os.getpid()])
    pilot = host_quiet.run_traced(bench(binary, "pilot", 3, 1, False))
    pilot_raw = json.loads(pilot.pop("stdout"))
    (run_dir / "pilot.json").write_text(json.dumps(pilot_raw) + "\n")
    schedule, record = long_launch.freeze(MODE, pilot_raw)
    (run_dir / "barrier-step.schedule").write_text(schedule, newline="\n")
    (run_dir / "barrier-step.freeze.json").write_text(json.dumps(record) + "\n")
    runs, replaced = [], False
    while len([r for r in runs if r["valid"]]) < PROCESSES:
        run = host_quiet.run_traced(bench(binary, run_dir / "barrier-step.schedule", 15, 3, True))
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
    return (p["work_per_step"], p["num_warps"], p["grid"])


def point_fails(lists):
    """Rule 1 or 2 on a point's launch lists in any process, or rule 3 across them."""
    meds = []
    for v in lists:
        q = st.quantiles(v, n=4)
        m = st.median(v)
        if (q[2] - q[0]) / m > capture_posthoc.IQR_RATIO or max(v) / min(
            v
        ) > capture_posthoc.SPIKE_RATIO:
            return True
        meds.append(m)
    return (max(meds) - min(meds)) / st.median(meds) > capture_posthoc.REPRO_RANGE


def slope(ts):
    xs = list(FIT)
    ys = [ts[b] for b in FIT]
    mx, my = st.mean(xs), st.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    worst = max(abs(y - (a + b * x)) / y for x, y in zip(xs, ys))
    return b, worst


def fixed_sync(profile):
    return {
        q["num_warps"]: q["overhead_ns"]["value"]
        for q in profile["measured"]["pipeline"]["fixed_sync_ns"]
        if q["tokens"] == 1
    }


def judge(record, profile):
    runs = [r["raw"] for r in record["runs"] if r["valid"]]
    if len(runs) != PROCESSES:
        return {"status": "not decided", "reason": f"{len(runs)} valid processes"}
    sync = fixed_sync(profile)
    by_point = {}
    for raw in runs:
        for p in raw["points"]:
            by_point.setdefault(key(p) + (p["barriers_per_step"],), []).append(p)
    out = {}
    for w in WORK:
        for warps in (4, 8):
            grids = sorted({k[2] for k in by_point if k[:2] == (w, warps)})
            for g in grids:
                pts = {b: by_point[(w, warps, g, b)] for b in BARRIERS}
                failing = [
                    b for b in BARRIERS if point_fails([p["launch_ns_samples"] for p in pts[b]])
                ]
                per = []
                for i in range(PROCESSES):
                    t = {b: st.median(pts[b][i]["ns_per_step_samples"]) for b in BARRIERS}
                    b, worst = slope(t)
                    per.append({"t": t, "d1": t[1] - t[0], "slope": b, "worst_residual": worst})
                res = {"failing_points": failing, "processes": per}
                if failing:
                    res["barrier_free"] = res["count"] = "not resolved: a point fails a rule"
                else:
                    half = HALF_SYNC * sync[warps]
                    if all(p["d1"] > SHARE * p["t"][1] and p["d1"] > half for p in per):
                        res["barrier_free"] = "does not pay the barrier cost"
                    elif all(p["d1"] < SHARE * p["t"][1] for p in per):
                        res["barrier_free"] = "pays it"
                    else:
                        res["barrier_free"] = "not resolved"
                    rise = [p["t"][14] - p["t"][1] for p in per]
                    if all(p["slope"] > 0 and r > SHARE * p["t"][1] for p, r in zip(per, rise)):
                        res["count"] = "matters"
                        res["linear"] = all(p["worst_residual"] <= LINEAR for p in per)
                    elif all(abs(r) < SHARE * p["t"][1] for p, r in zip(per, rise)):
                        res["count"] = "does not matter"
                    else:
                        res["count"] = "not resolved"
                res["fixed_sync_ns"] = sync[warps]
                out[f"W{w}_w{warps}_g{g}"] = res
    raws_rules = [capture_posthoc.dispersion_check(r)["rules"] for r in runs]
    return {
        "status": "judged",
        "rules_1_2_per_process": raws_rules,
        "rule_3": capture_posthoc.reproducibility_check(runs),
        "configurations": out,
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
        text = subprocess.check_output([cuobjdump(), "-sass", str(args.binary)], text=True)
        result = sass_check(text)
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
