"""Evaluate sync-loop-free runs as registered in docs/results/analysis/h1-free-gb10-registration.md.

python -m microbench.sync_loop_free_eval profiles/gb10.json run1.json run2.json run3.json
"""

import argparse
import itertools
import json
import math
import statistics as st

CALIBRATION = ((16384, 8, 8, 16), (16384, 4, 8, 16), (16384, 8, 32, 16))  # (SB, W, MI, G), in order
STEP_LIMIT, SPREAD_LIMIT = 0.05, 0.25
GATE_NS = 40.0
RULES = ("R0", "R1")
SHAPES = ("S1", "S2", "S3")


def mean_excess(shape, m, x):
    """E[max(0, x - d)] for a follower offset d of the given shape with median m."""
    if m <= 0:
        return x
    if shape == "S1":  # uniform on [0, w], median w / 2
        w = 2 * m
        return x - w / 2 if w <= x else x * x / (2 * w)
    if shape == "S2":  # exponential, median mu ln 2
        mu = m / math.log(2)
        return x - mu * (1 - math.exp(-x / mu))
    sigma = m / 0.6744897501960817  # half-normal, median 0.6745 sigma
    n = 2000
    return sum(
        max(0.0, x - d)
        * math.sqrt(2 / math.pi)
        / sigma
        * math.exp(-d * d / (2 * sigma * sigma))
        * (x / n)
        for d in ((i + 0.5) * x / n for i in range(n))
    )


def load(paths):
    rows = {}
    for path in paths:
        run = json.load(open(path))
        for p in run["sync_loop_free"]:
            key = (p["stage_bytes"], p["num_warps"], p["mma_groups"], p["group"])
            variant = "warm" if p["region"] == "warm" else ("stamp" if p["stamp"] else "cold")
            rows.setdefault(key, {}).setdefault(variant, []).append(p)
    return rows


def med(rows, field):
    return st.median(st.median(p[field]) for p in rows)


def summarize(rows, x, dram_gbps, groups):
    configs = {}
    for key, v in rows.items():
        sb, _, _, g = key
        step = med(v["cold"], "step_ns_samples")
        excess = st.median(
            st.median(c["stage_latency_ns_samples"]) - st.median(w["stage_latency_ns_samples"])
            for c, w in zip(v["cold"], v["warm"])
        )
        distorted = (
            abs(med(v["stamp"], "step_ns_samples") / step - 1) > STEP_LIMIT
            or abs(
                med(v["stamp"], "finish_spread_ns_samples")
                / med(v["cold"], "finish_spread_ns_samples")
                - 1
            )
            > SPREAD_LIMIT
        )
        configs[key] = {
            "step": step,
            "excess": excess,
            "median_offset": st.median(
                p["follower_offset_ns_quantiles"]["p50"] for p in v["stamp"]
            ),
            "distorted": distorted,
            "saturated": groups * sb / dram_gbps >= 0.5 * step,
        }
    return configs


def predict(rule, shape, cal, c, x, g):
    m = cal["median_offset"] if rule == "R0" else cal["median_offset"] / cal["step"] * c["step"]
    return (x + (g - 1) * mean_excess(shape, m, x)) / g


def beats(ea, eb):
    closer = sum(a < b for a, b in zip(ea, eb))
    return st.median(ea) < 0.5 * st.median(eb) and closer >= 2 / 3 * len(ea)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile")
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    memory = json.load(open(args.profile))["measured"]["memory"]
    x = memory["dram_latency_ns"]["value"] - memory["l2_latency_ns"]["value"]
    groups = json.load(open(args.runs[0]))["groups"]
    configs = summarize(load(args.runs), x, memory["dram_bw_read_gbps"]["value"], groups)
    print(f"X {x:.1f} ns, {groups} groups")
    for key, c in sorted(configs.items()):
        flags = " ".join(f for f in ("distorted", "saturated") if c[f])
        print(
            f"  SB {key[0] // 1024}K W {key[1]} MI {key[2]:2} G {key[3]:2}: step {c['step']:6.0f} excess {c['excess']:6.0f} "
            f"median offset {c['median_offset']:5.0f} {flags}"
        )
    cal_key = next((k for k in CALIBRATION if not configs[k]["distorted"]), None)
    if cal_key is None:
        print("every calibration candidate is distorted: no calibration, distributions only")
        return
    cal = configs[cal_key]
    held = [k for k, c in sorted(configs.items()) if k != cal_key and not c["saturated"]]
    print(f"calibration {cal_key}, {len(held)} held-out configs")
    combos = [(r, s) for r in RULES for s in SHAPES]
    err = {
        combo: [abs(predict(*combo, cal, configs[k], x, k[3]) - configs[k]["excess"]) for k in held]
        for combo in combos
    }
    err["model"] = [abs(x - configs[k]["excess"]) for k in held]
    for name, e in err.items():
        print(f"  {name}: median |err| {st.median(e):.0f} ns")
    passing = [c for c in combos if st.median(err[c]) <= GATE_NS and beats(err[c], err["model"])]
    print("pass the gate:", passing or "none")
    if not passing:
        print("no form passes: known limitation")
        return
    best = [c for c in passing if not any(beats(err[o], err[c]) for o in passing if o != c)]
    chosen = min(best, key=lambda c: (RULES.index(c[0]), SHAPES.index(c[1])))
    print(f"equally best: {best}; chosen by priority: {chosen}")
    r0 = [c for c in combos if c[0] == "R0"]
    r1 = [c for c in combos if c[0] == "R1"]
    kernel = any(all(beats(err[a], err[b]) for b in r0) for a in r1)
    card = any(all(beats(err[a], err[b]) for b in r1) for a in r0)
    print(f"R1 beats every R0: {kernel}; R0 beats every R1: {card}")
    for a, b in itertools.combinations(combos, 2):
        if beats(err[a], err[b]) or beats(err[b], err[a]):
            print(
                f"  {a if beats(err[a], err[b]) else b} beats {b if beats(err[a], err[b]) else a}"
            )


if __name__ == "__main__":
    main()
