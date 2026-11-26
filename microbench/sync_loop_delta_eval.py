"""Evaluate sync-loop-delta runs as registered in docs/results/analysis/h1-delta-rtx4070-registration.md.

python -m microbench.sync_loop_delta_eval --check-grid profiles/rtx4070.json 0,32,64,96,128,160,192,224,320,700
python -m microbench.sync_loop_delta_eval profiles/rtx4070.json run1.json run2.json run3.json
"""

import argparse
import itertools
import json
import statistics as st

LATE_LIMIT = 0.01


def fields(profile_path):
    memory = json.load(open(profile_path))["measured"]["memory"]
    return memory["l2_latency_ns"]["value"], memory["dram_latency_ns"]["value"]


def forms(l2l, ld):
    x = ld - l2l
    return {
        "model": lambda d: x,
        "step": lambda d: x if d < ld else 0.0,
        "remain": lambda d: max(0.0, x - d),
    }


def differing(h, a, b, offsets):
    return [i for i, d in enumerate(offsets) if abs(h[a](d) - h[b](d)) > 0.5]


def check_grid(h, offsets):
    """The forms must be separable on the grid: step and remain differ on at least two thirds of
    the points, and every pair differs on at least two."""
    for a, b in itertools.combinations(h, 2):
        n = len(differing(h, a, b, offsets))
        need = 2 / 3 * len(offsets) if {a, b} == {"step", "remain"} else 2
        if n < need:
            raise SystemExit(f"grid cannot separate {a} and {b}: they differ on {n}/{len(offsets)}")
        print(f"{a} and {b} differ on {n}/{len(offsets)} points")


def load(paths):
    runs = [json.load(open(p)) for p in paths]
    latency, late = {}, {}
    for r in runs:
        grid = r.get("grid", r["sm_count"])
        followers = grid - grid // r["group"]
        for p in r["sync_loop_delta"]:
            k = (p["region"], p["stage_bytes"], p["delta_ns"])
            ns = st.median(p["follower_latency_cycles_samples"]) / st.median(p["sm_ghz_samples"])
            latency.setdefault(k, []).append((ns, st.median(p["measured_offset_ns_samples"])))
            share = p["late_steps"] / (p["steps"] * followers * r["samples"])
            late[k] = max(late.get(k, 0.0), share)
    return latency, late


def judge(h, points):
    """points: (offset, excess). A form wins if, against each other form on the points where their
    predictions differ, its median absolute error is below half of the other's and it is closer on
    at least two thirds of them."""
    offsets = [d for d, _ in points]
    check_grid(h, offsets)
    err = {n: [abs(h[n](d) - e) for d, e in points] for n in h}
    print(
        "median |err| ns, all points: "
        + ", ".join(f"{n} {st.median(v):.0f}" for n, v in err.items())
    )
    if all(st.median(v) > 50 for v in err.values()):
        print("every form misses by more than 50 ns: all wrong")
        return
    for a in h:
        ok = True
        for b in h:
            if a == b:
                continue
            idx = differing(h, a, b, offsets)
            ea, eb = [err[a][i] for i in idx], [err[b][i] for i in idx]
            closer = sum(x < y for x, y in zip(ea, eb))
            beats = st.median(ea) < 0.5 * st.median(eb) and closer >= 2 / 3 * len(idx)
            ok &= beats
            print(
                f"  {a} against {b} on {len(idx)} differing points: closer on {closer}, median {st.median(ea):.0f} vs {st.median(eb):.0f}"
            )
        print(f"  {a} wins: {ok}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile")
    parser.add_argument("runs", nargs="*")
    parser.add_argument("--check-grid", metavar="NS_LIST")
    args = parser.parse_args()
    l2l, ld = fields(args.profile)
    h = forms(l2l, ld)
    print(f"L2l {l2l} Ld {ld} Ld-L2l {ld - l2l:.1f}")
    if args.check_grid:
        check_grid(h, [float(x) for x in args.check_grid.split(",")])
        return
    latency, late = load(args.runs)
    points = []
    for sb in sorted({k[1] for k in latency}):
        base = ("warm", sb, 0)
        if late[base] > LATE_LIMIT:
            print(f"{sb // 1024}K excluded: warm baseline late {late[base] * 100:.1f}%")
            continue
        warm = st.median(ns for ns, _ in latency[base])
        for d in sorted(k[2] for k in latency if k[0] == "cold" and k[1] == sb):
            k = ("cold", sb, d)
            if late[k] > LATE_LIMIT:
                print(f"  {sb // 1024}K delay {d} excluded: late {late[k] * 100:.2f}%")
                continue
            excess = st.median(ns for ns, _ in latency[k]) - warm
            offset = st.median(o for _, o in latency[k])
            points.append((offset, excess))
            print(
                f"  {sb // 1024}K delay {d:5} offset {offset:6.0f} excess {excess:6.0f} | "
                + " ".join(f"{n} {h[n](offset):5.0f}" for n in h)
            )
    if points:
        judge(h, points)


if __name__ == "__main__":
    main()
