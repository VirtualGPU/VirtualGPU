"""Evaluate sync-loop-burst runs as registered in docs/results/analysis/h1-burst-gb10-registration.md.

The model times come from the ignored test `sync_loop_burst` in crates/vgpu-core/tests/posthoc_sync_loop.rs.

  python -m microbench.sync_loop_burst_eval --write-grid grid.json
  python -m microbench.sync_loop_burst_eval --check-grid profiles/gb10.json model-on-grid.txt
  python -m microbench.sync_loop_burst_eval profiles/gb10.json model.txt run1.json run2.json run3.json
"""

import argparse
import json
import statistics as st

CALIBRATION = ("lock", 12, 16384)
GRID_N = (12, 36, 40, 44, 48)
GRID_SB = (16384, 32768, 49152, 65536)
SEEN_N = (
    48  # the lock points at 48 CTAs are the sync-loop-bytes configs the burst form was read from
)
MID = 4  # coherence mark at the middle step
RELOCK = (
    0.5  # a stag point whose median phase coherence at the middle step exceeds this is not judged
)
DIFFER = 0.02


def interpolate(points, x):
    points = sorted(points)
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


def measured(paths):
    turn, coherence = {}, {}
    for path in paths:
        for p in json.load(open(path))["sync_loop_burst"]:
            key = (p["dep"], p["start"], p["grid"], p["stage_bytes"])
            turn.setdefault(key, []).append(st.median(p["turnaround_ns_samples"]))
            if p["dep"] == "free":
                coherence.setdefault(key[1:], []).append(
                    [st.median(c) for c in p["coherence_samples"]]
                )
    excess = {}
    for (dep, start, n, sb), v in turn.items():
        if dep == "free":
            base = turn[("noload", "lock", n, sb)]
            excess[(start, n, sb)] = st.median(f - b for f, b in zip(v, base))
    marks = {k: [st.median(m) for m in zip(*v)] for k, v in coherence.items()}
    return excess, marks


def model(path):
    t = {}
    for line in open(path):
        dep, start, n, sb, ns, _ = line.split()
        t[(dep, start, int(n), int(sb))] = float(ns)
    return {
        (s, n, sb): t[("free", s, n, sb)] - t[("noload", "lock", n, sb)]
        for (d, s, n, sb) in t
        if d == "free"
    }


def judge(name, keys, meas, pred):
    diff = [k for k in keys if abs(pred["B-avg"][k] / pred["B-burst"][k] - 1) > DIFFER]
    err = {f: [abs(pred[f][k] / meas[k] - 1) for k in diff] for f in pred}
    if not diff:
        print(f"{name}: the forms agree on every point")
        return None
    for f, e in err.items():
        print(f"{name}: {f} median |err| {st.median(e) * 100:.1f}% on {len(diff)} differing points")
    winner = None
    for a, b in (("B-avg", "B-burst"), ("B-burst", "B-avg")):
        closer = sum(x < y for x, y in zip(err[a], err[b]))
        if st.median(err[a]) < 0.5 * st.median(err[b]) and closer >= 2 / 3 * len(diff):
            winner = a
    if all(st.median(e) > 0.25 for e in err.values()):
        print(f"{name}: both forms miss by more than 25%: both wrong")
        return "wrong"
    print(f"{name}: winner {winner or 'none'}")
    return winner


def fields(profile):
    memory = json.load(open(profile))["measured"]["memory"]
    curve = [(r["active_sms"], r["bw_gbps"]["value"]) for r in memory["l2_bw_by_active_sms"]]
    l2l, p1 = memory["l2_latency_ns"]["value"], interpolate(curve, 1)
    return l2l, p1, lambda n, sb: max(l2l + sb / p1, n * sb / interpolate(curve, n))


def judged_lock(keys):
    return [k for k in keys if k[0] == "lock" and k != CALIBRATION and k[1] != SEEN_N]


def write_grid(path):
    points = [
        {
            "dep": dep,
            "start": start,
            "grid": n,
            "num_warps": 8,
            "stage_bytes": sb,
            "turnaround_ns_samples": [0.0],
        }
        for n in GRID_N
        for sb in GRID_SB
        for dep, start in (("noload", "lock"), ("free", "lock"), ("free", "stag"))
    ]
    json.dump({"sync_loop_burst": points}, open(path, "w"))


def check_grid(profile, model_path):
    """Before measuring: the forms must differ (by more than DIFFER) on at least six judged lock
    points, since each comparison uses only the points where they differ."""
    _, _, burst = fields(profile)
    avg = model(model_path)
    keys = judged_lock(avg)
    diff = [k for k in keys if abs(avg[k] / burst(k[1], k[2]) - 1) > DIFFER]
    print(f"judged lock points {len(keys)}, forms differ on {len(diff)}: {sorted(diff)}")
    if len(diff) < 6:
        raise SystemExit("grid cannot separate the forms on the judged lock points")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", nargs="?")
    parser.add_argument("model", nargs="?")
    parser.add_argument("runs", nargs="*")
    parser.add_argument("--write-grid", metavar="PATH")
    parser.add_argument("--check-grid", action="store_true")
    args = parser.parse_args()
    if args.write_grid:
        write_grid(args.write_grid)
        return
    if args.check_grid:
        check_grid(args.profile, args.model)
        return
    l2l, p1, burst = fields(args.profile)
    meas, coherence = measured(args.runs)
    avg = model(args.model)

    n, sb = CALIBRATION[1:]
    if burst(n, sb) != l2l + sb / p1:
        raise SystemExit("calibration point is not on the port term")
    c0 = meas[CALIBRATION] - burst(n, sb)
    pred = {"B-avg": avg, "B-burst": {k: c0 + burst(k[1], k[2]) for k in meas}}
    print(f"L2l {l2l} P1 {p1} c0 {c0:.1f}")
    for k in sorted(meas):
        flag = " relocked" if k[0] == "stag" and coherence[k][MID] > RELOCK else ""
        flag += " seen, not judged" if k[0] == "lock" and k[1] == SEEN_N else ""
        print(
            f"  {k[0]} n {k[1]:2} {k[2] // 1024:2}K: E {meas[k]:6.0f}  B-avg {avg[k]:6.0f}  B-burst {pred['B-burst'][k]:6.0f}"
            f"  coherence at steps 0,16,64,256,S/2,S-1: {' '.join(f'{c:.2f}' for c in coherence[k])}{flag}"
        )
    lock = judged_lock(meas)
    stag = [k for k in meas if k[0] == "stag" and coherence[k][MID] <= RELOCK]
    w_lock = judge("lock", lock, meas, pred)
    w_stag = judge("stag", stag, meas, pred)
    if w_lock == "B-burst" and w_stag != "B-burst":
        print("mechanism: burst against the L2 bandwidth (lock only)")
    elif w_lock == "B-burst":
        print("B-burst wins with staggered starts too: not a burst; a steady share")
    else:
        print("mechanism not shown")


if __name__ == "__main__":
    main()
