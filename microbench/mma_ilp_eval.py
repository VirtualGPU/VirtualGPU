"""Evaluate mma-ilp runs as registered in docs/results/analysis/h7-mma-ilp-registration.md.

python -m microbench.mma_ilp_eval --check profiles/gb10.json profiles/rtx4070.json
python -m microbench.mma_ilp_eval profiles/gb10.json run1.json run2.json run3.json
"""

import argparse
import json
import statistics as st

GRID = [(i, w, c) for i in (1, 2, 4, 8) for w in (1, 2, 3, 4, 8) for c in (1, 2, 3) if w * c <= 16]
SEEN = {(1, 4, 1), (1, 8, 1), (4, 1, 1), (4, 2, 1), (4, 4, 1), (4, 8, 1)}
DIFFER = 0.02
KEY = "mma.sync:f16:f16:f32:16x8x16"


def interpolate(points, x):
    points = sorted(points)
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


def forms(profile):
    c = json.load(open(profile))["measured"]["compute"]
    curve = [
        (p["mma_warps_per_sm"], p["tflops_per_sm"]["value"] * 1e3)
        for p in c["mma_tflops_per_sm_by_warps"][KEY]
    ]
    dep = next(
        r["slope_ns_per_mma"]["value"]
        for r in c["serial_stage_overhead_ns"]
        if r["variant"] == "mma_only_dependent" and r["num_warps"] == 4
    )

    def m0(ilp, w, cps):
        return 4096 * w * cps / interpolate(curve, w * cps)

    return {"M0": m0, "M1": lambda ilp, w, cps: max(m0(ilp, w, cps), dep / ilp)}, dep


def differ(f, k):
    return abs(f["M1"](*k) / f["M0"](*k) - 1) > DIFFER


def check(gb10, rtx4070):
    f, _ = forms(gb10)
    held = [k for k in GRID if k not in SEEN]
    diff = [k for k in held if differ(f, k)]
    print(f"GB10: {len(held)} held-out points, forms differ on {len(diff)}: {diff}")
    if len(diff) < 6:
        raise SystemExit("too few points where the forms differ")
    g, _ = forms(rtx4070)
    worst = max(g["M1"](*k) / g["M0"](*k) for k in GRID)
    print(f"RTX 4070: largest M1 / M0 over the grid {worst:.3f}")
    if worst >= 1.05:
        raise SystemExit("the RTX 4070 prediction that M1 and M0 agree within 5% does not hold")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("profile")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        check(args.profile, args.runs[0])
        return
    f, dep = forms(args.profile)
    got = {}
    for path in args.runs:
        for p in json.load(open(path))["mma_ilp"]:
            got.setdefault((p["ilp"], p["num_warps"], p["ctas_per_sm"]), []).append(
                st.median(p["ns_per_mma_per_warp_samples"])
            )
    meas = {k: st.median(v) for k, v in got.items()}
    print(f"L_dep {dep} ns")
    for k in GRID:
        tag = " seen" if k in SEEN else (" differ" if differ(f, k) else "")
        print(
            f"  ILP {k[0]} W {k[1]} cps {k[2]}: measured {meas[k]:6.2f}  M0 {f['M0'](*k):6.2f}  M1 {f['M1'](*k):6.2f}{tag}"
        )
    held = [k for k in GRID if k not in SEEN]
    err = {n: {k: abs(f[n](*k) / meas[k] - 1) for k in held} for n in f}
    diff = [k for k in held if differ(f, k)]
    same = [k for k in held if not differ(f, k)]
    if diff:
        e0, e1 = [err["M0"][k] for k in diff], [err["M1"][k] for k in diff]
        closer = sum(a < b for a, b in zip(e1, e0))
        wins = st.median(e1) < 0.5 * st.median(e0) and closer >= 2 / 3 * len(diff)
        print(
            f"differing points {len(diff)}: M0 {st.median(e0) * 100:.1f}%  M1 {st.median(e1) * 100:.1f}%  M1 closer on {closer}: M1 wins {wins}"
        )
        if st.median(e0) > 0.25 and st.median(e1) > 0.25:
            print("both forms miss by more than 25% on the differing points: both wrong")
    if same:
        m = st.median(err["M0"][k] for k in same)
        print(
            f"points where the forms agree {len(same)}: M0 median |err| {m * 100:.1f}%: M0 holds there {m <= 0.10}"
        )


if __name__ == "__main__":
    main()
