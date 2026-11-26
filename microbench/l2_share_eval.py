"""Evaluate the single-buffer L2 sharing test as registered in docs/results/analysis/h1-l2-share-gb10-registration.md.

The model times come from the ignored test `sync_loop_burst` in crates/vgpu-core/tests/posthoc_sync_loop.rs,
run on a `sync-loop-burst-check` output.

  python -m microbench.l2_share_eval profiles/gb10.json model-on-grid.txt --check-grid
  python -m microbench.l2_share_eval profiles/gb10.json model.txt --paths p1.json p2.json p3.json --runs r1.json r2.json r3.json
"""

import argparse
import json
import statistics as st

CALIBRATION = (1, 24576)  # (n, SB)
DIFFER = 0.02
WRONG = 0.25


def interpolate(points, x):
    points = sorted(points)
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


def curves(paths):
    """Median over processes of each process's median, per path, one CTA per SM points only."""
    got = {}
    for path in paths:
        for p in json.load(open(path))["l2_paths"]:
            if p["grid"] == p["active_sms"]:
                got.setdefault((p["path"], p["active_sms"]), []).append(
                    st.median(p["gbps_samples"])
                )
    out = {}
    for (name, sms), v in got.items():
        out.setdefault(name, []).append((sms, st.median(v)))
    return {name: sorted(v) for name, v in out.items()}


def excess(paths):
    turn = {}
    for path in paths:
        for p in json.load(open(path))["sync_loop_burst"]:
            turn.setdefault((p["dep"], p["grid"], p["stage_bytes"]), []).append(
                st.median(p["turnaround_ns_samples"])
            )
    return {
        (n, sb): st.median(f - b for f, b in zip(v, turn[("noload", n, sb)]))
        for (dep, n, sb), v in turn.items()
        if dep == "free"
    }


def model(path):
    t = {}
    for line in open(path):
        dep, _, n, sb, ns, _ = line.split()
        t[(dep, int(n), int(sb))] = float(ns)
    return {(n, sb): t[("free", n, sb)] - t[("noload", n, sb)] for (d, n, sb) in t if d == "free"}


def forms(l2l, curve):
    p1 = interpolate(curve, 1)
    return {
        "burst": lambda n, sb: max(l2l + sb / p1, n * sb / interpolate(curve, n)),
        "share": lambda n, sb: l2l + sb / min(p1, interpolate(curve, n) / n),
    }


def beats(ea, eb):
    closer = sum(a < b for a, b in zip(ea, eb))
    return st.median(ea) < 0.5 * st.median(eb) and closer >= 2 / 3 * len(ea)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile")
    parser.add_argument("model")
    parser.add_argument("--paths", nargs="+")
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--check-grid", action="store_true")
    args = parser.parse_args()
    memory = json.load(open(args.profile))["measured"]["memory"]
    l2l = memory["l2_latency_ns"]["value"]
    if args.check_grid:
        # Before measuring, with the profile's ld.global.cg curve standing in for the cp.async one: every
        # pair of forms must differ on at least six held-out points.
        proxy = forms(
            l2l, [(r["active_sms"], r["bw_gbps"]["value"]) for r in memory["l2_bw_by_active_sms"]]
        )
        avg = model(args.model)
        pred = {"model": avg, **{n: {k: f(*k) for k in avg} for n, f in proxy.items()}}
        held = [k for k in avg if k != CALIBRATION]
        for a, b in (("model", "burst"), ("model", "share"), ("burst", "share")):
            n = sum(abs(pred[a][k] / pred[b][k] - 1) > DIFFER for k in held)
            print(f"{a} and {b} differ on {n}/{len(held)} held-out points")
            if n < 6:
                raise SystemExit(f"grid cannot separate {a} and {b}")
        return
    cur = curves(args.paths)
    for name, v in cur.items():
        print(f"{name}: " + ", ".join(f"{s} SM {g:.0f}" for s, g in v))
    meas, avg = excess(args.runs), model(args.model)
    shapes = forms(l2l, cur["cp_async"])
    c0 = meas[CALIBRATION] - shapes["burst"](*CALIBRATION)
    if abs(shapes["share"](*CALIBRATION) - shapes["burst"](*CALIBRATION)) > 1e-6:
        raise SystemExit("the calibration point must give both forms the same base")
    pred = {"model": avg, **{name: {k: c0 + f(*k) for k in meas} for name, f in shapes.items()}}
    ld = forms(l2l, cur["ld_global_cg"])
    c0_ld = meas[CALIBRATION] - ld["burst"](*CALIBRATION)
    print(
        f"L2l {l2l}  c0 {c0:.1f} (cp.async curve), {c0_ld:.1f} (ld.global.cg curve, described only)"
    )
    held = sorted(k for k in meas if k != CALIBRATION)
    for k in sorted(meas):
        print(
            f"  n {k[0]:2} {k[1] // 1024:2}K: E {meas[k]:6.0f}  "
            + "  ".join(f"{f} {pred[f][k]:6.0f}" for f in pred)
            + f"  (ld: burst {c0_ld + ld['burst'](*k):6.0f} share {c0_ld + ld['share'](*k):6.0f})"
            + ("  calibration" if k == CALIBRATION else "")
        )
    err = {f: {k: abs(pred[f][k] / meas[k] - 1) for k in held} for f in pred}
    for f in pred:
        print(
            f"{f}: median |err| {st.median(err[f].values()) * 100:.1f}% on {len(held)} held-out points"
        )
    if all(st.median(e.values()) > WRONG for e in err.values()):
        print("every form misses by more than 25%: all wrong")
        return
    winners = []
    for a in pred:
        ok = True
        for b in pred:
            if a == b:
                continue
            diff = [k for k in held if abs(pred[a][k] / pred[b][k] - 1) > DIFFER]
            won = bool(diff) and beats([err[a][k] for k in diff], [err[b][k] for k in diff])
            print(
                f"  {a} against {b} on {len(diff)} differing points: {'wins' if won else 'does not win'}"
            )
            ok &= won
        if ok:
            winners.append(a)
    print("winner:", winners[0] if winners else "none")


if __name__ == "__main__":
    main()
