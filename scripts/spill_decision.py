"""Registered decision on the spill reload latency when `spill-latency` is recaptured.

The rule is in docs/results/analysis/v0.3-spill-decision-registration.md. `freeze` writes the
predictions of the current core on the current profiles, before any long-launch capture of
`spill-latency`. `score` applies the rule to a recaptured profile and prints the verdict.

    python scripts/spill_decision.py freeze --profile gb10=profiles/gb10.json \\
        --profile rtx4070=profiles/rtx4070.json --out FROZEN.json
    python scripts/spill_decision.py score --frozen FROZEN.json \\
        --profile gb10=NEW.json --raw gb10=RUN_DIR/merged-spill-latency.json [--out OUT.json]

The lookup is a copy of `reload_table`, `reload_table_ns` and `Machine::spill_reload_ns` in
crates/vgpu-core/src/sim.rs, with the zero carveout miss curve as in the core test
`spill_reload_latency_at_the_l1_boundary_matches_the_microbenchmark`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

BATCH = "batch_before_dep"
HELD_OUT_COUNTS = (32, 68)
DECIDING = ("gb10", "rtx4070")
COUNT_MEDIAN_MAX = 0.10
COUNT_WORST_MAX = 0.25
BOUNDARY_MAPE_MAX = 0.20


def interpolate(points, x):
    """profile::interpolate: linear, clamped at both ends."""
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x < x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


def spill_points(profile):
    return profile["measured"]["memory"]["spill_reload_latency_by_count"]


def extra(q):
    return q["extra_ns_vs_zero"]["value"]


def footprint(q):
    return q["frame_bytes_per_thread"] * q["resident_threads_per_sm"]


def reload_table(points, regime):
    table = {}
    for q in points:
        if q["regime"] == regime and q["dependency_mode"] == BATCH:
            table.setdefault(footprint(q), []).append((q["reload_count"], max(extra(q), 0.0)))
    return sorted((f, sorted(c)) for f, c in table.items())


def reload_table_ns(table, count, at):
    per_frame = []
    for f, c in table:
        n, ns = c[-1]
        per_frame.append((f, ns * count / n if count > n > 0 else interpolate(c, count)))
    return interpolate(per_frame, at)


def miss_curve(profile):
    curve = next(
        c
        for c in profile["measured"]["memory"]["l1_to_l2_spill_threshold_by_carveout"]
        if c["smem_carveout_bytes"] == 0
    )
    out, top = [], 0.0
    for q in curve["miss_rate_by_working_set"]:
        top = min(max(top, q["estimated_miss_rate"]["value"]), 1.0)
        out.append((q["working_set_bytes"], max(top, 0.0)))
    return out


def boundary(profile):
    """The l1_boundary batch points with reloads, predicted by the lookup alone."""
    points = spill_points(profile)
    hit, l2 = reload_table(points, "l1_hit"), reload_table(points, "l2_spill")
    miss = miss_curve(profile)
    rows = []
    for q in points:
        if q["regime"] != "l1_boundary" or q["dependency_mode"] != BATCH or q["reload_count"] == 0:
            continue
        m = interpolate(miss, footprint(q))
        n, at = q["reload_count"], footprint(q)
        predicted = (1 - m) * reload_table_ns(hit, n, at) + m * reload_table_ns(l2, n, at)
        rows.append(row(q, predicted))
    return rows


def leave_one_out(profile):
    """Each batch point at 32 or 68 reloads from the same frame's other counts, per regime."""
    points = spill_points(profile)
    out = {}
    for regime in ("l1_hit", "l2_spill"):
        rows = []
        for q in points:
            if (
                q["regime"] != regime
                or q["dependency_mode"] != BATCH
                or q["reload_count"] not in HELD_OUT_COUNTS
            ):
                continue
            rest = sorted(
                (p["reload_count"], max(extra(p), 0.0))
                for p in points
                if p is not q
                and p["regime"] == regime
                and p["dependency_mode"] == BATCH
                and p["frame_bytes_per_thread"] == q["frame_bytes_per_thread"]
            )
            rows.append(row(q, interpolate(rest, q["reload_count"])))
        out[regime] = rows
    return out


def whole_core(profile, text):
    import recheck_mechanisms as rm

    mech = next(m for m in rm.MECHANISMS if m["name"] == "spill_l1_boundary")
    result = rm.check(mech, profile, text)
    return [] if result is None else result["rows"]


def row(q, predicted):
    measured = extra(q)
    return {
        "point": f"{q['frame_bytes_per_thread']}B x{q['reload_count']}",
        "measured": measured,
        "predicted": predicted,
        "error": predicted / measured - 1 if measured > 0 else None,
    }


def errors(rows):
    return [abs(r["error"]) for r in rows if r["error"] is not None]


def count_test(rows):
    e = errors(rows)
    if not e or len(e) < len(rows):
        return {"verdict": "not decided", "reason": "a held-out point has no extra time"}
    median, worst = st.median(e), max(e)
    ok = median <= COUNT_MEDIAN_MAX and worst <= COUNT_WORST_MAX
    return {"verdict": "pass" if ok else "fail", "median_abs_error": median, "worst": worst}


def boundary_test(rows):
    e = errors(rows)
    if len(e) != 4:
        return {"verdict": "not decided", "reason": f"{len(e)} of 4 boundary points"}
    mape = st.mean(e)
    return {"verdict": "pass" if mape < BOUNDARY_MAPE_MAX else "fail", "mape": mape}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def core_info():
    from virtualgpu import _core

    return _core.build_info()


def predictions(path):
    text = Path(path).read_text()
    profile = json.loads(text)
    return profile, {
        "path": str(path),
        "sha256": sha256(path),
        "revision": profile["provenance"].get("profile_revision"),
        "extra_ns_vs_zero": {
            f"{q['regime']} {q['dependency_mode']} {q['frame_bytes_per_thread']}B "
            f"x{q['reload_count']}": extra(q)
            for q in spill_points(profile)
        },
        "leave_one_out": leave_one_out(profile),
        "boundary_lookup": boundary(profile),
        "boundary_whole_core": whole_core(profile, text),
    }


def eligible(profile, raw_path):
    """Why a recaptured profile cannot be scored, or None."""
    protocol = profile["provenance"].get("capture_protocol")
    if protocol is None:
        return "not a long-launch profile"
    if protocol.get("rule_three", {}).get("spill-latency", {}).get("status") != "passed":
        return "spill-latency did not pass the capture rules"
    if raw_path is None:
        return "no spill-latency capture file given"
    raw = json.loads(Path(raw_path).read_text())
    raw = raw.get("workload", raw)
    if not raw.get("launch_cost_by_frame"):
        return "capture has no launch_cost_by_frame, so the launch cost is not taken out"
    return None


def score(frozen, name, path, raw_path):
    profile, now = predictions(path)
    reason = eligible(profile, raw_path)
    if reason:
        return {"name": name, "path": str(path), "sha256": now["sha256"], "not_run": reason}
    old = frozen["profiles"].get(name, {}).get("extra_ns_vs_zero", {})
    shift = {
        k: v / old[k] - 1
        for k, v in now["extra_ns_vs_zero"].items()
        if k in old and old[k] > 0 and k.startswith("l2_spill") and BATCH in k
    }
    return {
        **now,
        "name": name,
        "count_test": {r: count_test(rows) for r, rows in now["leave_one_out"].items()},
        "boundary_test": boundary_test(now["boundary_lookup"]),
        "l2_spill_batch_shift_vs_frozen": shift,
    }


def verdict(results):
    """Pass only when both reference GPUs ran and passed every test."""
    by_name = {r["name"]: r for r in results}
    lines = []
    for name in DECIDING:
        r = by_name.get(name)
        if r is None or "not_run" in r:
            lines.append(f"{name}: not run ({'no profile' if r is None else r['not_run']})")
            continue
        tests = [*r["count_test"].values(), r["boundary_test"]]
        if any(t["verdict"] == "not decided" for t in tests):
            lines.append(f"{name}: not decided")
        else:
            ok = all(t["verdict"] == "pass" for t in tests)
            lines.append(f"{name}: {'pass' if ok else 'fail'}")
    states = [line.split(": ")[1] for line in lines]
    if all(s == "pass" for s in states):
        overall = "pass"
    elif any(s == "fail" for s in states):
        overall = "fail"
    else:
        overall = "not decided"
    return overall, lines


def pairs(values):
    out = []
    for v in values:
        name, _, path = v.partition("=")
        if not path:
            raise SystemExit(f"expected NAME=PATH, got {v!r}")
        out.append((name, path))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--profile", action="append", required=True)
    f.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--frozen", type=Path, required=True)
    s.add_argument("--profile", action="append", required=True)
    s.add_argument("--raw", action="append", default=[])
    s.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    if args.command == "freeze":
        out = {
            "core": core_info(),
            "script_sha256": sha256(__file__),
            "profiles": {name: predictions(path)[1] for name, path in pairs(args.profile)},
        }
        args.out.write_text(json.dumps(out, indent=1) + "\n")
        print(f"Wrote {args.out}")
        return 0

    frozen = json.loads(args.frozen.read_text())
    raws = dict(pairs(args.raw))
    results = [score(frozen, n, p, raws.get(n)) for n, p in pairs(args.profile)]
    overall, lines = verdict(results)
    for line in lines:
        print(line)
    print(f"verdict: {overall}")
    if args.out:
        out = {
            "core": core_info(),
            "script_sha256": sha256(__file__),
            "frozen_sha256": sha256(args.frozen),
            "verdict": overall,
            "by_gpu": lines,
            "profiles": results,
        }
        args.out.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
