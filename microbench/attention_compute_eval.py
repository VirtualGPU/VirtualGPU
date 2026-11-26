"""Evaluate the compute-base sweep as registered in docs/results/analysis/h7-compute-registration.md.

The SASS comes from `cuobjdump -sass` of the cubins that `attention_compute cubins` writes.

  python -m microbench.attention_compute_eval profiles/gb10.json SASS_DIR run1.json run2.json run3.json
"""

import argparse
import json
import math
import re
import statistics as st
from collections import Counter
from pathlib import Path

KEY = "mma.sync:f16:f16:f32:16x8x16"
# SASS writes ldmatrix .x1, .x2, .x4 as LDSM.16.M88, LDSM.16.M88.2, LDSM.16.M88.4 (MT88 when transposed).
LDSM_BYTES = {"M88": 128, "MT88": 128, "2": 256, "4": 512}
DIFFER = 0.02
WRONG = 0.15
CLASSES = ("HMMA", "LDSM", "MUFU", "SHFL", "BAR")


def interpolate(points, x):
    points = sorted(points)
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


def loop_counts(sass):
    """Per-warp instruction counts of the longest backward-branch loop that holds HMMA."""
    instr = []
    for line in sass.splitlines():
        m = re.search(r"/\*([0-9a-f]{4,})\*/\s+(.*?);", line)
        if m:
            instr.append((int(m.group(1), 16), m.group(2).strip()))
    where = {a: i for i, (a, _) in enumerate(instr)}
    best = []
    for i, (_, text) in enumerate(instr):
        m = re.search(r"BRA\b.*?0x([0-9a-f]+)\s*$", text)
        if m and int(m.group(1), 16) in where and where[int(m.group(1), 16)] < i:
            body = [t for _, t in instr[where[int(m.group(1), 16)] : i + 1]]
            if any(t.split()[0].startswith("HMMA") or " HMMA" in t for t in body) and len(
                body
            ) > len(best):
                best = body
    counts, ldsm_bytes = Counter(), 0
    for text in best:
        op = re.sub(r"^@!?U?P\w+\s+", "", text).split()[0]
        base = op.split(".")[0]
        if base == "LDSM":
            last = op.split(".")[-1]
            if last not in LDSM_BYTES:
                raise ValueError(f"unexpected ldmatrix form {op}")
            ldsm_bytes += LDSM_BYTES[last]
        counts[base if base in CLASSES else "OTHER"] += 1
    if not best:
        raise ValueError("no MMA loop found")
    return counts, ldsm_bytes


def rates(profile):
    p = json.load(open(profile))["measured"]
    c, mem = p["compute"], p["memory"]
    curve = lambda name: [(q["warps_per_sm"], q["rate"]["value"]) for q in c[name]]  # noqa: E731
    sync = {
        (q["num_warps"], q["tokens"]): q["overhead_ns"]["value"]
        for q in p["pipeline"]["fixed_sync_ns"]
    }
    return {
        "mma": [
            (q["mma_warps_per_sm"], q["tflops_per_sm"]["value"] * 1e3)
            for q in c["mma_tflops_per_sm_by_warps"][KEY]
        ],
        "sfu": curve("sfu_issue_rate_per_sm_gips"),
        "shfl": curve("shfl_issue_rate_per_sm_gips"),
        "issue": c["non_mma_issue_rate_per_sm_gips"]["value"],
        "smem": mem["smem_bw_per_sm_gbps"]["value"],
        "sync": lambda w: min(sync.items(), key=lambda kv: (abs(kv[0][0] - w), abs(kv[0][1] - 1)))[
            1
        ],
    }


def forms(counts, ldsm_bytes, w, r):
    t_mma = counts["HMMA"] * w * 4096 / interpolate(r["mma"], w)
    t_alu = counts["OTHER"] * w / r["issue"]
    t_sfu = counts["MUFU"] * w / interpolate(r["sfu"], w)
    t_shfl = counts["SHFL"] * w / interpolate(r["shfl"], w)
    t_smem = ldsm_bytes * w / r["smem"]
    issue = (counts["OTHER"] + counts["MUFU"] + counts["SHFL"]) * w / r["issue"]
    issue_all = issue + (counts["HMMA"] + counts["LDSM"]) * w / r["issue"]
    s = r["sync"](w)
    return {
        "F-max": t_mma + max(issue, t_smem, t_sfu, t_shfl) + s,
        "F-sum": t_mma + t_alu + t_sfu + t_shfl + t_smem + s,
        "F-dep": t_mma + max(t_smem, t_alu + t_sfu + t_shfl) + s,
        "F-issue": t_mma + max(issue_all, t_smem, t_sfu, t_shfl) + s,
    }


def mma_per_iteration(config_id, w):
    """m16n8k16 MMAs per warp in one loop iteration, from the config's tile."""
    dims = dict(re.findall(r"([mnk])(\d+)", config_id.split("-")[0]))
    m, n = int(dims["m"]), int(dims["n"])
    if "k" in dims:  # GEMM: one BM x BN x BK step
        return m // 16 * (n // 8) * (int(dims["k"]) // 16) / w
    d = 128  # attention: QK (m x n over D) plus PV (m x D over n)
    return (m // 16 * (n // 8) * (d // 16) + m // 16 * (d // 8) * (n // 16)) / w


def name_of(row):
    if row["family"] == "gemm":
        return f"gemm-{row['config_id']}"
    e, a, r = row["knobs"]
    return f"attention-{row['config_id']}-e{e}a{a}r{r}"


def judge(label, points):
    """points: {name: (measured, {form: pred})}."""
    names = list(next(iter(points.values()))[1])
    err = {f: {k: abs(p[f] / m - 1) for k, (m, p) in points.items()} for f in names}
    for f in names:
        print(f"{label}: {f} median |err| {st.median(err[f].values()) * 100:.1f}%")
    if all(st.median(e.values()) > WRONG for e in err.values()):
        print(f"{label}: every form misses by more than 15%: all wrong")
        return
    winners = []
    for a in names:
        ok = True
        for b in names:
            if a == b:
                continue
            diff = [k for k, (_, p) in points.items() if abs(p[a] / p[b] - 1) > DIFFER]
            ea, eb = [err[a][k] for k in diff], [err[b][k] for k in diff]
            ok &= (
                bool(diff)
                and st.median(ea) < 0.5 * st.median(eb)
                and sum(x < y for x, y in zip(ea, eb)) >= 2 / 3 * len(diff)
            )
        if ok:
            winners.append(a)
    print(f"{label}: winner {winners[0] if winners else 'none'}")


def quantile(v, p):
    s = sorted(v)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def acquisition(runs):
    """The acquisition rules of 2026-11-27: lists with IQR over median above 10% at most 5%; lists with
    max over min above 1.5 at most 15%; points whose process medians span more than 10% at most 5%."""
    lists = [row["kernel_ns_samples"] for rows in runs for row in rows]
    by_point = {}
    for rows in runs:
        for row in rows:
            by_point.setdefault(name_of(row), []).append(st.median(row["kernel_ns_samples"]))
    r1 = sum((quantile(v, 0.75) - quantile(v, 0.25)) / st.median(v) > 0.10 for v in lists) / len(
        lists
    )
    r2 = sum(max(v) / min(v) > 1.5 for v in lists) / len(lists)
    r3 = sum((max(v) - min(v)) / st.median(v) > 0.10 for v in by_point.values()) / len(by_point)
    ok = r1 <= 0.05 and r2 <= 0.15 and r3 <= 0.05
    print(
        f"acquisition: IQR {r1 * 100:.1f}% (<=5)  max/min {r2 * 100:.1f}% (<=15)  cross-process {r3 * 100:.1f}% (<=5): {'passed' if ok else 'failed'}"
    )
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("profile")
    ap.add_argument("sass_dir", type=Path)
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    r = rates(args.profile)
    runs = [json.load(open(p))["rows"] for p in args.runs]
    if not acquisition(runs):
        print("the run is void under the acquisition rules: archive it as invalid and repeat")
        return
    points = {"attention": {}, "gemm": {}}
    base = {}
    for rows in zip(*runs):
        row = rows[0]
        name = name_of(row)
        counts, ldsm = loop_counts((args.sass_dir / f"{name}.sass").read_text())
        w = 8 if "-w8-" in row["config_id"] else 4
        # The compiler may unroll the loop: scale the body to one iteration by its MMA count.
        unroll = counts["HMMA"] / mma_per_iteration(row["config_id"], w)
        if unroll < 1 or unroll != int(unroll):
            raise ValueError(f"{name}: {counts['HMMA']} HMMA is not a whole number of iterations")
        counts = Counter({k: v / unroll for k, v in counts.items()})
        ldsm /= unroll
        waves = math.ceil(math.prod(row["grid"]) / row["sm_count"])
        per_iter = (
            st.median(x["kernel_ns_median"] for x in rows) / waves / row["iterations_per_cta"]
        )
        pred = forms(counts, ldsm, w, r)
        points[row["family"]][name] = (per_iter, pred)
        print(
            f"  {name:42} HMMA {counts['HMMA']:4} LDSM {counts['LDSM']:3} MUFU {counts['MUFU']:3} SHFL {counts['SHFL']:2} "
            f"OTHER {counts['OTHER']:4}  measured {per_iter:7.0f}  "
            + "  ".join(f"{f} {v:7.0f}" for f, v in pred.items())
        )
        if row["family"] == "attention":
            key = row["config_id"]
            if tuple(row["knobs"]) == (0, 0, 0):
                base[key] = (per_iter, pred)
    for name, (m, p) in points["attention"].items():
        key = name.split("-e")[0].removeprefix("attention-")
        if name.endswith("-e0a0r0"):
            continue
        bm, bp = base[key]
        print(
            f"  delta {name:36} measured {m - bm:6.0f}  "
            + "  ".join(f"{f} {p[f] - bp[f]:6.0f}" for f in p)
        )
    judge("attention", points["attention"])
    judge("gemm", points["gemm"])


if __name__ == "__main__":
    main()
