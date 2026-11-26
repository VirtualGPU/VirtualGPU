"""Evaluate sync-loop-delta runs as registered in docs/results/analysis/h1-delta-rtx4070-registration.md.

python -m microbench.sync_loop_delta_eval profiles/rtx4070.json run1.json run2.json run3.json
"""

import json
import statistics as st
import sys

prof = json.load(open(sys.argv[1]))["measured"]["memory"]
L2l, Ld = prof["l2_latency_ns"]["value"], prof["dram_latency_ns"]["value"]
runs = [json.load(open(p)) for p in sys.argv[2:]]
F, late = {}, {}
for r in runs:
    n = r.get("grid", r["sm_count"]) - r.get("grid", r["sm_count"]) // r["group"]
    for p in r["sync_loop_delta"]:
        k = (p["region"], p["stage_bytes"], p["delta_ns"])
        F.setdefault(k, []).append(
            (
                st.median(p["follower_latency_cycles_samples"]) / st.median(p["sm_ghz_samples"]),
                st.median(p["measured_offset_ns_samples"]),
            )
        )
        late[k] = max(late.get(k, 0), p["late_steps"] / (p["steps"] * n * r["samples"]))
H = {
    "model": lambda d: Ld - L2l,
    "step": lambda d: (Ld - L2l) if d < Ld else 0.0,
    "remain": lambda d: max(0.0, Ld - L2l - d),
}
print(f"L2l {L2l} Ld {Ld} Ld-L2l {Ld - L2l:.1f}")
err = {n: [] for n in H}
for sb in sorted({k[1] for k in F}):
    w = st.median(x for x, _ in F[("warm", sb, 0)])
    if late[("warm", sb, 0)] > 0.01:
        print(f"{sb // 1024}K excluded: warm baseline late {late[('warm', sb, 0)] * 100:.1f}%")
        continue
    e_sb = {n: [] for n in H}
    for d in sorted(k[2] for k in F if k[0] == "cold" and k[1] == sb):
        k = ("cold", sb, d)
        if late[k] > 0.01:
            print(f"  {sb // 1024}K delta {d} excluded: late {late[k] * 100:.2f}%")
            continue
        ex = st.median(x for x, _ in F[k]) - w
        off = st.median(o for _, o in F[k])
        row = {n: H[n](off) for n in H}
        for n in H:
            e_sb[n].append(abs(row[n] - ex))
            err[n].append(abs(row[n] - ex))
        print(
            f"  {sb // 1024}K delta {d:5} offset {off:6.0f} excess {ex:6.0f} | "
            + " ".join(f"{n} {row[n]:5.0f}" for n in H)
        )
    print(
        f"{sb // 1024}K median |err| ns: "
        + ", ".join(f"{n} {st.median(v):.0f}" for n, v in e_sb.items() if v)
    )
names = list(H)
if err["model"] and all(st.median(v) > 50 for v in err.values()):
    print("every form misses by more than 50 ns: all wrong")
if err["model"]:
    print(
        "all judged points",
        len(err["model"]),
        "median |err| ns:",
        ", ".join(f"{n} {st.median(err[n]):.0f}" for n in names),
    )
    for a in names:
        others = [b for b in names if b != a]
        wins = all(
            st.median(err[a]) < 0.5 * st.median(err[b])
            and sum(x < y for x, y in zip(err[a], err[b])) >= 2 / 3 * len(err[a])
            for b in others
        )
        print(
            f"  {a} wins: {wins}  ("
            + ", ".join(
                f"better than {b} on {sum(x < y for x, y in zip(err[a], err[b]))}/{len(err[a])}"
                for b in others
            )
            + ")"
        )
