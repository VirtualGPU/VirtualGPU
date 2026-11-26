"""Evaluate the attention issue time stamps as designed for H1 (post-hoc, development shape).

python -m microbench.attention_stamps_eval docs/results/analysis/attention-stamps-gb10-posthoc.json
"""

import json
import statistics as st
import sys

import numpy as np

prof = json.load(open("profiles/gb10.json"))["measured"]["memory"]
L2l, Ld = prof["l2_latency_ns"]["value"], prof["dram_latency_ns"]["value"]
P1 = prof["l2_bw_by_active_sms"][0]["bw_gbps"]["value"]
X = Ld - L2l
runs = [{"rows": rows} for rows in json.load(open(sys.argv[1]))["processes"]]


def med(cid, var):
    return st.median(
        next(
            r["kernel_ns_median"]
            for r in run["rows"]
            if r["config_id"] == cid and r["variant"] == var
        )
        for run in runs
    )


for cid in ("m128n32-w8-s2", "m128n64-w8-s2", "m128n32-w8-s3", "m128n64-w8-s3"):
    bn = 32 if "n32" in cid else 64
    iters = 2048 // bn
    own = 2 * bn * 128 * 2 / P1
    paid = (med(cid, "unchanged") - med(cid, "noload")) / 11 / iters
    overhead = med(cid, "timestamp") / med(cid, "unchanged") - 1
    preds, stats = [], {"w0": [], "w1": []}
    lead_share = []
    for run in runs:
        row = next(r for r in run["rows"] if r["config_id"] == cid and r["variant"] == "timestamp")
        g0, g1 = row["grid"][0], row["grid"][1]
        s = np.array(row["stamps_ns"], dtype=np.int64).reshape(
            g1, g0, iters
        )  # [head_batch, query_block, iter]
        t = s[:, :, :-1]  # stamp j-1 = issue of tile j, j = 1..iters-1
        d = t - t.min(axis=1, keepdims=True)
        lead = d == 0
        extra = np.where(lead, X, np.maximum(0.0, X - d))
        preds.append(L2l + own + extra.mean())
        for hb in range(g1):
            stats["w0" if hb < 3 else "w1"].append(d[hb][~lead[hb]])
        first = np.argmin(t, axis=1)  # which query block leads each tile
        lead_share.append(np.bincount(first.ravel(), minlength=g0).max() / first.size)
    P = st.median(preds)
    print(
        f"{cid}: timestamp overhead {overhead * 100:+.1f}%  measured paid/iter {paid:.0f} ns  predicted {P:.0f} ns  ratio {P / paid:.3f}  ({'within' if abs(P / paid - 1) <= 0.15 else 'outside'} 15%)"
    )
    for w, arrs in stats.items():
        a = np.concatenate(arrs)
        print(
            f"   {'wave 0' if w == 'w0' else 'waves 1-10'}: follower offset ns median {np.median(a):.0f} p10 {np.percentile(a, 10):.0f} p90 {np.percentile(a, 90):.0f} max {a.max():.0f}; share with d < X {np.mean(a < X) * 100:.0f}%"
        )
    print(f"   most frequent leader leads {st.median(lead_share) * 100:.0f}% of tiles")
