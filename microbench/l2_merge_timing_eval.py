"""Readings of the l2-merge episode timing experiment, as registered in
docs/results/analysis/v0.3-gb10-first-capture-l2-diagnosis.md (section 5).

    python microbench/l2_merge_timing_eval.py RUN1.json RUN2.json RUN3.json
"""

import json
import statistics as st
import sys
from pathlib import Path

SLOW = 1.15
SEPARABLE = 0.25
AGREE = 0.10
DIFFER = 0.15
MIN_GAPS = 10


def raw(path):
    record = json.loads(Path(path).read_text())
    return record.get("raw", record)


def episodes(block):
    """Onset indices of slow episodes and their lengths in one block."""
    t = block["launch_ns_samples"]
    slow = [x > SLOW * st.median(t) for x in t]
    onsets = [i for i, s in enumerate(slow) if s and (i == 0 or not slow[i - 1])]
    lengths = []
    for i in onsets:
        n = 0
        while i + n < len(slow) and slow[i + n]:
            n += 1
        lengths.append(n)
    return onsets, lengths, sum(slow) / len(slow)


def rel(a, b):
    return abs(a / b - 1)


def readings(runs):
    per = {}
    for run in runs:
        for block in run["blocks"]:
            d = per.setdefault(
                block["steps"], {"gs": [], "gt": [], "per_sample": [], "lengths": [], "slow": []}
            )
            onsets, lengths, share = episodes(block)
            host = block["host_start_ms"]
            d["gs"] += [b - a for a, b in zip(onsets, onsets[1:])]
            d["gt"] += [host[b] - host[a] for a, b in zip(onsets, onsets[1:])]
            d["per_sample"] += [b - a for a, b in zip(host, host[1:])]
            d["lengths"] += lengths
            d["slow"].append(share)
    long_steps, short_steps = max(per), min(per)
    A, B = per[long_steps], per[short_steps]
    out = {
        str(steps): {
            "gaps": len(d["gs"]),
            "G_s": st.median(d["gs"]) if d["gs"] else None,
            "G_t_ms": st.median(d["gt"]) if d["gt"] else None,
            "T_ms": st.median(d["per_sample"]),
            "slow_share": st.mean(d["slow"]),
            "episode_lengths": sorted(d["lengths"]),
        }
        for steps, d in per.items()
    }
    TA, TB = st.median(A["per_sample"]), st.median(B["per_sample"])
    if rel(TA, TB) < SEPARABLE and rel(TB, TA) < SEPARABLE:
        reading = "not separable"
    elif min(len(A["gs"]), len(B["gs"])) < MIN_GAPS:
        reading = "not enough episodes"
    else:
        gs = rel(st.median(A["gs"]), st.median(B["gs"]))
        gt = rel(st.median(A["gt"]), st.median(B["gt"]))
        if gt <= AGREE and gs >= DIFFER:
            reading = "fixed wall-clock period"
        elif gs <= AGREE and gt >= DIFFER:
            reading = "fixed sample count"
        else:
            reading = "neither"
    out["reading"] = reading
    return out


def main():
    if len(sys.argv) != 4:
        raise SystemExit("pass the first three valid processes")
    print(json.dumps(readings([raw(p) for p in sys.argv[1:]]), indent=1))


if __name__ == "__main__":
    main()
