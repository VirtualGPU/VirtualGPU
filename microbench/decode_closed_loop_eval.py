"""Readings of the GB10 strided closed-loop test, as registered in
docs/results/analysis/v0.3-gb10-decode-compression.md.

    python microbench/decode_closed_loop_eval.py RUN1.json RUN2.json RUN3.json

Each file is a capture_posthoc record of `decode-closed-loop` (or its raw output); pass the
first three valid processes.
"""

import json
import statistics as st
import sys
from pathlib import Path

CONTROL = 0.02
SUPPORT = 0.05
SUPPORT_PAIRS = 3
REJECT = 0.02


def raw(path):
    record = json.loads(Path(path).read_text())
    return record.get("raw", record)


def key(p):
    return (p["segment_bytes"], p["active_ctas"], p["stage_bytes"], p["tokens"])


def bandwidths(runs):
    """Per point, the median over processes of the point's median bandwidth."""
    per = {}
    for run in runs:
        for p in run["decode_closed_loop"]:
            per.setdefault(key(p), []).append(st.median(p["gbps_samples"]))
    return {k: st.median(v) for k, v in per.items()}


def gaps(bw):
    """g = bw(4 buffers) / bw(1 buffer) - 1 per (segment, grid, stage)."""
    pairs = sorted({k[:3] for k in bw})
    return {k: bw[(*k, 4)] / bw[(*k, 1)] - 1 for k in pairs}


def readings(runs):
    bw = bandwidths(runs)
    g = gaps(bw)
    control = [v for k, v in g.items() if k[0] == 0]
    strided = {s: [v for k, v in g.items() if k[0] == s] for s in sorted({k[0] for k in g} - {0})}
    if any(abs(v) > CONTROL for v in control):
        reading = "control not reproduced"
    elif any(sum(v >= SUPPORT for v in vs) >= SUPPORT_PAIRS for vs in strided.values()):
        reading = "H-gran supported"
    elif all(abs(v) < REJECT for vs in strided.values() for v in vs):
        reading = "H-gran rejected"
    else:
        reading = "inconclusive"
    return {
        "reading": reading,
        "gaps": [
            {"segment_bytes": k[0], "active_ctas": k[1], "stage_bytes": k[2], "g": v}
            for k, v in g.items()
        ],
        "bandwidth_gbps": [
            {
                "segment_bytes": k[0],
                "active_ctas": k[1],
                "stage_bytes": k[2],
                "tokens": k[3],
                "bw": v,
            }
            for k, v in sorted(bw.items())
        ],
    }


def main():
    if len(sys.argv) != 4:
        raise SystemExit("pass the first three valid processes")
    print(json.dumps(readings([raw(p) for p in sys.argv[1:]]), indent=1))


if __name__ == "__main__":
    main()
