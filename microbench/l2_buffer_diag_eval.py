"""Readings of the GB10 L2 buffer experiment, as registered in
docs/results/analysis/v0.3-gb10-first-capture-l2-diagnosis.md (section 3).

    python microbench/l2_buffer_diag_eval.py RUN1.json RUN2.json RUN3.json

Each file is a capture_posthoc record of `l2-buffer-diag` (or its raw output); pass the first
three valid processes.
"""

import json
import statistics as st
import sys
from pathlib import Path

MIB = 1 << 20
STABLE = 0.05
PLACEMENT_SPAN = 0.20
SESSION_SPAN = 0.05
PROCESS_SPAN = 0.10
COMPRESSION_HIT = 0.67


def raw(path):
    record = json.loads(Path(path).read_text())
    return record.get("raw", record)


def bandwidths(run):
    """(fill, visit, buffer) -> {size in MiB: median bandwidth}."""
    out = {}
    for p in run["steady"]:
        key = (p["fill"], p["visit"], p["buffer"])
        out.setdefault(key, {})[p["working_set_bytes"] // MIB] = st.median(p["gbps_samples"])
    return out


def hit(b):
    return (1 / b[36] - 1 / b[60]) / (1 / b[20] - 1 / b[60])


def span(values):
    return (max(values) - min(values)) / st.median(values)


def readings(runs):
    per = [bandwidths(r) for r in runs]
    buffers = sorted({k[2] for k in per[0]})
    levels, stable, hits = [], 0, {"constant": [], "random": []}
    for bw in per:
        lv = []
        for b in buffers:
            v1, v2 = bw[("constant", 1, b)][36], bw[("constant", 2, b)][36]
            stable += abs(v1 - v2) / ((v1 + v2) / 2) <= STABLE
            lv.append((v1 + v2) / 2)
            for fill in hits:
                hits[fill] += [hit(bw[(fill, v, b)]) for v in (1, 2)]
        levels.append(lv)
    spans = [span(lv) for lv in levels]
    process_medians = [st.median(lv) for lv in levels]
    everything = [x for lv in levels for x in lv]
    result = {
        "levels_36mib_gbps": levels,
        "within_process_span": spans,
        "process_medians_gbps": process_medians,
        "between_process_span": span(process_medians),
        "stable_buffers": stable,
        "hit_36mib": {f: {"median": st.median(h), "max": max(h)} for f, h in hits.items()},
    }
    if all(abs(x / st.median(everything) - 1) <= SESSION_SPAN for x in everything):
        result["reading"] = "not reproduced"
    else:
        placement = sum(s >= PLACEMENT_SPAN for s in spans) >= 2 and stable >= 15
        session = all(s <= SESSION_SPAN for s in spans) and span(process_medians) > PROCESS_SPAN
        result["reading"] = (
            "placement" if placement else "session state" if session else "neither supported"
        )
    result["compression"] = (
        max(hits["random"]) <= COMPRESSION_HIT and st.median(hits["constant"]) > COMPRESSION_HIT
    )
    result["one_pass"] = [
        {
            k: (st.median(v) if isinstance(v, list) else v)
            for k, v in p.items()
            if k in ("fill", "buffer", "pass1_gbps_samples", "pass2_gbps_samples")
        }
        for r in runs
        for p in r["one_pass"]
    ]
    return result


def main():
    if len(sys.argv) != 4:
        raise SystemExit("pass the first three valid processes")
    print(json.dumps(readings([raw(p) for p in sys.argv[1:]]), indent=1))


if __name__ == "__main__":
    main()
