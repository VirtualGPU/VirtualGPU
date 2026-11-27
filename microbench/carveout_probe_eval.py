"""Readings of the GB10 carveout probe, as registered in
docs/results/analysis/v0.3-gb10-carveout-probe-registration.md.

    python microbench/carveout_probe_eval.py RUN1.json RUN2.json RUN3.json

Each file is a capture_posthoc record of `carveout-probe` (or its raw output); pass the
first three valid processes.
"""

import json
import statistics as st
import sys
from pathlib import Path

KIB = 1024
PROFILE_ZERO_THRESHOLD = 128378  # GB10 profile, zero-carveout threshold_bytes_per_sm
REFERENCE = 8 * KIB
MATCH = 4 * KIB
APART = 8 * KIB
CONSISTENT = 8 * KIB
MIN_CROSSINGS = 8
FRAME = 512


def raw(path):
    record = json.loads(Path(path).read_text())
    return record.get("raw", record)


def cliff(points):
    """The L1 cliff of one launch shape, as collect_profile.py computes
    threshold_bytes_per_sm: per sample, L1 is the median of the first three points, L2 of
    the last four, the cliff the first crossing of their midpoint; then the median over
    samples."""
    pts = sorted(points, key=lambda p: p["ws_bytes"])
    n = min(len(p["latency_ns_samples"]) for p in pts)
    wss = [float(p["ws_bytes"]) for p in pts]
    knees = []
    for si in range(n):
        lats = [float(p["latency_ns_samples"][si]) for p in pts]
        l1, l2 = st.median(lats[:3]), st.median(lats[-4:])
        mid = 0.5 * (l1 + l2)
        knee = wss[0]
        for i in range(1, len(pts)):
            if lats[i - 1] <= mid <= lats[i] and lats[i] > lats[i - 1]:
                frac = (mid - lats[i - 1]) / (lats[i] - lats[i - 1])
                knee = wss[i - 1] + frac * (wss[i] - wss[i - 1])
                break
        else:
            if lats[-1] <= mid:
                knee = wss[-1]
        knees.append(float(round(knee)))
    return st.median(knees)


def cliffs(run):
    by = {}
    for p in run["l1_cliff"]:
        by.setdefault(p["config"], []).append(p)
    return {name: cliff(points) for name, points in by.items()}


def transitions(run):
    """Per context: the footprint where m(w) crosses 0.5, or the side it stays on."""
    t = {}
    for p in run["spill_transition"]:
        t[(p["context"], p["warps"], p["frame_bytes_per_thread"])] = st.median(p["step_ns_samples"])
    out = []
    for context in sorted({k[0] for k in t}):
        warps = sorted({k[1] for k in t if k[0] == context})
        m = [
            (t[(context, w, FRAME)] - t[(context, w, 128)])
            / (t[(context, w, 4096)] - t[(context, w, 128)])
            for w in warps
        ]
        foot = [FRAME * 32 * w for w in warps]
        crossing = None
        for i in range(1, len(m)):
            if (m[i - 1] - 0.5) * (m[i] - 0.5) <= 0 and m[i] != m[i - 1]:
                crossing = foot[i - 1] + (0.5 - m[i - 1]) / (m[i] - m[i - 1]) * (
                    foot[i] - foot[i - 1]
                )
                break
        side = (
            None
            if crossing is not None
            else ("below" if max(m) < 0.5 else "above" if min(m) >= 0.5 else "mixed")
        )
        out.append({"context": context, "m": m, "footprints": foot, "T": crossing, "side": side})
    return out


def readings(runs):
    per = [cliffs(r) for r in runs]
    c = {name: st.median(p[name] for p in per) for name in per[0]}
    result = {"cliffs_per_process": per, "cliffs": c}
    if abs(c["R0"] - PROFILE_ZERO_THRESHOLD) > REFERENCE:
        result["step1"] = "reference not reproduced"
    elif abs(c["S"] - c["R0"]) <= MATCH and abs(c["S"] - c["R16"]) > APART:
        result["step1"] = "labelled by request"
    elif abs(c["S"] - c["R16"]) <= MATCH or (
        abs(c["S"] - c["R8"]) <= MATCH and abs(c["R8"] - c["R0"]) > APART
    ):
        result["step1"] = "labelled by carveout"
    else:
        result["step1"] = "neither"
    contexts = [x | {"process": i + 1} for i, r in enumerate(runs) for x in transitions(r)]
    crossed = [x["T"] for x in contexts if x["T"] is not None]
    result["contexts"] = contexts
    if len(crossed) < MIN_CROSSINGS:
        result["step2"] = "no transition in the swept range"
        result["T"] = None
    else:
        result["T"] = st.median(crossed)
        result["step2"] = (
            "consistent" if abs(result["T"] - c["S"]) <= CONSISTENT else "inconsistent"
        )
    return result


def main():
    if len(sys.argv) != 4:
        raise SystemExit("pass the first three valid processes")
    print(json.dumps(readings([raw(p) for p in sys.argv[1:]]), indent=1))


if __name__ == "__main__":
    main()
