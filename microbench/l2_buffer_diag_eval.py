"""Readings of the GB10 L2 buffer experiment, as re-registered in
docs/results/analysis/v0.3-gb10-first-capture-l2-diagnosis.md (section 7).

    python microbench/l2_buffer_diag_eval.py gate RUN.json      # exit 1 if the process is void
    python microbench/l2_buffer_diag_eval.py RUN1.json ... RUN5.json

Each file is a capture_posthoc record of `l2-buffer-diag` (or its raw output); pass the first
five valid processes.
"""

import json
import statistics as st
import sys
from pathlib import Path

MIB = 1 << 20
STABLE = 0.05
PLACEMENT_SPAN = 0.20
REPRODUCED_SPAN = 0.20
UNSTEADY_IQR = 0.10
PROCESSES = 5
SESSION_SPAN = 0.05
PROCESS_SPAN = 0.10
COMPRESSION_HIT = 0.67


def gate(record):
    """Rules 1 and 2 over the steady arm's 20 and 60 MiB lists only, and the host gate."""
    try:
        from capture_posthoc import dispersion_check
    except ImportError:  # imported as microbench.l2_buffer_diag_eval
        from microbench.capture_posthoc import dispersion_check

    run = record.get("raw", record)
    references = [p for p in run["steady"] if p["working_set_bytes"] // MIB in (20, 60)]
    check = dispersion_check({"references": references})
    host = record.get("host_quiet_check") or {}
    return check["status"] == "passed" and host.get("status") == "passed", check


def iqr(values):
    q1, _, q3 = st.quantiles(values, n=4, method="inclusive")
    return (q3 - q1) / st.median(values)


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
    lists = [
        {
            (p["fill"], p["visit"], p["buffer"]): p["gbps_samples"]
            for p in r["steady"]
            if p["working_set_bytes"] // MIB == 36
        }
        for r in runs
    ]
    control = [
        st.median(
            next(p for p in r["control"] if p["working_set_bytes"] // MIB == 36)["gbps_samples"]
        )
        for r in runs
    ]
    control_iqr = [
        iqr(next(p for p in r["control"] if p["working_set_bytes"] // MIB == 36)["gbps_samples"])
        for r in runs
    ]
    buffers = sorted({k[2] for k in per[0]})
    levels, kept, hits = [], 0, {"constant": [], "random": []}
    unsteady = {"constant": 0, "random": 0}
    for bw, li in zip(per, lists):
        lv = []
        for b in buffers:
            for fill in unsteady:
                unsteady[fill] += sum(iqr(li[(fill, v, b)]) > UNSTEADY_IQR for v in (1, 2))
            v1, v2 = bw[("constant", 1, b)][36], bw[("constant", 2, b)][36]
            steady = all(iqr(li[("constant", v, b)]) <= UNSTEADY_IQR for v in (1, 2))
            kept += steady and abs(v1 - v2) / ((v1 + v2) / 2) <= STABLE
            lv.append((v1 + v2) / 2)
            for fill in hits:
                hits[fill] += [hit(bw[(fill, v, b)]) for v in (1, 2)]
        levels.append(lv)
    spans = [span(lv) for lv in levels]
    process_medians = [st.median(lv) for lv in levels]
    visits = len(runs) * len(buffers) * 2
    result = {
        "control_36mib_gbps": control,
        "control_span": span(control),
        "control_iqr_36mib": control_iqr,
        "levels_36mib_gbps": levels,
        "within_process_span": spans,
        "process_medians_gbps": process_medians,
        "between_process_span": span(process_medians),
        "buffers_keeping_level": kept,
        "unsteady_share": {f: n / visits for f, n in unsteady.items()},
        "hit_36mib": {f: {"median": st.median(h), "max": max(h)} for f, h in hits.items()},
    }
    if span(control) < REPRODUCED_SPAN:
        result["reading"] = "not reproduced"
    else:
        placement = sum(s >= PLACEMENT_SPAN for s in spans) >= 3 and kept >= 25
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
    if len(sys.argv) == 3 and sys.argv[1] == "gate":
        ok, check = gate(json.loads(Path(sys.argv[2]).read_text()))
        print(json.dumps(check))
        raise SystemExit(0 if ok else 1)
    if len(sys.argv) != PROCESSES + 1:
        raise SystemExit(f"pass the first {PROCESSES} valid processes")
    print(json.dumps(readings([raw(p) for p in sys.argv[1:]]), indent=1))


if __name__ == "__main__":
    main()
