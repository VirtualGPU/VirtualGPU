"""Expected change of GB10 predictions on the v0.2 formal IRs under the candidate barrier charge:
per step fixed_sync + max(n - 1, 0) x m(warps), n the loop barrier op count, m the barrier-step
empty-step slope at all SMs (7.80 ns at 4 warps, 10.92 at 8). Opened labels are not read.
Usage (repo root): python expected_change.py IR_ROOT PROFILE OUT.json"""

import json
import statistics as st
import sys
from pathlib import Path

from virtualgpu import _core

root, profile_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
base_text = profile_path.read_text()
M = {4: 7.80, 8: 10.92}
cache = {}


def edited(warps, n):
    key = (warps, n)
    if key not in cache:
        p = json.loads(base_text)
        m = M[min(M, key=lambda w: abs(w - warps))]
        for e in p["measured"]["pipeline"]["fixed_sync_ns"]:
            e["overhead_ns"]["value"] += max(n - 1, 0) * m
        cache[key] = json.dumps(p)
    return cache[key]


rows = []
for group in ("gb10-gemm", "gb10-attention"):
    for f in sorted((root / group).rglob("*.json")):
        ir = json.loads(f.read_text())
        n = sum(
            o.get("count", 0) for o in ir["body"]["pipeline_loop"]["ops"] if o["kind"] == "barrier"
        )
        w = ir["launch"]["num_warps"]
        r0 = dict(_core.predict(f.read_text(), base_text, l2_state="warm", draws=0, seed=17))
        r1 = dict(_core.predict(f.read_text(), edited(w, n), l2_state="warm", draws=0, seed=17))
        if r0["status"] != "ok" or r1["status"] != "ok":
            continue
        rows.append(
            {
                "group": group,
                "file": str(f.relative_to(root)),
                "barriers": n,
                "warps": w,
                "base_ns": r0["time_ns"],
                "form_ns": r1["time_ns"],
            }
        )
summary = {}
for r in rows:
    summary.setdefault((r["group"], r["barriers"], r["warps"]), []).append(
        r["form_ns"] / r["base_ns"] - 1
    )
table = {
    f"{g} n={n} w={w}": {"configs": len(v), "median": st.median(v), "min": min(v), "max": max(v)}
    for (g, n, w), v in sorted(summary.items())
}
out.write_text(json.dumps({"summary": table, "rows": rows}, indent=1))
for k, v in table.items():
    print(
        f"{k:32} {v['configs']:4}  median {v['median'] * 100:+.2f}%  range {v['min'] * 100:+.2f}% to {v['max'] * 100:+.2f}%"
    )
