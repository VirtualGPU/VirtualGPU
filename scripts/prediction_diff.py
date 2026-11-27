"""Compare two `virtualgpu predict` outputs config by config.

Usage: python scripts/prediction_diff.py BASE.json CANDIDATE.json
"""

import json
import sys
from pathlib import Path
from statistics import median


def read(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return {(row["case_id"], row["config_id"]): row for row in rows}


def prediction(row):
    timings = row.get("draws", {}) | {"nominal": row.get("nominal_latency_ns")}
    return row["status"], {k: float(v).hex() for k, v in timings.items() if v is not None}


def main(argv):
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    base, candidate = map(read, argv)
    if not base or base.keys() != candidate.keys():
        print("Both files need the same nonempty set of case and config IDs", file=sys.stderr)
        return 2
    deltas = []
    unchanged = 0
    print("| Case | Config | Base status | Candidate status | Nominal change (%) |")
    print("| --- | --- | --- | --- | ---: |")
    for (case, config), old in sorted(base.items()):
        new = candidate[case, config]
        if prediction(old) == prediction(new):
            unchanged += 1
            continue
        delta = "n/a"
        if old["status"] == new["status"] == "ok":
            change = 100 * (new["nominal_latency_ns"] / old["nominal_latency_ns"] - 1)
            deltas.append(change)
            delta = f"{change:+.3f}"
        print(f"| {case} | {config} | {old['status']} | {new['status']} | {delta} |")
    print(f"\nBit-identical predictions: {unchanged}/{len(base)}")
    if deltas:
        print(
            f"Changed ok rows: median {median(deltas):+.3f}%, "
            f"max absolute {max(map(abs, deltas)):.3f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
