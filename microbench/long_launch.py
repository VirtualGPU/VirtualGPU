"""Pilot, freeze and verify launch schedules for any `vgpu_bench` mode.

A mode run with `--long-read-schedule pilot` sizes each point until its
fastest launch reaches 1.25 ms and prints every trial; `freeze` turns that
output into a schedule (one length per point, in emission order) plus a
record of each point's identity; a capture run with the schedule file is
checked by `verify`: same points in the same order, the frozen lengths, and
every launch at least 1.0 ms. A point is any object in the output that has
`launch_ns_samples`, in document order (see docs/components/long-launch-capture.md).

    python microbench/long_launch.py freeze MODE PILOT.json SCHEDULE.txt FREEZE.json
    python microbench/long_launch.py verify FREEZE.json CAPTURE.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PILOT_MIN_NS = 1_250_000
CAPTURE_MIN_NS = 1_000_000
# Fields that come from timing, not from the point's geometry.
TIMED_FIELDS = {"sizing_trials", "steps", "launch_ns_samples", "transfer_bytes"}


def points(node, path=()):
    """Yield (path, point) for every timed point in document order."""
    if isinstance(node, dict):
        if "launch_ns_samples" in node:
            yield path, node
            return
        for key, value in node.items():
            yield from points(value, path + (key,))
    elif isinstance(node, list):
        for value in node:
            yield from points(value, path)


def identity(path, point):
    """A point's geometry: where it sits and every field that is not a timing result."""
    fields = {
        k: v for k, v in point.items() if k not in TIMED_FIELDS and not k.endswith("_samples")
    }
    return {"field": "/".join(path), **fields}


def freeze(mode, pilot):
    rows = list(points(pilot))
    if not rows:
        raise ValueError(f"{mode}: no timed points in the pilot output")
    frozen = []
    for path, point in rows:
        fastest = min(point["launch_ns_samples"])
        if fastest < PILOT_MIN_NS:
            raise ValueError(f"{mode} {identity(path, point)}: pilot reached only {fastest:.0f} ns")
        frozen.append(
            {"point": identity(path, point), "steps": point["steps"], "pilot_min_ns": fastest}
        )
    schedule = "".join(f"{row['steps']}\n" for row in frozen)
    record = {
        "mode": mode,
        "pilot_min_ns": PILOT_MIN_NS,
        "capture_min_ns": CAPTURE_MIN_NS,
        "schedule_sha256": hashlib.sha256(schedule.encode()).hexdigest(),
        "points": frozen,
    }
    return schedule, record


def verify(record, capture):
    rows = list(points(capture))
    if len(rows) != len(record["points"]):
        raise ValueError(
            f"{record['mode']}: {len(rows)} points captured, {len(record['points'])} frozen"
        )
    shortest = None
    for (path, point), frozen in zip(rows, record["points"], strict=True):
        if identity(path, point) != frozen["point"]:
            raise ValueError(
                f"{record['mode']}: point {identity(path, point)} differs from the freeze"
            )
        if point["steps"] != frozen["steps"]:
            raise ValueError(
                f"{record['mode']} {frozen['point']}: ran {point['steps']} steps, froze {frozen['steps']}"
            )
        fastest = min(point["launch_ns_samples"])
        if fastest < record["capture_min_ns"]:
            raise ValueError(f"{record['mode']} {frozen['point']}: a launch took {fastest:.0f} ns")
        shortest = fastest if shortest is None else min(shortest, fastest)
    return {"mode": record["mode"], "points": len(rows), "shortest_launch_ns": shortest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("mode")
    f.add_argument("pilot", type=Path)
    f.add_argument("schedule", type=Path)
    f.add_argument("record", type=Path)
    v = sub.add_parser("verify")
    v.add_argument("record", type=Path)
    v.add_argument("capture", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        for path in (args.schedule, args.record):
            if path.exists():
                raise FileExistsError(path)
        schedule, record = freeze(args.mode, json.loads(args.pilot.read_text()))
        args.schedule.write_text(schedule)
        args.record.write_text(json.dumps(record, indent=1) + "\n")
    else:
        print(
            json.dumps(
                verify(json.loads(args.record.read_text()), json.loads(args.capture.read_text()))
            )
        )


if __name__ == "__main__":
    main()
