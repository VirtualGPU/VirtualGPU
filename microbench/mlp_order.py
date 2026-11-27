"""GB10 MLP slow-segment timing: the full MLP sweep in given orders, with wall clock.

Reads an orders manifest (`orders.json`: per order the frozen points in execution
order, each with its key, steps and the SHA-256 of its steps file, and the list of
orders to execute), writes one point-order file per order for
`vgpu_bench --mode mlp-order`, and runs each listed order as its own process.
Waits for a quiet host first; while each process runs, /proc/stat is recorded
every 10 ms and processes every second (host_quiet.run_traced), and a failed host
check voids the capture. Run under the GPU lock.

    python microbench/mlp_order.py BINARY orders.json OUTPUT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from shutil import which

sys.path.insert(0, str(Path(__file__).resolve().parent))

import host_quiet  # noqa: E402
from repeat_mlp import host_snapshot, require_quiet, telemetry, warm_device, write_json  # noqa: E402

WARMUP = 3


def order_lines(order):
    """The point-order file of one order, after checking its steps against the manifest digest."""
    steps = "".join(f"{p['steps']}\n" for p in order["points"])
    if hashlib.sha256(steps.encode()).hexdigest() != order["steps_sha256"]:
        raise ValueError(f"{order['steps_file']}: steps differ from the manifest digest")
    lines = []
    for p in order["points"]:
        issue_mode, bytes_in_flight, ctas, bytes_per_cta = p["key"]
        if bytes_in_flight != ctas * bytes_per_cta:
            raise ValueError(f"inconsistent key {p['key']}")
        lines.append(
            f"{p['field']} {issue_mode} {bytes_in_flight} {ctas} {bytes_per_cta} {p['steps']}\n"
        )
    return "".join(lines)


def check_result(order, result):
    """The process ran the manifest's points in its order."""
    ran = [
        (
            p["field"],
            [p["issue_mode"], p["bytes_in_flight"], p["active_ctas"], p["bytes_per_cta"]],
            p["steps"],
        )
        for p in result["points"]
    ]
    want = [(p["field"], list(p["key"]), p["steps"]) for p in order["points"]]
    if ran != want:
        raise ValueError("executed points differ from the manifest")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--warmup-seconds", type=int, default=120)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text())
    if manifest["warmups_per_point"] != WARMUP:
        raise ValueError(f"the runner uses {WARMUP} warmups per point")
    files = {}
    for name, order in manifest["orders"].items():
        path = args.output.with_name(f"{args.output.stem}-{name}.order")
        path.write_text(order_lines(order))
        files[name] = path
    smi = which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    before = host_snapshot(smi)
    require_quiet(before)
    warm, stable = warm_device(
        smi,
        str(args.binary),
        args.warmup_seconds,
        timeout_report=args.output.with_suffix(".thermal-timeout.json"),
    )
    gate = host_quiet.wait_for_quiet_host([os.getpid()])
    runs = []
    for i, name in enumerate(manifest["execution"]):
        cmd = [
            str(args.binary),
            "--mode",
            "mlp-order",
            "--samples",
            str(manifest["samples_per_point"]),
            "--warmup",
            str(WARMUP),
            "--point-order",
            str(files[name]),
        ]
        run = host_quiet.run_traced(cmd)
        result = json.loads(run.pop("stdout"))
        check_result(manifest["orders"][name], result)
        runs.append({"process": i, "order": name, **run, "gpu": telemetry(smi), "result": result})
    write_json(
        args.output,
        {
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "base_freeze_sha256": manifest["base_freeze_sha256"],
            "cpu_count": os.cpu_count(),
            "host_before": before,
            "thermal_warmup": warm,
            "thermal_stability": stable,
            "host_quiet_gate": gate,
            "runs": runs,
            "host_after": host_snapshot(smi),
        },
    )
    if not all(run["host_check"]["status"] == "passed" for run in runs):
        sys.exit("Host CPU was not quiet, capture void")


if __name__ == "__main__":
    main()
