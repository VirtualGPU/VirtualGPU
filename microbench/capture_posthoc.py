"""Capture one post-hoc microbenchmark mode behind the GPU lock and the thermal gate.

Wraps `vgpu_bench --mode <mode>` with the quiet-host checks, warmup gate and
telemetry of repeat_mlp.py, and writes the raw output with its receipt.

Usage (inside `flock /tmp/virtualgpu-gpu.lock`):
    python microbench/capture_posthoc.py --binary vgpu_bench_sm121 --mode wave-rescan \
        --revision <commit> --output wave-rescan.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from shutil import which

from repeat_mlp import (
    WARMUP,
    command,
    digest,
    host_snapshot,
    require_quiet,
    telemetry,
    utc,
    warm_device,
    write_json,
)

MODES = (
    "wave-rescan",
    "b-stream",
    "staged-mma",
    "sync-loop-delta",
    "sync-loop-delta-flag",
    "l2-paths",
)


def capture(args):
    smi = which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    before = host_snapshot(smi)
    require_quiet(before)
    query = json.loads(command(str(args.binary), "--mode", "query", "--device", "0"))
    warm, stable = warm_device(
        smi,
        str(args.binary),
        args.warmup_seconds,
        timeout_report=args.output.with_suffix(".thermal-timeout.json"),
    )
    run_args = [str(args.binary), "--mode", args.mode, "--device", "0"]
    run_args += ["--samples", str(args.samples), "--warmup", str(WARMUP)]
    with tempfile.TemporaryFile(mode="w+t") as out, tempfile.TemporaryFile(mode="w+t") as err:
        started = utc()
        process = subprocess.Popen(run_args, stdout=out, stderr=err, text=True)
        snaps = []
        try:
            while process.poll() is None:
                snaps.append(telemetry(smi))
                apps = command(
                    smi, "--query-compute-apps=pid,process_name", "--format=csv,noheader"
                )
                require_quiet({"compute_apps": apps}, own_pid=process.pid)
                time.sleep(0.5)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
        err.seek(0)
        if process.returncode:
            raise RuntimeError(f"Microbenchmark exited with {process.returncode}: {err.read()}")
        out.seek(0)
        raw = json.load(out)
    after = host_snapshot(smi)
    require_quiet(after)
    return {
        "post_hoc": True,
        "source_revision": args.revision,
        "binary_sha256": digest(args.binary),
        "device_query": query,
        "mode": args.mode,
        "command": run_args,
        "environment": {k: v for k, v in os.environ.items() if k.startswith("VGPU_")},
        "host": platform.node(),
        "platform": platform.platform(),
        "started_utc": started,
        "finished_utc": after["utc"],
        "clock_policy": "record_only",
        "before": before,
        "after": after,
        "thermal_warmup": warm,
        "thermal_stability": stable,
        "telemetry": snaps,
        "raw": raw,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=15)
    ap.add_argument("--warmup-seconds", type=float, default=60)
    args = ap.parse_args()
    write_json(args.output, capture(args))


if __name__ == "__main__":
    main()
