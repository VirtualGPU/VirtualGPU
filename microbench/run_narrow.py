"""Capture repeated narrow-load sweeps (H5) behind the GB10 thermal gate.

Run under the GPU lock, for example
`flock /tmp/virtualgpu-gpu.lock python microbench/run_narrow.py ...`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from shutil import which

sys.path.insert(0, str(Path(__file__).resolve().parent))

from repeat_mlp import (  # noqa: E402
    command,
    host_snapshot,
    require_quiet,
    telemetry,
    utc,
    warm_device,
    write_json,
)

MODES = {"gmem-narrow": "gmem_load_narrow", "sync-loop-narrow": "sync_loop_narrow"}
SAMPLES = 15
WARMUP = 3


def run_once(smi, binary, mode):
    args = [
        str(binary),
        "--mode",
        mode,
        "--device",
        "0",
        "--samples",
        str(SAMPLES),
        "--warmup",
        str(WARMUP),
    ]
    with tempfile.TemporaryFile(mode="w+t") as out, tempfile.TemporaryFile(mode="w+t") as err:
        started = utc()
        process = subprocess.Popen(args, stdout=out, stderr=err, text=True)
        trace = []
        try:
            while process.poll() is None:
                trace.append(telemetry(smi))
                apps = command(
                    smi, "--query-compute-apps=pid,process_name", "--format=csv,noheader"
                )
                require_quiet({"compute_apps": apps}, own_pid=process.pid)
                time.sleep(0.5)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
        out.seek(0)
        err.seek(0)
        if process.returncode != 0:
            raise RuntimeError(f"{mode} failed: {err.read()}")
        return {
            "started_utc": started,
            "finished_utc": utc(),
            "telemetry": trace,
            "result": json.loads(out.read())[MODES[mode]],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("mode", choices=sorted(MODES))
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-seconds", type=int, default=120)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    smi = which("nvidia-smi")
    before = host_snapshot(smi)
    require_quiet(before)
    query = json.loads(command(str(args.binary), "--mode", "query", "--device", "0"))
    warm, stable = warm_device(
        smi,
        str(args.binary),
        args.warmup_seconds,
        timeout_report=args.output.with_suffix(".thermal-timeout.json"),
    )
    repeats = [run_once(smi, args.binary, args.mode) for _ in range(args.repeats)]
    write_json(
        args.output,
        {
            "mode": args.mode,
            "samples": SAMPLES,
            "warmup": WARMUP,
            "device_query": query,
            "host_before": before,
            "thermal_warmup": warm,
            "thermal_stability": stable,
            "repeats": repeats,
            "host_after": host_snapshot(smi),
        },
    )


if __name__ == "__main__":
    main()
