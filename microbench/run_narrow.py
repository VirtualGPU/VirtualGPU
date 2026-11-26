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
    WINDOWS_SMI,
    command,
    host_snapshot,
    require_quiet,
    telemetry,
    utc,
    warm_device,
    write_json,
)

# GB10 shares its memory with the CPU: a large CPU process competes for it.
MAX_FOREIGN_RSS_KIB = 8 << 20
MIN_AVAILABLE_KIB = 16 << 20

# RTX 4070 under WSL: the Windows desktop shares the GPU. Wait for GPU
# utilization under 5% for 60 s before a capture, and void a capture whose
# sample lists are dispersed (largest over smallest above 1.3) in more than
# 10% of cases.
QUIET_UTILIZATION = 5
QUIET_SECONDS = 60
QUIET_TIMEOUT_SECONDS = 30 * 60
DISPERSION_RATIO = 1.3
DISPERSION_MAX_SHARE = 0.10

MODES = {
    "gmem-narrow": "gmem_load_narrow",
    "sync-loop-narrow": "sync_loop_narrow",
    "gmem-narrow-validate": "gmem_load_narrow",
    "gmem-narrow-calibrate": "gmem_load_narrow",
    "sync-loop-narrow-validate": "sync_loop_narrow",
    "stage-narrow": "stage_narrow",
    "stage-narrow-batched": "stage_narrow",
    "stage-narrow-validate": "stage_narrow",
    "stage-narrow-k1": "stage_narrow",
    "stage-narrow-k1-calibrate": "stage_narrow",
}
SAMPLES = 15
WARMUP = 3


def memory_snapshot():
    meminfo = dict(
        line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line
    )
    available = int(meminfo["MemAvailable"].split()[0])
    top = command("ps", "-eo", "pid,rss,comm", "--sort=-rss").splitlines()[1:6]
    return {"utc": utc(), "mem_available_kib": available, "top_rss": top}


def require_memory(snapshot):
    if snapshot["mem_available_kib"] < MIN_AVAILABLE_KIB:
        raise RuntimeError(f"Only {snapshot['mem_available_kib']} KiB of memory available")
    for line in snapshot["top_rss"]:
        pid, rss, name = line.split(None, 2)
        if int(rss) > MAX_FOREIGN_RSS_KIB:
            raise RuntimeError(f"Large CPU process holds unified memory: {pid} {name} {rss} KiB")


def wait_for_quiet_desktop():
    if not WINDOWS_SMI.is_file():
        return None
    start = time.monotonic()
    trace, quiet_since = [], None
    while True:
        util = int(
            command(
                str(WINDOWS_SMI), "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"
            ).splitlines()[0]
        )
        now = time.monotonic() - start
        trace.append({"utc": utc(), "utilization_gpu": util})
        quiet_since = (
            (quiet_since if quiet_since is not None else now) if util < QUIET_UTILIZATION else None
        )
        if quiet_since is not None and now - quiet_since >= QUIET_SECONDS:
            return trace
        if now > QUIET_TIMEOUT_SECONDS:
            raise RuntimeError("The Windows desktop did not stay quiet for 60 s within 30 minutes")
        time.sleep(1)


def dispersion(record):
    lists = [
        row["elapsed_ns"]
        for run in record["repeats"] + record["calibration_runs"]
        for row in run["result"]
    ]
    over = sum(max(v) / min(v) > DISPERSION_RATIO for v in lists)
    share = over / len(lists) if lists else 0.0
    return {
        "sample_lists": len(lists),
        "over_ratio": over,
        "share": share,
        "ratio": DISPERSION_RATIO,
        "max_share": DISPERSION_MAX_SHARE,
        "status": "failed" if share > DISPERSION_MAX_SHARE else "passed",
    }


def run_once(smi, binary, mode):
    memory = memory_snapshot()
    require_memory(memory)
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
            "memory_before": memory,
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
    parser.add_argument(
        "--calibration-mode",
        choices=sorted(MODES),
        help="Also run this mode as its own process before every repeat",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    smi = which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    before = host_snapshot(smi)
    require_quiet(before)
    quiet = wait_for_quiet_desktop()
    query = json.loads(command(str(args.binary), "--mode", "query", "--device", "0"))
    warm, stable = warm_device(
        smi,
        str(args.binary),
        args.warmup_seconds,
        timeout_report=args.output.with_suffix(".thermal-timeout.json"),
    )
    repeats, calibration = [], []
    for _ in range(args.repeats):
        if args.calibration_mode:
            calibration.append(run_once(smi, args.binary, args.calibration_mode))
        repeats.append(run_once(smi, args.binary, args.mode))
    record = {"repeats": repeats, "calibration_runs": calibration}
    check = dispersion(record)
    write_json(
        args.output,
        {
            "desktop_quiet_trace": quiet,
            "dispersion_check": check,
            "mode": args.mode,
            "samples": SAMPLES,
            "warmup": WARMUP,
            "device_query": query,
            "host_before": before,
            "thermal_warmup": warm,
            "thermal_stability": stable,
            "repeats": repeats,
            "calibration_mode": args.calibration_mode,
            "calibration_runs": calibration,
            "host_after": host_snapshot(smi),
        },
    )

    # Under WSL the desktop can perturb the GPU unseen: a dispersed capture is void.
    if quiet is not None and check["status"] != "passed":
        raise SystemExit(f"Dispersion check failed, capture void: {check}")


if __name__ == "__main__":
    main()
