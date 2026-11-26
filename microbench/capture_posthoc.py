"""Capture one post-hoc microbenchmark mode behind the GPU lock and the thermal gate.

Wraps `vgpu_bench --mode <mode>` with the quiet-host checks, warmup gate and
telemetry of repeat_mlp.py, and writes the raw output with its receipt.

It does not take the GPU lock itself: it must run inside an external
`flock /tmp/virtualgpu-gpu.lock`, and `wait_for_quiet_desktop` and
`dispersion_check` are meant to be called by other drivers while they hold it.

Usage:
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

try:
    from repeat_mlp import (
        WARMUP,
        WINDOWS_SMI,
        command,
        digest,
        host_snapshot,
        require_quiet,
        telemetry,
        utc,
        warm_device,
        write_json,
    )
except ImportError:  # imported as microbench.capture_posthoc
    from microbench.repeat_mlp import (
        WARMUP,
        WINDOWS_SMI,
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
    "dram-probe-sweep",
    "sync-loop-delta",
    "sync-loop-delta-flag",
    "l2-paths",
)
POWERSHELL = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
QUIET_PERCENT = 5
QUIET_SECONDS = 60
QUIET_TIMEOUT_SECONDS = 20 * 60
DISPERSION_RATIO = 1.3
DISPERSION_MAX_SHARE = 0.10
# Timing and bandwidth samples only; other *_samples fields (offsets, spreads) vary by design.
DISPERSION_SUFFIXES = ("_ns_samples", "gbps_samples")


def windows_engines():
    """Per-process GPU engine utilization above 0.5% from the Windows performance counters.

    Provenance only: a failure is recorded as an error string, never raised.
    """
    if not POWERSHELL.is_file():
        return None
    script = (
        "(Get-Counter '\\GPU Engine(*)\\Utilization Percentage').CounterSamples | "
        "Where-Object CookedValue -gt 0.5 | ForEach-Object { $_.InstanceName + ' ' + $_.CookedValue }"
    )
    try:
        return command(str(POWERSHELL), "-NoProfile", "-Command", script).splitlines()
    except (OSError, subprocess.SubprocessError) as error:
        return {"error": str(error)}


def percent(text):
    """nvidia-smi utilization field, or None when it reads N/A or anything non-numeric."""
    try:
        return int(text.strip().strip("[]"))
    except ValueError:
        return None


def wait_for_quiet_desktop():
    """On WSL, wait until the Windows GPU, decoder and encoder stay below 5% for 60 s.

    A field that reads N/A does not count; the GPU field itself must be numeric.
    Call it while holding the GPU lock, or a colleague's run counts as desktop use.
    """
    if not WINDOWS_SMI.is_file():
        return {"status": "not_applicable"}
    query = "--query-gpu=utilization.gpu,utilization.decoder,utilization.encoder"
    start = time.monotonic()
    quiet_since, trace = None, []
    while True:
        raw = command(str(WINDOWS_SMI), query, "--format=csv,noheader,nounits").split(",")
        values = [percent(v) for v in raw]
        now = time.monotonic()
        trace.append({"utc": utc(), "gpu_decoder_encoder_percent": values})
        known = [v for v in values if v is not None]
        if values[0] is not None and max(known) < QUIET_PERCENT:
            quiet_since = quiet_since if quiet_since is not None else now
            if now - quiet_since >= QUIET_SECONDS:
                return {"status": "passed", "trace": trace, "engines": windows_engines()}
        else:
            quiet_since = None
        if now - start > QUIET_TIMEOUT_SECONDS:
            raise RuntimeError(
                f"Windows desktop GPU use did not stay below {QUIET_PERCENT}% for {QUIET_SECONDS} s"
            )
        time.sleep(1)


def sample_lists(node, suffixes=DISPERSION_SUFFIXES):
    if isinstance(node, dict):
        for key, value in node.items():
            if key.endswith(suffixes) and isinstance(value, list):
                yield key, value
            else:
                yield from sample_lists(value, suffixes)
    elif isinstance(node, list):
        for item in node:
            yield from sample_lists(item, suffixes)


def valid_samples(values):
    return len(values) >= 2 and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0 for x in values
    )


def dispersion_check(raw, suffixes=DISPERSION_SUFFIXES):
    """Share of timing or bandwidth sample lists whose max over min exceeds 1.3.

    Above 10% the capture is void. So is a capture with no checkable list, or with
    a list that is not at least two positive numbers.
    """
    found = [v for _, v in sample_lists(raw, suffixes)]
    lists = [v for v in found if valid_samples(v)]
    spread = [v for v in lists if max(v) / min(v) > DISPERSION_RATIO]
    share = len(spread) / len(lists) if lists else 1.0
    failed = not lists or len(lists) != len(found) or share > DISPERSION_MAX_SHARE
    return {
        "status": "failed" if failed else "passed",
        "sample_lists": len(lists),
        "invalid_lists": len(found) - len(lists),
        "suffixes": list(suffixes),
        "over_ratio": len(spread),
        "share": share,
        "ratio": DISPERSION_RATIO,
        "max_share": DISPERSION_MAX_SHARE,
    }


def capture(args):
    smi = which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    before = host_snapshot(smi)
    require_quiet(before)
    query = json.loads(command(str(args.binary), "--mode", "query", "--device", "0"))
    quiet = wait_for_quiet_desktop()
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
        "desktop_quiet_gate": quiet,
        "dispersion_check": dispersion_check(raw),
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
    record = capture(args)
    write_json(args.output, record)
    if record["dispersion_check"]["status"] != "passed":
        raise SystemExit(f"Dispersion check failed, capture void: {record['dispersion_check']}")


if __name__ == "__main__":
    main()
