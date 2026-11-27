"""Capture one post-hoc microbenchmark mode behind the GPU lock and the thermal gate.

Wraps `vgpu_bench --mode <mode>` with the quiet-host checks, warmup gate and
telemetry of repeat_mlp.py, and writes the raw output with its receipt.

It does not take the GPU lock itself: it must run inside an external
`flock /tmp/virtualgpu-gpu.lock`, and `wait_for_quiet_desktop` and
`dispersion_check` are meant to be called by other drivers while they hold it.

Usage:
    python microbench/capture_posthoc.py check RUN.json [RUN.json ...]
    python microbench/capture_posthoc.py --binary vgpu_bench_sm121 --mode wave-rescan \
        --revision <commit> --output wave-rescan.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from shutil import which

try:
    import host_quiet
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
    from microbench import host_quiet
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
    "wave-rescan-long",
    "wave-rescan-v3",
    "b-stream",
    "staged-mma",
    "dram-probe-sweep",
    "sync-loop-delta",
    "sync-loop-delta-flag",
    "l2-paths",
    "mma-ilp",
)
POWERSHELL = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
QUIET_PERCENT = 5
QUIET_SECONDS = 60
QUIET_TIMEOUT_SECONDS = 20 * 60
# Capture rules (adopted 2026-11-27, after an independent audit; fixed in advance).
IQR_RATIO = 0.10  # 1) IQR over median above this ...
IQR_MAX_SHARE = 0.05  # ... in at most this share of lists
SPIKE_RATIO = 1.5  # 2) max over min above this ...
SPIKE_MAX_SHARE = 0.15  # ... in at most this share of lists
REPRO_RANGE = 0.10  # 3) across runs, range of a point's medians over their median above this ...
REPRO_MAX_SHARE = 0.05  # ... in at most this share of points
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


def sample_lists(node, suffixes=DISPERSION_SUFFIXES, path=()):
    """Yield (path, samples) for every timing or bandwidth sample list in a raw capture."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key.endswith(suffixes) and isinstance(value, list):
                yield path + (key,), value
            else:
                yield from sample_lists(value, suffixes, path + (key,))
    elif isinstance(node, list):
        for index, item in enumerate(node):
            yield from sample_lists(item, suffixes, path + (index,))


def valid_samples(values):
    return len(values) >= 2 and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0 for x in values
    )


def iqr_ratio(values):
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return (q3 - q1) / statistics.median(values)


def dispersion_check(raw, suffixes=DISPERSION_SUFFIXES):
    """Rules 1 and 2 on one run.

    1) at most 5% of lists may have IQR over median above 10%;
    2) at most 15% of lists may have max over min above 1.5.
    A run with no checkable list, or with a list that is not at least two positive
    numbers, fails as well.
    """
    found = [v for _, v in sample_lists(raw, suffixes)]
    lists = [v for v in found if valid_samples(v)]
    n = max(len(lists), 1)
    wide = sum(iqr_ratio(v) > IQR_RATIO for v in lists)
    spiky = sum(max(v) / min(v) > SPIKE_RATIO for v in lists)
    rules = {
        "iqr": {"share": wide / n, "limit": IQR_MAX_SHARE, "ratio": IQR_RATIO},
        "spike": {"share": spiky / n, "limit": SPIKE_MAX_SHARE, "ratio": SPIKE_RATIO},
    }
    failed = (
        not lists
        or len(lists) != len(found)
        or any(r["share"] > r["limit"] for r in rules.values())
    )
    return {
        "status": "failed" if failed else "passed",
        "sample_lists": len(lists),
        "invalid_lists": len(found) - len(lists),
        "suffixes": list(suffixes),
        "rules": rules,
    }


def reproducibility_check(raws, suffixes=DISPERSION_SUFFIXES):
    """Rule 3 across runs of the same geometry: at most 5% of points may have a range of
    their per-run medians above 10% of the median of those medians."""
    if len(raws) < 2:
        return {"status": "not_applicable", "runs": len(raws)}
    per_run = [dict(sample_lists(raw, suffixes)) for raw in raws]
    keys = set(per_run[0])
    if any(set(run) != keys for run in per_run) or not keys:
        return {"status": "failed", "runs": len(raws), "reason": "runs differ in geometry"}
    spread = 0
    for key in keys:
        meds = [statistics.median(run[key]) for run in per_run]
        spread += (max(meds) - min(meds)) / statistics.median(meds) > REPRO_RANGE
    share = spread / len(keys)
    return {
        "status": "failed" if share > REPRO_MAX_SHARE else "passed",
        "runs": len(raws),
        "points": len(keys),
        "share": share,
        "limit": REPRO_MAX_SHARE,
        "range": REPRO_RANGE,
    }


def capture(args):
    smi = which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    before = host_snapshot(smi)
    require_quiet(before)
    query = json.loads(command(str(args.binary), "--mode", "query", "--device", "0"))
    quiet = wait_for_quiet_desktop()
    # GPUs that share memory with the CPU (GB10): CPU work slows DRAM-bound measurements.
    gate_host = query.get("integrated", False) or args.host_quiet
    warm, stable = warm_device(
        smi,
        str(args.binary),
        args.warmup_seconds,
        timeout_report=args.output.with_suffix(".thermal-timeout.json"),
    )
    host_gate = host_quiet.wait_for_quiet_host([os.getpid()]) if gate_host else None
    run_args = [str(args.binary), "--mode", args.mode, "--device", "0"]
    run_args += ["--samples", str(args.samples), "--warmup", str(WARMUP)]
    if args.l2_gbps:
        run_args += ["--l2-gbps", str(args.l2_gbps)]
    with tempfile.TemporaryFile(mode="w+t") as out, tempfile.TemporaryFile(mode="w+t") as err:
        started = utc()
        process = subprocess.Popen(run_args, stdout=out, stderr=err, text=True)
        snaps, host_samples, next_host = [], [], 0.0
        try:
            while process.poll() is None:
                if gate_host and time.monotonic() >= next_host:
                    host_samples.append(host_quiet.sample())
                    next_host = time.monotonic() + 1.0
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
        "host_quiet_gate": host_gate,
        "host_quiet_check": (
            host_quiet.check(host_samples + [host_quiet.sample()], [os.getpid(), process.pid])
            if gate_host
            else None
        ),
        "thermal_warmup": warm,
        "thermal_stability": stable,
        "telemetry": snaps,
        "raw": raw,
    }


def check_files(paths):
    """Rules 1 to 3 on archived captures (the raw output under "raw", or the file itself)."""
    raws = []
    for path in paths:
        record = json.loads(Path(path).read_text())
        raws.append(record.get("raw", record))
    return {
        "runs": {str(p): dispersion_check(raw) for p, raw in zip(paths, raws)},
        "reproducibility": reproducibility_check(raws),
    }


def main_check(argv):
    """`capture_posthoc.py check RUN.json [RUN.json ...]`: exit 1 if any rule fails."""
    result = check_files(argv)
    print(json.dumps(result, indent=1))
    statuses = [r["status"] for r in result["runs"].values()]
    statuses.append(result["reproducibility"]["status"])
    raise SystemExit(1 if "failed" in statuses else 0)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        main_check(sys.argv[2:])
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=15)
    ap.add_argument("--warmup-seconds", type=float, default=60)
    ap.add_argument("--l2-gbps", type=float, help="profile l2_bw_read_gbps, for wave-rescan-long")
    ap.add_argument(
        "--host-quiet",
        action="store_true",
        help="Gate on a quiet host CPU even if the GPU is not integrated",
    )
    args = ap.parse_args()
    record = capture(args)
    write_json(args.output, record)
    if record["dispersion_check"]["status"] != "passed":
        raise SystemExit(f"Dispersion check failed, capture void: {record['dispersion_check']}")
    host = record["host_quiet_check"]
    if host is not None and host["status"] != "passed":
        raise SystemExit(f"Host CPU was not quiet, capture void: {host}")


if __name__ == "__main__":
    main()
