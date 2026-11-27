"""Capture and reduce independent MLP sweeps for the reference profiles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    import host_quiet
except ImportError:  # imported as microbench.repeat_mlp
    from microbench import host_quiet

CONTIGUOUS = ("dram_bw_by_bytes_in_flight", "l2_bw_by_bytes_in_flight")
STRIDED = "dram_bw_by_bytes_in_flight_strided"
MLP_MODES = {"mlp": CONTIGUOUS, "strided-mlp": (STRIDED,)}
ACTIVE_MODES = {"dram-bw": ("dram_bw_by_active_sms",), "l2": ("l2_bw_by_active_sms",)}
MODES = {**MLP_MODES, **ACTIVE_MODES}
ACTIVE_FIELDS = {fields[0] for fields in ACTIVE_MODES.values()}
REPEATS = 5
SAMPLES = 15
WARMUP = 3
TELEMETRY = (
    "uuid",
    "name",
    "driver_version",
    "temperature.gpu",
    "pstate",
    "power.draw",
    "power.limit",
    "utilization.gpu",
    "clocks.current.sm",
    "clocks.current.memory",
    "clocks_event_reasons.hw_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.sw_thermal_slowdown",
    "clocks_event_reasons.hw_power_brake_slowdown",
    "clocks_event_reasons.sw_power_cap",
)
WINDOW_SECONDS = 30
THERMAL_TIMEOUT_SECONDS = 20 * 60
THERMAL_EVENT_FIELDS = TELEMETRY[-5:]
WINDOWS_SMI = Path("/mnt/c/Windows/System32/nvidia-smi.exe")


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path, data):
    raw = (json.dumps(data, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as output:
        output.write(raw)


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def telemetry(smi):
    raw = command(
        smi, f"--query-gpu={','.join(TELEMETRY)}", "--format=csv,noheader,nounits", "--id=0"
    )
    (row,) = csv.reader(raw.splitlines(), skipinitialspace=True)
    if len(row) != len(TELEMETRY):
        raise ValueError("Unexpected nvidia-smi telemetry format")
    return {"utc": utc(), **dict(zip(TELEMETRY, row))}


def host_snapshot(smi):
    output = {
        "utc": utc(),
        "uptime": command("uptime"),
        "nvidia_smi": command(smi),
        "compute_apps": command(
            smi, "--query-compute-apps=pid,process_name", "--format=csv,noheader"
        ),
    }
    if WINDOWS_SMI.is_file():
        output["windows_nvidia_smi"] = command(WINDOWS_SMI)
    return output


def require_quiet(snapshot, own_pid=None):
    foreign = [
        line
        for line in snapshot["compute_apps"].splitlines()
        if line.partition(",")[0].strip() != str(own_pid)
    ]
    if foreign:
        raise RuntimeError(f"Other compute app activity: {foreign}")
    if "gameviewer.exe" in snapshot.get("windows_nvidia_smi", "").lower():
        raise RuntimeError(
            "Windows GameViewer is using the GPU. Wait for the operator to clear it."
        )


def point_key(field, row):
    if field in ACTIVE_FIELDS:
        return (row["active_sms"] if "active_sms" in row else row["target_sms"],)
    layout = ("segment_bytes", "stride_bytes") if field == STRIDED else ("issue_mode",)
    return tuple(row[key] for key in (*layout, "bytes_in_flight", "active_ctas", "bytes_per_cta"))


def point_maps(profile, mode, raw, sample_count=SAMPLES):
    maps = {}
    for field in MODES[mode]:
        expected = [point_key(field, row) for row in profile["measured"]["memory"][field]]
        records = raw["read_curve"] if field in ACTIVE_FIELDS else raw[field]
        observed = {point_key(field, row): row["gbps_samples"] for row in records}
        if len(observed) != len(records) or set(observed) != set(expected):
            raise ValueError(f"{field}: captured geometry differs from base profile")
        for key, samples in observed.items():
            if (
                not isinstance(samples, list)
                or len(samples) != sample_count
                or not all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in samples)
            ):
                raise ValueError(f"{field} {key}: expected {sample_count} finite positive samples")
        maps[field] = observed
    return maps


def thermal_windows(rows, end_seconds):
    start, midpoint = end_seconds - 2 * WINDOW_SECONDS, end_seconds - WINDOW_SECONDS
    return (
        [row for row in rows if start <= row["elapsed_seconds"] < midpoint],
        [row for row in rows if midpoint <= row["elapsed_seconds"] < end_seconds],
    )


def thermal_stability(windows):
    if not all(windows):
        return None
    if any(
        not row["temperature.gpu"].isdecimal()
        or not row["clocks.current.sm"].isdecimal()
        or int(row["clocks.current.sm"]) <= 0
        or any(row.get(key, "").lower() != "not active" for key in THERMAL_EVENT_FIELDS)
        for window in windows
        for row in window
    ):
        return None
    means = [statistics.mean(int(row["temperature.gpu"]) for row in window) for window in windows]
    clocks = [
        statistics.median(int(row["clocks.current.sm"]) for row in window) for window in windows
    ]
    if abs(means[1] - means[0]) > 0.5 or 100 * abs(clocks[1] - clocks[0]) > clocks[0]:
        return None
    return {
        "temperature_mean_c": means,
        "sm_clock_median_mhz": clocks,
        "samples_per_window": list(map(len, windows)),
    }


def warm_device(smi, binary, seconds, *, timeout_report, long_read=False):
    start = time.monotonic()
    minimum = max(seconds, 2 * WINDOW_SECONDS)
    if minimum > THERMAL_TIMEOUT_SECONDS:
        raise ValueError("Requested warmup exceeds the thermal gate timeout")
    trace = []
    while True:
        elapsed = time.monotonic() - start
        if elapsed >= minimum:
            gate_end = min(elapsed, THERMAL_TIMEOUT_SECONDS)
            result = thermal_stability(thermal_windows(trace, gate_end))
            if result is not None:
                return trace, {"elapsed_seconds": gate_end, **result}
            if elapsed >= THERMAL_TIMEOUT_SECONDS:
                write_json(timeout_report, {"utc": utc(), "warmup_telemetry": trace})
                raise RuntimeError("Thermal/clock gate did not pass within twenty minutes")
        require_quiet(host_snapshot(smi))
        if long_read:
            command(binary, "--mode", "read-soak")
        else:
            command(binary, "--mode", "mlp", "--samples", "1", "--warmup", str(WARMUP))
        snap = telemetry(smi)
        trace.append({**snap, "elapsed_seconds": time.monotonic() - start})


def warmup_evidence(record):
    passing = record.get("thermal_stability")
    if passing is not None:
        return {"status": "passed", **passing}
    rows = record.get("thermal_warmup", [])
    if not rows:
        return {"status": "not_applied"}
    start = datetime.fromisoformat(rows[0]["utc"])
    trace = [
        {**row, "elapsed_seconds": (datetime.fromisoformat(row["utc"]) - start).total_seconds()}
        for row in rows
    ]
    windows = thermal_windows(trace, 2 * WINDOW_SECONDS)
    means = [
        statistics.mean(int(row["temperature.gpu"]) for row in window)
        for window in windows
        if window
    ]
    return {
        "status": "fixed_warmup_only",
        "sample_span_seconds": trace[-1]["elapsed_seconds"],
        "adjacent_30s_temperature_mean_c": means,
        "temperature_mean_delta_c": abs(means[1] - means[0]) if len(means) == 2 else None,
    }


def points_raw(points):
    """Sample lists of one run in a fixed order, so runs can be compared point by point."""
    return {
        "points": [
            {"gbps_samples": points[field][key]}
            for field in sorted(points)
            for key in sorted(points[field], key=repr)
        ]
    }


def dispersion_for_points(points):
    """Capture rules 1 and 2 on one run (microbench/capture_posthoc.py)."""
    from microbench.capture_posthoc import dispersion_check

    return dispersion_check(points_raw(points), suffixes=("gbps_samples",))


def reproducibility_for_runs(point_runs):
    """Capture rule 3 across the repeats of one benchmark."""
    from microbench.capture_posthoc import reproducibility_check

    return reproducibility_check([points_raw(p) for p in point_runs], suffixes=("gbps_samples",))


def capture(args):
    from shutil import which

    schedule = getattr(args, "long_read_schedule", None)
    sample_count = 3 if schedule == "pilot" else SAMPLES
    smi = which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    before = host_snapshot(smi)
    require_quiet(before)
    profile = json.loads(args.profile.read_bytes())
    use_desktop_gate = profile["provenance"]["is_wsl"]
    desktop_gate = {"status": "not_applicable"}
    if use_desktop_gate or schedule is not None:
        from microbench.capture_posthoc import wait_for_quiet_desktop

        desktop_gate = wait_for_quiet_desktop()
        if use_desktop_gate and desktop_gate["status"] != "passed":
            raise RuntimeError("Windows desktop quiet gate is unavailable")
        before = host_snapshot(smi)
        require_quiet(before)
    if schedule is not None:
        before["memory"] = command("free", "-b")
        before["cpu_processes"] = command(
            "ps", "-eo", "pid,comm,pcpu,rss", "--sort=-pcpu"
        ).splitlines()[:12]
    initial = telemetry(smi)
    if initial["name"] != profile["identity"]["sku"]:
        # NVIDIA tools may omit the vendor prefix used by the profile.
        if initial["name"].removeprefix("NVIDIA ") != profile["identity"]["sku"].removeprefix(
            "NVIDIA "
        ):
            raise ValueError("Capture device name differs from the profile")
    query = json.loads(command(args.binary, "--mode", "query", "--device", "0"))
    target = f"sm_{query['major']}{query['minor']}"
    if target != profile["identity"]["arch"] or query["sm_count"] != profile["specs"]["sm_count"]:
        raise ValueError("Capture device architecture or SM count differs from profile")
    warm, stable = warm_device(
        smi,
        args.binary,
        args.warmup_seconds,
        timeout_report=args.output.with_suffix(".thermal-timeout.json"),
        long_read=schedule is not None,
    )
    # GPUs that share memory with the CPU (GB10): CPU work slows DRAM-bound sweeps.
    gate_host = bool(query.get("integrated"))
    host_gate = host_quiet.wait_for_quiet_host([os.getpid()]) if gate_host else None
    run_args = [
        str(args.binary),
        "--mode",
        args.mode,
        "--device",
        "0",
        "--samples",
        str(sample_count),
        "--warmup",
        str(WARMUP),
    ]
    if schedule is not None:
        run_args.extend(("--long-read-schedule", str(schedule)))
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
                activity = {"compute_apps": apps}
                if WINDOWS_SMI.is_file():
                    activity["windows_nvidia_smi"] = command(WINDOWS_SMI)
                require_quiet(activity, own_pid=process.pid)
                time.sleep(0.2)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
        err.seek(0)
        if process.returncode:
            raise RuntimeError(
                f"Microbenchmark exited with status {process.returncode}: {err.read()}"
            )
        out.seek(0)
        raw = json.load(out)
    after = host_snapshot(smi)
    if schedule is not None:
        after["memory"] = command("free", "-b")
        after["cpu_processes"] = command(
            "ps", "-eo", "pid,comm,pcpu,rss", "--sort=-pcpu"
        ).splitlines()[:12]
    require_quiet(after)
    host_roots = [os.getpid(), process.pid]
    if gate_host:
        host_samples.append(host_quiet.sample())
    # Rules 1 and 2 apply to every sweep, long-launch ones included; a pilot has 3 samples.
    dispersion = dispersion_for_points(point_maps(profile, args.mode, raw, sample_count))
    if not snaps:
        raise ValueError("No telemetry was recorded during the sweep")
    return {
        "source_revision": args.revision,
        "source_bundle_sha256": args.source_sha256,
        "base_profile_sha256": digest(args.profile),
        "binary_sha256": digest(args.binary),
        "nvcc": command(args.nvcc, "--version"),
        "target": target,
        "device_query": query,
        "mode": args.mode,
        "session_id": args.session_id,
        "repeat": args.repeat,
        "command": run_args,
        "initial_gpu": initial,
        "host": platform.node(),
        "platform": platform.platform(),
        "started_utc": started,
        "finished_utc": after["utc"],
        "clock_policy": "record_only",
        "before": before,
        "after": after,
        "desktop_quiet_gate": desktop_gate,
        "dispersion_check": dispersion,
        "host_quiet_gate": host_gate,
        "host_quiet_roots": host_roots if gate_host else None,
        "host_samples": host_quiet.slim(host_samples, host_roots) if gate_host else None,
        "host_quiet_check": host_quiet.check(host_samples, host_roots) if gate_host else None,
        "thermal_warmup": warm,
        "thermal_stability": stable,
        "telemetry": snaps,
        "raw": raw,
    }


def active_clock_samples(telemetry_rows, domain):
    readings = [
        int(value)
        for snap in telemetry_rows
        if (value := snap[f"clocks.current.{domain}"]).isdecimal() and int(value) > 0
    ]
    peak = max(readings, default=0)
    return [value for value in readings if value >= 0.5 * peak]


def clock_range(readings):
    return [min(readings), max(readings)] if readings else None


def slowdown_samples(telemetry_rows):
    return sum(
        any(row[key].lower() == "active" for key in TELEMETRY[-4:-1]) for row in telemetry_rows
    )


def reduce_runs(base, paths, revision):
    from microbench.collect_profile import make_measured

    profile = json.loads(base.read_bytes())
    records = [(path, json.loads(path.read_bytes())) for path in paths]
    include_active = any(item["mode"] in ACTIVE_MODES for _, item in records)
    modes = MODES if include_active else MLP_MODES
    expected = Counter((mode, i) for mode in modes for i in range(1, REPEATS + 1))
    if Counter((item["mode"], item["repeat"]) for _, item in records) != expected:
        raise ValueError("Expected repeats 1 through 5 for each benchmark")
    for attr in (
        "source_revision",
        "source_bundle_sha256",
        "base_profile_sha256",
        "binary_sha256",
        "nvcc",
        "target",
        "host",
    ):
        if len({item[attr] for _, item in records}) != 1:
            raise ValueError(f"Conflicting capture provenance for {attr}")
    origin = records[0][1]
    if profile["provenance"]["is_wsl"] and any(
        item.get("desktop_quiet_gate", {}).get("status") != "passed" for _, item in records
    ):
        raise ValueError("Windows repeats require a passing desktop quiet gate")
    if any(
        item.get("dispersion_check", {}).get("status") != "passed"
        or dispersion_for_points(point_maps(profile, item["mode"], item["raw"]))
        != item["dispersion_check"]
        for _, item in records
    ):
        raise ValueError("Every repeat requires recorded and reproducible capture rules 1 and 2")
    if any(
        item.get("device_query", {}).get("integrated")
        and (
            (item.get("host_quiet_check") or {}).get("status") != "passed"
            or host_quiet.check(item["host_samples"], item["host_quiet_roots"])
            != item["host_quiet_check"]
        )
        for _, item in records
    ):
        raise ValueError(
            "Repeats on an integrated GPU require a recorded, passing host quiet check"
        )
    reproducibility = {}
    for mode in modes:
        runs = [
            point_maps(profile, mode, item["raw"]) for _, item in records if item["mode"] == mode
        ]
        reproducibility[mode] = reproducibility_for_runs(runs)
        if reproducibility[mode]["status"] != "passed":
            raise ValueError(f"{mode}: repeats fail capture rule 3: {reproducibility[mode]}")
    if include_active and (
        not origin.get("session_id")
        or any(item.get("session_id") != origin["session_id"] for _, item in records)
        or any(not item.get("thermal_stability") for _, item in records)
    ):
        raise ValueError("MLP and active-SM repeats require one session and a passing thermal gate")
    if (
        origin["base_profile_sha256"] != digest(base)
        or origin["target"] != profile["identity"]["arch"]
    ):
        raise ValueError("Base profile or architecture changed")
    if len({item["initial_gpu"]["uuid"] for _, item in records}) != 1:
        raise ValueError("Captures came from different GPUs")
    if any(item["clock_policy"] != "record_only" or not item["telemetry"] for _, item in records):
        raise ValueError("Capture is missing the approved clock policy or telemetry")
    memory = profile["measured"]["memory"]
    previous = dict(profile["provenance"])
    evidence = {
        "base_profile_sha256": digest(base),
        "previous_provenance": previous,
        "source_revision": origin["source_revision"],
        "source_bundle_sha256": origin["source_bundle_sha256"],
        "binary_sha256": origin["binary_sha256"],
        "nvcc": origin["nvcc"],
        "target": origin["target"],
        "host": origin["host"],
        "gpu": origin["initial_gpu"],
        "new_revision": revision,
        "procedure": "Median of five per-process medians; existing bootstrap over those five medians.",
        "samples_per_run": SAMPLES,
        "warmups_per_point": WARMUP,
        "runs": [],
        "reproducibility": reproducibility,
        "fields": {},
        "ceilings": {},
    }
    if include_active:
        evidence["session_id"] = origin["session_id"]
    for mode, fields in modes.items():
        subset = sorted(
            (item["repeat"], path, item) for path, item in records if item["mode"] == mode
        )
        maps = [point_maps(profile, mode, item["raw"]) for _, _, item in subset]
        snapshots = [row for _, _, item in subset for row in item["telemetry"]]
        events = slowdown_samples(snapshots)
        for index, path, item in subset:
            evidence["runs"].append(
                {
                    "file": path.name,
                    "sha256": digest(path),
                    "mode": mode,
                    "repeat": index,
                    "command": item["command"],
                    "started_utc": item["started_utc"],
                    "finished_utc": item["finished_utc"],
                    "warmup": warmup_evidence(item),
                    "dispersion_check": item["dispersion_check"],
                    **(
                        {"desktop_quiet_gate_status": item["desktop_quiet_gate"]["status"]}
                        if profile["provenance"]["is_wsl"]
                        else {}
                    ),
                }
            )
        for field in fields:
            clock = "memory" if field.startswith("dram_") else "sm"
            clocks = clock_range(active_clock_samples(snapshots, clock))
            notes = []
            for point in memory[field]:
                key = point_key(field, point)
                samples = [run[field][key] for run in maps]
                medians = [statistics.median(group) for group in samples]
                pooled = [value for group in samples for value in group]
                old = point["bw_gbps"]
                measured = make_measured(
                    medians,
                    old["unit"],
                    old["method"],
                    old["perturb_group"],
                    clocks,
                    False,
                    events,
                )
                if origin["target"] == "sm_121":
                    measured["rel_uncertainty"] = max(0.15, measured["rel_uncertainty"])
                point["bw_gbps"] = measured
                q1, _, q3 = statistics.quantiles(pooled)
                notes.append(
                    {
                        "geometry": key,
                        "old_gbps": old["value"],
                        "new_gbps": measured["value"],
                        "repeat_medians_gbps": medians,
                        "repeat_range_gbps": [min(medians), max(medians)],
                        "pooled_median_gbps": statistics.median(pooled),
                        "pooled_iqr_gbps": q3 - q1,
                    }
                )
            evidence["fields"][field] = {
                "points": notes,
                "active_clock_range_mhz": clocks,
                "slowdown_samples": events,
            }
    for level, field in zip(("dram", "l2"), CONTIGUOUS):
        scalar = f"{level}_bw_read_gbps"
        sweep = max(memory[f"{level}_bw_by_active_sms"], key=lambda row: row["active_sms"])[
            "bw_gbps"
        ]
        peak = max(memory[field], key=lambda row: row["bw_gbps"]["value"])["bw_gbps"]
        old = memory[scalar]
        depended_on_mlp = old["method"].startswith("cuda_mlp_")
        if depended_on_mlp or include_active:
            chosen, source = (
                (peak, field)
                if peak["value"] > sweep["value"]
                else (sweep, f"{level}_bw_by_active_sms")
            )
        else:
            chosen, source = old, "unchanged_independent_measurement"
        evidence["ceilings"][scalar] = {
            "old_gbps": old["value"],
            "active_sm_sweep_gbps": sweep["value"],
            "repeat_mlp_peak_gbps": peak["value"],
            "new_gbps": chosen["value"],
            "source": source,
        }
        memory[scalar] = chosen
    snapshots = [row for _, item in records for row in item["telemetry"]]
    sm_clocks = active_clock_samples(snapshots, "sm")
    mem_clocks = active_clock_samples(snapshots, "memory")
    temperatures = [
        int(row["temperature.gpu"]) for row in snapshots if row["temperature.gpu"].isdecimal()
    ]
    evidence["measurement_telemetry"] = {
        "temperature_range_c": clock_range(temperatures),
        "active_sm_clock_range_mhz": clock_range(sm_clocks),
        "active_mem_clock_range_mhz": clock_range(mem_clocks),
        "slowdown_samples": slowdown_samples(snapshots),
        "sw_power_cap_samples": sum(row[TELEMETRY[-1]].lower() == "active" for row in snapshots),
    }
    provenance = profile["provenance"]
    ended = max(item["finished_utc"] for _, item in records)
    stamp = datetime.fromisoformat(ended).strftime("%Y%m%dT%H%M%SZ")
    provenance.update(
        profile_revision=revision,
        calibration_run_id=f"cal-mlp-{stamp}-{origin['target']}",
        timestamp_utc=ended,
        driver_version=origin["initial_gpu"]["driver_version"],
    )
    return profile, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    sample = commands.add_parser("capture")
    sample.add_argument("profile", type=Path)
    sample.add_argument("binary", type=Path)
    sample.add_argument("mode", choices=MODES)
    sample.add_argument("repeat", type=int, choices=range(1, REPEATS + 1))
    sample.add_argument("output", type=Path)
    sample.add_argument("--revision", required=True)
    sample.add_argument("--session-id")
    sample.add_argument("--source-sha256", required=True)
    sample.add_argument("--nvcc", type=Path, required=True)
    sample.add_argument("--warmup-seconds", type=int, default=0)
    reduce = commands.add_parser("reduce")
    reduce.add_argument("profile", type=Path)
    reduce.add_argument("revision")
    reduce.add_argument("output", type=Path)
    reduce.add_argument("manifest", type=Path)
    reduce.add_argument("runs", nargs="+", type=Path)
    reduce.add_argument("--capture-harness", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "capture":
        if args.output.exists():
            raise FileExistsError(args.output)
        record = capture(args)
        write_json(args.output, record)
        if record["dispersion_check"]["status"] == "failed":
            raise SystemExit(f"Dispersion check failed, capture void: {record['dispersion_check']}")
        host = record["host_quiet_check"]
        if host is not None and host["status"] != "passed":
            raise SystemExit(f"Host CPU was not quiet, capture void: {host}")
    else:
        profile, manifest = reduce_runs(args.profile, args.runs, args.revision)
        manifest["capture_harness_sha256"] = digest(args.capture_harness)
        write_json(args.manifest, manifest)
        profile["provenance"]["calibration_manifest_digest"] = digest(args.manifest)
        write_json(args.output, profile)


if __name__ == "__main__":
    main()
