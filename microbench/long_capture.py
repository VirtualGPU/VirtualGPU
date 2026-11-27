"""Five-round long-launch capture (docs/components/long-launch-capture.md, section 6).

    python microbench/long_capture.py pilot BINARY RUN_DIR      # size and freeze every mode
    python microbench/long_capture.py capture BINARY RUN_DIR    # 5 rounds; rerun to resume
    python microbench/long_capture.py reduce RUN_DIR            # merged raws for build_profile

`pilot` writes each mode's schedule and freeze record and a manifest that binds
them, the binary and the protocol before any timed capture. `capture` runs the
rounds: each round passes the gates once (thermal; host on GPUs that share memory
with the CPU; desktop under WSL), then runs every mode that still lacks valid
rounds as its own process under its frozen schedule with the SM clock probe. A
mode-round that fails after timing (frozen-schedule check of the output, rules 1
and 2, host check) is void and kept; extra rounds run only the modes short of 5
valid rounds, at most 2 extra rounds per capture. An interruption before a mode's
output is recorded (a gate that does not pass, a process that cannot run) stops
the command; rerunning it resumes at that round, gates first, and does not count.
Rule 3 is judged across each mode's 5 valid rounds. Device exceptions
(microbench/capture_exceptions.py) leave rules 1 and 3 only.

`reduce` writes, per mode, a raw of the usual shape whose sample lists are the 5
round medians in round order, every other field kept, and each timed point's
`launch` (frozen length, shortest launch over all valid rounds, median of round
medians, and each round's SM clock before and after); points a device exception
covers also carry the exceptions' names and the range of their round medians. It
writes no profile.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import host_quiet  # noqa: E402
import long_launch  # noqa: E402
from capture_exceptions import exceptions_for, exempt  # noqa: E402
from capture_posthoc import dispersion_check, reproducibility_check  # noqa: E402

MODES = (
    "mlp",
    "strided-mlp",
    "dram-bw",
    "l2",
    "closed-loop",
    "l2-merge",
    "mma",
    "gmem-issue",
    "gmem-issue-controlled",
    "sync-loop",
    "closed-loop-mma",
    "spill-latency",
    "mma-chain",
)
ROUNDS = 5
EXTRA_ROUNDS = 2
SAMPLES = 15
WARMUP = 3
PILOT_SAMPLES = 3
TIMING = ("launch_ns_samples",)
# Raw output fields whose profile field has another name (exceptions are registered by
# profile field); every other raw field has its profile field's name.
PROFILE_FIELD = {
    ("dram-bw", "read_curve"): "dram_bw_by_active_sms",
    ("l2", "read_curve"): "l2_bw_by_active_sms",
    ("l2", "capacity_sweep"): "l2_capacity_sweep",
    ("l2-rescan", "points"): "l2_rescan_hit_rate_by_working_set",
    ("spill-latency", "reload_latency_by_count"): "spill_reload_latency_by_count",
    ("mma", "mma"): "mma_tflops_per_sm_by_warps",
    ("mma", "fp32_fma"): "fp32_fma_tflops",
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, data):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(data, indent=1) + "\n")


def command(binary, mode, schedule, samples, device=0, probe=True):
    cmd = [str(binary), "--mode", mode, "--device", str(device), "--samples", str(samples)]
    cmd += ["--warmup", str(WARMUP), "--launch-schedule", str(schedule)]
    return cmd + (["--clock-probe"] if probe else [])


# Exceptions and rules ------------------------------------------------------------


def judged(mode, raw, device, rule, l2_bytes=0):
    """A copy of `raw` without the timed points the device's exceptions exempt from `rule`."""
    out = copy.deepcopy(raw)

    def strip(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                strip(value, path + (key,))
        elif isinstance(node, list):
            field = PROFILE_FIELD.get((mode, path[0]), path[0]) if path else None
            node[:] = [
                item
                for item in node
                if not (
                    isinstance(item, dict)
                    and "launch_ns_samples" in item
                    and exempt(device, field, item, rule, l2_bytes)
                )
            ]
            for item in node:
                strip(item, path)

    strip(out, ())
    return out


def rules_one_two(mode, raw, device, l2_bytes=0):
    """Rule 1 without the points exempt from it, rule 2 on every list."""
    every = dispersion_check(raw, TIMING)
    rule1 = dispersion_check(judged(mode, raw, device, 1, l2_bytes), TIMING)["rules"]["iqr"]
    rules = {"iqr": rule1, "spike": every["rules"]["spike"]}
    failed = (
        every["sample_lists"] == 0
        or every["invalid_lists"] > 0
        or any(r["failing"] > r["allowed"] for r in rules.values())
    )
    return {**every, "status": "failed" if failed else "passed", "rules": rules}


def rule_three(mode, raws, device, l2_bytes=0):
    return reproducibility_check([judged(mode, r, device, 3, l2_bytes) for r in raws], TIMING)


# Round state ---------------------------------------------------------------------


def valid_rounds(state, mode):
    return [
        r["round"] for r in state["rounds"] if r["results"].get(mode, {}).get("status") == "valid"
    ]


def next_action(state):
    """("run", round, mode), ("new_round", modes), ("done",) or ("failed", reason)."""
    if state["rounds"]:
        last = state["rounds"][-1]
        missing = [m for m in last["modes"] if m not in last["results"]]
        if missing:
            return ("run", last["round"], missing[0])
    short = [m for m in state["modes"] if len(valid_rounds(state, m)) < ROUNDS]
    if not short:
        return ("done",)
    if len(state["rounds"]) >= ROUNDS + EXTRA_ROUNDS:
        return (
            "failed",
            f"modes short of {ROUNDS} valid rounds after {EXTRA_ROUNDS} extra rounds: {short}",
        )
    return ("new_round", short)


def judge_output(mode, receipt, raw, host_check, device, l2_bytes=0):
    """Status of one mode-round after its process: valid, or void with the reason."""
    try:
        long_launch.verify(receipt, raw)
    except ValueError as error:
        return {"status": "void", "reason": f"frozen schedule: {error}"}
    rules = rules_one_two(mode, raw, device, l2_bytes)
    if rules["status"] != "passed":
        return {"status": "void", "reason": "rules 1 and 2", "rules_one_two": rules}
    if host_check["status"] != "passed":
        return {"status": "void", "reason": "host check", "rules_one_two": rules}
    return {"status": "valid", "rules_one_two": rules}


def capture(run_dir, manifest, state, run, gates, telemetry=dict):
    """Advance `state` until done or failed; `run(cmd)` runs one process
    (host_quiet.run_traced), `gates()` passes the round gates or raises, `telemetry()`
    reads the GPU's driver, power limit, clocks and throttle state after each process."""
    device, l2_bytes = manifest["device"]["name"], manifest["device"]["l2_size_bytes"]
    resumed = True
    while True:
        action = next_action(state)
        if action[0] in ("done", "failed"):
            break
        if action[0] == "new_round":
            state["rounds"].append(
                {
                    "round": len(state["rounds"]) + 1,
                    "modes": action[1],
                    "gates": [gates()],
                    "results": {},
                }
            )
            resumed = False
            save(run_dir, state)
            continue
        _, number, mode = action
        current = state["rounds"][-1]
        if resumed:  # an interrupted round passes its gates again
            current["gates"].append(gates())
            resumed = False
        entry = manifest["modes"][mode]
        schedule = run_dir / entry["schedule"]
        receipt = json.loads((run_dir / entry["record"]).read_text())
        long_launch.check_schedule(receipt, schedule.read_bytes())
        out = run(command(manifest["binary"], mode, schedule, manifest["samples"]))
        raw = json.loads(out.pop("stdout"))
        stem = f"round{number}-{mode}"
        (run_dir / f"{stem}.json").write_text(json.dumps(raw) + "\n")
        (run_dir / f"{stem}.trace.json").write_text(json.dumps(out) + "\n")
        current["results"][mode] = {
            "raw": f"{stem}.json",
            "host_check": out["host_check"],
            "gpu": telemetry(),
            **judge_output(mode, receipt, raw, out["host_check"], device, l2_bytes),
        }
        save(run_dir, state)
    if action[0] == "failed":
        state["status"], state["reason"] = "failed", action[1]
    else:
        state["rule_three"] = {
            mode: rule_three(mode, load_valid(run_dir, state, mode), device, l2_bytes)
            for mode in state["modes"]
        }
        passed = all(r["status"] == "passed" for r in state["rule_three"].values())
        state["status"] = "passed" if passed else "rule_three_failed"
    save(run_dir, state)
    return state


def save(run_dir, state):
    tmp = run_dir / "capture-state.json.tmp"
    tmp.write_text(json.dumps(state, indent=1) + "\n")
    tmp.replace(run_dir / "capture-state.json")


def load_valid(run_dir, state, mode):
    rounds = {r["round"]: r for r in state["rounds"]}
    return [
        json.loads((run_dir / rounds[n]["results"][mode]["raw"]).read_text())
        for n in valid_rounds(state, mode)
    ]


# Reduction -----------------------------------------------------------------------


def merge(raws):
    """One raw of the usual shape from the valid rounds: sample lists become the round
    medians in round order, timed points gain `launch`, other fields are kept: equal across
    rounds, or numbers measured per process (clock calibrations), which take their median."""

    def walk(nodes):
        first = nodes[0]
        if isinstance(first, dict):
            out = {}
            for key, value in first.items():
                values = [n[key] for n in nodes]
                if key.endswith("_samples") and isinstance(value, list):
                    out[key] = [statistics.median(v) for v in values]
                elif isinstance(value, (dict, list)):
                    out[key] = walk(values)
                elif key.startswith("sm_mhz_"):
                    out[key] = values
                elif all(v == value for v in values):
                    out[key] = value
                elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
                    out[key] = statistics.median(values)  # per-process calibrations
                else:
                    raise ValueError(f"{key} differs between rounds")
            if "launch_ns_samples" in first:
                launches = [n["launch_ns_samples"] for n in nodes]
                out["launch"] = {
                    "length": first["steps"],
                    "min_ns": min(min(v) for v in launches),
                    "median_ns": statistics.median(statistics.median(v) for v in launches),
                    "sm_mhz_before": [n.get("sm_mhz_before") for n in nodes],
                    "sm_mhz_after": [n.get("sm_mhz_after") for n in nodes],
                }
            return out
        if isinstance(first, list):
            if any(len(n) != len(first) for n in nodes):
                raise ValueError("rounds differ in their points")
            return [walk([n[i] for n in nodes]) for i in range(len(first))]
        return first

    return walk(raws)


def mark_exceptions(mode, merged, device, l2_bytes=0):
    """Name, on each merged timed point a device exception covers, the exceptions and
    the range of its round medians, which its uncertainty must cover."""
    for path, point in long_launch.points(merged):
        field = PROFILE_FIELD.get((mode, path[0]), path[0])
        names = [e["name"] for e in exceptions_for(device, field, point, l2_bytes)]
        if names:
            medians = point["launch_ns_samples"]
            point["exceptions"] = names
            point["round_median_range_ns"] = [min(medians), max(medians)]
    return merged


def reduce(run_dir):
    state = json.loads((run_dir / "capture-state.json").read_text())
    manifest = json.loads((run_dir / "freeze-manifest.json").read_text())
    if "rule_three" not in state:
        raise ValueError("the capture has not finished its rounds")
    device, l2_bytes = manifest["device"]["name"], manifest["device"]["l2_size_bytes"]
    for mode in state["modes"]:
        raw = mark_exceptions(mode, merge(load_valid(run_dir, state, mode)), device, l2_bytes)
        write_new(run_dir / f"merged-{mode}.json", raw)


# Commands ------------------------------------------------------------------------


def query(binary):
    import subprocess

    return json.loads(subprocess.check_output([str(binary), "--mode", "query"], text=True))


def pilot(binary, run_dir, modes):
    run_dir.mkdir(parents=True, exist_ok=True)
    device = query(binary)
    entries = {}
    for mode in modes:
        out = host_quiet.run_traced(command(binary, mode, "pilot", PILOT_SAMPLES, probe=False))
        raw = json.loads(out["stdout"])
        write_new(run_dir / f"pilot-{mode}.json", raw)
        schedule, record = long_launch.freeze(mode, raw)
        (run_dir / f"{mode}.schedule").write_text(schedule, newline="\n")
        write_new(run_dir / f"{mode}.freeze.json", record)
        entries[mode] = {
            "schedule": f"{mode}.schedule",
            "schedule_sha256": sha256(schedule.encode()),
            "record": f"{mode}.freeze.json",
            "record_sha256": sha256((run_dir / f"{mode}.freeze.json").read_bytes()),
        }
    write_new(
        run_dir / "freeze-manifest.json",
        {
            "binary": str(binary),
            "binary_sha256": sha256(Path(binary).read_bytes()),
            "device": {k: device[k] for k in ("name", "integrated", "l2_size_bytes", "sm_count")},
            "rounds": ROUNDS,
            "extra_rounds": EXTRA_ROUNDS,
            "samples": SAMPLES,
            "warmup": WARMUP,
            "modes": entries,
        },
    )


def round_gates(binary, run_dir, manifest, warmup_seconds):
    """The gates a round passes once, before its first mode."""
    from shutil import which

    from capture_posthoc import wait_for_quiet_desktop
    from repeat_mlp import warm_device

    smi = which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"

    def gates():
        report = run_dir / "thermal-timeout.json"
        warm, stable = warm_device(smi, str(binary), warmup_seconds, timeout_report=report)
        record = {"thermal_stability": stable}
        if manifest["device"]["integrated"]:
            record["host_quiet"] = host_quiet.wait_for_quiet_host([os.getpid()])
        record["desktop_quiet"] = wait_for_quiet_desktop()
        return record

    return gates


def check_manifest(run_dir, binary):
    manifest = json.loads((run_dir / "freeze-manifest.json").read_text())
    if sha256(Path(binary).read_bytes()) != manifest["binary_sha256"]:
        raise ValueError("the binary differs from the frozen manifest")
    for entry in manifest["modes"].values():
        for name in ("schedule", "record"):
            if sha256((run_dir / entry[name]).read_bytes()) != entry[f"{name}_sha256"]:
                raise ValueError(f"{entry[name]} differs from the frozen manifest")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("pilot")
    p.add_argument("binary", type=Path)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    c = sub.add_parser("capture")
    c.add_argument("binary", type=Path)
    c.add_argument("run_dir", type=Path)
    c.add_argument("--warmup-seconds", type=int, default=120)
    r = sub.add_parser("reduce")
    r.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    if args.action == "pilot":
        pilot(args.binary, args.run_dir, args.modes)
    elif args.action == "capture":
        manifest = check_manifest(args.run_dir, args.binary)
        path = args.run_dir / "capture-state.json"
        state = (
            json.loads(path.read_text())
            if path.exists()
            else {"modes": list(manifest["modes"]), "rounds": []}
        )
        from collect_profile import sample_nvidia_smi

        state = capture(
            args.run_dir,
            manifest,
            state,
            host_quiet.run_traced,
            round_gates(args.binary, args.run_dir, manifest, args.warmup_seconds),
            lambda: sample_nvidia_smi(0),
        )
        print(json.dumps({"status": state["status"], "rounds": len(state["rounds"])}))
        if state["status"] != "passed":
            sys.exit(f"capture {state['status']}")
    else:
        reduce(args.run_dir)


if __name__ == "__main__":
    main()
