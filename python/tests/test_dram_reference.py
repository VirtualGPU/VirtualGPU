"""CPU-only coverage of opt-in GB10 block reference capture and profile validation."""

import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from virtualgpu.profile import GB10_REFERENCE_GEOMETRIES, dram_mlp_geometry, validate_profile

from microbench import collect_profile, dram_reference, long_capture, long_launch
from scripts.check_profiles import check

ROOT = Path(__file__).resolve().parents[2]
GB10 = ROOT / "profiles/gb10.json"
GPU = "GPU-01234567-1234-1234-1234-0123456789ab"
OPTS = {"device_uuid": GPU, "protocol": dram_reference.PROTOCOL, "source_revision": "b" * 40}
TIMESTAMP = 1795737856463


def probe(value, timestamp):
    return {
        "mode": "dram-probe",
        "sm_count": 48,
        "samples": 15,
        "bytes": 256 * 1024**2,
        "unix_ms_end": timestamp,
        "candidates": [
            {
                "name": "P3a",
                "issue": "ld_global_v4",
                "ctas_per_sm": 4,
                "threads_per_cta": 256,
                "loads_in_flight_per_thread": 4,
                "bytes_in_flight_per_cta": 16384,
                "occupancy": 6,
                "gbps_samples": [value] * 15,
            }
        ],
    }


def workload(count=15):
    return {
        "samples": count,
        "dram_bw_by_bytes_in_flight": [
            {
                "issue_mode": mode,
                "bytes_in_flight": bif,
                "active_ctas": ctas,
                "bytes_per_cta": size,
                "gbps_samples": [245.0] * count,
                "steps": 128,
                "launch_ns_samples": [1.3e6] * count,
            }
            for mode, bif, ctas, size in sorted(GB10_REFERENCE_GEOMETRIES)
        ],
    }


def block(value_after=242.0):
    return {
        "before": probe(238.0, TIMESTAMP),
        "workload": workload(),
        "after": probe(value_after, TIMESTAMP + 1000),
    }


def write(path, obj):
    data = json.dumps(obj).encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def evidence(tmp_path):
    manifest = {"binary_sha256": "c" * 64, "dram_reference": OPTS}
    digest = write(tmp_path / "freeze-manifest.json", manifest)
    rounds = []
    for number, raw, status in (
        (1, block(), "valid"),
        (2, block(250.0), "valid"),
        (3, block(), "void"),
    ):
        name = f"block-{number}.json"
        sha = write(tmp_path / name, raw)
        raw_name = f"mlp-{number}.json"
        write(tmp_path / raw_name, raw["workload"])
        rounds.append(
            {
                "round": number,
                "results": {
                    "mlp": {
                        "status": status,
                        "raw": raw_name,
                        "dram_reference": {"raw": name, "sha256": sha},
                    }
                },
            }
        )
    state = {"status": "passed", "rounds": rounds, "dram_reference_manifest_sha256": digest}
    return manifest, digest, state


def base_profile(digest):
    profile = json.loads(GB10.read_text())
    profile["provenance"]["capture_protocol"] = {
        "minimum_launch_ns": 1e6,
        "freeze_manifest_sha256": digest,
        "rounds": [
            {"round": i, "results": {"mlp": {"status": status}}}
            for i, status in enumerate(("valid", "valid", "void"), 1)
        ],
        "rule_three": {},
    }
    return profile


def test_bracketed_capture_preserves_raw_mlp_and_freeze(tmp_path):
    payload = block()
    cmd = long_capture.command("fake", "mlp", "frozen.schedule", 15)
    expected = ["dram-probe", "mlp", "dram-probe"]
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args[args.index("--mode") + 1])
        return Mock(
            returncode=0, stdout=json.dumps(payload["before" if len(calls) == 1 else "after"])
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(dram_reference.subprocess, "run", fake_run)

        def run_mlp(args, **kwargs):
            calls.append(args[args.index("--mode") + 1])
            assert args == cmd
            return json.dumps(payload["workload"])

        patch.setattr(dram_reference.subprocess, "check_output", run_mlp)
        assert dram_reference.run_block(cmd) == payload
    assert calls == expected

    schedule, freeze = long_launch.freeze("mlp", workload(3))
    (tmp_path / "mlp.schedule").write_text(schedule)
    write(tmp_path / "mlp.freeze.json", freeze)
    manifest = {
        "binary": "fake",
        "samples": 15,
        "device": {"name": "NVIDIA GB10", "integrated": True, "l2_size_bytes": 24 * 1024**2},
        "dram_reference": OPTS,
        "modes": {"mlp": {"schedule": "mlp.schedule", "record": "mlp.freeze.json"}},
    }

    def run_capture(args):
        assert args[:2] == dram_reference.block_command([]) and "--clock-probe" in args
        return {"stdout": json.dumps(payload), "host_check": {"status": "passed"}}

    result = long_capture.capture(
        tmp_path, manifest, {"modes": ["mlp"], "rounds": []}, run_capture, dict
    )
    assert result["status"] == "passed"
    for rnd in result["rounds"]:
        mlp = rnd["results"]["mlp"]
        assert json.loads((tmp_path / mlp["raw"]).read_bytes()) == payload["workload"]
        receipt = mlp["dram_reference"]
        assert (
            hashlib.sha256((tmp_path / receipt["raw"]).read_bytes()).hexdigest()
            == receipt["sha256"]
        )


def test_profile_pairs_only_valid_blocks_and_leaves_absolute_values(tmp_path):
    manifest, digest, state = evidence(tmp_path)
    profile = base_profile(digest)
    absolute = copy.deepcopy(profile["measured"])
    dram_reference.attach(profile, tmp_path, manifest, digest, state)
    validate_profile(profile)
    assert [(ref["status"], ref["value_gbps"]) for ref in profile["references"].values()] == [
        ("valid", 240.0),
        ("drift", 244.0),
        ("void", 240.0),
    ]
    count = 0
    for point in profile["measured"]["memory"]["dram_bw_by_bytes_in_flight"]:
        if observations := point.pop("reference_observations", []):
            assert observations == [{"reference_id": f"{digest}:1", "value_gbps": 245.0}]
            assert (
                observations[0]["value_gbps"] / profile["references"][f"{digest}:1"]["value_gbps"]
                > 1
            )
            count += 1
    assert count == 14
    assert profile["measured"] == absolute


def test_old_capture_cannot_get_references_and_block_digest_is_checked(tmp_path):
    manifest, digest, state = evidence(tmp_path)
    profile = base_profile(digest)
    old = {k: v for k, v in state.items() if not k.startswith("dram_reference")}
    dram_reference.attach(profile, tmp_path, {}, digest, old)
    assert "references" not in profile
    with pytest.raises(ValueError, match="opt-in freeze"):
        dram_reference.attach(profile, tmp_path, manifest, digest, old)
    (tmp_path / "block-1.json").write_text("{}")
    with pytest.raises(ValueError, match="block digest"):
        dram_reference.attach(profile, tmp_path, manifest, digest, state)


@pytest.mark.parametrize(
    "change, error",
    [
        ("unknown", "missing or ineligible"),
        ("void", "missing or ineligible"),
        ("geometry", "outside the registered plateau"),
        ("duplicate", "repeated reference"),
        ("device", "require the NVIDIA GB10"),
        ("drift", "3% probe drift bound"),
        ("value", "mean of the two probe medians"),
        ("time", "invalid probe timestamps"),
        ("gate", "status does not match the MLP capture round"),
        ("l2", "should not be valid"),
    ],
)
def test_check_profiles_rejects_invalid_references(tmp_path, change, error):
    manifest, digest, state = evidence(tmp_path)
    profile = base_profile(digest)
    dram_reference.attach(profile, tmp_path, manifest, digest, state)
    curve = profile["measured"]["memory"]["dram_bw_by_bytes_in_flight"]
    paired = next(p for p in curve if "reference_observations" in p)
    obs = paired["reference_observations"][0]
    ref = profile["references"][obs["reference_id"]]
    if change == "unknown":
        obs["reference_id"] = "unknown"
    elif change == "void":
        obs["reference_id"] = f"{digest}:3"
    elif change == "geometry":
        next(p for p in curve if "reference_observations" not in p)["reference_observations"] = [
            obs
        ]
    elif change == "duplicate":
        paired["reference_observations"].append(obs)
    elif change == "device":
        profile["identity"]["sku"] = "Other device"
    elif change == "drift":
        ref["status"] = "drift"
    elif change == "value":
        ref["value_gbps"] = 241.0
    elif change == "time":
        ref["before"]["utc"] = ref["after"]["utc"]
    elif change == "gate":
        profile["provenance"]["capture_protocol"]["rounds"][0]["results"]["mlp"]["status"] = "void"
    else:
        profile["measured"]["memory"]["l2_bw_by_bytes_in_flight"][0]["reference_observations"] = [
            obs
        ]
    path = tmp_path / "profile.json"
    write(path, profile)
    assert error in "\n".join(check(path)[0])


def test_collector_uses_reference_receipts_only_for_opted_in_profiles(tmp_path, monkeypatch):
    manifest, digest, state = evidence(tmp_path)
    binary = tmp_path / "binary"
    binary.write_bytes(b"captured")
    manifest["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    digest = write(tmp_path / "freeze-manifest.json", manifest)
    state["dram_reference_manifest_sha256"] = digest
    state["modes"] = list(collect_profile.BUILD_MODES)
    state["rule_three"] = {}
    write(tmp_path / "capture-state.json", state)
    for mode in state["modes"]:
        write(tmp_path / f"merged-{mode}.json", {})

    # The absolute profile builder and GPU dispatch query are independent of reference pairing.
    monkeypatch.setattr(
        collect_profile, "build_profile", lambda *args: json.loads(GB10.read_text())
    )
    monkeypatch.setattr(
        collect_profile, "run_bench_with_clock_polling", lambda *args, **kw: ({}, [])
    )
    monkeypatch.setattr("microbench.profile_launch.attach_all", lambda *args: None)
    monkeypatch.setattr(dram_reference, "require_lock", lambda: None)
    monkeypatch.setattr(dram_reference, "current_uuid", lambda: GPU)
    profile = collect_profile.long_launch_profile(binary, 0, tmp_path)
    assert len(profile["references"]) == 3
    assert profile["provenance"]["profile_revision"] == "0.23.0"


def test_exclusive_capture_lock_is_required(tmp_path, monkeypatch):
    if not Path("/proc/self/fdinfo").is_dir():
        pytest.skip("Linux exposes inherited flock state in fdinfo")
    import fcntl

    lock = tmp_path / "gpu.lock"
    monkeypatch.setattr(dram_reference, "GPU_LOCK", lock)
    with lock.open("a") as fd:
        with pytest.raises(RuntimeError, match="exclusive flock"):
            dram_reference.require_lock()
        fcntl.flock(fd, fcntl.LOCK_EX)
        dram_reference.require_lock()


def test_p3a_probe_geometry_matches_reviewed_archive():
    path = ROOT / "docs/results/analysis/v0.3-gb10-dram-probe/formal/session-2.json"
    raw = json.loads(path.read_text())["blocks"][0]
    assert {dram_mlp_geometry(p) for p in raw["workload"]["points"]} == GB10_REFERENCE_GEOMETRIES
    before = dram_reference.probe_samples(raw["probe_before"])
    assert len(before["gbps_samples"]) == 15
    raw["probe_before"]["candidates"][2]["loads_in_flight_per_thread"] = 8
    with pytest.raises(ValueError, match="registered P3a geometry"):
        dram_reference.probe_samples(raw["probe_before"])
