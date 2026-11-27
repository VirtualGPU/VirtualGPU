"""CPU-only checks of spill-latency read across local memory allocations
(docs/results/analysis/v0.3-spill-allocation-design-registration.md)."""

import copy
import importlib.util
import json
import statistics
from pathlib import Path

import pytest

from microbench import capture_exceptions, long_capture, long_launch, profile_launch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("spill_decision", ROOT / "scripts/spill_decision.py")
sd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sd)

N = long_launch.ALLOCATIONS["spill-latency"]
FIELDS = long_launch.DESIGN_FIELDS["spill-latency"]["reload_latency_by_count"]


def capture_point(rc, launches, steps=200, mode="serial_chain", frame=4096):
    point = {
        "regime": "l2_spill",
        "dependency_mode": mode,
        "num_warps": 4,
        "resident_threads_per_sm": 128,
        "frame_bytes_per_thread": frame,
        "reload_count": rc,
        "steps": steps,
        "launch_ns_samples": [1.3e6] * 15,
        "iter_time_ns_samples": [1.3e6 / steps] * 15,
    }
    if launches is not None:
        point["allocation_launch_ns"] = launches
    return point


def record_for(points):
    pilot = {
        "samples": 3,
        "reload_latency_by_count": [
            {**p, "launch_ns_samples": [2.1e6 if p["reload_count"] else 1.3e6] * 3} for p in points
        ],
    }
    for p in pilot["reload_latency_by_count"]:
        p.pop("allocation_launch_ns", None)
    return long_launch.freeze("spill-latency", pilot)[1]


def test_pilot_sizes_reload_points_to_twice_the_floor():
    # v0.3-spill-allocation-redo-registration.md: every reload point at 2.0 ms, nothing else.
    def pilot(rc_ns, zero_ns, cost_ns):
        cost = {"dependency_mode": "serial_chain", "frame_bytes_per_thread": 1024, "steps": 256}
        return {
            "samples": 3,
            "reload_latency_by_count": [
                {**capture_point(0, None), "launch_ns_samples": [zero_ns] * 3},
                {**capture_point(128, None, frame=1024), "launch_ns_samples": [rc_ns] * 3},
            ],
            "launch_cost_by_frame": [{**cost, "launch_ns_samples": [cost_ns] * 3}],
        }

    long_launch.freeze("spill-latency", pilot(2.0e6, 1.25e6, 1.25e6))
    with pytest.raises(ValueError, match="reload_count': 128.*pilot reached only 1990000"):
        long_launch.freeze("spill-latency", pilot(1.99e6, 1.3e6, 1.3e6))
    with pytest.raises(ValueError, match="pilot reached only"):
        long_launch.freeze("spill-latency", pilot(2.0e6, 1.2e6, 1.3e6))
    with pytest.raises(ValueError, match="pilot reached only"):
        long_launch.freeze("spill-latency", pilot(2.0e6, 1.3e6, 1.2e6))
    assert long_launch.pilot_target("l2", {"reload_count": 8}) == long_launch.PILOT_MIN_NS


def test_verify_checks_every_launch_of_every_allocation():
    good = [[1.3e6] * 15 for _ in range(N)]
    points = [capture_point(0, None), capture_point(8, good)]
    record = record_for(points)
    capture = {"samples": 15, "reload_latency_by_count": copy.deepcopy(points)}
    assert long_launch.verify(record, capture)["shortest_launch_ns"] == 1.3e6
    short = copy.deepcopy(capture)
    short_point = short["reload_latency_by_count"][1]
    short_point["allocation_launch_ns"][7][3] = 0.9e6
    ranked = [sorted(one) for one in short_point["allocation_launch_ns"]]
    short_point["launch_ns_samples"] = [sum(one[j] for one in ranked) / N for j in range(15)]
    with pytest.raises(ValueError, match="a launch took 900000"):
        long_launch.verify(record, short)
    missing = copy.deepcopy(capture)
    missing["reload_latency_by_count"][1]["allocation_launch_ns"].pop()
    with pytest.raises(ValueError, match=f"expected {N} allocations"):
        long_launch.verify(record, missing)
    zero = copy.deepcopy(capture)
    zero["reload_latency_by_count"][0]["allocation_launch_ns"] = good
    with pytest.raises(ValueError, match="read in one allocation"):
        long_launch.verify(record, zero)


def test_samples_must_be_rank_means_of_the_unsorted_allocations():
    # allocation_launch_ns keeps launch order; the samples are the means by rank.
    launches = [[1.3e6 + 1e3 * ((a * 7 + j * 5) % 15) for j in range(15)] for a in range(N)]
    point = capture_point(8, launches)
    ranked = [sorted(one) for one in launches]
    point["launch_ns_samples"] = [round(sum(one[j] for one in ranked) / N, 1) for j in range(15)]
    assert point["allocation_launch_ns"][0] != sorted(point["allocation_launch_ns"][0])
    record = record_for([capture_point(0, None), point])
    capture = {"samples": 15, "reload_latency_by_count": [capture_point(0, None), point]}
    long_launch.verify(record, capture)
    by_position = copy.deepcopy(capture)
    by_position["reload_latency_by_count"][1]["launch_ns_samples"] = [
        sum(one[j] for one in launches) / N for j in range(15)
    ]
    with pytest.raises(ValueError, match="rank"):
        long_launch.verify(record, by_position)


def test_merge_keeps_each_round_s_allocations_and_their_shortest_launch():
    rounds = []
    for r in range(5):
        launches = [[1.3e6 + 1e3 * a] * 15 for a in range(N)]
        launches[2][0] = 1.05e6 - r
        rounds.append({"samples": 15, "reload_latency_by_count": [capture_point(8, launches)]})
    merged = long_capture.merge(rounds)["reload_latency_by_count"][0]
    assert len(merged["allocation_launch_ns"]) == 5
    assert merged["launch"]["min_ns"] == 1.05e6 - 4
    assert merged["launch_ns_samples"] == [1.3e6] * 5


def test_serial_1_kib_reload_points_leave_rules_one_and_three_on_gb10_only():
    field = "spill_reload_latency_by_count"

    def point(mode, frame, rc):
        return {"dependency_mode": mode, "frame_bytes_per_thread": frame, "reload_count": rc}

    gb10 = "NVIDIA GB10"
    assert capture_exceptions.exempt(gb10, field, point("serial_chain", 1024, 8), 1)
    assert capture_exceptions.exempt(gb10, field, point("serial_chain", 1024, 128), 3)
    assert not capture_exceptions.exempt(gb10, field, point("serial_chain", 1024, 128), 2)
    assert not capture_exceptions.exempt(gb10, field, point("serial_chain", 1024, 0), 1)
    assert not capture_exceptions.exempt(gb10, field, point("batch_before_dep", 1024, 68), 1)
    assert not capture_exceptions.exempt(gb10, field, point("serial_chain", 2048, 8), 1)
    assert not capture_exceptions.exempt(
        "NVIDIA GeForce RTX 4070", field, point("serial_chain", 1024, 8), 1
    )


def merged_from(profile):
    """A merged spill-latency raw of the new method whose values reproduce `profile`."""
    rows = []
    for q in sd.spill_points(profile):
        steps = 200
        iter_ns = q["iter_time_ns"]["value"]
        row = {k: q[k] for k in ("regime", *FIELDS)}
        row |= {
            "steps": steps,
            "launch_ns_samples": [iter_ns * steps] * 5,
            "iter_time_ns_samples": [iter_ns] * 5,
            "launch": {
                "length": steps,
                "min_ns": 1.1e6,
                "median_ns": 1.3e6,
                "sm_mhz_before": [2500.0] * 5,
                "sm_mhz_after": [2500.0] * 5,
            },
        }
        if q["reload_count"] > 0:
            levels = (0.9, 1.0, 1.1)
            row["allocation_launch_ns"] = [
                [[iter_ns * steps * levels[a % 3]] * 15 for a in range(N)] for _ in range(5)
            ]
        rows.append(row)
    costs = [
        {
            "dependency_mode": m,
            "frame_bytes_per_thread": f,
            "steps": 256,
            "launch_ns_samples": [1.3e6] * 5,
            "launch_cost_ns_samples": [5e3] * 5,
        }
        for f in (128, 256, 512, 1024, 2048, 4096)
        for m in ("serial_chain", "batch_before_dep")
    ]
    return {"samples": 5, "reload_latency_by_count": rows, "launch_cost_by_frame": costs}


def test_attach_stores_the_allocation_span_and_widens_the_uncertainty():
    profile = json.loads((ROOT / "profiles/gb10.json").read_text())
    merged = merged_from(profile)
    profile_launch.attach(profile, "spill-latency", merged)
    q = next(
        p
        for p in sd.spill_points(profile)
        if p["reload_count"] == 128 and p["frame_bytes_per_thread"] == 4096
    )
    assert q["allocation_span"] == pytest.approx(0.2)
    assert q["iter_time_ns"]["rel_uncertainty"] >= 0.1
    zero = next(p for p in sd.spill_points(profile) if p["reload_count"] == 0)
    assert "allocation_span" not in zero


def test_spill_decision_reads_a_profile_built_this_way(tmp_path):
    original = json.loads((ROOT / "profiles/gb10.json").read_text())
    profile = copy.deepcopy(original)
    merged = merged_from(profile)
    for p in merged["reload_latency_by_count"]:
        names = [
            e["name"]
            for e in capture_exceptions.exceptions_for(
                "NVIDIA GB10", "spill_reload_latency_by_count", p
            )
        ]
        if names:
            p["exceptions"] = names
    profile_launch.attach(profile, "spill-latency", merged)
    flagged = [p for p in sd.spill_points(profile) if p.get("l1_boundary_two_state")]
    assert [
        (p["dependency_mode"], p["frame_bytes_per_thread"], p["reload_count"]) for p in flagged
    ] == [("serial_chain", 1024, rc) for rc in (8, 32, 68, 128)]
    profile["provenance"]["capture_protocol"] = {
        "minimum_launch_ns": 1e6,
        "freeze_manifest_sha256": "0" * 64,
        "rounds": [],
        "rule_three": {"spill-latency": {"status": "passed"}},
    }
    path = tmp_path / "gb10.json"
    path.write_text(json.dumps(profile))
    raw = tmp_path / "merged-spill-latency.json"
    raw.write_text(json.dumps(merged))
    assert sd.eligible(profile, raw) is None
    _, now = sd.predictions(path)
    _, before = sd.predictions(ROOT / "profiles/gb10.json")
    assert now["extra_ns_vs_zero"] == before["extra_ns_vs_zero"]
    assert now["boundary_lookup"] == before["boundary_lookup"]
    assert statistics.median(r["error"] for r in now["leave_one_out"]["l2_spill"]) == pytest.approx(
        statistics.median(r["error"] for r in before["leave_one_out"]["l2_spill"])
    )
