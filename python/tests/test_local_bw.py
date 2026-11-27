"""CPU-only tests of the local-bw driver and its long-launch registration."""

import json
from pathlib import Path

import pytest

from microbench import local_bw as lb
from microbench import long_launch

ROOT = Path(__file__).resolve().parents[2]
PROFILE = json.loads((ROOT / "profiles/gb10-0.23.0.json").read_text())

POINTS = (
    [
        ("load", 4, "l1", w, f)
        for w, f in ((1, 128), (2, 128), (4, 128), (8, 128), (16, 64), (32, 32))
    ]
    + [
        ("store", 4, "l1", w, f)
        for w, f in ((1, 128), (2, 128), (4, 128), (8, 128), (16, 64), (32, 32))
    ]
    + [
        ("load", 16, "l1", w, f)
        for w, f in ((1, 128), (2, 128), (4, 128), (8, 128), (16, 64), (32, 32))
    ]
    + [("load", 4, "l1", 8, 32), ("load", 4, "l1", 8, 64)]
    + [("load", 4, "l2", 16, 1024), ("load", 4, "l2", 8, 2048), ("load", 4, "l2", 4, 4096)]
)


def head(k):
    op, width, tier, warps, frame = k
    return {
        "op": op,
        "width_bytes": width,
        "tier": tier,
        "num_warps": warps,
        "frame_bytes_per_thread": frame,
    }


def test_l2_points_need_twice_the_floor_in_the_pilot():
    def pilot(fastest):
        pts = [head(k) | {"steps": 1000, "launch_ns_samples": [fastest] * 3} for k in POINTS]
        return {"samples": 3, "points": pts}

    with pytest.raises(ValueError, match="pilot reached only"):
        long_launch.freeze("local-bw", pilot(1.5e6))
    schedule, record = long_launch.freeze("local-bw", pilot(2.1e6))
    assert len(schedule.split()) == len(POINTS)
    assert record["points"][-1]["point"] == {"field": "points", **head(POINTS[-1])}


def test_every_point_is_read_in_fifteen_allocations():
    point = head(POINTS[0]) | {"steps": 1000, "launch_ns_samples": [1.2e6] * 3}
    with pytest.raises(ValueError, match="expected 15 allocations"):
        long_launch.check_allocations("local-bw", point, 3)
    point["allocation_launch_ns"] = [[1.2e6] * 3 for _ in range(15)]
    assert long_launch.check_allocations("local-bw", point, 3) == 1.2e6


def sass(local_ops):
    lines = ["        Function : _ZN8local_bw6kernelILi128ELi4ELb0EEEvPji"]
    addr = 0x100
    for op in ["SHFL.IDX R3, R3, RZ, 0x1f"] + local_ops + ["IADD3 R5, R5, 0x1, RZ"]:
        lines.append(f"        /*{addr:04x}*/                   {op} ;")
        addr += 0x10
    lines.append(f"        /*{addr:04x}*/              @P0 BRA 0x100 ;")
    return "\n".join(lines)


def test_assembler_check_wants_eight_local_ops_of_the_width():
    name = lb.instance_name(128, 4, False)
    good = lb.functions(sass(["LDL R4, [R2]"] * 8))[name]
    assert lb.check_instance(good, 4, False)["status"] == "passed"
    assert lb.check_instance(good, 16, False)["status"] == "failed"
    promoted = lb.functions(sass(["LDL R4, [R2]"] * 2))[name]
    assert lb.check_instance(promoted, 4, False)["status"] == "failed"


def capture(rate, store_rate=None, spread=0.0):
    """Three processes; every point's per-allocation medians give `rate(key)` GB/s per SM."""
    runs = []
    for _ in range(3):
        pts = []
        for k in POINTS:
            r = store_rate if (store_rate and k[0] == "store") else rate(k)
            steps = 10000
            ns = lb.bytes_per_step(head(k)) * steps / r
            allocs = [[ns * (1 + spread * (a % 2))] * 15 for a in range(15)]
            pts.append(
                head(k)
                | {"steps": steps, "launch_ns_samples": [ns] * 15, "allocation_launch_ns": allocs}
            )
        runs.append({"valid": True, "raw": {"samples": 15, "points": pts}})
    return {"runs": runs}


def tiered(k):
    if k[2] == "l2":
        return 100.0
    return {1: 60.0, 2: 120.0, 4: 240.0, 8: 300.0, 16: 305.0, 32: 304.0}[k[3]]


def test_readings_on_a_saturating_two_tier_capture():
    out = lb.judge(capture(tiered), PROFILE)
    assert out["curves"]["load4"]["saturated"] is True
    assert out["curves"]["load4"]["value_gbps_per_sm"] == pytest.approx(305.0)
    assert out["frame_independence"] == "clean"
    assert out["tiers"]["separate"] is True
    assert out["one_rate_for_loads_and_stores"] is True
    assert out["points"]["load/4/l1/8/128"]["allocation"] == "independent"


def test_store_rate_off_by_more_than_ten_percent_and_allocation_spread():
    out = lb.judge(capture(tiered, store_rate=250.0, spread=0.2), PROFILE)
    assert out["one_rate_for_loads_and_stores"] is False
    assert out["points"]["load/4/l2/4/4096"]["allocation"] == "dependent"


def test_fewer_than_three_valid_processes_is_not_decided():
    rec = capture(tiered)
    rec["runs"][0]["valid"] = False
    assert lb.judge(rec, PROFILE)["status"] == "not decided"
