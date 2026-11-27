"""CPU-only tests of the barrier-step driver: assembler check, freeze and reading."""

import json
from pathlib import Path

from microbench import barrier_step as bs
from microbench import long_launch

ROOT = Path(__file__).resolve().parents[2]

SASS = """
        Function : _Z19barrier_step_kernelILi2ELi64EEvPfi
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0010*/                   ISETP.GE.AND P0, PT, R0, 0x1, PT ;
{body}
        /*ff00*/                   EXIT ;
"""


def sass_for(bars, ffma, extra=()):
    lines, addr = [], 0x100
    for op in ["FFMA R2, R2, R3, R4"] * ffma + ["BAR.SYNC.DEFER_BLOCKING 0x0"] * bars + list(extra):
        lines.append(f"        /*{addr:04x}*/                   {op} ;")
        addr += 0x10
    lines.append(f"        /*{addr:04x}*/                   IADD3 R5, R5, 0x1, RZ ;")
    lines.append(
        f"        /*{addr + 0x10:04x}*/                   ISETP.NE.AND P0, PT, R5, R6, PT ;"
    )
    lines.append(f"        /*{addr + 0x20:04x}*/              @P0 BRA 0x100 ;")
    return SASS.format(body="\n".join(lines))


def test_assembler_check_counts_the_loop():
    f = bs.functions(sass_for(2, 64))["_Z19barrier_step_kernelILi2ELi64EEvPfi"]
    assert bs.check_instance(f, 2, 64)["status"] == "passed"
    assert bs.check_instance(f, 1, 64)["status"] == "failed"
    stray = bs.functions(sass_for(2, 64, ["LDG.E R7, [R8.64]"]))
    r = bs.check_instance(stray["_Z19barrier_step_kernelILi2ELi64EEvPfi"], 2, 64)
    assert r["status"] == "failed" and "LDG.E" in r["other"]


def test_freeze_keys_points_by_barriers_work_warps_and_grid():
    def point(b, w, warps, g):
        return {
            "barriers_per_step": b,
            "work_per_step": w,
            "num_warps": warps,
            "grid": g,
            "steps": 1000,
            "launch_ns_samples": [1.3e6] * 3,
        }

    pilot = {"samples": 3, "points": [point(0, 0, 4, 1), point(14, 512, 8, 48)]}
    schedule, record = long_launch.freeze("barrier-step", pilot)
    assert schedule == "1000\n1000\n"
    assert record["points"][1]["point"] == {
        "field": "points",
        "barriers_per_step": 14,
        "work_per_step": 512,
        "num_warps": 8,
        "grid": 48,
    }


def capture(t_of_b, noisy=None):
    """Three valid processes whose per-step time is t_of_b(B) for every configuration."""
    runs = []
    for _ in range(3):
        pts = []
        for b in bs.BARRIERS:
            for w in bs.WORK:
                for warps in (4, 8):
                    for g in (1, 48):
                        t = t_of_b(b)
                        steps = 10000
                        launch = [t * steps] * 15
                        if noisy and (b, w, warps, g) == noisy:
                            launch = [t * steps * (2.0 if k == 0 else 1.0) for k in range(15)]
                        pts.append(
                            {
                                "barriers_per_step": b,
                                "work_per_step": w,
                                "num_warps": warps,
                                "grid": g,
                                "steps": steps,
                                "launch_ns_samples": launch,
                                "ns_per_step_samples": [x / steps for x in launch],
                            }
                        )
        runs.append({"valid": True, "raw": {"samples": 15, "points": pts}})
    return {"runs": runs}


PROFILE = json.loads((ROOT / "profiles/gb10-0.23.0.json").read_text())


def test_linear_barrier_cost_reads_as_matters_and_barrier_free():
    out = bs.judge(capture(lambda b: 200.0 + 30.0 * b), PROFILE)["configurations"]
    r = out["W64_w8_g48"]
    assert r["barrier_free"] == "does not pay the barrier cost"
    assert r["count"] == "matters" and r["linear"]


def test_flat_cost_reads_as_pays_and_does_not_matter():
    out = bs.judge(capture(lambda b: 500.0), PROFILE)["configurations"]
    assert out["W512_w4_g1"]["barrier_free"] == "pays it"
    assert out["W512_w4_g1"]["count"] == "does not matter"


def test_a_point_failing_a_rule_leaves_its_configuration_unresolved():
    out = bs.judge(capture(lambda b: 200.0 + 30.0 * b, noisy=(4, 0, 4, 1)), PROFILE)
    r = out["configurations"]["W0_w4_g1"]
    assert r["failing_points"] == [4]
    assert r["barrier_free"].startswith("not resolved") and r["count"].startswith("not resolved")
    assert out["configurations"]["W0_w8_g1"]["count"] == "matters"


def test_fewer_than_three_valid_processes_is_not_decided():
    rec = capture(lambda b: 200.0)
    rec["runs"][2]["valid"] = False
    assert bs.judge(rec, PROFILE)["status"] == "not decided"
