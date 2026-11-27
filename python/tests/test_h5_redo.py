"""CPU-only checks of the H5 GB10 redo under the long-launch protocol
(docs/results/analysis/v0.3-gb10-h5-redo-registration.md)."""

import json

import pytest

from microbench import long_capture, long_launch
from validation import h5_candidates as h5
from validation import h5_dram_form as dram
from validation import h5_redo as redo
from validation import h5_staging_forms as sf

RESOURCES = """Resource usage:
 Function void narrow::gmem_narrow_kernel<1, 16, true, 0>(unsigned char const*, unsigned int*):
  REG:40 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:400 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function void narrow::stage_batched_kernel<1, 2, 32, 0, 4>(unsigned char const*, float*):
  REG:96 STACK:16 SHARED:0 LOCAL:8 CONSTANT[0]:400 TEXTURE:0 SURFACE:0 SAMPLER:0
"""


def gmem_row(level="dram", fit=2, registered=256 << 20):
    return {
        "width_bytes": 1,
        "aligned": True,
        "level": level,
        "ctas_per_sm": 2,
        "ctas_per_sm_fit": fit,
        "num_warps": 8,
        "loads_in_flight": 16,
        "registered_request_bytes": registered,
    }


def stage_row(fit=1):
    return {
        "variant": "load_only",
        "width_bytes": 2,
        "stage_bytes": 4096,
        "mma_per_warp": 0,
        "ctas_per_sm": 1,
        "ctas_per_sm_fit": fit,
        "num_warps": 4,
    }


def test_h5_modes_run_only_when_named_and_size_dram_points_to_three_ms():
    assert set(long_capture.OPT_IN_MODES) == {"h5-dram", "h5-staging"}
    assert not set(long_capture.OPT_IN_MODES) & set(long_capture.MODES)
    dram_point = gmem_row()
    assert long_launch.pilot_target("h5-dram", dram_point) == 3_000_000
    assert long_launch.pilot_target("h5-dram", {**dram_point, "level": "l2"}) == 1_250_000
    assert long_launch.pilot_target("h5-dram", {"stage_bytes": 2048}) == 3_000_000  # sync, DRAM
    assert long_launch.pilot_target("h5-staging", stage_row()) == 1_250_000
    identity = long_launch.identity("h5-dram", ("gmem_load_narrow",), dram_point)
    assert identity["registered_request_bytes"] == 256 << 20


def test_freeze_rejects_a_dram_point_under_three_ms():
    def pilot(ms):
        return {
            "samples": 3,
            "calibration": [{**gmem_row(), "steps": 100, "launch_ns_samples": [ms * 1e6] * 3}],
            "gmem_load_narrow": [],
            "sync_loop_narrow": [],
        }

    long_launch.freeze("h5-dram", pilot(3.0))
    with pytest.raises(ValueError, match="pilot reached only"):
        long_launch.freeze("h5-dram", pilot(2.9))


def test_resources_and_exclusions():
    resources = redo.kernel_resources(RESOURCES)
    assert resources[("gmem_narrow_kernel", ("1", "16", "true", "0"))] == (0, 0)
    assert redo.exclusion("calibration", gmem_row(), resources) is None
    assert "fits 1 CTAs" in redo.exclusion("calibration", gmem_row(fit=1), resources)
    # The load-only variant is template argument 0; argument 1 (store-only) had local memory.
    store = {**stage_row(), "variant": "store_only", "stage_bytes": 4096}
    assert "local memory 8 B" in redo.exclusion("stage_narrow", store, resources)
    with pytest.raises(ValueError, match="no resource usage"):
        redo.exclusion("stage_narrow", stage_row(), resources)


def test_merged_keys_match_the_original_point_keys():
    assert redo.gmem_key(gmem_row()) == dram.CALIBRATION
    l2 = redo.gmem_key(gmem_row(level="l2", registered=0))
    assert l2 == dram.gmem_point(1, 16, "l2", True, 2, 8)
    assert sf.key(redo.stage_key(stage_row())) == sf.key(sf.CAL_B2)


def test_only_a_capture_with_five_valid_rounds_and_rule_three_passed_is_judged():
    rounds = [{"results": {"h5-dram": {"status": "valid"}, "h5-staging": {"status": "valid"}}}] * 5
    state = {
        "rounds": rounds,
        "rule_three": {"h5-dram": {"status": "passed"}, "h5-staging": {"status": "failed"}},
    }
    assert redo.judged_modes(state) == {"h5-dram": True, "h5-staging": False}
    state["rounds"] = rounds[:4]
    assert redo.judged_modes(state) == {"h5-dram": False, "h5-staging": False}


def test_k1_scoring_uses_this_freezes_exclusions_and_the_registered_pass_rule():
    cal1, cal2 = sf.key(sf.CAL_B1), sf.key(sf.CAL_B2)
    grid = [1000.0 * (1 + i) for i in range(len(sf.Q_GRID))]
    points = {
        cal1: {"c0_current": 900.0, "b1_grid": grid, "l2_load_steps_x_loads_per_thread": 0},
        cal2: {
            "c0_current": 900.0,
            "b1_grid": [900.0] * len(grid),
            "l2_load_steps_x_loads_per_thread": 10,
        },
    }
    order = [cal1, cal2]
    for width in (1, 2, 4, 16):
        for mma in (0, 8):
            p = {
                "variant": "load_only",
                "width_bytes": width,
                "stage_bytes": 6144,
                "mma_per_warp": mma,
                "ctas_per_sm": 2,
                "num_warps": 4,
            }
            points[sf.key(p)] = {
                "c0_current": 1000.0,
                "b1_grid": [1000.0] * len(grid),
                "l2_load_steps_x_loads_per_thread": 10,
            }
            order.append(sf.key(p))
    excluded = {order[-1]: "fits 1 CTAs per SM, not 2"}
    frozen = {
        "staging": {
            "calibration_b1": sf.CAL_B1,
            "calibration_b2": sf.CAL_B2,
            "q_grid": sf.Q_GRID,
            "points": points,
            "order": order,
            "excluded": excluded,
        }
    }
    # tau = (1000 - 900) / 10 = 10, so every point is predicted 1000 + 100 = 1100.
    rounds = {k: [1100.0] * 5 for k in order}
    rounds[cal1] = [grid[3]] * 5
    rounds[cal2] = [1000.0] * 5
    out = redo.score_staging(frozen, rounds)
    assert out["tau_ns_per_load"] == pytest.approx(10.0)
    k1 = out["forms"]["k1_b1_b2"]
    assert out["passed"] and k1["judged"]["mape"] == pytest.approx(0.0)
    assert order[-1] not in k1["errors"] and len(k1["errors"]) == 7


def test_dram_scoring_takes_r_from_the_five_round_medians_of_the_calibration_point():
    cal = h5.point_key(dram.CALIBRATION)
    points = {cal: {"t_core_ns": 1000.0, "dram_sector_bytes": 0.0, "dram_requests": 5000.0}}
    rounds = {cal: [2000.0, 2100.0, 1900.0, 2050.0, 1950.0]}
    for width in (1, 2, 4, 16):
        for aligned in (True, False):
            for level in ("dram", "l2"):
                p = dram.gmem_point(width, 4, level, aligned, 1, 4, 128 << 20)
                key = h5.point_key(p)
                points[key] = {
                    "t_core_ns": 500.0,
                    "dram_sector_bytes": 0.0,
                    "dram_requests": 5000.0 if level == "dram" else 0,
                }
                rounds[key] = [2000.0] * 5 if level == "dram" else [500.0] * 5
    frozen = {
        "bw_strided_128_gbps": 222.7,
        "dram": {
            "calibration": dram.CALIBRATION,
            "calibration_runs": 5,
            "points": points,
            "excluded": {},
        },
    }
    out = redo.score_dram(frozen, rounds)
    assert out["rate_warp_requests_per_ns"] == pytest.approx(5000.0 / 2000.0)
    assert out["passed"]
    json.dumps(out)
