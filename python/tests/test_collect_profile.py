"""Tests for the non-MMA issue rate in microbench/collect_profile.py."""

from __future__ import annotations

from microbench.collect_profile import non_mma_issue_rate, peak_rate_samples


def _curve(medians: dict[int, float]) -> list[dict]:
    return [
        {"warps_per_sm": w, "rate_gwarp_inst_per_sm_samples": [m * 0.99, m, m * 1.01]}
        for w, m in medians.items()
    ]


def test_peak_rate_samples_takes_highest_median() -> None:
    samples = peak_rate_samples(_curve({4: 3.0, 8: 7.4, 16: 7.6, 32: 7.5}))
    assert samples == [7.6 * 0.99, 7.6, 7.6 * 1.01]


def test_non_mma_issue_rate_measured_from_alu_curve() -> None:
    raw = {"alu_issue_rate_per_sm_gips": _curve({8: 7.4, 16: 7.6})}
    mv = non_mma_issue_rate(raw, 1.9, [1890.0, 1905.0], 0)
    assert mv["source"] == "measured"
    assert mv["method"] == "cuda_ptx_ffma_lop3_mix"
    assert mv["unit"] == "Ginst/s"
    assert abs(mv["value"] - 7.6) < 1e-9


def test_non_mma_issue_rate_estimated_without_alu_curve() -> None:
    mv = non_mma_issue_rate({}, 1.9, None, 0)
    assert mv["source"] == "estimated"
    assert mv["value"] == 7.6
