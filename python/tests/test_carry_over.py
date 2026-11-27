"""CPU-only checks of a long-launch capture whose only failure is rule 3 in one mode
(docs/components/long-launch-capture.md, section 6): the mode's fields are carried over from
an earlier profile, marked provisional, and the consumers do not take them as measured."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from microbench import collect_profile as cp

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "profiles/gb10.json"
FIELD = "spill_reload_latency_by_count"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def state(rule_three, valid=5, status="rule_three_failed", modes=("dram-bw", "spill-latency")):
    rounds = [
        {
            "round": n + 1,
            "results": {m: {"status": "valid" if n < valid else "void"} for m in modes},
        }
        for n in range(5)
    ]
    out = {"status": status, "modes": list(modes), "rounds": rounds}
    if rule_three is not None:
        out["rule_three"] = {m: {"status": rule_three.get(m, "passed")} for m in modes}
    return out


def test_only_one_listed_mode_failing_rule_three_is_carried():
    assert cp.carried_mode(state({}, status="passed")) is None
    assert cp.carried_mode(state({"spill-latency": "failed"})) == "spill-latency"
    with pytest.raises(SystemExit, match="rule 3 failed in"):
        cp.carried_mode(state({"spill-latency": "failed", "dram-bw": "failed"}))
    with pytest.raises(SystemExit, match="cannot be carried over"):
        cp.carried_mode(state({"dram-bw": "failed"}))


def test_a_capture_that_failed_rule_one_or_lacks_rounds_writes_no_profile():
    # Rules 1 and 2, the launch minimum and the host check void rounds; a capture that runs
    # out of extra rounds ends "failed" and never reaches rule 3.
    with pytest.raises(SystemExit, match="did not pass"):
        cp.carried_mode(state(None, status="failed"))
    with pytest.raises(SystemExit, match="short of 5 rounds"):
        cp.carried_mode(state({"spill-latency": "failed"}, valid=4))


def test_long_launch_profile_refuses_before_any_gpu_call(tmp_path):
    (tmp_path / "capture-state.json").write_text(
        json.dumps(state({"spill-latency": "failed", "dram-bw": "failed"}))
    )
    with pytest.raises(SystemExit, match="rule 3 failed in"):
        cp.long_launch_profile(Path("vgpu_bench"), 0, tmp_path, SOURCE)
    (tmp_path / "capture-state.json").write_text(json.dumps(state({"spill-latency": "failed"})))
    with pytest.raises(SystemExit, match="give --carry-from"):
        cp.long_launch_profile(Path("vgpu_bench"), 0, tmp_path)
    (tmp_path / "capture-state.json").write_text(json.dumps(state({}, status="passed")))
    with pytest.raises(SystemExit, match="only failure is rule 3"):
        cp.long_launch_profile(Path("vgpu_bench"), 0, tmp_path, SOURCE)


def carried_profile():
    """A stand-in for the new profile: the source with every spill value changed, then the
    spill field carried over from the source."""
    profile = json.loads(SOURCE.read_text())
    for v in cp.measured_values(profile["measured"]["memory"][FIELD]):
        v["value"] += 1.0
    profile["provenance"]["capture_protocol"] = {
        "minimum_launch_ns": 1e6,
        "freeze_manifest_sha256": "0" * 64,
        "rounds": [],
        "rule_three": {"dram-bw": {"status": "passed"}, "spill-latency": {"status": "failed"}},
    }
    record = cp.carry_over(profile, "spill-latency", SOURCE)
    profile["provenance"]["capture_protocol"]["carried_over"] = [record]
    return profile, record


def test_carried_fields_equal_the_source_and_every_value_is_marked():
    source = json.loads(SOURCE.read_text())
    profile, record = carried_profile()
    sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    revision = source["provenance"]["profile_revision"]
    assert record == {
        "mode": "spill-latency",
        "fields": [f"memory.{FIELD}"],
        "from_profile_revision": revision,
        "from_profile_sha256": sha,
    }
    mark = {
        "reason": "rule_three_failed",
        "mode": "spill-latency",
        "from_profile_revision": revision,
        "from_profile_sha256": sha,
    }
    carried = profile["measured"]["memory"][FIELD]
    values = cp.measured_values(carried)
    assert len(values) == 3 * len(carried)
    assert all(v.pop("provisional") == mark for v in values)
    assert json.dumps(carried, sort_keys=True) == json.dumps(
        source["measured"]["memory"][FIELD], sort_keys=True
    )
    cp.validate_profile(carried_profile()[0])


def test_carry_over_refuses_another_device_or_an_already_carried_source(tmp_path):
    other = json.loads(SOURCE.read_text())
    other["identity"]["sku"] = "NVIDIA RTX 4070"
    path = tmp_path / "other.json"
    path.write_text(json.dumps(other))
    with pytest.raises(SystemExit, match="is a profile of"):
        cp.carry_over(json.loads(SOURCE.read_text()), "spill-latency", path)
    path.write_text(json.dumps(carried_profile()[0]))
    with pytest.raises(SystemExit, match="no measured values to carry"):
        cp.carry_over(json.loads(SOURCE.read_text()), "spill-latency", path)


def test_check_profiles_skips_carried_launches_and_requires_their_records():
    check = load("check_profiles")
    profile, _ = carried_profile()
    assert check.carried_errors(profile) == []
    assert not [e for e in check.launch_errors(profile) if FIELD in e]

    unmarked = copy.deepcopy(profile)
    unmarked["measured"]["memory"][FIELD][0]["iter_time_ns"].pop("provisional")
    assert any("not marked" in e for e in check.carried_errors(unmarked))

    passed = copy.deepcopy(profile)
    passed["provenance"]["capture_protocol"]["rule_three"]["spill-latency"]["status"] = "passed"
    assert any("rule 3 failed in []" in e for e in check.carried_errors(passed))

    stray = copy.deepcopy(profile)
    stray["provenance"]["capture_protocol"].pop("carried_over")
    assert any("provisional outside carried_over" in e for e in check.carried_errors(stray))


def test_spill_decision_does_not_run_on_carried_spill_fields(tmp_path):
    sd = load("spill_decision")
    profile, _ = carried_profile()
    raw = tmp_path / "merged-spill-latency.json"
    raw.write_text(json.dumps({"launch_cost_by_frame": [{}]}))
    assert sd.eligible(profile, raw) == (
        "spill fields carried over from another profile, marked provisional"
    )
    # Even a rule 3 result rewritten to passed does not make carried values eligible.
    profile["provenance"]["capture_protocol"]["rule_three"]["spill-latency"]["status"] = "passed"
    assert "carried over" in sd.eligible(profile, raw)


def test_recheck_marks_a_mechanism_on_carried_values_as_not_rechecked():
    rm = load("recheck_mechanisms")
    profile, record = carried_profile()
    spill = next(m for m in rm.MECHANISMS if m["name"] == "spill_l1_boundary")
    result = rm.check(spill, profile, json.dumps(profile))
    assert result["carried_over"]["from_profile_sha256"] == record["from_profile_sha256"]
    table = rm.markdown({"profiles": {"gb10": {"mechanisms": {"spill_l1_boundary": result}}}})
    assert "not rechecked" in table
    lsu = next(m for m in rm.MECHANISMS if m["name"] == "lsu_pipe")
    assert "carried_over" not in rm.check(lsu, profile, json.dumps(profile))
