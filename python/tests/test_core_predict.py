import json
from pathlib import Path

import pytest
from virtualgpu import _core

ROOT = Path(__file__).resolve().parents[2]
IR = (ROOT / "schemas/examples/gemm_sm89.json").read_text()
PROFILE = (ROOT / "profiles/rtx4070.json").read_text()


def test_predict_example_gemm_on_rtx4070():
    r = _core.predict(IR, PROFILE)
    assert r["status"] == "ok"
    assert r["time_ns"] > r["makespan_ns"] > 0
    assert r["ctas_per_sm"] >= 1
    assert 0 < r["tail_efficiency"] <= 1


def test_predict_rejects_bad_l2_state():
    with pytest.raises(ValueError):
        _core.predict(IR, PROFILE, l2_state="lukewarm")


def test_predict_reports_cta_that_does_not_fit_as_unsupported():
    ir = json.loads(IR)
    ir["resources"]["static_smem_bytes"] = 131 * 1024
    r = _core.predict(json.dumps(ir), PROFILE)
    assert r["status"] == "unsupported"
    assert "TooMuchSharedMemory" in r["reason"]
    assert "time_ns" not in r


def test_schemas_are_embedded():
    for name in ("kernel_ir", "gpu_profile"):
        assert json.loads(_core.schema(name))["$schema"]


def test_draws_are_shared_and_include_the_nominal_draw():
    a = _core.predict(IR, PROFILE, draws=8, seed=3)
    b = _core.predict(IR, PROFILE, draws=8, seed=3)
    assert list(a["draws"]) == [str(i) for i in range(9)]
    assert a["draws"]["0"] == a["time_ns"]
    assert a["draws"] == b["draws"]
    assert len(set(a["draws"].values())) > 1
    c = _core.predict(IR, PROFILE, draws=8, seed=4)
    assert c["draws"]["0"] == a["draws"]["0"]
    assert c["draws"] != a["draws"]


def test_pair_verdict_uses_the_five_percent_band():
    assert _core.pair_verdict([90.0, 94.0], [100.0, 100.0]) == "a_faster"
    assert _core.pair_verdict([90.0, 96.0], [100.0, 100.0]) == "indistinguishable"
    with pytest.raises(ValueError):
        _core.pair_verdict([1.0], [1.0, 2.0])


def test_predictions_carry_the_build():
    info = _core.build_info()
    assert set(info) == {"commit", "dirty", "result_class"}
    assert info["result_class"] == ("diagnostic" if info["dirty"] else "formal")
    assert _core.predict(IR, PROFILE, draws=0)["build"] == info


def test_predict_batch_matches_predict_exactly():
    too_big = json.loads(IR)
    too_big["resources"]["static_smem_bytes"] = 131 * 1024
    irs = [IR, json.dumps(too_big), IR]
    batch = _core.predict_batch(irs, PROFILE, l2_state="warm", draws=4, seed=5)
    assert batch == [_core.predict(ir, PROFILE, l2_state="warm", draws=4, seed=5) for ir in irs]
    assert [r["status"] for r in batch] == ["ok", "unsupported", "ok"]


def test_predict_batch_names_a_malformed_variant():
    with pytest.raises(ValueError, match="variant 1"):
        _core.predict_batch([IR, "{}"], PROFILE)


def test_causal_attention_is_unvalidated_only_without_measured_pipes():
    attn = (ROOT / "schemas/examples/flash_attention_causal_gqa_sm89.json").read_text()
    r = _core.predict(attn, PROFILE, draws=2)
    assert r["status"] == "ok"
    assert "unvalidated_feature" not in r["flags"]
    profile = json.loads(PROFILE)
    del profile["measured"]["compute"]["shfl_issue_rate_per_sm_gips"]
    unmeasured = _core.predict(attn, json.dumps(profile), draws=2)
    assert "unvalidated_feature" in unmeasured["flags"]
    assert set(r["bound_fraction"]) >= {"sfu", "shfl"}
