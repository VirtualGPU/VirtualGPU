"""The mechanism re-check: its synthetic KernelIR routes bytes as intended and its numbers hold."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "recheck_mechanisms", ROOT / "scripts/recheck_mechanisms.py"
)
recheck = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recheck)
PROFILE = (ROOT / "profiles/gb10.json").read_text()


def predict(kernel, cache):
    r = recheck._core.predict(json.dumps(kernel), PROFILE, l2_state=cache, draws=0, seed=17)
    assert r["status"] == "ok", r
    return r


def test_a_stream_reads_dram_and_a_resident_stage_hits_l2():
    stream = predict(
        recheck.kernel(grid=48, warps=4, steps=256, load=recheck.stream(48, 4096)), "cold"
    )
    assert stream["dram_bytes"] == stream["load_bytes"] > 0
    warm = predict(recheck.kernel(grid=48, warps=4, steps=256, load=recheck.resident(4096)), "warm")
    assert warm["l2_hit_bytes"] == warm["load_bytes"] > 0


def test_per_step_time_drops_the_launch_cost():
    k = [
        recheck.kernel(grid=48, warps=4, steps=n, load=recheck.resident(4096))
        for n in recheck.LENGTHS
    ]
    (step,) = recheck.per_step(k, PROFILE, "warm")
    assert 0 < step < predict(k[0], "warm")["time_ns"] / recheck.LENGTHS[0]


def test_every_mechanism_runs_on_both_reference_profiles():
    for name in ("gb10", "rtx4070"):
        text = (ROOT / f"profiles/{name}.json").read_text()
        profile = json.loads(text)
        for mech in recheck.MECHANISMS:
            r = recheck.check(mech, profile, text)
            if r is None:
                continue
            assert r["points"] == len(r["rows"]) > 0
            assert all(row["predicted"] > 0 for row in r["rows"]), mech["name"]
            assert mech["name"] in recheck.EXPECTED


def test_check_passes_on_the_baseline_and_fails_on_a_regression():
    baseline = json.loads(
        (ROOT / "docs/results/analysis/v0.3-mechanism-recheck-baseline.json").read_text()
    )
    recheck.check_core_revision(baseline, recheck._core.build_info())
    profiles = [(n, ROOT / b["path"]) for n, b in baseline["profiles"].items()]
    result = recheck.run(profiles)
    lines, failures = recheck.compare(baseline, result, 0.005)
    assert lines and not failures
    worse = json.loads(json.dumps(result))
    worse["profiles"]["gb10"]["mechanisms"]["miss_merging"]["mape"] += 0.01
    del worse["profiles"]["rtx4070"]["mechanisms"]["lsu_pipe"]
    _, failures = recheck.compare(baseline, worse, 0.005)
    assert any("gb10 miss_merging: mape" in f for f in failures)
    assert any("rtx4070 lsu_pipe: points" in f for f in failures)


def test_check_core_revision_tracks_compiled_sources_across_documentation_commits(tmp_path):
    def git(*args):
        return recheck.git_output(tmp_path, *args)

    def commit():
        git("add", ".")
        git(
            "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "test"
        )
        return git("rev-parse", "HEAD").decode().strip()

    def build(rev):
        return {"commit": rev, "dirty": False, "result_class": "formal"}

    git("init", "-q")
    cargo = tmp_path / "Cargo.lock"
    cargo.write_text("old")
    original = commit()
    baseline = {
        "core": build(original),
        "core_source_sha256": recheck.compiled_source_sha(tmp_path, original),
    }
    (tmp_path / "README.md").write_text("docs only")
    docs = commit()
    recheck.check_core_revision(baseline, build(docs), tmp_path)
    with pytest.raises(ValueError, match="differs from checkout"):
        recheck.check_core_revision(baseline, build(original), tmp_path)
    with pytest.raises(ValueError, match="lacks a clean core build or compiled-source SHA"):
        recheck.check_core_revision({"core": build(original)}, build(docs), tmp_path)
    with pytest.raises(ValueError, match="clean formal build"):
        recheck.check_core_revision(baseline, build(docs) | {"dirty": True}, tmp_path)

    cargo.write_text("new")
    with pytest.raises(ValueError, match="uncommitted compiled-source changes"):
        recheck.check_core_revision(baseline, build(docs), tmp_path)
    changed = commit()
    with pytest.raises(ValueError, match="Baseline core SHA .* differs from current"):
        recheck.check_core_revision(baseline, build(changed), tmp_path)

    for revision in (docs, changed):
        git("checkout", "-q", revision)
        shallow = tmp_path / f"shallow-{revision}"
        subprocess.run(
            ["git", "clone", "-q", "--depth=1", tmp_path.as_uri(), str(shallow)], check=True
        )
        if revision == docs:
            recheck.check_core_revision(baseline, build(docs), shallow)
        else:
            with pytest.raises(ValueError, match="Baseline core SHA .* differs from current"):
                recheck.check_core_revision(baseline, build(changed), shallow)
