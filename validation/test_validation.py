"""CPU-only checks for the manifest, baseline, and scoring."""

import importlib.util
import json
import math
import os
from pathlib import Path
import stat
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from virtualgpu.frontend.compiler import ARTIFACT_HASH_VERSION

from validation import h5_candidates, h5_dram_form, v02_score
from validation.baseline import gemm_work, latency_ns, predict
from validation.experiment import core_records, development_gates, load_case, validate_core_builds
from validation.holdout import read_verified_report, require_release, seal_report, sha256
from validation.neusight import TileFeatures, fit, leave_one_case_out
from validation.phase2_capture import SECOND_NS, SparkSession, thermal_stability, thermal_windows
from validation.posthoc import published_seal, read_posthoc_reports, read_sealed_bytes
from validation.provenance import read_core_revision, require_formal_build
from validation.score import check_artifact, kendall_tau_b, median_ci95, observed_pair, predicted_pair, score
from validation.suite import development_manifest, manifest


class ManifestTest(unittest.TestCase):
    def test_frozen_manifest(self):
        frozen = json.loads(Path(__file__).with_name("suite.json").read_text())
        self.assertEqual(frozen, manifest())
        configs = {entry["config_id"] for entry in frozen["configs"]}
        splits = {"development": set(), "holdout": set()}
        for case in frozen["cases"]:
            self.assertTrue(set(case["config_ids"]) <= configs)
            splits[case["split"]].add(tuple(case["shape"]))
        self.assertFalse(splits["development"] & splits["holdout"])
        self.assertEqual(len(frozen["cases"][0]["config_ids"]), 36)

    def test_materialized_development_boundaries(self):
        suite = development_manifest({"specs": {"sm_count": 46},
                                      "provenance": {"profile_revision": "test"}})
        self.assertTrue(all(case["split"] == "development" for case in suite["cases"]))
        boundaries = [case for case in suite["cases"] if case["family"] == "sm-round"]
        self.assertEqual([case["shape"][1] // 64 for case in boundaries], [45, 46, 47])
        self.assertTrue(all(len(case["config_ids"]) == 36 for case in boundaries))


class BaselineTest(unittest.TestCase):
    def test_wave_tail_rounds_up(self):
        inputs = dict(resident_ctas_per_sm=2, sm_count=4, flops_per_cta=10,
                      bytes_per_cta=20, flops_per_second=1e9, bytes_per_second=1e9,
                      launch_ns=50)
        self.assertEqual(latency_ns(cta_count=8, **inputs), 210)
        self.assertEqual(latency_ns(cta_count=9, **inputs), 370)

    def test_masked_gemm(self):
        count, flops, size = gemm_work((17, 65, 33), (16, 64, 32))
        self.assertEqual(count, 4)
        self.assertEqual(flops, 2 * 16 * 64 * 64)
        self.assertEqual(size, 2 * (17 * 33 * 2 + 33 * 65 * 2 + 17 * 65) / 4)

    def test_profile_contract(self):
        path = Path(__file__).parents[1] / "schemas/examples/profile_example.json"
        profile = json.loads(path.read_text())
        measurement = {
            "case_id": "cube", "shape": [4096, 4096, 4096], "gpu": {"arch": "sm_89"},
            "configs": {
                "a": {"status": "ok", "num_warps": 4, "registers_per_thread": 96,
                      "shared_bytes": 49152, "tile": [128, 128, 32], "mma_shapes": ["16x8x16"]},
                "b": {"status": "rejected", "reason": "fixture"},
            },
        }
        a, b = predict(measurement, profile)
        self.assertEqual(a["profile_revision"], profile["provenance"]["profile_revision"])
        self.assertGreater(a["nominal_latency_ns"], 0)
        self.assertEqual(b["status"], "failed")


class ScoreTest(unittest.TestCase):
    def test_core_draws_include_every_shared_id(self):
        row = {"status": "ok", "nominal_latency_ns": 100, "draws": {"0": 100, "1": 110}}
        data = {"cases": {"case": {"cold": {"cfg": row}}}, "perturbation_draws": 1}
        self.assertEqual(core_records(data, "case", "cold", {"cfg": {}}), {"cfg": row})
        row["draws"] = {"0": 100}
        with self.assertRaisesRegex(ValueError, "invalid shared draw set"):
            core_records(data, "case", "cold", {"cfg": {}})

    def test_development_gates_include_regret_non_inferiority(self):
        cache = {
            "core": {"median_tau_b": 0.70, "mean_top_5_regret": 0.01},
            "roofline_wave": {"median_tau_b": 0.61, "mean_top_5_regret": 0.02},
            "neusight_style": {"median_tau_b": 0.65, "mean_top_5_regret": 0.005},
        }
        self.assertTrue(development_gates({"cold": cache})["cold"]["pass"])
        cache["neusight_style"]["mean_top_5_regret"] = 0.004
        self.assertFalse(development_gates({"cold": cache})["cold"]["pass"])

    def test_artifact_gate(self):
        toolchain = {
            "compiler": "triton", "compiler_version": "3.9.0", "assembler": "ptxas",
            "assembler_version": "12.9", "target_sm_compiled": "sm_89",
        }
        ir = {
            "metadata": {
                "config": {},
                "toolchain": dict(toolchain, artifact_hash_version=ARTIFACT_HASH_VERSION,
                                  ptx_sha256="matching", ttgir_sha256="aot"),
            },
            "launch": {"grid": [8, 1, 1], "dynamic_smem_bytes": 0},
            "resources": {"static_smem_bytes": 4096, "registers_per_thread": 96},
        }
        jit = {
            "artifact_hash_version": ARTIFACT_HASH_VERSION, "ptx_sha256": "matching",
            "ttgir_sha256": "jit",
            "grid": [8], "shared_bytes": 4096, "registers_per_thread": 96,
            "toolchain": dict(toolchain, enable_fp_fusion=True, ptx_options=None,
                              disable_ptxas_opt=False),
        }
        self.assertEqual(check_artifact(jit, ir),
                         {"status": "matched", "differences": [], "ttgir_match": False})
        jit["artifact_hash_version"] = "1"
        ir["metadata"]["toolchain"]["artifact_hash_version"] = "1"
        self.assertEqual(check_artifact(jit, ir)["differences"],
                         ["artifact_hash_version", "aot_artifact_hash_version"])
        jit["artifact_hash_version"] = ARTIFACT_HASH_VERSION
        ir["metadata"]["toolchain"]["artifact_hash_version"] = ARTIFACT_HASH_VERSION
        jit["ptx_sha256"] = "different"
        jit["registers_per_thread"] = 100
        self.assertEqual(check_artifact(jit, ir)["differences"],
                         ["ptx_sha256", "registers_per_thread"])
        jit["toolchain"]["enable_fp_fusion"] = False
        self.assertIn("enable_fp_fusion", check_artifact(jit, ir)["differences"])

    def test_measurement_intervals_and_gaps(self):
        self.assertEqual(median_ci95([[100, 100], [110, 110]]), (105, 105))
        blocks = {("session", index): 100 for index in range(3)}
        result = score({"a": blocks, "b": dict.fromkeys(blocks, 103)}, {})
        self.assertEqual(result["configs"]["a"]["measured_median_ci95_ns"], (100, 100))
        self.assertEqual(result["configs"]["a"]["relative_ci95_half_width"], 0)
        self.assertAlmostEqual(result["configs"]["b"]["gap_to_measured_best"], 0.03)

    def test_nominal_baseline_pair_margin(self):
        blocks = {("session", index): 100 for index in range(3)}
        records = {key: {"status": "ok", "nominal_latency_ns": value}
                   for key, value in (("a", 100), ("b", 104), ("c", 106))}
        result = score(dict.fromkeys(records, blocks), records, nominal_pair_margin=0.05)
        self.assertEqual(result["pair_confusion"],
                         {"indistinguishable": {"indistinguishable": 2, "a_faster": 1}})

    def test_tau_b_with_ties(self):
        actual = {"a": 1, "b": 2, "c": 2}
        self.assertAlmostEqual(kendall_tau_b(actual, {"a": 1, "b": 2, "c": 3}), 2 / 6**0.5)
        self.assertEqual(kendall_tau_b(actual, {"a": 3, "b": 2, "c": 2}), -1)
        self.assertIsNone(kendall_tau_b(actual, {"a": 1}))

    def test_pair_rules(self):
        actual = {("session", block): 100 for block in range(8)}
        close = dict.fromkeys(actual, 103)
        faster = dict.fromkeys(actual, 90)
        self.assertEqual(observed_pair(actual, close, 100, 1)[0], "indistinguishable")
        self.assertEqual(observed_pair(actual, faster, 100, 1)[0], "b_faster")
        self.assertEqual(predicted_pair({"0": 4, "1": 2}, {"0": 3, "1": 3}), "indistinguishable")
        self.assertEqual(predicted_pair({"0": 3, "1": 2}, {"0": 3, "1": 3}), "indistinguishable")
        self.assertEqual(predicted_pair({"0": 4}, {"0": 3}), "b_faster")
        self.assertEqual(predicted_pair({"0": 4, "1": 4}, {"0": 3}), "unavailable")
        self.assertEqual(predicted_pair({"0": 100, "1": 100}, {"0": 106, "1": 104}), "indistinguishable")
        self.assertEqual(predicted_pair({"0": 100, "1": 100}, {"0": 105, "1": 110}), "a_faster")
        self.assertEqual(predicted_pair({"0": 100}, {"0": 106}, math.log(1.10)), "indistinguishable")
        self.assertEqual(predicted_pair({"0": 100}, {"0": 100}, 0), "indistinguishable")

    def test_shortlist_oracle_includes_abstentions(self):
        block = ("session", 0)
        timings = {"a": {block: 100}, "b": {block: 120}, "c": {block: 180}}
        records = {
            "a": {"status": "unsupported"},
            "b": {"status": "ok", "nominal_latency_ns": 40},
            "c": {"status": "ok", "nominal_latency_ns": 50},
        }
        result = score(timings, records, k=1, samples=10)
        self.assertEqual(result["shortlist"], ["b"])
        self.assertFalse(result["top_k_hit"])
        self.assertAlmostEqual(result["top_k_regret"], 0.2)
        self.assertEqual(result["kendall_tau_b"], 1)
        self.assertEqual(result["scored_configs"], 2)
        self.assertEqual(result["measured_configs"], 3)


class NeuSightTest(unittest.TestCase):
    def test_fit_uses_only_development_cases_other_than_scored_case(self):
        tile = TileFeatures((1.0, 2.0), 30.0, 3, 5.0)
        examples = [
            {"case_id": "a", "split": "development", "tile": tile, "measured_ns": 20.0},
            {"case_id": "b", "split": "development", "tile": tile, "measured_ns": 185.0},
        ]
        fits = leave_one_case_out(examples)
        self.assertAlmostEqual(fits["a"].predict_ns(tile), 185.0)
        self.assertAlmostEqual(fits["b"].predict_ns(tile), tile.kernel_floor_ns)
        with self.assertRaisesRegex(ValueError, "only development"):
            fit([dict(examples[0], split="holdout")])


class MeasurementTest(unittest.TestCase):
    def test_clean_cold_replacement_preserves_warm_labels(self):
        case = {"case_id": "shape", "shape": [1, 2, 3], "config_ids": ["cfg"]}
        report = {
            "case_id": "shape", "split": "development", "shape": [1, 2, 3],
            "manifest_sha256": "manifest", "session_id": "old",
            "gpu": {}, "software": {}, "kernel_sha256": "kernel",
            "configs": {"cfg": {"status": "ok"}},
        }
        protocol = {"clock_policy": "record_only", "timer": "graph", "l2_bytes": 4,
                    "eviction_bytes": 8, "blocks": 1}

        def sample(cache, ns):
            return dict(cache=cache, config_id="cfg", block=0, latency_ns=ns)

        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            cold = results / "replacement"
            aot = results / "aot" / "shape"
            cold.mkdir()
            aot.mkdir(parents=True)
            (aot / "cfg.json").write_text("{}")
            (results / "shape-session-1.json").write_text(json.dumps(dict(
                report, protocol=protocol, samples=[sample("warm", 100), sample("cold", 150)],
            )))
            (cold / "shape-session-1.json").write_text(json.dumps(dict(
                report, session_id="new",
                protocol=dict(protocol, cache_states=["cold"], cold_flush="read_cg",
                              eviction_bytes=16),
                samples=[sample("cold", 120)],
            )))
            with patch("validation.experiment.check_artifact", return_value={"status": "matched"}):
                with self.assertRaisesRegex(ValueError, "require read-only"):
                    load_case(case, results, "manifest", 1)
                data = load_case(case, results, "manifest", 1, cold)
        self.assertEqual(data.timings["warm"]["cfg"], {("old", 0): 100})
        self.assertEqual(data.timings["cold"]["cfg"], {("new", 0): 120})
        self.assertEqual(data.matched_config_sessions, 2)


class ThermalStabilityTest(unittest.TestCase):
    def sample(self, second, temp, clock, reasons):
        return {"monotonic_ns": second * SECOND_NS, "temperature_c": temp,
                "sm_mhz": clock, "clock_event_reasons": reasons}

    def test_window_boundaries_and_thresholds(self):
        rows = [self.sample(t, (75, 80)[t % 2], 2000, 0)
                for t in range(30)]
        rows += [self.sample(t, (76, 80)[t % 2], 2020, 0)
                 for t in range(30, 60)]
        rows.append(self.sample(60, 100, 3000, 4))
        earlier, later = thermal_windows(rows, 60 * SECOND_NS)
        self.assertEqual(earlier, rows[:30])
        self.assertEqual(later, rows[30:60])
        self.assertEqual(thermal_stability((earlier, later)), {
            "temperature_mean_c": [77.5, 78],
            "sm_clock_median_mhz": [2000, 2020],
            "samples_per_window": [30, 30],
        })
        for temperature, clock in ((79, 2020), (78, 2021), (78, 1979)):
            with self.subTest(temperature=temperature, clock=clock):
                second = [self.sample(t, temperature, clock, 0) for t in range(30, 60)]
                self.assertIsNone(thermal_stability((earlier, second)))

    def test_rejects_slowdown_or_missing_telemetry_in_either_window(self):
        clean = [self.sample(1, 78, 2500, 0)]
        for event in (4, 8, 32, 64, 128, {"unavailable": "Unsupported"}):
            with self.subTest(event=event):
                affected = [self.sample(2, 78, 2500, event)]
                self.assertIsNone(thermal_stability((clean, affected)))
                self.assertIsNone(thermal_stability((affected, clean)))
        self.assertIsNone(thermal_stability((clean, [self.sample(2, 78, None, 0)])))
        self.assertIsNone(thermal_stability((clean, [self.sample(2, None, 2500, 0)])))
        self.assertIsNone(thermal_stability((clean, [])))

    def test_timeout_keeps_all_three_telemetry_fields(self):
        rows = [self.sample(t, 78, 2500, 4) for t in range(1201)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeout.json"
            session = SimpleNamespace(soaks=[], thermal_trace=path)
            with patch("validation.phase2_capture.time.monotonic_ns",
                       side_effect=[0, 0, 1200 * SECOND_NS, 1200 * SECOND_NS]):
                with self.assertRaisesRegex(RuntimeError, "twenty minutes"):
                    SparkSession.stabilize(session, lambda: None, rows, True)
            self.assertEqual(session.soaks, [])
            payload = json.loads(path.read_bytes())["warmup_telemetry"]
            self.assertEqual(len(payload), 1200)
            self.assertEqual(payload[-1], {
                "elapsed_ns": 1199 * SECOND_NS, "temperature_c": 78,
                "sm_mhz": 2500, "clock_event_reasons": 4,
            })


class ProvenanceTest(unittest.TestCase):
    revision = "a" * 40

    def test_expected_revision_from_file_or_full_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "expected-revision.txt"
            path.write_text(self.revision + "\n")
            self.assertEqual(read_core_revision(str(path)), self.revision)
        self.assertEqual(read_core_revision(self.revision), self.revision)
        with self.assertRaises(ValueError):
            read_core_revision("abc")

    def test_all_formal_records_require_matching_clean_builds(self):
        clean = {"commit": self.revision, "dirty": False, "result_class": "formal"}
        require_formal_build(clean, self.revision, "candidate")
        rows = {"ok": {"status": "ok", "build": clean}, "failed": {"status": "failed", "build": clean}}
        bundle = {"cases": {"decode": {"warm": rows}}}
        validate_core_builds(bundle, self.revision)
        for bad in (None, dict(clean, dirty=True), dict(clean, result_class="diagnostic"),
                    dict(clean, commit="b" * 40)):
            with self.subTest(build=bad):
                rows["failed"]["build"] = bad
                with self.assertRaisesRegex(ValueError, "decode warm failed"):
                    validate_core_builds(bundle, self.revision)
        rows["failed"]["build"] = clean
        with self.assertRaisesRegex(ValueError, "Diagnostic prediction bundles"):
            validate_core_builds(dict(bundle, result_class="diagnostic"), self.revision)


class HoldoutSealTest(unittest.TestCase):
    def test_report_digest_verified_and_permissions_restored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.json"
            report = {"split": "holdout", "manifest_sha256": "fixture",
                      "timestamp_utc": "2026-01-01T00:00:00+00:00", "samples": []}
            receipt = seal_report(path, report)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0)
            audit = {"manifest_sha256": "fixture", "reports": {path.name: receipt}}
            self.assertEqual(read_verified_report(path, audit), report)
            receipt["sha256"] = "invalid"
            with self.assertRaisesRegex(ValueError, "digest does not match"):
                read_verified_report(path, audit)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0)

    def test_release_requires_tag_before_archive_read(self):
        revision = "a" * 40
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "archive.json"
            with patch("validation.holdout.subprocess.run", return_value=SimpleNamespace(
                returncode=1, stdout="",
            )):
                with self.assertRaisesRegex(ValueError, "phase1-final"):
                    require_release(revision, archive, "unused")
            archive.write_bytes(b"[]")
            with patch("validation.holdout.subprocess.run", return_value=SimpleNamespace(
                returncode=0, stdout=revision + "\n",
            )):
                with self.assertRaisesRegex(ValueError, "archive digest"):
                    require_release(revision, archive, "invalid")
                self.assertEqual(require_release(revision, archive, sha256(b"[]")), [])


class PostHocAccessTest(unittest.TestCase):
    def test_only_published_release_with_matching_seal_is_eligible(self):
        seal = b'{"reports": {}}'
        revision = "a" * 40
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "validation/holdout-seal.json"
            path.parent.mkdir()
            path.write_bytes(seal)
            okay = lambda data: SimpleNamespace(returncode=0, stdout=data)
            with patch("validation.posthoc.REPO_ROOT", root):
                with self.assertRaisesRegex(ValueError, "published Phase 1 and Phase 2"):
                    published_seal(path.with_name("attention-holdout-seal.json"))
                with patch("validation.posthoc.subprocess.run", side_effect=[
                    okay(revision.encode()), okay(seal), SimpleNamespace(returncode=1, stdout=b""),
                ]):
                    with self.assertRaisesRegex(ValueError, "published seal and report"):
                        published_seal(path)
                with patch("validation.posthoc.subprocess.run", side_effect=[
                    okay(revision.encode()), okay(seal), okay(b"unrelated report"),
                ]):
                    with self.assertRaisesRegex(ValueError, "does not match"):
                        published_seal(path)
                with patch("validation.posthoc.subprocess.run", side_effect=[
                    okay(revision.encode()), okay(seal), okay(f"`{sha256(seal)}`".encode()),
                ]):
                    self.assertEqual(published_seal(path)[2:], (revision, {"reports": {}}))

    def test_report_mode_is_restored_before_bytes_are_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.json"
            receipt = seal_report(path, {"timestamp_utc": "fixture"})
            fdopen = os.fdopen

            def check_mode(fd, *args, **kwargs):
                self.assertEqual(stat.S_IMODE(os.fstat(fd).st_mode), 0)
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0)
                return fdopen(fd, *args, **kwargs)

            with patch("validation.posthoc.os.fdopen", side_effect=check_mode):
                self.assertEqual(sha256(read_sealed_bytes(path)), receipt["sha256"])
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0)

    def test_gb10_digest_and_access_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = "secondary/sealed/fixture.json"
            path = root / name
            path.parent.mkdir(parents=True)
            (root / "validation").mkdir()
            report = {"timestamp_utc": "fixture", "gpu": {"arch": "sm_121"}, "cohort": "secondary"}
            receipt = seal_report(path, report)
            release = ("seal", "report", "a" * 40, {"reports": {name: receipt}})
            revision = "b" * 40
            with patch("validation.posthoc.REPO_ROOT", root):
                self.assertEqual(read_posthoc_reports([path], release, revision, "diagnosis", "hash"), [report])
                receipt["sha256"] = "invalid"
                with self.assertRaisesRegex(ValueError, "digest does not match"):
                    read_posthoc_reports([path], release, revision, "diagnosis", "hash")
            events = (root / "results/post-hoc-access.jsonl").read_text().splitlines()
            self.assertEqual(len(events), 2)
            self.assertEqual(json.loads(events[0])["prediction_revision"], revision)
            self.assertEqual(json.loads(events[0])["purpose"], "diagnosis")
            summary = (root / "validation/post-hoc-access-log.md").read_text().splitlines()
            self.assertEqual(len(summary), 2)
            for event, line in zip(events, summary, strict=True):
                self.assertIn(json.loads(event)["timestamp_utc"], line)
                for value in (revision, "diagnosis", "seal", name):
                    self.assertIn(value, line)
                self.assertNotIn("prediction_sha256", line)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0)


_V02_REVISION = "a" * 40
_V02_BUILD = {"commit": _V02_REVISION, "dirty": False, "result_class": "formal"}
_V02_UTC = "2026-11-26T01:48:34.277968+00:00"
_V02_DIGEST = "b" * 64


class V02ScoreTest(unittest.TestCase):
    def test_unpublished_tag_or_amendment_prevents_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(v02_score, "git", side_effect=[_V02_REVISION.encode(), _V02_REVISION.encode(), b""]):
                with self.assertRaisesRegex(ValueError, "published tag"):
                    v02_score.require_release(Path(tmp), "amendment")
            published = f"{_V02_REVISION}\trefs/tags/{v02_score.TAG}\n".encode()
            main = f"{_V02_REVISION} refs/heads/main".encode()
            calls = [_V02_REVISION.encode(), _V02_REVISION.encode(), published, _V02_REVISION.encode(),
                     main, _V02_REVISION.encode(), _V02_REVISION.encode(), b"",
                     b"unpublished evidence", b"different published evidence"]
            with patch.object(v02_score, "git", side_effect=calls), patch.object(v02_score.subprocess, "run"):
                with self.assertRaisesRegex(ValueError, "published on origin/main"):
                    v02_score.require_release(Path(tmp), "amendment")

    def test_matching_published_archive_and_evidence_are_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = json.dumps({name: {} for name in v02_score.COHORTS}).encode()
            metadata = {
                "build": _V02_BUILD, "result_class": "formal", "tag": v02_score.TAG, "tag_object": _V02_REVISION,
                "draws": v02_score.DRAWS, "seed": v02_score.SEED, "pair_margin": v02_score.PRACTICAL_MARGIN,
                "binary_sha256": _V02_DIGEST, "fit_sha256": _V02_DIGEST, "recorded_utc": _V02_UTC,
                "aot_index_sha256": sha256(index),
                "cohort_inputs": {name: {} for name in v02_score.COHORTS},
            }
            data = {**metadata, "cohorts": {name: {} for name in v02_score.COHORTS}}
            archive = json.dumps(data).encode()
            evidence = json.dumps({**metadata, "prediction_archive_sha256": sha256(archive)}).encode()
            (root / "predictions.json").write_bytes(archive)
            (root / "aot-index.json").write_bytes(index)
            (root / "prediction-evidence.json").write_bytes(evidence)
            calls = [_V02_REVISION.encode(), _V02_REVISION.encode(), f"{_V02_REVISION} refs/tags/{v02_score.TAG}".encode(),
                     _V02_REVISION.encode(), f"{_V02_REVISION} refs/heads/main".encode(),
                     _V02_REVISION.encode(), _V02_REVISION.encode(), b"", evidence, evidence]
            with patch.object(v02_score, "git", side_effect=calls), patch.object(v02_score.subprocess, "run"):
                revision, _, records, returned_index, _ = v02_score.require_release(root, "amendment")
            self.assertEqual(revision, _V02_REVISION)
            self.assertEqual(records, data)
            self.assertEqual(returned_index, json.loads(index))

    def test_stale_local_main_prevents_release_even_with_the_published_tag(self):
        ref = _V02_REVISION.encode()
        published_tag = f"{_V02_REVISION} refs/tags/{v02_score.TAG}".encode()
        published_main = f"{'b' * 40} refs/heads/main".encode()
        calls = [ref, ref, published_tag, ref, published_main]
        with patch.object(v02_score, "git", side_effect=calls) as git:
            with self.assertRaisesRegex(ValueError, "origin/main differs from published main"):
                v02_score.require_release(Path("unused"), "amendment")
        git.assert_called_with("ls-remote", "--exit-code", "origin", "refs/heads/main")

    def test_each_retry_marker_requires_history_and_full_linkage(self):
        slot = "example-session-2.json"
        for field in ("report_path", "repeat_of", "retry_receipt_sha256"):
            with self.subTest(field=field):
                seal = {"reports": {slot: {field: "fixture"}}}
                with self.assertRaisesRegex(ValueError, "failed report history"):
                    v02_score.sealed_retry_slots(seal)
                seal["failed_reports"] = {slot: [{}]}
                self.assertEqual(v02_score.sealed_retry_slots(seal), {slot})
                with self.assertRaisesRegex(ValueError, "incomplete retry linkage"):
                    v02_score.retry_receipts("rtx4070-gemm", {"case_id": "example"}, 2, slot,
                                            seal, Path("unused"), None)

    def test_retry_is_bound_to_the_failed_receipt_and_standalone_digest(self):
        slot = "example-session-2.json"
        failure = {"sha256": "c" * 64, "timestamp_utc": "2026-11-26T01:40:00Z",
                   "status": "protocol_failed", "complete_blocks": 0}
        replacement = {"sha256": _V02_DIGEST, "timestamp_utc": _V02_UTC, "status": "complete", "complete_blocks": 10}
        seal = {"source": _V02_BUILD, "manifest_sha256": _V02_DIGEST, "profile_sha256": _V02_DIGEST,
                "wave_boundary_sha256": _V02_DIGEST, "aot_sha256_by_case": {"example": _V02_DIGEST},
                "failed_reports": {slot: [{"receipt": failure, "report_path": "original.json"}]}}
        retry = {"case_id": "example", "session": 2, "order_seed": 102, "retry_attempt": 1,
                 "original_report": slot, "retry_of_sha256": failure["sha256"], "original_receipt": failure,
                 "retry_report": "retry.json", "retry_receipt": replacement, "capture_revision": _V02_REVISION,
                 "manifest_sha256": _V02_DIGEST, "profile_sha256": _V02_DIGEST, "wave_boundary_sha256": _V02_DIGEST,
                 "aot_sha256": _V02_DIGEST, "original_seal": v02_score.COHORTS["rtx4070-gemm"]}
        raw = json.dumps(retry).encode()
        seal["reports"] = {slot: {**replacement, "report_path": "retry.json", "order_seed": 102,
                                  "repeat_of": failure["sha256"],
                                  "retry_receipt_sha256": sha256(raw)}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = ("rtx4070-gemm", {"case_id": "example"}, 2, slot, seal, root)
            self.assertEqual(v02_score.retry_receipts(*args, raw),
                             (root / "retry.json", (root / "original.json", failure)))
            with self.assertRaisesRegex(ValueError, "receipt digest mismatch"):
                v02_score.retry_receipts(*args, raw + b" ")
            failure["status"] = "complete"
            with self.assertRaisesRegex(ValueError, "protocol failure"):
                v02_score.retry_receipts(*args, raw)

    def test_all_digests_precede_scoring_and_failed_report_is_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, failed = root / "valid.json", root / "failed.json"
            valid_receipt = seal_report(first, {"timestamp_utc": _V02_UTC})
            failed_receipt = seal_report(failed, {"timestamp_utc": _V02_UTC})
            data = {"cohort": {"case": {"reports": {2: (first, valid_receipt)},
                                        "failed": [(failed, failed_receipt)]}}}
            snapshots = v02_score.verify_reports(data)
            self.assertEqual(set(snapshots["cohort"]["case"]), {2})
            self.assertEqual(first.stat().st_mode & 0o777, 0)
            self.assertEqual(failed.stat().st_mode & 0o777, 0)
            failed_receipt["sha256"] = _V02_DIGEST
            with self.assertRaisesRegex(ValueError, "failed.json: report digest mismatch"):
                v02_score.verify_reports(data)

    def test_devices_are_gated_independently_and_missing_predictions_keep_oracle(self):
        def forecast(order):
            return {key: {"status": "ok", "nominal_latency_ns": value,
                          "draws": {str(i): value for i in range(33)}}
                    for key, value in zip(order, (1., 2., 3.), strict=True)}

        measured = {key: {("session", block): ns for block in range(10)}
                    for key, ns in (("a", 1.), ("b", 2.), ("c", 3.))}
        case = {"case_id": "example", "config_ids": ["a", "b", "c"]}
        good, reversed_ = forecast("abc"), forecast("cba")
        forecasts = {cache: {"core": good, "neusight_style": reversed_, "roofline_wave": reversed_}
                     for cache in v02_score.CACHES}
        weak = dict(good, a={"status": "unsupported"})
        failing = dict(forecasts, warm={**forecasts["warm"], "core": weak})
        collection = {name: {"example": {"case": case, "forecasts": forecasts, "failed": []}}
                      for name in v02_score.COHORTS}
        collection["gb10-attention"]["example"]["forecasts"] = failing
        snapshots = {name: {"example": {}} for name in v02_score.COHORTS}
        status_by_config = {key: {"1": "ok", "2": "ok", "3": "ok"} for key in measured}
        probes = {"1": {"median_ns": {"warm": 4., "cold": 8.}}}
        with patch.object(v02_score, "report_timings",
                          return_value=({cache: measured for cache in v02_score.CACHES}, status_by_config, probes)):
            cohorts, devices = v02_score.evaluate(collection, snapshots)
        self.assertEqual(devices, {"rtx4070": True, "gb10": False})
        core = cohorts["gb10-attention"]["cases"]["example"]["cache"]["warm"]["core"]
        self.assertEqual(core["top_k_regret"], 1.0)
        self.assertEqual(core["missing_or_invalid_predictions"], {"a": "unsupported"})
        self.assertEqual((core["measured_configs"], core["scored_configs"]), (3, 2))
        self.assertEqual(cohorts["gb10-attention"]["cases"]["example"]["cache_control_ns"], probes)

    def test_missing_eligible_shortlist_fails_each_shortlist_gate(self):
        measured = {"a": {("session", 0): 1.}, "b": {("session", 0): 2.}}
        forward = {"a": {"status": "ok", "nominal_latency_ns": 1., "draws": {"0": 1.}},
                   "b": {"status": "ok", "nominal_latency_ns": 2., "draws": {"0": 2.}}}
        reversed_ = {"a": {"status": "ok", "nominal_latency_ns": 2.},
                     "b": {"status": "ok", "nominal_latency_ns": 1.}}
        empty = {key: {"status": "unsupported"} for key in measured}
        timings = {cache: measured for cache in v02_score.CACHES}
        statuses = {key: {"1": "ok"} for key in measured}
        good = {cache: {"core": forward, "roofline_wave": reversed_, "neusight_style": reversed_}
                for cache in v02_score.CACHES}
        missing = dict(good, warm={**good["warm"], "core": empty})
        cases = {"good": v02_score.scored_case(timings, good, statuses),
                 "missing": v02_score.scored_case(timings, missing, statuses)}
        gates = v02_score.aggregate(cases)["gates"]
        self.assertEqual(gates["warm"]["cases_missing_core_shortlist"], ["missing"])
        self.assertFalse(gates["warm"]["absolute_regret_pass"])
        self.assertFalse(gates["warm"]["pass"])
        for baseline in gates["warm"]["baselines"].values():
            self.assertTrue(baseline["ranking_pass"])
            self.assertFalse(baseline["non_inferiority_pass"])
        self.assertTrue(gates["cold"]["pass"])

    def test_cache_read_probe_rejects_missing_invalid_or_unseparated_samples(self):
        values = {"warm": [8., 2., 2.], "cold": [3., 3., 3.]}
        self.assertEqual(v02_score.check_cache_control(values, "fixture"),
                         {"samples_ns": values, "median_ns": {"warm": 2., "cold": 3.}})
        for bad in (None, {}, {"warm": [2.]}, {"warm": [], "cold": [3.]},
                    {"warm": [float("inf")], "cold": [3.]},
                    {"warm": [2.], "cold": [float("nan")]},
                    {"warm": [2.], "cold": [0.]},
                    {"warm": [2.], "cold": [2.]},
                    {"warm": [3.], "cold": [2.]}):
            with self.subTest(values=bad), self.assertRaisesRegex(ValueError, "read probe"):
                v02_score.check_cache_control(bad, "fixture")

    def test_legacy_report_validates_all_blocks_and_lists_invalid_candidates(self):
        case = {"case_id": "example", "config_ids": ["ok", "bad", "absent"],
                "batch": 1, "query_heads": 4, "kv_heads": 2, "seq_len": 65, "head_dim": 64, "causal": True}
        capture = {"kernel_key": "kernel", "source": _V02_BUILD,
                   "configs": {"ok": {"status": "ok"}, "bad": {"status": "ok"}}}
        entry = {"case": case, "capture": capture, "capture_sha256": _V02_DIGEST,
                 "receipt_manifest": _V02_DIGEST, "reports": {}}
        reports = {}
        for order in (1, 2, 3):
            protocol = {"order_seed": 100 + order, "blocks": 10, "cache_states": list(v02_score.CACHES),
                        "timer": "cuda_graph_single_launch", "cold_flush": "read_cg",
                        "l2_bytes": 32 * 1024**2, "eviction_bytes": 128 * 1024**2}
            reports[order] = {
                "timestamp_utc": _V02_UTC, "session_id": str(order), "case_id": case["case_id"],
                "split": "holdout", "shape": [1, 4, 2, 65, 64, True], "manifest_sha256": _V02_DIGEST,
                "aot_sha256": _V02_DIGEST, "kernel_key": "kernel", "source": _V02_BUILD,
                "gpu": {"arch": "sm_89", "uuid": "fixture"}, "protocol": protocol,
                "cache_control_ns": {"warm": [2., 3.], "cold": [4., 5.]},
                "configs": {"ok": {"status": "ok"}, "bad": {"status": "launch_failed"}},
                "samples": [{"block": block, "cache": cache, "config_id": "ok", "latency_ns": 10.}
                            for block in range(10) for cache in v02_score.CACHES],
            }
            entry["reports"][order] = (None, {"timestamp_utc": _V02_UTC})

        def snapshots():
            return {order: json.dumps(record).encode() for order, record in reports.items()}

        with patch.object(v02_score, "check_artifact", return_value={"status": "matched", "ttgir_match": True}):
            timings, statuses, probes = v02_score.report_timings(
                "rtx4070-attention", "example", entry, snapshots(), set(), {},
            )
            self.assertEqual(len(timings["warm"]["ok"]), 30)
            self.assertEqual(probes["2"]["median_ns"], {"warm": 2.5, "cold": 4.5})
            self.assertEqual(set(statuses["bad"].values()), {"launch_failed"})
            self.assertEqual(set(statuses["absent"].values()), {"missing"})
            reports[2]["samples"].pop()
            with self.assertRaisesRegex(ValueError, "incomplete randomized blocks"):
                v02_score.report_timings("rtx4070-attention", "example", entry, snapshots(), set(), {})

            reports[2].pop("cache_control_ns")
            with self.assertRaisesRegex(ValueError, "example/2: missing warm or cold read probe"):
                v02_score.report_timings("rtx4070-attention", "example", entry, snapshots(), set(), {})

    def test_legacy_receipt_needs_digest_and_utc(self):
        v02_score.check_receipt({"sha256": _V02_DIGEST, "timestamp_utc": _V02_UTC}, "legacy", legacy=True)
        with self.assertRaisesRegex(ValueError, "complete 10-block receipt"):
            v02_score.check_receipt({"sha256": _V02_DIGEST, "timestamp_utc": _V02_UTC}, "new")
        with self.assertRaisesRegex(ValueError, "UTC timestamp"):
            v02_score.check_receipt({"sha256": _V02_DIGEST, "timestamp_utc": "2026-11-26T01:48:34"}, "legacy", legacy=True)


class NarrowLoadCandidatesTest(unittest.TestCase):
    def test_sectors_per_request(self):
        self.assertEqual(h5_candidates.sectors_per_request(2, 32, 0), 2)
        self.assertEqual(h5_candidates.sectors_per_request(2, 32, 2), 3)
        self.assertEqual(h5_candidates.sectors_per_request(2, 24, 0), 2)
        self.assertEqual(h5_candidates.sectors_per_request(16, 32, 16), 17)
        self.assertEqual(h5_candidates.sectors_per_request(1, 32, 1), 2)

    def test_points_match_the_benchmark(self):
        points = h5_candidates.gmem_narrow_points() + h5_candidates.sync_loop_narrow_points()
        keys = [h5_candidates.point_key(p) for p in points]
        self.assertEqual((len(points), len(set(keys))), (1290, 1290))
        for point in h5_candidates.CALIBRATIONS.values():
            self.assertIn(h5_candidates.point_key(point), keys)

    def test_geometry_fits_the_benchmark_buffer(self):
        specs = json.loads(h5_candidates.PROFILE.read_text())["specs"]
        for p in h5_candidates.gmem_narrow_points() + h5_candidates.sync_loop_narrow_points():
            g = h5_candidates.geometry(p, specs)
            rows = g["grid"] // p.get("share", 1) * g["rows_per_cta"]
            self.assertLessEqual(rows * g["pitch"], 1088 << 20)
            self.assertEqual(g["pitch"] % 128, 0 if p["aligned"] else p["width_bytes"])

    def test_calibration_inverts_the_grid(self):
        xs, ys = [0.1, 0.2, 0.4], [400.0, 200.0, 100.0]
        k, clamped = h5_candidates.solve(xs, ys, 150.0, True)
        self.assertFalse(clamped)
        self.assertAlmostEqual(h5_candidates.interpolate(xs, ys, k, True), 150.0)
        self.assertEqual(h5_candidates.solve(xs, ys, 50.0, True), (0.4, True))


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("cuda", "pynvml", "torch")),
    "the v0.2 capture modules import the CUDA driver bindings, pynvml and torch",
)
class V02ArchiveTest(unittest.TestCase):
    def test_frozen_frontend_runs_without_caches(self):
        from validation import v02_predict

        with patch.object(v02_predict, "extract_gemm_kernel_ir") as gemm:
            v02_predict.frozen_ir("gemm", {"shape": [256, 256, 256]},
                                  {"tile": [64, 64, 32], "group_m": 1, "num_warps": 4, "num_stages": 2}, "sm_89")
        self.assertIs(gemm.call_args.kwargs["use_cache"], False)
        spec = (None, (8, 8, 512, 64, True, 0.18, 64, 32), {"num_warps": 4, "num_stages": 2})
        with patch.object(v02_predict.attention, "launch_spec", return_value=spec), \
                patch.object(v02_predict, "extract_attention_kernel_ir") as attn:
            v02_predict.frozen_ir("attention", {"batch": 1}, {}, "sm_89")
        self.assertIs(attn.call_args.kwargs["use_cache"], False)

    def test_frozen_archive_imports_the_checkout(self):
        import virtualgpu

        from validation import v02_predict

        v02_predict.require_imports_from_checkout(virtualgpu)
        installed = SimpleNamespace(__name__="virtualgpu", __file__="/usr/lib/python3/site-packages/virtualgpu/__init__.py")
        with self.assertRaisesRegex(ValueError, "not this checkout"):
            v02_predict.require_imports_from_checkout(installed)

    def test_capture_refuses_uncommitted_profiles(self):
        import subprocess

        from validation import v02_capture

        real = subprocess.check_output

        def git(args, **kwargs):
            if "profiles/" in args:
                return b" M profiles/gb10.json\n"
            if args[1] == "show":  # the approved manifest: the checkout's own
                return (v02_capture.REPO_ROOT / args[2].split(":", 1)[1]).read_bytes()
            return real(args, **kwargs)

        clean = {"commit": "0" * 40, "python_dirty": False, "validation_dirty": False}
        with patch.object(v02_capture.subprocess, "run"), \
                patch.object(v02_capture.subprocess, "check_output", side_effect=git), \
                patch.object(v02_capture, "source_info", return_value=clean):
            for device in ("gb10", "rtx4070"):
                with self.subTest(device=device), self.assertRaisesRegex(ValueError, "clean capture checkout"):
                    v02_capture.inputs("gemm", device)

    def test_capture_receipt_counts_config_statuses(self):
        from validation.v02_capture import config_status_counts

        report = {"configs": {"a": {"status": "ok"}, "b": {"status": "artifact_mismatch"},
                              "c": {"status": "ok"}, "d": {"status": "aot_compile_failed"}}}
        self.assertEqual(config_status_counts(report),
                         {"aot_compile_failed": 1, "artifact_mismatch": 1, "ok": 2})


class DramNarrowFormTest(unittest.TestCase):
    def frozen(self):
        cal = h5_candidates.point_key(h5_dram_form.CALIBRATION)
        dram = h5_candidates.point_key(h5_dram_form.gmem_point(2, 4, "dram", True, 1, 4, 128 << 20))
        l2 = h5_candidates.point_key(h5_dram_form.gmem_point(2, 4, "l2", True, 1, 4))
        point = lambda t, s, n: {"t_core_ns": t, "dram_sector_bytes": s, "dram_requests": n}
        return {"calibration": h5_dram_form.CALIBRATION, "bw_strided_128_gbps": 200.0,
                "points": {cal: point(100.0, 20000.0, 400.0), dram: point(50.0, 4000.0, 300.0),
                           l2: point(80.0, 0.0, 0.0)}}, cal, dram, l2

    def test_points_are_new_and_include_the_calibration(self):
        points = h5_dram_form.gmem_points() + h5_dram_form.sync_points()
        keys = [h5_candidates.point_key(p) for p in points]
        self.assertEqual((len(keys), len(set(keys))), (325, 325))
        self.assertEqual(keys[0], h5_candidates.point_key(h5_dram_form.CALIBRATION))
        old = {h5_candidates.point_key(p) for p in h5_candidates.gmem_narrow_points()}
        for p in points[1:]:
            q = {k: v for k, v in p.items() if k != "dram_request_bytes"}
            if p["level"] == "dram" and p["bench"] == "gmem-narrow":
                self.assertNotIn(h5_candidates.point_key(q), old)

    def test_rate_comes_from_the_calibration_point(self):
        frozen, cal, dram, l2 = self.frozen()
        result = h5_dram_form.score(frozen, {cal: 200.0, dram: 150.0, l2: 80.0})
        self.assertTrue(result["identifiable"])
        self.assertAlmostEqual(result["rate_warp_requests_per_ns"], 2.0)
        self.assertAlmostEqual(result["groups"]["gmem_dram"]["form_errors"][dram], 0.0)
        self.assertTrue(result["groups"]["gmem_l2_regression"]["prediction_unchanged"])

    def test_calibration_below_the_floor_is_not_identifiable(self):
        frozen, cal, dram, l2 = self.frozen()
        result = h5_dram_form.score(frozen, {cal: 90.0, dram: 150.0, l2: 80.0})
        self.assertFalse(result["identifiable"])
        self.assertFalse(result["passed"])
