"""Preregistered Ascend 910B3 sealed holdout: freeze predictions, capture timings, and score."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from validation import attention_baseline, baseline, neusight
from validation.pair_precision import summarize as pair_precision_summary
from validation.score import PRACTICAL_MARGIN, pair_observations
from validation.score import score as score_case

REPO_ROOT = Path(__file__).resolve().parent.parent
GEMM_MANIFEST = REPO_ROOT / "validation" / "ascend-910b3-gemm-holdout.json"
ATTN_MANIFEST = REPO_ROOT / "validation" / "ascend-910b3-attention-holdout.json"
FROZEN_FIT_PATH = REPO_ROOT / "validation" / "v02-frozen-fit.json"
PROFILE_PATH = REPO_ROOT / "profiles" / "ascend910b3.json"
DRAWS = 32
SEED = 17
CACHES = ("warm", "cold")
METHODS = ("core", "roofline_wave", "neusight_style")
SESSIONS = (1, 2, 3)
BLOCKS_PER_SESSION = 10


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_canonical_json(path: Path, data: object) -> str:
    raw = (json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return sha256_bytes(raw)


def _ascend_mma_peak_flops(profile: dict) -> float:
    specs = profile["specs"]
    rates = profile["measured"]["compute"]["mma_tflops_per_sm_by_warps"]
    key = "cube.mmad:f16:f16:f32:16x16x16"
    curve = rates.get(key)
    if not curve:
        raise ValueError(f"Missing Ascend FP16 MMA curve {key}")
    return specs["sm_count"] * max(pt["tflops_per_sm"]["value"] for pt in curve) * 1e12


def _ascend_gemm_roofline(shape: list[int], cfg: dict, profile: dict) -> float:
    specs = profile["specs"]
    measured = profile["measured"]
    count, flops, size = baseline.gemm_work(shape, cfg["tile"])
    return baseline.latency_ns(
        cta_count=count,
        resident_ctas_per_sm=1,
        sm_count=specs["sm_count"],
        flops_per_cta=flops,
        bytes_per_cta=size,
        flops_per_second=_ascend_mma_peak_flops(profile),
        bytes_per_second=measured["memory"]["dram_bw_read_gbps"]["value"] * 1e9,
        launch_ns=measured["overhead"]["launch_overhead_us"]["value"] * 1e3,
    )


def _ascend_gemm_tile_features(
    shape: list[int], cfg: dict, profile: dict, cache: str
) -> neusight.TileFeatures:
    specs = profile["specs"]
    measured = profile["measured"]
    sm_count = specs["sm_count"]
    capacity = sm_count * 1
    count, flops, requested = baseline.gemm_work(shape, cfg["tile"])
    waves = math.ceil(count / capacity)
    memory = measured["memory"]
    dram_bw = memory["dram_bw_read_gbps"]["value"] * 1e9
    l2_bw = memory["l2_bw_read_gbps"]["value"] * 1e9
    l2_capacity = specs["l2_size_bytes"]
    fp_rate = _ascend_mma_peak_flops(profile)
    m, n, k = shape
    working_set = 2 * (m * k + k * n + m * n)
    warm = cache == "warm"
    compulsory = max(working_set - l2_capacity, 0) if warm else working_set
    compute_ns = count * flops / fp_rate * 1e9
    l2_ns = count * requested / l2_bw * 1e9
    dram_ns = compulsory / dram_bw * 1e9
    wave_floor_ns = max(compute_ns, l2_ns, dram_ns) / waves
    bm, bn, bk = cfg["tile"]
    values = (
        math.log2(flops / 1024),
        math.log2(requested / 1024),
        math.log2(working_set / l2_capacity),
        count / (waves * capacity),
        math.log2(capacity / sm_count),
        math.log2(compute_ns / l2_ns),
        math.log2(bm),
        math.log2(bn),
        math.log2(bk),
        math.log2(cfg["num_warps"]),
        math.log2(cfg["num_stages"]),
        math.log2(cfg["group_m"]),
        float(warm),
    )
    launch_ns = measured["overhead"]["launch_overhead_us"]["value"] * 1e3
    return neusight.TileFeatures(values, wave_floor_ns, waves, launch_ns)


def _ascend_attn_roofline(case: dict, cfg: dict, profile: dict) -> float:
    specs = profile["specs"]
    measured = profile["measured"]
    work = attention_baseline.attention_work(case, cfg)
    return baseline.latency_ns(
        cta_count=work.ctas,
        resident_ctas_per_sm=1,
        sm_count=specs["sm_count"],
        flops_per_cta=work.flops_per_cta,
        bytes_per_cta=work.bytes_per_cta,
        flops_per_second=_ascend_mma_peak_flops(profile),
        bytes_per_second=measured["memory"]["dram_bw_read_gbps"]["value"] * 1e9,
        launch_ns=measured["overhead"]["launch_overhead_us"]["value"] * 1e3,
    )


def _ascend_attn_tile_features(
    case: dict, cfg: dict, profile: dict, cache: str
) -> neusight.TileFeatures:
    specs = profile["specs"]
    measured = profile["measured"]
    work = attention_baseline.attention_work(case, cfg)
    flops = _ascend_mma_peak_flops(profile)
    dram = measured["memory"]["dram_bw_read_gbps"]["value"] * 1e9
    l2 = measured["memory"]["l2_bw_read_gbps"]["value"] * 1e9
    launch = measured["overhead"]["launch_overhead_us"]["value"] * 1e3
    resident = 1
    capacity = specs["sm_count"] * resident
    waves = math.ceil(work.ctas / capacity)
    warm = cache == "warm"
    footprint = work.working_set_bytes
    l2_capacity = specs["l2_size_bytes"]
    compulsory = max(footprint - l2_capacity, 0) if warm else footprint
    compute_ns = work.ctas * work.flops_per_cta / flops * 1e9
    l2_ns = work.ctas * work.bytes_per_cta / l2 * 1e9
    dram_ns = compulsory / dram * 1e9
    features = (
        math.log2(work.flops_per_cta),
        math.log2(work.bytes_per_cta),
        math.log2(footprint / l2_capacity),
        work.ctas / (waves * capacity),
        math.log2(resident),
        math.log2(compute_ns / l2_ns),
        math.log2(cfg["block_m"]),
        math.log2(cfg["block_n"]),
        math.log2(case["head_dim"]),
        math.log2(case["query_heads"] / case["kv_heads"]),
        math.log2(cfg["num_warps"]),
        math.log2(cfg["num_stages"]),
        float(case["causal"]),
        float(warm),
    )
    return neusight.TileFeatures(features, max(compute_ns, l2_ns, dram_ns) / waves, waves, launch)


def load_frozen_fits() -> tuple[dict[str, neusight.EfficiencyFit], str]:
    raw = FROZEN_FIT_PATH.read_bytes()
    data = json.loads(raw)
    fits: dict[str, neusight.EfficiencyFit] = {}
    for family in ("gemm", "attention"):
        row = data["fits"][family]
        fits[family] = neusight.EfficiencyFit(
            np.asarray(row["center"], dtype=float),
            np.asarray(row["scale"], dtype=float),
            np.asarray(row["weights"], dtype=float),
            float(row["intercept"]),
        )
    return fits, sha256_bytes(raw)


def _get_triton_kernels():
    from validation.ascend_kernels import dense_gemm
    from validation.attention import fused_attention

    return dense_gemm, fused_attention


def cmd_predict(args: argparse.Namespace) -> None:
    """Freeze CPU-only core + baseline predictions BEFORE opening any holdout timings."""
    from virtualgpu import _core
    from virtualgpu.frontend.triton_ascend_attention import extract_ascend_attention_kernel_ir
    from virtualgpu.frontend.triton_ascend_gemm import extract_ascend_gemm_kernel_ir

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_raw = PROFILE_PATH.read_text(encoding="utf-8")
    profile = json.loads(profile_raw)
    profile_sha = sha256_bytes(profile_raw.encode("utf-8"))
    fits, fit_sha = load_frozen_fits()
    dense_gemm, fused_attention = _get_triton_kernels()

    manifests = {
        "gemm": (GEMM_MANIFEST, json.loads(GEMM_MANIFEST.read_text(encoding="utf-8"))),
        "attention": (ATTN_MANIFEST, json.loads(ATTN_MANIFEST.read_text(encoding="utf-8"))),
    }

    archive: dict = {
        "schema_version": 1,
        "recorded_utc": utc(),
        "core_build": _core.build_info(),
        "profile_revision": profile["provenance"]["profile_revision"],
        "profile_sha256": profile_sha,
        "fit_sha256": fit_sha,
        "draws": DRAWS,
        "seed": SEED,
        "pair_margin": PRACTICAL_MARGIN,
        "manifest_sha256": {fam: sha256_file(path) for fam, (path, _) in manifests.items()},
        "cohorts": {},
    }

    for family, (_path, manifest) in manifests.items():
        cfg_map = {c["config_id"]: c for c in manifest["configs"]}
        cohort_preds: dict = {}
        for case in manifest["cases"]:
            cid = case["case_id"]
            methods = {cache: {m: {} for m in METHODS} for cache in CACHES}
            ok_cnt = 0
            for key in case["config_ids"]:
                cfg = cfg_map[key]
                try:
                    if family == "gemm":
                        m, n, k = case["shape"]
                        bm, bn, bk = cfg["tile"]
                        ir = extract_ascend_gemm_kernel_ir(
                            dense_gemm,
                            M=m,
                            N=n,
                            K=k,
                            config={
                                "M": m,
                                "N": n,
                                "K": k,
                                "BM": bm,
                                "BN": bn,
                                "BK": bk,
                                "GROUP_M": cfg["group_m"],
                                "multibuffer": cfg["multibuffer"],
                            },
                            target_arch="ascend910b3",
                            use_cache=False,
                        )
                    else:
                        ir = extract_ascend_attention_kernel_ir(
                            fused_attention,
                            Z=case["batch"],
                            H=case["query_heads"],
                            H_kv=case["kv_heads"],
                            N_CTX=case["seq_len"],
                            D=case["head_dim"],
                            config={
                                "BM": cfg["block_m"],
                                "BN": cfg["block_n"],
                                "multibuffer": cfg["multibuffer"],
                            },
                            target_arch="ascend910b3",
                            causal=case["causal"],
                            use_cache=False,
                        )
                    ir_text = json.dumps(ir, sort_keys=True, separators=(",", ":"))
                    ir_sha = sha256_bytes(ir_text.encode("utf-8"))
                    tc = ir["metadata"]["toolchain"]
                    ext = ir["resources"]["vendor_extensions"]["ascend"]
                except Exception as exc:
                    fail_row = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
                    for cache in CACHES:
                        for m_name in METHODS:
                            methods[cache][m_name][key] = dict(fail_row)
                    continue

                for cache in CACHES:
                    pred = _core.predict(
                        ir_text, profile_raw, l2_state=cache, draws=DRAWS, seed=SEED
                    )
                    if pred["status"] == "ok":
                        pred["nominal_latency_ns"] = pred["time_ns"]
                    pred["ir_sha256"] = ir_sha
                    pred["hivm_sha256"] = tc.get("hivm_sha256")
                    pred["ub_bytes"] = ext.get("ub_bytes")
                    pred["l0c_bytes"] = ext.get("l0c_bytes")
                    methods[cache]["core"][key] = pred

                    if family == "gemm":
                        roof_ns = _ascend_gemm_roofline(case["shape"], cfg, profile)
                        tf = _ascend_gemm_tile_features(case["shape"], cfg, profile, cache)
                    else:
                        roof_ns = _ascend_attn_roofline(case, cfg, profile)
                        tf = _ascend_attn_tile_features(case, cfg, profile, cache)
                    methods[cache]["roofline_wave"][key] = {
                        "status": "ok",
                        "nominal_latency_ns": roof_ns,
                    }
                    methods[cache]["neusight_style"][key] = {
                        "status": "ok",
                        "nominal_latency_ns": fits[family].predict_ns(tf),
                    }
                if methods["cold"]["core"][key]["status"] == "ok":
                    ok_cnt += 1
            cohort_preds[cid] = methods
            print(f"[PREDICT] {family}/{cid}: {ok_cnt}/{len(case['config_ids'])} ok", flush=True)
        archive["cohorts"][family] = cohort_preds

    pred_path = out_dir / "predictions.json"
    pred_sha = write_canonical_json(pred_path, archive)
    freeze_receipt = {
        "schema_version": 1,
        "recorded_utc": archive["recorded_utc"],
        "core_build": archive["core_build"],
        "profile_revision": archive["profile_revision"],
        "profile_sha256": archive["profile_sha256"],
        "fit_sha256": archive["fit_sha256"],
        "manifest_sha256": archive["manifest_sha256"],
        "prediction_archive_sha256": pred_sha,
        "draws": DRAWS,
        "seed": SEED,
        "pair_margin": PRACTICAL_MARGIN,
    }
    freeze_path = Path(args.freeze_out)
    write_canonical_json(freeze_path, freeze_receipt)
    print(f"[FREEZE] wrote {pred_path} (sha256={pred_sha}) and {freeze_path}", flush=True)


def _npu_smi_telemetry() -> dict:
    try:
        out = subprocess.check_output(
            ["npu-smi", "info", "-t", "common", "-i", "0"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
        return {"status": 0, "stdout": out.strip()}
    except Exception as exc:
        return {"status": -1, "error": str(exc)}


def cmd_capture(args: argparse.Namespace) -> None:
    """Capture 3-session x 10-block randomized warm and clean-cold timings on Ascend 910B3."""
    import torch
    import torch.nn.functional as F
    import torch_npu  # noqa: F401
    import triton

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = Path(args.freeze)
    if not freeze_path.is_file():
        raise FileNotFoundError(f"Freeze receipt must exist before capture: {freeze_path}")
    freeze_data = json.loads(freeze_path.read_text(encoding="utf-8"))

    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    l2_bytes = int(profile["specs"]["l2_size_bytes"])
    evict_bytes = 2 * l2_bytes  # 384 MiB > 192 MiB L2

    # Clean read-only L2 eviction kernel (no dirty writes into the timed window).
    from validation.ascend_kernels import read_flush

    flush_block = 4096
    flush_blocks = triton.cdiv(evict_bytes // 4, flush_block)
    flush_buf = torch.ones((flush_blocks * flush_block,), dtype=torch.int32, device="npu:0")
    flush_sink = torch.empty((flush_blocks,), dtype=torch.int32, device="npu:0")

    def evict_l2() -> None:
        read_flush[(flush_blocks,)](flush_buf, flush_sink, flush_block)

    # Warm-vs-cold read probe (64 MiB buffer <= 192 MiB L2)
    probe_bytes = 64 * 1024 * 1024
    probe_blocks = triton.cdiv(probe_bytes // 4, flush_block)
    probe_buf = torch.ones((probe_blocks * flush_block,), dtype=torch.int32, device="npu:0")
    probe_sink = torch.empty((probe_blocks,), dtype=torch.int32, device="npu:0")
    ev0 = torch.npu.Event(enable_timing=True)
    ev1 = torch.npu.Event(enable_timing=True)

    def run_cache_probe() -> dict[str, list[float]]:
        def do_read() -> None:
            read_flush[(probe_blocks,)](probe_buf, probe_sink, flush_block)

        do_read()
        torch.npu.synchronize()
        warm_ns, cold_ns = [], []
        for _ in range(6):
            do_read()
            ev0.record()
            do_read()
            ev1.record()
            torch.npu.synchronize()
            warm_ns.append(ev0.elapsed_time(ev1) * 1e6)
            evict_l2()
            ev0.record()
            do_read()
            ev1.record()
            torch.npu.synchronize()
            cold_ns.append(ev0.elapsed_time(ev1) * 1e6)
        return {"warm_ns": warm_ns, "cold_ns": cold_ns}

    dense_gemm, fused_attention = _get_triton_kernels()
    manifests = {
        "gemm": (GEMM_MANIFEST, json.loads(GEMM_MANIFEST.read_text(encoding="utf-8"))),
        "attention": (ATTN_MANIFEST, json.loads(ATTN_MANIFEST.read_text(encoding="utf-8"))),
    }

    seal: dict = {
        "schema_version": 1,
        "device": "ascend910b3",
        "freeze_sha256": sha256_file(freeze_path),
        "profile_sha256": freeze_data["profile_sha256"],
        "manifest_sha256": freeze_data["manifest_sha256"],
        "software": {
            "triton": triton.__version__,
            "torch": torch.__version__,
            "python": sys.version.split()[0],
        },
        "reports": {},
    }

    evict_l2()
    torch.npu.synchronize()

    for family, (_mpath, manifest) in manifests.items():
        if args.family and args.family != family:
            continue
        cfg_map = {c["config_id"]: c for c in manifest["configs"]}
        for case in manifest["cases"]:
            cid = case["case_id"]
            torch.manual_seed(manifest["input_seed"])
            if family == "gemm":
                m, n, k = case["shape"]
                a = torch.randn((m, k), dtype=torch.float16, device="npu:0") * 0.1
                b = torch.randn((k, n), dtype=torch.float16, device="npu:0") * 0.1
                c = torch.empty((m, n), dtype=torch.float16, device="npu:0")
                ref = torch.matmul(a.float(), b.float()).half()
            else:
                z = case["batch"]
                hq = case["query_heads"]
                hk = case["kv_heads"]
                seq = case["seq_len"]
                d = case["head_dim"]
                causal = case["causal"]
                scale_log2 = math.log2(math.e) / math.sqrt(d)
                q = torch.randn((z, hq, seq, d), dtype=torch.float16, device="npu:0") * 0.2
                k_t = torch.randn((z, hk, seq, d), dtype=torch.float16, device="npu:0") * 0.2
                v_t = torch.randn((z, hk, seq, d), dtype=torch.float16, device="npu:0") * 0.2
                o = torch.empty_like(q)
                lse = torch.empty((z, hq, seq), dtype=torch.float32, device="npu:0")
                rep = hq // hk
                k_rep = k_t.repeat_interleave(rep, dim=1) if rep > 1 else k_t
                v_rep = v_t.repeat_interleave(rep, dim=1) if rep > 1 else v_t
                ref = F.scaled_dot_product_attention(
                    q.float(), k_rep.float(), v_rep.float(), is_causal=causal
                ).half()

            # Build runnable config closures & verify correctness once before sessions.
            runnables = {}
            cfg_statuses = {}
            for key in case["config_ids"]:
                cfg = cfg_map[key]
                try:
                    if family == "gemm":
                        bm, bn, bk = cfg["tile"]
                        grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
                        mb = cfg["multibuffer"]
                        gm = cfg["group_m"]

                        def run_fn(
                            _grid=grid,
                            _a=a,
                            _b=b,
                            _c=c,
                            _m=m,
                            _n=n,
                            _k=k,
                            _bm=bm,
                            _bn=bn,
                            _bk=bk,
                            _gm=gm,
                            _mb=mb,
                        ):
                            return dense_gemm[_grid](
                                _a, _b, _c, _m, _n, _k, _bm, _bn, _bk, _gm, multibuffer=_mb
                            )

                        c.zero_()
                        run_fn()
                        torch.npu.synchronize()
                        max_err = float((c - ref).abs().max().item())
                        if not math.isfinite(max_err) or max_err > 0.25:
                            raise RuntimeError(f"accuracy check failed: max_err={max_err}")
                    else:
                        bm, bn = cfg["block_m"], cfg["block_n"]
                        mb = cfg["multibuffer"]
                        grid = (triton.cdiv(seq, bm), z * hq, 1)

                        def run_fn(
                            _grid=grid,
                            _q=q,
                            _kt=k_t,
                            _vt=v_t,
                            _o=o,
                            _lse=lse,
                            _hq=hq,
                            _hk=hk,
                            _seq=seq,
                            _d=d,
                            _c=causal,
                            _sc=scale_log2,
                            _bm=bm,
                            _bn=bn,
                            _mb=mb,
                        ):
                            return fused_attention[_grid](
                                _q,
                                _kt,
                                _vt,
                                _o,
                                _lse,
                                _hq,
                                _hk,
                                _seq,
                                _d,
                                _c,
                                _sc,
                                _bm,
                                _bn,
                                multibuffer=_mb,
                            )

                        o.zero_()
                        run_fn()
                        torch.npu.synchronize()
                        max_err = float((o - ref).abs().max().item())
                        if not math.isfinite(max_err) or max_err > 0.15:
                            raise RuntimeError(f"accuracy check failed: max_err={max_err}")

                    # Warmup compile & steady state
                    for _ in range(2):
                        run_fn()
                    torch.npu.synchronize()
                    runnables[key] = run_fn
                    cfg_statuses[key] = {"status": "ok", "max_abs_err": round(max_err, 6)}
                except Exception as exc:
                    cfg_statuses[key] = {
                        "status": "compile_or_run_failed",
                        "reason": f"{type(exc).__name__}: {str(exc)[:200]}",
                    }

            active_keys = sorted(runnables.keys())
            print(
                f"[CAPTURE] {family}/{cid}: {len(active_keys)}/{len(case['config_ids'])} runnable",
                flush=True,
            )

            for session in SESSIONS:
                order_seed = 100 + session
                rng = random.Random(order_seed)
                tel_before = _npu_smi_telemetry()
                probe = run_cache_probe()
                samples = []
                for block in range(BLOCKS_PER_SESSION):
                    # Interleave (cache, config) in randomized order per block
                    trials = [(cache, key) for cache in CACHES for key in active_keys]
                    rng.shuffle(trials)
                    for cache, key in trials:
                        fn = runnables[key]
                        if cache == "warm":
                            fn()
                        else:
                            evict_l2()
                        ev0.record()
                        fn()
                        ev1.record()
                        torch.npu.synchronize()
                        ns = ev0.elapsed_time(ev1) * 1e6
                        samples.append(
                            {
                                "cache": cache,
                                "config_id": key,
                                "block": block,
                                "latency_ns": ns,
                            }
                        )
                tel_after = _npu_smi_telemetry()
                slot = f"{family}/{cid}-session-{session}.json"
                report = {
                    "schema_version": 1,
                    "family": family,
                    "case_id": cid,
                    "split": "holdout",
                    "session_id": session,
                    "order_seed": order_seed,
                    "manifest_sha256": freeze_data["manifest_sha256"][family],
                    "profile_sha256": freeze_data["profile_sha256"],
                    "protocol": {
                        "timer": "torch.npu.Event",
                        "cache_states": list(CACHES),
                        "cold_flush": "read_only_triton_sum",
                        "l2_bytes": l2_bytes,
                        "eviction_bytes": evict_bytes,
                        "blocks": BLOCKS_PER_SESSION,
                    },
                    "cache_probe": probe,
                    "telemetry": {"before": tel_before, "after": tel_after},
                    "configs": cfg_statuses,
                    "samples": samples,
                }
                rep_path = out_dir / "sealed" / slot
                rep_sha = write_canonical_json(rep_path, report)
                seal["reports"][slot] = {
                    "sha256": rep_sha,
                    "timestamp_utc": utc(),
                    "status": "complete",
                    "complete_blocks": BLOCKS_PER_SESSION,
                    "cold_probe_ratio": round(
                        statistics.median(probe["cold_ns"]) / statistics.median(probe["warm_ns"]),
                        3,
                    ),
                }
            # Release tensors before next case
            del ref
            torch.npu.empty_cache()

    seal_path = Path(args.seal_out)
    write_canonical_json(seal_path, seal)
    print(f"[SEAL] wrote {seal_path} ({len(seal['reports'])} reports)", flush=True)


def cmd_score(args: argparse.Namespace) -> None:
    """Verify freeze + seal digests and score all four Ascend 910B3 holdout cohorts once."""
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))
    seal = json.loads(Path(args.seal).read_text(encoding="utf-8"))
    pred_path = Path(args.predictions)
    if sha256_file(pred_path) != freeze["prediction_archive_sha256"]:
        raise ValueError("Prediction archive SHA-256 does not match freeze receipt")
    if seal["freeze_sha256"] != sha256_file(Path(args.freeze)):
        raise ValueError("Seal freeze_sha256 does not match freeze receipt")
    if sha256_file(PROFILE_PATH) != freeze["profile_sha256"]:
        raise ValueError("Profile SHA-256 changed since freeze")

    predictions = json.loads(pred_path.read_text(encoding="utf-8"))
    sealed_root = Path(args.sealed_dir)

    manifests = {
        "gemm": json.loads(GEMM_MANIFEST.read_text(encoding="utf-8")),
        "attention": json.loads(ATTN_MANIFEST.read_text(encoding="utf-8")),
    }
    for fam, path in (("gemm", GEMM_MANIFEST), ("attention", ATTN_MANIFEST)):
        if sha256_file(path) != freeze["manifest_sha256"][fam]:
            raise ValueError(f"{fam} manifest SHA-256 changed since freeze")

    results: dict = {
        "schema_version": 1,
        "scored_utc": utc(),
        "freeze_sha256": sha256_file(Path(args.freeze)),
        "seal_sha256": sha256_file(Path(args.seal)),
        "profile_revision": freeze["profile_revision"],
        "cohorts": {},
    }

    all_pass = True
    for family in ("gemm", "attention"):
        manifest = manifests[family]
        for cache in CACHES:
            cohort_key = f"ascend910b3-{family}-{cache}"
            per_case: dict = {}
            tau_by_m: dict[str, list[float]] = {m: [] for m in METHODS}
            reg_by_m: dict[str, list[float]] = {m: [] for m in METHODS}
            tau_gains: dict[str, list[float]] = {"roofline_wave": [], "neusight_style": []}
            pair_conf: dict[str, dict] = {m: {} for m in METHODS}
            med_abs_err_core: list[float] = []
            probe_ratios: list[float] = []

            for case in manifest["cases"]:
                cid = case["case_id"]
                timings: dict[str, dict[tuple[int, int], float]] = {}
                for session in SESSIONS:
                    slot = f"{family}/{cid}-session-{session}.json"
                    rep_path = sealed_root / slot
                    if sha256_file(rep_path) != seal["reports"][slot]["sha256"]:
                        raise ValueError(f"Sealed report digest mismatch: {slot}")
                    rep = json.loads(rep_path.read_text(encoding="utf-8"))
                    probe_ratios.append(seal["reports"][slot]["cold_probe_ratio"])
                    for s in rep["samples"]:
                        if s["cache"] == cache:
                            timings.setdefault(s["config_id"], {})[session, s["block"]] = s[
                                "latency_ns"
                            ]

                obs = pair_observations(timings, samples=1000, seed=SEED)
                case_scores = {}
                for m_name in METHODS:
                    recs = predictions["cohorts"][family][cid][cache][m_name]
                    margin = None if m_name == "core" else PRACTICAL_MARGIN
                    sc = score_case(
                        timings,
                        recs,
                        k=5,
                        samples=1000,
                        seed=SEED,
                        nominal_pair_margin=margin,
                        observations=obs,
                    )
                    case_scores[m_name] = sc
                    if sc["kendall_tau_b"] is not None:
                        tau_by_m[m_name].append(sc["kendall_tau_b"])
                    if sc["top_k_regret"] is not None:
                        reg_by_m[m_name].append(sc["top_k_regret"])
                    for o_lbl, row in sc["pair_confusion"].items():
                        pair_conf[m_name].setdefault(o_lbl, Counter()).update(row)

                for base_m in ("roofline_wave", "neusight_style"):
                    t_c = case_scores["core"]["kendall_tau_b"]
                    t_b = case_scores[base_m]["kendall_tau_b"]
                    if t_c is not None and t_b is not None:
                        tau_gains[base_m].append(t_c - t_b)

                errs = [
                    abs(row["nominal_latency_ns"] - row["measured_median_ns"])
                    / row["measured_median_ns"]
                    for row in case_scores["core"]["configs"].values()
                    if row.get("nominal_latency_ns") is not None and row.get("measured_median_ns")
                ]
                if errs:
                    med_abs_err_core.append(statistics.median(errs))

                per_case[cid] = {
                    "eligible_configs": len(timings),
                    "core_tau_b": case_scores["core"]["kendall_tau_b"],
                    "roofline_tau_b": case_scores["roofline_wave"]["kendall_tau_b"],
                    "neusight_tau_b": case_scores["neusight_style"]["kendall_tau_b"],
                    "core_top5_regret": case_scores["core"]["top_k_regret"],
                    "roofline_top5_regret": case_scores["roofline_wave"]["top_k_regret"],
                    "neusight_top5_regret": case_scores["neusight_style"]["top_k_regret"],
                    "core_med_abs_err": statistics.median(errs) if errs else None,
                }

            med_tau = {m: statistics.median(tau_by_m[m]) for m in METHODS}
            mean_reg = {m: statistics.mean(reg_by_m[m]) for m in METHODS}
            pair_prec = {
                m: pair_precision_summary({k: dict(v) for k, v in pair_conf[m].items()})
                for m in METHODS
            }
            pass_tau = all(
                med_tau["core"] - med_tau[b] >= 0.05 for b in ("roofline_wave", "neusight_style")
            )
            pass_reg = mean_reg["core"] <= 0.01 and all(
                mean_reg["core"] <= mean_reg[b] + 0.005 for b in ("roofline_wave", "neusight_style")
            )
            cohort_pass = pass_tau and pass_reg
            all_pass = all_pass and cohort_pass

            results["cohorts"][cohort_key] = {
                "decision": "pass" if cohort_pass else "fail",
                "pass_tau": pass_tau,
                "pass_regret": pass_reg,
                "median_kendall_tau_b": med_tau,
                "mean_top5_regret": mean_reg,
                "median_paired_tau_gain": {b: statistics.median(v) for b, v in tau_gains.items()},
                "core_median_abs_rel_error": statistics.median(med_abs_err_core),
                "cold_probe_median_ratio": statistics.median(probe_ratios),
                "pair_precision": pair_prec,
                "cases": per_case,
            }
            core_pp = pair_prec["core"]
            core_prec = (
                core_pp["agree"] / core_pp["ordered_calls"] if core_pp["ordered_calls"] else 0.0
            )
            print(
                f"[SCORE] {cohort_key:28s}: {('PASS' if cohort_pass else 'FAIL'):4s} | "
                f"tau core={med_tau['core']:.3f} roof={med_tau['roofline_wave']:.3f} neu={med_tau['neusight_style']:.3f} | "
                f"regret core={mean_reg['core'] * 100:.2f}% roof={mean_reg['roofline_wave'] * 100:.2f}% neu={mean_reg['neusight_style'] * 100:.2f}% | "
                f"prec={core_prec * 100:.1f}%",
                flush=True,
            )

    results["overall_decision"] = "pass" if all_pass else "fail"
    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] overall={results['overall_decision']} -> {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--output", required=True)
    p_pred.add_argument("--freeze-out", required=True)

    p_cap = sub.add_parser("capture")
    p_cap.add_argument("--freeze", required=True)
    p_cap.add_argument("--output", required=True)
    p_cap.add_argument("--seal-out", required=True)
    p_cap.add_argument("--family", choices=("gemm", "attention"), default=None)

    p_sc = sub.add_parser("score")
    p_sc.add_argument("--freeze", required=True)
    p_sc.add_argument("--predictions", required=True)
    p_sc.add_argument("--seal", required=True)
    p_sc.add_argument("--sealed-dir", required=True)
    p_sc.add_argument("--out", required=True)

    args = parser.parse_args()
    if args.cmd == "predict":
        cmd_predict(args)
    elif args.cmd == "capture":
        cmd_capture(args)
    else:
        cmd_score(args)


if __name__ == "__main__":
    main()
