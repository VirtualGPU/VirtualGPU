"""CPU-only tests for offline autotuning, CLI autotune, and online pruning hooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import triton
import triton.language as tl
import virtualgpu
from virtualgpu.autotune import AutotuneReport, autotune_offline, prune_configs_by
from virtualgpu.cli import main as cli_main
from virtualgpu.frontend.compiler import MockTensor

REPO_ROOT = Path(__file__).resolve().parents[3]
PROFILE_EXAMPLE_PATH = REPO_ROOT / "schemas" / "examples" / "profile_example.json"


_TEST_CONFIGS = [
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
        num_warps=4,
        num_stages=5,
    ),
]


@triton.autotune(configs=_TEST_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _autotuned_gemm(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = acc.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


class _DeterministicCore:
    """Deterministic stand-in for virtualgpu._core supporting predict, pair_verdict, and build_info."""

    __version__ = "0.1.0-test"

    @staticmethod
    def build_info() -> dict[str, Any]:
        return {"commit": "deadbeef", "dirty": False, "result_class": "formal"}

    @staticmethod
    def predict(
        kernel_ir_json: str,
        profile_json: str,
        l2_state: str = "cold",
        draws: int = 32,
        seed: int = 17,
    ) -> dict[str, Any]:
        ir = json.loads(kernel_ir_json)
        if ir["resources"]["static_smem_bytes"] > 100_000:
            return {
                "status": "unsupported",
                "l2_state": l2_state,
                "dispatch_policy": "round_robin",
                "reason": "TooMuchSharedMemory",
            }
        cfg = ir["metadata"]["config"]
        bm = cfg["BLOCK_SIZE_M"]
        bn = cfg["BLOCK_SIZE_N"]
        # Make 128x128 fastest (10000 ns), 128x64 nearly tied (10200 ns, within 5%), 64x64 slower (15000 ns).
        if bm == 128 and bn == 128:
            base = 10000.0
        elif bm == 128 and bn == 64:
            base = 10200.0
        else:
            base = 15000.0
        by_draw = {str(i): base * (1.0 + 0.002 * (i % 5)) for i in range(draws + 1)}
        return {
            "status": "ok",
            "l2_state": l2_state,
            "dispatch_policy": "round_robin",
            "time_ns": base,
            "draws": by_draw,
            "makespan_ns": base * 0.9,
            "ctas_per_sm": 1,
            "occupancy_limiter": "Registers",
            "waves": 2,
            "tail_efficiency": 1.0,
            "load_bytes": 4096,
            "l2_hit_bytes": 2048,
            "dram_bytes": 2048,
            "bound_fraction": {"mma": 0.75, "dram": 0.25},
            "flags": [],
            "diagnostics": [],
        }

    @staticmethod
    def pair_verdict(a: list[float], b: list[float]) -> str:
        if all(x * 1.05 < y for x, y in zip(a, b)):
            return "a_faster"
        if all(y * 1.05 < x for x, y in zip(a, b)):
            return "b_faster"
        return "indistinguishable"


class _AlwaysUnsupportedCore(_DeterministicCore):
    @staticmethod
    def predict(
        kernel_ir_json: str,
        profile_json: str,
        l2_state: str = "cold",
        draws: int = 32,
        seed: int = 17,
    ) -> dict[str, Any]:
        return {
            "status": "unsupported",
            "l2_state": l2_state,
            "dispatch_policy": "round_robin",
            "reason": "UnsupportedFeature",
        }


def test_autotune_offline_ranks_triton_autotuner_and_reports_ties(tmp_path: Path) -> None:
    report = autotune_offline(
        _autotuned_gemm,
        shape=(1024, 1024, 512),
        profile=PROFILE_EXAMPLE_PATH,
        l2_state="cold",
        draws=8,
        seed=17,
        top_k=2,
        cache_dir=tmp_path / "cache",
        core_module=_DeterministicCore,
    )
    assert isinstance(report, AutotuneReport)
    assert report.status == "ok"
    assert len(report.shortlist) == 2
    assert report.best_config is not None
    assert report.best_config.kwargs["BLOCK_SIZE_M"] == 128
    assert report.best_config.kwargs["BLOCK_SIZE_N"] == 128
    assert "verifying the top 2" in report.advice

    # 128x128 (10000 ns) and 128x64 (10200 ns) are within 5% -> indistinguishable
    assert len(report.indistinguishable_pairs) == 1
    assert report.ranking[0]["primary_bottleneck"] == {"resource": "mma", "fraction": 0.75}
    # Fourth config (128x128x64 s=5) exceeds 100 KB SMEM -> unsupported at end of ranking
    assert report.ranking[-1]["status"] == "unsupported"
    assert report.ranking[-1]["reason"] == "TooMuchSharedMemory"
    assert report.provenance["build"]["commit"] == "deadbeef"
    assert report.provenance["profile_revision"] == "0.1.0-example"
    assert json.loads(report.to_json())["status"] == "ok"
    assert "VirtualGPU autotune" in report.summary()

    # top_k=1 must explicitly state unverified on hardware
    report_k1 = autotune_offline(
        _autotuned_gemm,
        shape=(1024, 1024, 512),
        profile=PROFILE_EXAMPLE_PATH,
        draws=8,
        top_k=1,
        cache_dir=tmp_path / "cache",
        core_module=_DeterministicCore,
    )
    assert len(report_k1.shortlist) == 1
    assert "Unverified on hardware" in report_k1.advice


def test_cli_autotune_writes_report_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(virtualgpu, "_core", _DeterministicCore, raising=False)
    out_file = tmp_path / "autotune_report.json"
    rc = cli_main(
        [
            "autotune",
            "--kernel",
            "tests.frontend.test_autotune:_autotuned_gemm",
            "--shape",
            "1024",
            "1024",
            "512",
            "--profile",
            str(PROFILE_EXAMPLE_PATH),
            "--cache",
            "cold",
            "--draws",
            "8",
            "--top-k",
            "2",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--out",
            str(out_file),
        ]
    )
    assert rc == 0
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["status"] == "ok"
    assert len(data["shortlist"]) == 2
    assert data["provenance"]["top_k"] == 2


def test_online_pruner_shortlists_and_falls_back_loudly_on_total_failure(tmp_path: Path) -> None:
    reports: list[AutotuneReport] = []
    prune_dict = prune_configs_by(
        _autotuned_gemm.fn,
        profile=PROFILE_EXAMPLE_PATH,
        top_k=2,
        draws=8,
        cache_dir=tmp_path / "cache",
        report_sink=reports,
        core_module=_DeterministicCore,
    )
    pruner = prune_dict["early_config_prune"]
    nargs = {
        "a_ptr": MockTensor("f16", 0x10000),
        "b_ptr": MockTensor("f16", 0x210000),
        "c_ptr": MockTensor("f16", 0x410000),
        "M": 1024,
        "N": 1024,
        "K": 512,
        "stride_am": 512,
        "stride_ak": 1,
        "stride_bk": 1024,
        "stride_bn": 1,
        "stride_cm": 1024,
        "stride_cn": 1,
    }
    kept = pruner(_TEST_CONFIGS, nargs)
    assert len(kept) == 2
    assert kept[0].kwargs["BLOCK_SIZE_M"] == 128 and kept[0].kwargs["BLOCK_SIZE_N"] == 128
    assert len(reports) == 1
    assert reports[0].status == "ok"

    # Total modeling failure must warn loudly and fall back to all candidate configs.
    fail_pruner = prune_configs_by(
        _autotuned_gemm.fn,
        profile=PROFILE_EXAMPLE_PATH,
        top_k=2,
        draws=8,
        cache_dir=tmp_path / "cache",
        core_module=_AlwaysUnsupportedCore,
    )["early_config_prune"]
    with pytest.warns(UserWarning, match="falling back to all candidate configs"):
        fallback = fail_pruner(_TEST_CONFIGS, nargs)
    assert fallback == _TEST_CONFIGS

    # strict=True raises instead of falling back.
    strict_pruner = prune_configs_by(
        _autotuned_gemm.fn,
        profile=PROFILE_EXAMPLE_PATH,
        top_k=2,
        draws=8,
        strict=True,
        cache_dir=tmp_path / "cache",
        core_module=_AlwaysUnsupportedCore,
    )["early_config_prune"]
    with pytest.raises(RuntimeError, match="could not model any candidate config"):
        strict_pruner(_TEST_CONFIGS, nargs)


def test_virtualgpu_autotune_decorator_env_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VIRTUALGPU_PROFILE", raising=False)
    plain = virtualgpu.autotune(configs=_TEST_CONFIGS, key=["M", "N", "K"])(_autotuned_gemm.fn)
    assert not hasattr(plain, "vgpu_reports")
    assert plain.early_config_prune is None

    monkeypatch.setenv("VIRTUALGPU_PROFILE", str(PROFILE_EXAMPLE_PATH))
    monkeypatch.setenv("VIRTUALGPU_TOP_K", "2")
    wrapped = virtualgpu.autotune(
        configs=_TEST_CONFIGS,
        key=["M", "N", "K"],
        cache_dir=tmp_path / "cache",
        core_module=_DeterministicCore,
    )(_autotuned_gemm.fn)
    assert hasattr(wrapped, "vgpu_reports")
    assert wrapped.early_config_prune is not None
    assert wrapped._do_bench is not None


def test_kernel_ir_disk_cache_and_predict_batch_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from virtualgpu.frontend import compiler as comp_mod
    from virtualgpu.frontend import triton_gemm as tg_mod

    batch_calls = 0

    class _BatchCore(_DeterministicCore):
        @staticmethod
        def predict_batch(
            kernel_ir_jsons: list[str],
            profile_json: str,
            l2_state: str = "cold",
            draws: int = 32,
            seed: int = 17,
        ) -> list[dict[str, Any]]:
            nonlocal batch_calls
            batch_calls += 1
            return [
                _DeterministicCore.predict(
                    ir_j, profile_json, l2_state=l2_state, draws=draws, seed=seed
                )
                for ir_j in kernel_ir_jsons
            ]

    cache_dir = tmp_path / "cache"
    comp_mod._MEMORY_CACHE.clear()
    tg_mod._IR_MEMORY_CACHE.clear()
    r1 = autotune_offline(
        _autotuned_gemm,
        shape=(1024, 1024, 512),
        profile=PROFILE_EXAMPLE_PATH,
        draws=4,
        top_k=2,
        cache_dir=cache_dir,
        core_module=_BatchCore,
    )
    assert r1.status == "ok"
    assert batch_calls == 1

    # Clear memory caches and poison both triton.compile and scalar evaluator:
    # second run must hit on-disk KernelIR cache cleanly.
    comp_mod._MEMORY_CACHE.clear()
    tg_mod._IR_MEMORY_CACHE.clear()
    monkeypatch.setattr(
        triton,
        "compile",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should hit IR cache")),
    )
    monkeypatch.setattr(
        tg_mod,
        "evaluate_ttgir_cta_offsets",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should skip scalar_eval")),
    )

    r2 = autotune_offline(
        _autotuned_gemm,
        shape=(1024, 1024, 512),
        profile=PROFILE_EXAMPLE_PATH,
        draws=4,
        top_k=2,
        cache_dir=cache_dir,
        core_module=_BatchCore,
    )
    assert r2.status == "ok"
    assert batch_calls == 2
    assert r2.shortlist == r1.shortlist
