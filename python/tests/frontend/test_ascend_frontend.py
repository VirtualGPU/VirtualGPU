"""Tests for the Triton-Ascend HIVM frontend parser and KernelIR assembly."""

from __future__ import annotations

import json
from pathlib import Path

from virtualgpu.frontend.ascend_compiler import (
    AscendCompiledArtifacts,
    bishengir_target_arch,
    canonical_ascend_arch,
    hash_hivm,
    is_ascend_arch,
)
from virtualgpu.frontend.hivm import parse_hivm_summary
from virtualgpu.frontend.triton_ascend_attention import assemble_ascend_attention_kernel_ir
from virtualgpu.frontend.triton_ascend_gemm import (
    _reorder_for_autoblockify,
    assemble_ascend_gemm_kernel_ir,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

# Captured post-hivm-graph-sync-solver HIVM MLIR from Triton-Ascend 3.2.2 on Ascend 910B3
# for a clean 128x128x64 FP16 GEMM with multibuffer=True (double-buffered L1 cbuf).
SAMPLE_HIVM_GEMM_DOUBLE_BUF = """\
// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @gemm_clean(%arg0: i64, %arg1: memref<?xi8, #hivm.address_space<gm>>, %arg2: memref<?xi8, #hivm.address_space<gm>>, %arg3: memref<?xf16, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, mix_mode = "mix", parallel_mode = "simd"} {
  %c0_i64_0 = arith.constant 0 : i64
  %c16384_i64 = arith.constant 16384 : i64
  %c32768_i64 = arith.constant 32768 : i64
  %c49152_i64 = arith.constant 49152 : i64
  %c0_i64_2 = arith.constant 0 : i64
  %c128 = arith.constant 128 : index
  %c64 = arith.constant 64 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c20_i32 = arith.constant 20 : i32
  scf.for %arg15 = %3 to %1 step %c20_i32 : i32 {
    %32 = hivm.hir.pointer_cast(%c0_i64_2) : memref<8x8x16x16xf32, #hivm.address_space<cc>>
    %cast = memref.cast %32 : memref<8x8x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    scf.for %arg16 = %c0_i32 to %11 step %c1_i32 : i32 {
      %58 = hivm.hir.pointer_cast(%c16384_i64, %c49152_i64) : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %58 {hivm.multi_buffer = 2 : i32} : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      %59 = hivm.hir.pointer_cast(%c0_i64_0, %c32768_i64) : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %59 {hivm.multi_buffer = 2 : i32} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_3 : memref<128x64xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_5 : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_4 : memref<64x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_6 : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true} ins(%cast_5, %cast_6, %61, %c128, %c64, %c128 : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
    } {fixpipe_for_mmad_result_already_inserted = true, pipeline.cubeonly}
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%cast : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%reinterpret_cast : memref<128x128xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>)
  } {autoblockify.subloop}
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}
"""

SAMPLE_HIVM_GEMM_SINGLE_BUF = """\
func.func @gemm_clean(%arg0: i64) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %c0_i64 = arith.constant 0 : i64
  %c16384_i64 = arith.constant 16384 : i64
  %c128 = arith.constant 128 : index
  %c64 = arith.constant 64 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c20_i32 = arith.constant 20 : i32
  scf.for %arg15 = %3 to %1 step %c20_i32 : i32 {
    %32 = hivm.hir.pointer_cast(%c0_i64) : memref<8x8x16x16xf32, #hivm.address_space<cc>>
    scf.for %arg16 = %c0_i32 to %11 step %c1_i32 : i32 {
      %52 = hivm.hir.pointer_cast(%c0_i64) : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
      %54 = hivm.hir.pointer_cast(%c16384_i64) : memref<8x4x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.mmadL1 ins(%cast_5, %cast_6, %61, %c128, %c64, %c128 : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
    }
  } {autoblockify.subloop}
  return
}
"""

SAMPLE_HIVM_MIX_CV = """\
func.func @attn_mix_aic(%arg0: i64, %arg2: memref<?xi8, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix} {
  %c0 = arith.constant 0 : index
  %c65536 = arith.constant 65536 : index
  %c131072 = arith.constant 131072 : index
  %c0_i64 = arith.constant 0 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c64 = arith.constant 64 : index
  %c20_i32 = arith.constant 20 : i32
  scf.for %arg23 = %3 to %1 step %c20_i32 : i32 {
    %ws0 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<4x64x64xf32, #hivm.address_space<gm>>
    %ws1 = memref_ext.alloc_workspace() from %arg2 offset = [%c65536] : from memref<?xi8, #hivm.address_space<gm>> to memref<4x64x64xf32, #hivm.address_space<gm>>
    %ws2 = memref_ext.alloc_workspace() from %arg2 offset = [%c131072] : from memref<?xi8, #hivm.address_space<gm>> to memref<4x64x64xf16, #hivm.address_space<gm>>
    scf.for %arg24 = %c0 to %11 step %c64 {
      %cb = hivm.hir.pointer_cast(%c0_i64, %c8192_i64) : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %cb {hivm.multi_buffer = 2 : i32} : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
      %cc = hivm.hir.pointer_cast(%c0_i64) : memref<4x4x16x16xf32, #hivm.address_space<cc>>
      hivm.hir.mmadL1 ins(%cast_a, %cast_b, %true, %c64, %c64, %c64 : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_c : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
      hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
      hivm.hir.fixpipe ins(%cast_c : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%ws0 : memref<4x64x64xf32, #hivm.address_space<gm>>)
    }
  } {autoblockify.subloop}
  return
}
func.func @attn_mix_aiv(%arg0: i64, %arg2: memref<?xi8, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c32768_i64 = arith.constant 32768 : i64
  %c20_i32 = arith.constant 20 : i32
  scf.for %arg23 = %3 to %1 step %c20_i32 : i32 {
    scf.for %arg24 = %c0 to %11 step %c64 {
      %ub = hivm.hir.pointer_cast(%c32768_i64) : memref<32x64xf32, #hivm.address_space<ub>>
      hivm.hir.vmul ins(%ub, %cst : memref<32x64xf32, #hivm.address_space<ub>>, f32) outs(%ub : memref<32x64xf32, #hivm.address_space<ub>>)
      hivm.hir.vexp ins(%ub : memref<32x64xf32, #hivm.address_space<ub>>) outs(%ub : memref<32x64xf32, #hivm.address_space<ub>>)
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
    }
  } {autoblockify.subloop}
  return
}
"""


def test_ascend_arch_helpers() -> None:
    assert is_ascend_arch("ascend910b3")
    assert is_ascend_arch("Ascend910B3")
    assert not is_ascend_arch("sm_89")
    assert canonical_ascend_arch("Ascend910B3") == "ascend910b3"
    assert bishengir_target_arch("ascend910b3") == "Ascend910B3"


def test_parse_hivm_double_vs_single_buffer() -> None:
    dbl = parse_hivm_summary(
        SAMPLE_HIVM_GEMM_DOUBLE_BUF, requested_multibuffer=True, requested_num_stages=3
    )
    assert dbl.mix_mode == "aic"
    assert dbl.autoblockify_stride == 20
    assert dbl.num_stages == 2
    assert dbl.prefetch_distance == 1
    assert dbl.async_copy == "async_element"
    assert dbl.l1_bytes == 65536
    assert dbl.l0c_bytes == 65536
    assert dbl.ub_bytes == 0
    assert len(dbl.mma_ops) == 1
    assert dbl.mma_ops[0].tile_mnk == (128, 128, 64)
    assert dbl.mma_ops[0].inst_mnk == (16, 16, 16)
    assert dbl.mma_ops[0].count == 8 * 8 * 4
    assert "ascend_stages_clamped_to_double_buffer" in dbl.diagnostics

    sng = parse_hivm_summary(
        SAMPLE_HIVM_GEMM_SINGLE_BUF, requested_multibuffer=False, requested_num_stages=1
    )
    assert sng.mix_mode == "aic"
    assert sng.num_stages == 1
    assert sng.prefetch_distance == 0
    assert sng.async_copy == "none"
    assert sng.l1_bytes == 32768
    assert sng.l0c_bytes == 65536


def test_parse_hivm_mix_cv_workspace_and_vector_ops() -> None:
    s = parse_hivm_summary(SAMPLE_HIVM_MIX_CV, requested_multibuffer=True, requested_num_stages=2)
    assert s.mix_mode == "mix"
    assert s.workspace_multibuffer == 4
    assert s.workspace_bytes == 163840
    assert s.ub_bytes == 32768 + 32 * 64 * 4
    assert s.loop_vec_ops == 1
    assert s.loop_sfu_ops == 1
    assert s.loop_sync_blocks == 2
    assert s.loop_cv_sync_hops == 1
    assert s.aiv_loop_max_rows == 32
    assert s.diagnostics == []


def test_autoblockify_reordering_preserves_all_tiles() -> None:
    logical = [(i // 4, i % 4) for i in range(45)]
    phys_grid, trips, ordered = _reorder_for_autoblockify(logical, 20)
    assert phys_grid == 20
    assert sum(trips) == 45
    assert trips[:5] == [3, 3, 3, 3, 3]
    assert trips[5:] == [2] * 15
    assert set(ordered) == set(logical)
    assert ordered[0] == logical[0]
    assert ordered[1] == logical[20]
    assert ordered[2] == logical[40]


def test_assemble_and_predict_ascend_gemm_end_to_end() -> None:
    from virtualgpu import _core

    artifacts = AscendCompiledArtifacts(
        kernel_name="gemm_clean",
        target_arch="ascend910b3",
        triton_version="3.5.0",
        ttir="",
        ttadapter="",
        hivm_ir=SAMPLE_HIVM_GEMM_DOUBLE_BUF,
        ttir_sha256="0" * 64,
        hivm_sha256=hash_hivm(SAMPLE_HIVM_GEMM_DOUBLE_BUF),
        mix_mode="mix",
        workspace_size=0,
        bs_task_type=20,
        requested_num_stages=2,
        requested_multibuffer=True,
    )
    ir = assemble_ascend_gemm_kernel_ir(
        artifacts,
        M=2048,
        N=2048,
        K=2048,
        config={"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8, "num_stages": 2},
    )
    assert ir["metadata"]["target_vendor"] == "ascend"
    assert ir["metadata"]["target_arch"] == "ascend910b3"
    assert ir["launch"]["grid"] == [20, 1, 1]
    assert ir["body"]["outer_loop"]["trip_count_mode"] == "per_cta"
    assert sum(ir["body"]["outer_loop"]["trip_counts"]) == 256
    assert ir["resources"]["static_smem_bytes"] == 65536
    assert "ub_bytes" not in ir["resources"]
    assert "l0c_bytes" not in ir["resources"]
    assert ir["resources"]["vendor_extensions"]["ascend"]["l0c_bytes"] == 65536
    assert "unvalidated_target_ascend" not in ir["diagnostics"]

    profile_text = (REPO_ROOT / "profiles" / "ascend910b3.json").read_text(encoding="utf-8")
    pred = _core.predict(json.dumps(ir), profile_text, l2_state="cold", draws=8, seed=17)
    assert pred["status"] == "ok", pred
    assert pred["time_ns"] > 0.0
    assert pred["ctas_per_sm"] == 1
    assert "unvalidated_feature" not in pred["flags"]
    assert "unvalidated_target_ascend" not in pred["diagnostics"]

    # Unvalidated Ascend targets still carry unvalidated_target_ascend in frontend and core.
    other_artifacts = AscendCompiledArtifacts(
        kernel_name="gemm_clean",
        target_arch="ascend910c",
        triton_version="3.5.0",
        ttir="",
        ttadapter="",
        hivm_ir=SAMPLE_HIVM_GEMM_DOUBLE_BUF,
        ttir_sha256="0" * 64,
        hivm_sha256=hash_hivm(SAMPLE_HIVM_GEMM_DOUBLE_BUF),
        mix_mode="mix",
        workspace_size=0,
        bs_task_type=20,
        requested_num_stages=2,
        requested_multibuffer=True,
    )
    ir_other = assemble_ascend_gemm_kernel_ir(
        other_artifacts,
        M=2048,
        N=2048,
        K=2048,
        config={"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 8, "num_stages": 2},
    )
    assert "unvalidated_target_ascend" in ir_other["diagnostics"]
    ir_other_no_diag = dict(ir_other, diagnostics=[])
    pred_other = _core.predict(
        json.dumps(ir_other_no_diag), profile_text, l2_state="cold", draws=4, seed=17
    )
    assert pred_other["status"] == "ok"
    assert "unvalidated_feature" in pred_other["flags"]
    assert "unvalidated_target_ascend" in pred_other["diagnostics"]

    # On-chip UB/L0C overflow returns a dedicated TooMuchOnChipBuffer reason.
    ir_bad_l0c = json.loads(json.dumps(ir))
    ir_bad_l0c["resources"]["vendor_extensions"]["ascend"]["l0c_bytes"] = 256 * 1024
    pred_bad = _core.predict(
        json.dumps(ir_bad_l0c), profile_text, l2_state="cold", draws=4, seed=17
    )
    assert pred_bad["status"] == "unsupported"
    assert "TooMuchOnChipBuffer" in pred_bad["reason"]
    assert "l0c" in pred_bad["reason"]


def test_assemble_and_predict_ascend_attention_causal_gqa() -> None:
    from virtualgpu import _core

    artifacts = AscendCompiledArtifacts(
        kernel_name="fused_attention",
        target_arch="ascend910b3",
        triton_version="3.5.0",
        ttir="",
        ttadapter="",
        hivm_ir=SAMPLE_HIVM_MIX_CV,
        ttir_sha256="0" * 64,
        hivm_sha256=hash_hivm(SAMPLE_HIVM_MIX_CV),
        mix_mode="mix",
        workspace_size=163840,
        bs_task_type=32,
        requested_num_stages=2,
        requested_multibuffer=True,
    )
    ir_causal = assemble_ascend_attention_kernel_ir(
        artifacts,
        Z=2,
        H=16,
        H_kv=4,
        N_CTX=1024,
        D=64,
        config={"BM": 64, "BN": 64, "multibuffer": True},
        causal=True,
    )
    assert ir_causal["metadata"]["target_vendor"] == "ascend"
    assert ir_causal["launch"]["grid"] == [20, 1, 1]
    assert sum(ir_causal["body"]["outer_loop"]["trip_counts"]) == 2 * 16 * (1024 // 64)
    assert ir_causal["body"]["pipeline_loop"]["trip_count_mode"] == "per_outer_iter"
    trips = ir_causal["body"]["pipeline_loop"]["trip_counts"]
    assert len(trips) == 512
    assert min(trips) == 1 and max(trips) == 16
    asc_ext = ir_causal["resources"]["vendor_extensions"]["ascend"]
    assert asc_ext["workspace_multibuffer"] == 4
    assert asc_ext["workspace_loop_bytes"] == 2 * (163840 // 4)
    assert asc_ext["cv_sync_hops_per_iter"] == 1
    assert "unvalidated_target_ascend" not in ir_causal["diagnostics"]

    ir_full = assemble_ascend_attention_kernel_ir(
        artifacts,
        Z=2,
        H=16,
        H_kv=4,
        N_CTX=1024,
        D=64,
        config={"BM": 64, "BN": 64, "multibuffer": True},
        causal=False,
    )
    assert ir_full["body"]["pipeline_loop"]["trip_count_mode"] == "uniform"
    assert ir_full["body"]["pipeline_loop"]["trip_counts"] == 16

    profile_text = (REPO_ROOT / "profiles" / "ascend910b3.json").read_text(encoding="utf-8")
    p_causal = _core.predict(json.dumps(ir_causal), profile_text, l2_state="cold", draws=8, seed=17)
    p_full = _core.predict(json.dumps(ir_full), profile_text, l2_state="cold", draws=8, seed=17)
    assert p_causal["status"] == "ok", p_causal
    assert p_full["status"] == "ok", p_full
    assert p_causal["time_ns"] < p_full["time_ns"]
    assert "unvalidated_feature" not in p_causal["flags"]
    assert "unvalidated_target_ascend" not in p_causal["diagnostics"]
    assert "unvalidated_feature" not in p_full["flags"]
    assert "unvalidated_target_ascend" not in p_full["diagnostics"]


# Captured pattern from dense_gemm with tl.constexpr M, N, K (m128n64k32):
# - fractal NZ cc is (N1=4, M1=8, M0=16, N0=16) -> logical (M=128, N=64)
# - fractal NZ cbuf A is (K1=2, M1=8, M0=16, K0=16) -> logical (M=128, K=32)
# - AIC-only masked load emits hivm.hir.vbrc into cbuf (not AIV vector ALU)
SAMPLE_HIVM_GEMM_CONSTEXPR_RECT = """\
func.func @dense_gemm(%arg0: i64) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %c0_i64 = arith.constant 0 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c16384_i64 = arith.constant 16384 : i64
  %c24576_i64 = arith.constant 24576 : i64
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c20_i32 = arith.constant 20 : i32
  scf.for %arg15 = %3 to %1 step %c20_i32 : i32 {
    %29 = hivm.hir.pointer_cast(%c0_i64) : memref<4x8x16x16xf32, #hivm.address_space<cc>>
    %cast = memref.cast %29 : memref<4x8x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    scf.for %arg16 = %c0_i32 to %11 step %c1_i32 : i32 {
      %56 = hivm.hir.pointer_cast(%c8192_i64, %c24576_i64) : memref<4x2x16x16xf16, #hivm.address_space<cbuf>>
      %57 = hivm.hir.pointer_cast(%c0_i64, %c16384_i64) : memref<2x8x16x16xf16, #hivm.address_space<cbuf>>
      %cast_5 = memref.cast %57 : memref<2x8x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, #hivm.address_space<cbuf>>
      %cast_10 = memref.cast %56 : memref<4x2x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, #hivm.address_space<cbuf>>
      hivm.hir.vbrc ins(%cst : f16) outs(%collapse_shape : memref<4096xf16, #hivm.address_space<cbuf>>)
      hivm.hir.mmadL1 ins(%cast_5, %cast_10, %72, %21, %64, %27 : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
    }
  } {autoblockify.subloop}
  return
}
"""


def test_parse_hivm_constexpr_fractal_nz_and_aic_vbrc() -> None:
    s = parse_hivm_summary(
        SAMPLE_HIVM_GEMM_CONSTEXPR_RECT, requested_multibuffer=True, requested_num_stages=2
    )
    assert s.mix_mode == "aic"
    assert s.ub_bytes == 0
    assert s.loop_vec_ops == 0
    assert s.loop_sfu_ops == 0
    assert len(s.mma_ops) == 1
    assert s.mma_ops[0].tile_mnk == (128, 64, 32)
    assert s.mma_ops[0].count == (128 // 16) * (64 // 16) * (32 // 16)
