"""Parse post-scheduling BiShengIR HIVM MLIR emitted by Triton-Ascend."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from virtualgpu.frontend.ttgir import dtype_bits


@dataclass(frozen=True)
class HivmMmaOp:
    """One Cube matrix-multiply operation extracted from HIVM IR."""

    op_class: str
    tile_mnk: tuple[int, int, int]
    inst_mnk: tuple[int, int, int]
    a_dtype: str
    b_dtype: str
    acc_dtype: str
    count: int


@dataclass(frozen=True)
class HivmSummary:
    """Structured hardware-resource and pipeline summary from post-pass HIVM IR."""

    mix_mode: str  # "aic" | "aiv" | "mix"
    autoblockify_stride: int | None
    num_stages: int
    prefetch_distance: int
    async_copy: str  # "none" | "async_element"
    workspace_multibuffer: int
    l1_bytes: int
    l0c_bytes: int
    ub_bytes: int
    workspace_bytes: int
    workspace_loop_store_bytes: int
    workspace_loop_load_bytes: int
    mma_ops: list[HivmMmaOp] = field(default_factory=list)
    loop_vec_ops: int = 0
    loop_sfu_ops: int = 0
    loop_vec_insts: float = 0.0
    loop_sfu_insts: float = 0.0
    aiv_inner_loop_count: int = 0
    aiv_loop_max_rows: int = 0
    loop_sync_blocks: int = 0
    loop_cv_sync_hops: int = 0
    loop_barriers: int = 0
    diagnostics: list[str] = field(default_factory=list)
    unsupported_features: list[str] = field(default_factory=list)


_DTYPE_NORM: dict[str, str] = {
    "i1": "i1",
    "bool": "i1",
    "f16": "f16",
    "fp16": "f16",
    "bf16": "bf16",
    "f32": "f32",
    "fp32": "f32",
    "i8": "i8",
    "si8": "i8",
    "ui8": "u8",
    "i16": "i16",
    "i32": "i32",
    "i64": "i64",
}

_RE_CONST_INT = re.compile(
    r"^\s*(%[A-Za-z0-9_]+)\s*=\s*arith\.constant\s+(-?\d+)\s*:\s*(?:i\d+|index)\b"
)
_RE_FUNC_CORE = re.compile(r"hivm\.func_core_type\s*=\s*#hivm\.func_core_type<(AIC|AIV)>")
_RE_POINTER_CAST = re.compile(
    r"hivm\.hir\.pointer_cast\(([^)]+)\)\s*:\s*memref<([0-9x]+)x([a-zA-Z0-9_]+)(?:,\s*strided<[^>]+>)?,\s*#hivm\.address_space<(cbuf|cc|ub|ca|cb)>"
)
_RE_MULTI_BUFFER = re.compile(r"hivm\.multi_buffer\s*=\s*(\d+)\s*:")
_RE_ALLOC_WORKSPACE = re.compile(
    r"(%[A-Za-z0-9_]+)\s*=\s*memref_ext\.alloc_workspace\(\)\s*from\s+%[A-Za-z0-9_]+\s*offset\s*=\s*\[([^\]]+)\]\s*:\s*from\s+memref<[^\n]+?>\s+to\s+memref<([0-9x]+)x([a-zA-Z0-9_]+)"
)
_RE_MMAD_L1 = re.compile(
    r"hivm\.hir\.mmadL1\b[^\n]*?ins\(([^:]+):\s*memref<[^>]*x([a-zA-Z0-9_]+),\s*#hivm\.address_space<cbuf>>,\s*memref<[^>]*x([a-zA-Z0-9_]+),\s*#hivm\.address_space<cbuf>>[^)]*\)\s*outs\((%[A-Za-z0-9_]+)\s*:\s*memref<[^>]*x([a-zA-Z0-9_]+),\s*#hivm\.address_space<cc>>"
)
_RE_MEMREF_ALIAS = re.compile(
    r"^\s*(%[A-Za-z0-9_]+)\s*=\s*memref\.(?:subview|collapse_shape|expand_shape|reinterpret_cast|cast)\s+(%[A-Za-z0-9_]+)\b"
)
_RE_AUTOBLOCKIFY_FOR = re.compile(
    r"scf\.for\s+%[A-Za-z0-9_]+\s*=\s*[^\s]+\s+to\s+[^\s]+\s+step\s+(%[A-Za-z0-9_]+)"
)
_RE_VEC_SFU = re.compile(r"\bhivm\.hir\.(?:vexp|vln|vrec|vrsqrt|vsqrt)\b")
_RE_VEC_ALU = re.compile(
    r"\bhivm\.hir\.(?:vadd|vsub|vmul|vdiv|vmax|vmin|vbrc|vreduce|vcast|vsel|vcmp|vrelu|vabs|vmla|vmadd|vaxpy)\b"
)
_RE_OUTS_SHAPE = re.compile(r"outs\([^:]+:\s*memref<([0-9x]+)x[a-zA-Z0-9_]+")
_RE_SYNC_BLOCK = re.compile(r"\bhivm\.hir\.sync_block_(?:set|wait)\b")
_RE_PIPE_BARRIER = re.compile(r"\bhivm\.hir\.pipe_barrier\b")
_RE_WORKSPACE_STORE = re.compile(
    r"\bhivm\.hir\.(?:store|fixpipe)\b[^\n]*outs\((%[A-Za-z0-9_]+)\s*:\s*memref<([0-9x?]+)x([a-zA-Z0-9_]+)"
)
_RE_WORKSPACE_LOAD = re.compile(
    r"\bhivm\.hir\.(?:load|nd2nz)\b[^\n]*ins\((%[A-Za-z0-9_]+)\s*:\s*memref<([0-9x?]+)x([a-zA-Z0-9_]+)"
)


def _norm_dtype(raw: str) -> str:
    return _DTYPE_NORM.get(raw, raw)


def _shape_numel(shape_str: str) -> int | None:
    dims = shape_str.split("x")
    total = 1
    for d in dims:
        if d == "?" or not d.isdigit():
            return None
        total *= int(d)
    return total


def _memref_bytes(shape_str: str, dtype_str: str) -> int | None:
    numel = _shape_numel(shape_str)
    if numel is None:
        return None
    norm = _norm_dtype(dtype_str)
    if norm == "i1":
        return max(1, (numel + 7) // 8)
    bits = dtype_bits(norm)
    return (numel * bits + 7) // 8


def _parse_constants(ir_text: str) -> dict[str, int]:
    consts: dict[str, int] = {}
    for line in ir_text.splitlines():
        m = _RE_CONST_INT.match(line)
        if m:
            consts[m.group(1)] = int(m.group(2))
    return consts


def _extract_functions(ir_text: str) -> list[tuple[str, str]]:
    """Return list of (core_type, function_body) for each func.func in HIVM IR."""
    lines = ir_text.splitlines()
    funcs: list[tuple[str, str]] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.lstrip().startswith("func.func "):
            m_core = _RE_FUNC_CORE.search(line)
            core_type = m_core.group(1) if m_core else "UNKNOWN"
            body: list[str] = [line]
            depth = line.count("{") - line.count("}")
            i += 1
            while i < n and depth > 0:
                cur = lines[i]
                depth += cur.count("{") - cur.count("}")
                body.append(cur)
                i += 1
            funcs.append((core_type, "\n".join(body)))
        else:
            i += 1
    return funcs


def _extract_scf_for_blocks(func_text: str) -> list[tuple[str, str, str]]:
    """Return list of (header_line, body_text, closing_line) for scf.for loops."""
    lines = func_text.splitlines()
    loops: list[tuple[str, str, str]] = []
    for start_idx, line in enumerate(lines):
        if "scf.for " in line and line.rstrip().endswith("{"):
            depth = 1
            j = start_idx + 1
            body_lines: list[str] = []
            closing = ""
            while j < len(lines):
                cur = lines[j]
                stripped = cur.strip()
                if stripped.endswith("{"):
                    depth += 1
                if stripped.startswith("}"):
                    depth -= 1
                    if depth == 0:
                        closing = cur
                        break
                body_lines.append(cur)
                j += 1
            loops.append((line, "\n".join(body_lines), closing))
    return loops


def _cube_inst_mnk(a_dtype: str, b_dtype: str) -> tuple[int, int, int]:
    if a_dtype in ("i8", "u8") and b_dtype in ("i8", "u8"):
        return (16, 16, 32)
    return (16, 16, 16)


def parse_hivm_summary(
    hivm_ir: str,
    *,
    requested_multibuffer: bool = True,
    requested_num_stages: int = 2,
    fallback_workspace_bytes: int = 0,
) -> HivmSummary:
    """Extract post-scheduling Ascend pipeline, tile, and SRAM usage from HIVM IR."""
    funcs = _extract_functions(hivm_ir)
    core_types = {ct for ct, _ in funcs}
    if "AIC" in core_types and "AIV" in core_types:
        mix_mode = "mix"
    elif "AIC" in core_types:
        mix_mode = "aic"
    elif "AIV" in core_types:
        mix_mode = "aiv"
    else:
        mix_mode = "aic" if "hivm.hir.mmadL1" in hivm_ir else "aiv"

    l1_bytes = 0
    l0c_bytes = 0
    ub_bytes = 0
    workspace_bytes = max(0, int(fallback_workspace_bytes))
    workspace_multibuffer = 0
    max_cbuf_multibuffer = 1
    autoblockify_stride: int | None = None
    mma_ops: list[HivmMmaOp] = []
    loop_vec_ops = 0
    loop_sfu_ops = 0
    loop_vec_insts = 0.0
    loop_sfu_insts = 0.0
    aiv_inner_loop_count = 0
    aiv_loop_max_rows = 0
    loop_sync_blocks = 0
    aic_loop_sync_blocks = 0
    aiv_loop_sync_blocks = 0
    loop_barriers = 0
    ws_loop_store_bytes = 0
    ws_loop_load_bytes = 0

    for _core_type, ftext in funcs:
        consts = _parse_constants(ftext)

        # Build SSA alias table for memref view/cast chains (subview -> collapse_shape -> cast).
        aliases: dict[str, str] = {}
        for line in ftext.splitlines():
            ma = _RE_MEMREF_ALIAS.match(line)
            if ma:
                aliases[ma.group(1)] = ma.group(2)

        def _resolve_root(var: str, _aliases: dict[str, str] = aliases) -> str:
            seen: set[str] = set()
            cur = var
            while cur in _aliases and cur not in seen:
                seen.add(cur)
                cur = _aliases[cur]
            return cur

        # Track static 2D tile shapes of pointer_cast buffers (e.g., fractal 4x4x16x16 -> (64, 64)).
        ptr_tile_shapes: dict[str, tuple[int, int]] = {}
        for line in ftext.splitlines():
            m = _RE_POINTER_CAST.search(line)
            if not m:
                continue
            off_args = [tok.strip() for tok in m.group(1).split(",")]
            shape_str, dtype_str, space = m.group(2), m.group(3), m.group(4)
            buf_bytes = _memref_bytes(shape_str, dtype_str)
            if buf_bytes is None:
                continue
            offsets = [consts.get(a, 0) for a in off_args]
            high_water = max(offsets) + buf_bytes
            if space == "cbuf":
                l1_bytes = max(l1_bytes, high_water)
                if len(off_args) > 1:
                    max_cbuf_multibuffer = max(max_cbuf_multibuffer, len(off_args))
            elif space == "cc":
                l0c_bytes = max(l0c_bytes, high_water)
            elif space == "ub":
                ub_bytes = max(ub_bytes, high_water)

            lhs_m = re.match(r"^\s*(%[A-Za-z0-9_]+)\s*=", line)
            if lhs_m and space in ("cbuf", "cc"):
                dims = [int(x) for x in shape_str.split("x") if x.isdigit()]
                if len(dims) == 4:
                    # Ascend fractal NZ layout for logical (Rows, Cols) is (Cols1, Rows1, Rows0, Cols0):
                    #   cc (C): (N1, M1, M0, N0) -> (M, N)
                    #   cbuf A: (K1, M1, M0, K0) -> (M, K)
                    #   cbuf B: (N1, K1, K0, N0) -> (K, N)
                    ptr_tile_shapes[lhs_m.group(1)] = (dims[1] * dims[2], dims[0] * dims[3])
                elif len(dims) == 2:
                    ptr_tile_shapes[lhs_m.group(1)] = (dims[0], dims[1])

        # Explicit multi_buffer annotations on cbuf memrefs.
        for line in ftext.splitlines():
            if "#hivm.address_space<cbuf>" in line:
                mb_m = _RE_MULTI_BUFFER.search(line)
                if mb_m:
                    max_cbuf_multibuffer = max(max_cbuf_multibuffer, int(mb_m.group(1)))

        # Workspace allocations & ring depth.
        ws_vars: dict[str, int] = {}
        for m in _RE_ALLOC_WORKSPACE.finditer(ftext):
            var_name = m.group(1)
            off_sym = m.group(2).strip()
            shape_str = m.group(3)
            dtype_str = m.group(4)
            b = _memref_bytes(shape_str, dtype_str)
            if b is None:
                continue
            off_val = consts.get(off_sym, 0)
            workspace_bytes = max(workspace_bytes, off_val + b)
            dims = [int(x) for x in shape_str.split("x") if x.isdigit()]
            if len(dims) >= 3:
                workspace_multibuffer = max(workspace_multibuffer, dims[0])
                per_slot = b // max(dims[0], 1)
            else:
                workspace_multibuffer = max(workspace_multibuffer, 1)
                per_slot = b
            ws_vars[var_name] = per_slot

        # Extract loops: identify autoblockify outer loop and inner reduction/pipeline loops.
        loops = _extract_scf_for_blocks(ftext)
        inner_bodies: list[str] = []
        for header, body, closing in loops:
            if "autoblockify.subloop" in closing:
                m_step = _RE_AUTOBLOCKIFY_FOR.search(header)
                if m_step and m_step.group(1) in consts:
                    autoblockify_stride = consts[m_step.group(1)]
            else:
                inner_bodies.append(body)

        # Scan inner loops (or whole function if unrolled/loop-free) for MMA, vector, and workspace ops.
        scan_regions = inner_bodies if inner_bodies else [ftext]
        # Avoid double counting nested loops by picking innermost loops.
        primary_regions: list[str] = []
        for idx, reg in enumerate(scan_regions):
            if not any(
                idx != j and reg in other and len(other) > len(reg)
                for j, other in enumerate(scan_regions)
            ):
                primary_regions.append(reg)

        if _core_type == "AIV" and inner_bodies:
            aiv_inner_loop_count += len(primary_regions)

        # When a core function has multiple sequential peer loops (e.g., causal attention's
        # steady-state off-diagonal loop followed by a 1-iteration diagonal masked loop),
        # pick the steady-state unmasked loop (fewest mask ops, tie-broken by first) for
        # per-iteration compute and cross-core sync accounting.
        if len(primary_regions) > 1:
            steady_regions = [
                min(
                    primary_regions,
                    key=lambda r: r.count("hivm.hir.vcmp") + r.count("hivm.hir.vsel"),
                )
            ]
        else:
            steady_regions = primary_regions

        for reg in steady_regions:
            for m in _RE_MMAD_L1.finditer(reg):
                ins_args = [tok.strip() for tok in m.group(1).split(",")]
                a_dt = _norm_dtype(m.group(2))
                b_dt = _norm_dtype(m.group(3))
                out_var = m.group(4).strip()
                acc_dt = _norm_dtype(m.group(5))
                # mmadL1 ins(lhs, rhs, init_flag, M, K, N)
                tile_m = consts.get(ins_args[3], 0) if len(ins_args) >= 6 else 0
                tile_k = consts.get(ins_args[4], 0) if len(ins_args) >= 6 else 0
                tile_n = consts.get(ins_args[5], 0) if len(ins_args) >= 6 else 0
                # Fall back to static L0C/L1 fractal tile dims when M/N/K are runtime-clamped.
                cc_shape = ptr_tile_shapes.get(_resolve_root(out_var))
                if cc_shape is not None:
                    if tile_m <= 0:
                        tile_m = cc_shape[0]
                    if tile_n <= 0:
                        tile_n = cc_shape[1]
                if tile_k <= 0 and len(ins_args) >= 1:
                    a_shape = ptr_tile_shapes.get(_resolve_root(ins_args[0]))
                    if a_shape is not None:
                        tile_k = a_shape[1]
                if tile_m > 0 and tile_n > 0 and tile_k > 0:
                    im, in_, ik = _cube_inst_mnk(a_dt, b_dt)
                    cnt = max(
                        1,
                        math.ceil(tile_m / im) * math.ceil(tile_n / in_) * math.ceil(tile_k / ik),
                    )
                    mma_ops.append(
                        HivmMmaOp(
                            op_class="cube.mmad",
                            tile_mnk=(tile_m, tile_n, tile_k),
                            inst_mnk=(im, in_, ik),
                            a_dtype=a_dt,
                            b_dtype=b_dt,
                            acc_dtype=acc_dt,
                            count=cnt,
                        )
                    )

            # AIC-only functions do not execute on the Vector unit; any hivm.hir.vbrc inside an
            # AIC function is an L1 (cbuf) zero-padding helper rather than AIV vector ALU work.
            if _core_type != "AIC":
                for line in reg.splitlines():
                    if _RE_VEC_SFU.search(line):
                        loop_sfu_ops += 1
                        om = _RE_OUTS_SHAPE.search(line)
                        numel = _shape_numel(om.group(1)) if om else None
                        loop_sfu_insts += max(1.0, (numel or 32) / 32.0)
                        if _core_type == "AIV" and om:
                            dims = [int(x) for x in om.group(1).split("x") if x.isdigit()]
                            if len(dims) >= 2:
                                aiv_loop_max_rows = max(aiv_loop_max_rows, dims[0])
                    elif _RE_VEC_ALU.search(line):
                        loop_vec_ops += 1
                        om = _RE_OUTS_SHAPE.search(line)
                        numel = _shape_numel(om.group(1)) if om else None
                        loop_vec_insts += max(1.0, (numel or 32) / 32.0)
                        if _core_type == "AIV" and om:
                            dims = [int(x) for x in om.group(1).split("x") if x.isdigit()]
                            if len(dims) >= 2:
                                aiv_loop_max_rows = max(aiv_loop_max_rows, dims[0])

            sb_cnt = len(_RE_SYNC_BLOCK.findall(reg))
            loop_sync_blocks += sb_cnt
            if _core_type == "AIC":
                aic_loop_sync_blocks += sb_cnt
            elif _core_type == "AIV":
                aiv_loop_sync_blocks += sb_cnt
            loop_barriers += len(_RE_PIPE_BARRIER.findall(reg))

            if ws_vars:
                for ms in _RE_WORKSPACE_STORE.finditer(reg):
                    root = _resolve_root(ms.group(1))
                    if root in ws_vars:
                        ws_loop_store_bytes += ws_vars[root]
                for ml in _RE_WORKSPACE_LOAD.finditer(reg):
                    root = _resolve_root(ml.group(1))
                    if root in ws_vars:
                        ws_loop_load_bytes += ws_vars[root]

    loop_cv_sync_hops = max(aic_loop_sync_blocks, aiv_loop_sync_blocks) if mix_mode == "mix" else 0

    num_stages = max_cbuf_multibuffer
    prefetch_distance = max(0, num_stages - 1)
    async_copy = "async_element" if num_stages >= 2 else "none"

    diagnostics: list[str] = []
    unsupported_features: list[str] = []
    if requested_num_stages > 2 and num_stages == 2:
        diagnostics.append("ascend_stages_clamped_to_double_buffer")
    if requested_multibuffer and requested_num_stages >= 2 and num_stages == 1:
        diagnostics.append("unpipelined_fallback")

    return HivmSummary(
        mix_mode=mix_mode,
        autoblockify_stride=autoblockify_stride,
        num_stages=num_stages,
        prefetch_distance=prefetch_distance,
        async_copy=async_copy,
        workspace_multibuffer=workspace_multibuffer,
        l1_bytes=l1_bytes,
        l0c_bytes=l0c_bytes,
        ub_bytes=ub_bytes,
        workspace_bytes=workspace_bytes,
        workspace_loop_store_bytes=ws_loop_store_bytes,
        workspace_loop_load_bytes=ws_loop_load_bytes,
        mma_ops=mma_ops,
        loop_vec_ops=loop_vec_ops,
        loop_sfu_ops=loop_sfu_ops,
        loop_vec_insts=loop_vec_insts,
        loop_sfu_insts=loop_sfu_insts,
        aiv_inner_loop_count=aiv_inner_loop_count,
        aiv_loop_max_rows=aiv_loop_max_rows,
        loop_sync_blocks=loop_sync_blocks,
        loop_cv_sync_hops=loop_cv_sync_hops,
        loop_barriers=loop_barriers,
        diagnostics=diagnostics,
        unsupported_features=unsupported_features,
    )
