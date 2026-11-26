"""Extract post-pipelining structure from Triton TTGIR and PTX."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MmaInfo:
    """Hardware MMA instruction details extracted from TTGIR and PTX."""

    op_class: str
    inst_mnk: tuple[int, int, int]
    a_dtype: str
    b_dtype: str
    acc_dtype: str


@dataclass(frozen=True)
class MmaDotInfo:
    """Per-tt.dot MMA operation extracted from a pipeline loop body."""

    op_class: str
    tile_mnk: tuple[int, int, int]
    inst_mnk: tuple[int, int, int]
    a_dtype: str
    b_dtype: str
    acc_dtype: str
    count: int


@dataclass(frozen=True)
class TtgirSummary:
    """Structured summary extracted from post-pipelining TTGIR + PTX."""

    num_warps: int
    warp_size: int
    cluster_shape: tuple[int, int, int]
    async_copy: str  # "none" | "async_element" | "async_tensor"
    num_stages: int
    prefetch_distance: int
    transfer_width_bytes: int
    mma: MmaInfo | None
    smem_load_bytes_per_iter: int
    loop_barriers: int
    diagnostics: list[str] = field(default_factory=list)
    unsupported_features: list[str] = field(default_factory=list)
    store_transfer_width_bytes: int | None = None
    mma_dots: list[MmaDotInfo] = field(default_factory=list)
    loop_alu_per_warp: int = 0
    loop_sfu_per_warp: int = 0
    loop_shfl_per_warp: int = 0
    prologue_smem_store_bytes: int = 0
    num_top_level_loops: int = 0
    loop_ptx_barriers: int = 0
    loop_ptx_st_shared_bytes: int = 0
    loop_ptx_ld_shared_bytes: int = 0


_DTYPE_BITS: dict[str, int] = {
    "f4": 4,
    "f8E4M3FN": 8,
    "f8E5M2": 8,
    "i8": 8,
    "f16": 16,
    "bf16": 16,
    "i16": 16,
    "f32": 32,
    "tf32": 32,
    "i32": 32,
    "f64": 64,
    "i64": 64,
}

_RE_NUM_WARPS = re.compile(r'"ttg\.num-warps"\s*=\s*(\d+)')
_RE_WARP_SIZE = re.compile(r'"ttg\.threads-per-warp"\s*=\s*(\d+)')
_RE_NUM_CTAS = re.compile(r'"ttg\.num-ctas"\s*=\s*(\d+)')
_RE_MULTIBUF_MEMDESC = re.compile(
    r"ttg\.local_alloc\s*:\s*\(\)\s*->\s*!ttg\.memdesc<(\d+)x(\d+)x(\d+)x([a-zA-Z0-9_]+)"
)
_RE_CONTIGUITY = re.compile(r"contiguity\s*=\s*(\d+)")
_RE_PTX_MMA_SYNC = re.compile(
    r"mma\.sync\.aligned\.m(\d+)n(\d+)k(\d+)\.[a-z.]+\.(f32|f16|bf16|s32)\.(f16|bf16|tf32|f8e4m3|f8e5m2|s8|u8)\.(f16|bf16|tf32|f8e4m3|f8e5m2|s8|u8)\.(f32|f16|bf16|s32)"
)
_RE_PTX_WGMMA = re.compile(r"wgmma\.mma_async\.sync\.aligned\.m(\d+)n(\d+)k(\d+)")
_RE_PTX_TCGEN05 = re.compile(r"tcgen05\.mma\.")
_RE_TTGIR_MMA_LAYOUT = re.compile(r"#ttg\.nvidia_mma<[^>]*instrShape\s*=\s*\[(\d+),\s*(\d+)\]")
_RE_TT_DOT = re.compile(
    r"tt\.dot\s+[^:]+:\s*tensor<(\d+)x(\d+)x([a-zA-Z0-9_]+),\s*(?:<[^>]*>|[^>])+>\s*\*\s*tensor<(\d+)x(\d+)x([a-zA-Z0-9_]+),\s*(?:<[^>]*>|[^>])+>\s*->\s*tensor<(\d+)x(\d+)x([a-zA-Z0-9_]+)"
)
_RE_LOCAL_LOAD = re.compile(r"ttg\.local_load\b[^:]*:\s*!ttg\.memdesc<(\d+)x(\d+)x([a-zA-Z0-9_]+)")
_RE_PROLOGUE_LOCAL_ALLOC_VAL = re.compile(
    r"ttg\.local_alloc\s+%[A-Za-z0-9_]+\s*:\s*\([^)]+\)\s*->\s*!ttg\.memdesc<(\d+)x(\d+)x([a-zA-Z0-9_]+)"
)
_RE_TTGIR_WAIT_NUM = re.compile(r"ttg\.async_wait\b[^{]*\{\s*num\s*=\s*(\d+)")
_RE_PTX_WAIT_GROUP = re.compile(r"\bcp\.async\.wait_group\s+(\d+)\s*;")
_RE_PTX_COMMIT_GROUP = re.compile(r"\bcp\.async\.commit_group\s*;")
_RE_PTX_MBAR_TRY_WAIT = re.compile(r"\bmbarrier\.try_wait\b")
_RE_PTX_MBAR_ARRIVE_TX = re.compile(r"\bmbarrier\.arrive\.expect_tx\b")
_RE_PTX_ANY_MMA = re.compile(r"\b(?:mma\.sync\.aligned|wgmma\.mma_async|tcgen05\.mma)\b")
_RE_PTX_SFU = re.compile(r"\b(?:ex2|lg2|rcp|rsqrt|sin|cos|sqrt)\.approx\b")
_RE_PTX_SHFL = re.compile(r"\bshfl\.sync\b")
_RE_PTX_FP_ALU = re.compile(
    r"^\s*(?:@!?%p\d+\s+)?(?:fma\.rn|mul\.(?:rn\.)?f|add\.(?:rn\.)?f|sub\.(?:rn\.)?f|max\.f|min\.f|cvt\.|selp\.|setp\.)"
)
_RE_TTGIR_ANY_DOT = re.compile(r"\b(?:tt\.dot|ttng\.warp_group_dot)\b")
_RE_PTX_LABEL = re.compile(r"^\s*(\$[A-Za-z0-9_]+):")
_RE_PTX_BRA = re.compile(r"\bbra(?:\.uni)?\s+(\$[A-Za-z0-9_]+)")
_RE_PTX_LD_GLOBAL = re.compile(
    r"\bld\.global(?:\.[a-z0-9_:]+)*?(?:\.(v[24]))?\.(?:b|u|s|f)(8|16|32|64)\b"
)
_RE_PTX_ST_GLOBAL = re.compile(
    r"\bst\.global(?:\.[a-z0-9_:]+)*?(?:\.(v[24]))?\.(?:b|u|s|f)(8|16|32|64)\b"
)
_RE_PTX_CP_ASYNC_SIZE = re.compile(
    r"\bcp\.async\.(?:ca|cg)\.shared(?:\.global|::cta\.global)[^;]*,\s*(0x[0-9a-fA-F]+|\d+)\b"
)
_RE_PTX_TMA_BULK = re.compile(r"\bcp\.async\.bulk\.tensor\b")
_RE_PTX_BAR_SYNC = re.compile(r"\bbar\.sync\b")
_RE_PTX_ST_SHARED = re.compile(
    r"\bst\.shared(?:\.[a-z0-9_:]+)*?(?:\.(v[24]))?\.(?:b|u|s|f)(8|16|32|64)\b"
)
_RE_PTX_LD_SHARED = re.compile(
    r"\bld\.shared(?:\.[a-z0-9_:]+)*?(?:\.(v[24]))?\.(?:b|u|s|f)(8|16|32|64)\b"
)


def dtype_bits(dtype: str) -> int:
    """Return bit width for a canonical dtype name."""
    if dtype in _DTYPE_BITS:
        return _DTYPE_BITS[dtype]
    raise ValueError(f"Unknown dtype: {dtype!r}")


def _extract_loop_bodies(ttgir: str) -> list[str]:
    """Return text inside each top-level scf.for loop body in tt.func."""
    lines = ttgir.splitlines()
    bodies: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if "scf.for " in line and line.rstrip().endswith("{"):
            body_lines: list[str] = []
            depth = 1
            i += 1
            while i < n:
                cur = lines[i]
                stripped = cur.strip()
                if stripped.endswith("{"):
                    depth += 1
                if stripped.startswith("}"):
                    depth -= 1
                    if depth == 0:
                        break
                body_lines.append(cur)
                i += 1
            bodies.append("\n".join(body_lines))
        i += 1
    return bodies


def _extract_loop_body(ttgir: str) -> str:
    """Return the text inside the primary scf.for loop, or empty string if none."""
    bodies = _extract_loop_bodies(ttgir)
    return bodies[0] if bodies else ""


def _count_ptx_loop_scalar_ops(ptx_loop: str) -> tuple[int, int, int]:
    """Count (alu_per_warp, sfu_per_warp, shfl_per_warp) inside a PTX loop body."""
    if not ptx_loop:
        return 0, 0, 0
    alu = 0
    sfu = 0
    shfl = 0
    for line in ptx_loop.splitlines():
        if _RE_PTX_SFU.search(line):
            sfu += 1
        elif _RE_PTX_SHFL.search(line):
            shfl += 1
        elif _RE_PTX_FP_ALU.match(line):
            alu += 1
    return alu, sfu, shfl


def _extract_ptx_loop_body(ptx: str) -> str:
    """Extract lines inside the primary PTX backward-branch loop body."""
    lines = ptx.splitlines()
    labels: dict[str, int] = {}
    for idx, line in enumerate(lines):
        m = _RE_PTX_LABEL.match(line)
        if m:
            labels[m.group(1)] = idx

    best_range: tuple[int, int] | None = None
    for idx, line in enumerate(lines):
        m = _RE_PTX_BRA.search(line)
        if m and m.group(1) in labels and labels[m.group(1)] < idx:
            start = labels[m.group(1)]
            # Prefer a loop containing cp.async instructions; otherwise the largest loop.
            span = (start, idx)
            if best_range is None:
                best_range = span
            else:
                cur_text = "\n".join(lines[start : idx + 1])
                best_text = "\n".join(lines[best_range[0] : best_range[1] + 1])
                cur_has_async = "cp.async." in cur_text
                best_has_async = "cp.async." in best_text
                if (cur_has_async and not best_has_async) or (
                    cur_has_async == best_has_async
                    and (idx - start) > (best_range[1] - best_range[0])
                ):
                    best_range = span
    if best_range is None:
        return ""
    return "\n".join(lines[best_range[0] : best_range[1] + 1])


def _derive_cp_async_prefetch_distance(
    ptx: str,
    ttgir_loop_body: str,
) -> tuple[int, list[str], list[str]]:
    """Derive effective in-flight stage overlap from loop wait_group / async_wait.

    Per the public PTX ISA specification, `cp.async.wait_group N` waits until at
    most N most-recent committed async groups remain pending. Dividing N by the
    number of commit groups per stage gives the number of stages still in flight
    during compute. Cross-checks PTX against post-pipelining TTGIR when both are
    present; if neither can resolve it, marks `prefetch_distance_unresolved` in
    `unsupported_features` so the model core refuses to guess.
    """
    diags: list[str] = []
    unsupported: list[str] = []

    ptx_dist: int | None = None
    ptx_loop = _extract_ptx_loop_body(ptx)
    if ptx_loop:
        ptx_waits = [int(m.group(1)) for m in _RE_PTX_WAIT_GROUP.finditer(ptx_loop)]
        ptx_commits = len(_RE_PTX_COMMIT_GROUP.findall(ptx_loop))
        if len(set(ptx_waits)) > 1:
            unsupported.append("split_async_wait_unsupported")
            return 0, diags, unsupported
        if ptx_waits and ptx_commits > 0:
            ptx_dist = ptx_waits[0] // ptx_commits

    ttgir_dist: int | None = None
    ttgir_waits = [int(m.group(1)) for m in _RE_TTGIR_WAIT_NUM.finditer(ttgir_loop_body)]
    ttgir_commits = ttgir_loop_body.count("ttg.async_commit_group")
    if len(set(ttgir_waits)) > 1:
        unsupported.append("split_async_wait_unsupported")
        return 0, diags, unsupported
    if ttgir_waits and ttgir_commits > 0:
        ttgir_dist = ttgir_waits[0] // ttgir_commits

    if ptx_dist is not None and ttgir_dist is not None:
        if ptx_dist != ttgir_dist:
            diags.append("prefetch_distance_mismatch")
        return ptx_dist, diags, unsupported
    if ptx_dist is not None:
        return ptx_dist, diags, unsupported
    if ttgir_dist is not None:
        return ttgir_dist, diags, unsupported

    unsupported.append("prefetch_distance_unresolved")
    return 0, diags, unsupported


def _derive_tma_prefetch_distance(
    ptx: str,
    ttgir: str,
    ttgir_loop_body: str,
) -> tuple[int, list[str], list[str]]:
    """Derive in-flight stage overlap for TMA mbarrier pipelines from PTX and TTGIR.

    Per public PTX ISA mbarrier semantics:
    - In PTX, count `mbarrier.arrive.expect_tx` before the loop's first
      `mbarrier.try_wait` (prologue prefills) versus inside the loop body
      (steady-state refills). After `try_wait` retires the oldest barrier,
      `(prologue_arrives // loop_arrives) - 1` prefilled stages remain in flight.
      If the loop issues the next `mbarrier.arrive.expect_tx` *before* the first
      MMA instruction, that newly issued stage is also in flight during compute
      (`+ 1`); if it issues *after* MMA, it is not (`+ 0`).
    - In TTGIR, cross-check using `ttng.async_tma_copy_global_to_local` counts
      before vs. inside `scf.for` alongside `ttng.wait_barrier` and relative
      order against `tt.dot` / `ttng.warp_group_dot`.
    If neither can resolve it cleanly, emit `tma_prefetch_unresolved` in
    `unsupported_features` rather than guessing.
    """
    diags: list[str] = []
    unsupported: list[str] = []

    ptx_dist: int | None = None
    m_wait = _RE_PTX_MBAR_TRY_WAIT.search(ptx)
    if m_wait:
        before_wait = ptx[: m_wait.start()]
        after_wait = ptx[m_wait.end() :]
        pro_arrives = len(_RE_PTX_MBAR_ARRIVE_TX.findall(before_wait))
        loop_arrive_matches = list(_RE_PTX_MBAR_ARRIVE_TX.finditer(after_wait))
        loop_arrives = len(loop_arrive_matches)
        m_mma = _RE_PTX_ANY_MMA.search(after_wait)
        if (
            m_mma is not None
            and loop_arrives > 0
            and pro_arrives >= loop_arrives
            and (pro_arrives % loop_arrives) == 0
        ):
            issue_before_mma = 1 if loop_arrive_matches[0].start() < m_mma.start() else 0
            ptx_dist = (pro_arrives // loop_arrives) - 1 + issue_before_mma

    ttgir_dist: int | None = None
    for_pos = ttgir.find("scf.for ")
    if for_pos != -1:
        prologue = ttgir[:for_pos]
        prologue_tma = prologue.count("ttng.async_tma_copy_global_to_local")
        loop_tma = ttgir_loop_body.count("ttng.async_tma_copy_global_to_local")
        wait_pos = ttgir_loop_body.find("ttng.wait_barrier")
        if (
            wait_pos != -1
            and loop_tma > 0
            and prologue_tma >= loop_tma
            and (prologue_tma % loop_tma) == 0
        ):
            after_wait_ir = ttgir_loop_body[wait_pos:]
            m_dot = _RE_TTGIR_ANY_DOT.search(after_wait_ir)
            first_tma_pos = after_wait_ir.find("ttng.async_tma_copy_global_to_local")
            if m_dot is not None and first_tma_pos != -1:
                issue_before_dot = 1 if first_tma_pos < m_dot.start() else 0
                ttgir_dist = (prologue_tma // loop_tma) - 1 + issue_before_dot

    if ptx_dist is not None and ttgir_dist is not None:
        if ptx_dist != ttgir_dist:
            diags.append("prefetch_distance_mismatch")
        return ptx_dist, diags, unsupported
    if ptx_dist is not None:
        return ptx_dist, diags, unsupported
    if ttgir_dist is not None:
        return ttgir_dist, diags, unsupported

    unsupported.append("tma_prefetch_unresolved")
    return 0, diags, unsupported


def _effective_ptx_width(widths: list[int]) -> int | None:
    """Return instruction-count-weighted effective transfer width in bytes."""
    if not widths:
        return None
    return max(1, round(sum(widths) / len(widths)))


def _extract_ptx_transfer_widths(ptx: str, ptx_loop: str) -> tuple[int | None, int | None]:
    """Derive effective per-instruction global load and store widths (bytes) from PTX."""
    if not ptx:
        return None, None

    load_src = ptx_loop if ptx_loop else ptx
    load_widths: list[int] = []
    for m in _RE_PTX_LD_GLOBAL.finditer(load_src):
        vec = 4 if m.group(1) == "v4" else (2 if m.group(1) == "v2" else 1)
        load_widths.append(vec * (int(m.group(2)) // 8))
    for m in _RE_PTX_CP_ASYNC_SIZE.finditer(load_src):
        size_b = int(m.group(1), 0)
        if size_b > 0:
            load_widths.append(size_b)
    if not load_widths and _RE_PTX_TMA_BULK.search(load_src):
        load_widths.append(128)
    # Fall back to whole PTX if the loop body had no explicit global load instructions.
    if not load_widths and load_src is ptx_loop:
        for m in _RE_PTX_LD_GLOBAL.finditer(ptx):
            vec = 4 if m.group(1) == "v4" else (2 if m.group(1) == "v2" else 1)
            load_widths.append(vec * (int(m.group(2)) // 8))
        for m in _RE_PTX_CP_ASYNC_SIZE.finditer(ptx):
            size_b = int(m.group(1), 0)
            if size_b > 0:
                load_widths.append(size_b)

    store_widths: list[int] = []
    for m in _RE_PTX_ST_GLOBAL.finditer(ptx):
        vec = 4 if m.group(1) == "v4" else (2 if m.group(1) == "v2" else 1)
        store_widths.append(vec * (int(m.group(2)) // 8))
    if not store_widths and _RE_PTX_TMA_BULK.search(ptx):
        store_widths.append(128)

    return _effective_ptx_width(load_widths), _effective_ptx_width(store_widths)


def _extract_mma(ttgir: str, ptx: str) -> MmaInfo | None:
    """Extract hardware MMA class, instruction shape, and dtypes."""
    m_dot = _RE_TT_DOT.search(ttgir)
    a_dtype = m_dot.group(3) if m_dot else "f16"
    b_dtype = m_dot.group(6) if m_dot else "f16"
    acc_dtype = m_dot.group(9) if m_dot else "f32"

    m_sync = _RE_PTX_MMA_SYNC.search(ptx)
    if m_sync:
        return MmaInfo(
            op_class="mma.sync",
            inst_mnk=(int(m_sync.group(1)), int(m_sync.group(2)), int(m_sync.group(3))),
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            acc_dtype=acc_dtype,
        )

    m_wgmma = _RE_PTX_WGMMA.search(ptx)
    if m_wgmma:
        return MmaInfo(
            op_class="wgmma",
            inst_mnk=(int(m_wgmma.group(1)), int(m_wgmma.group(2)), int(m_wgmma.group(3))),
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            acc_dtype=acc_dtype,
        )

    if _RE_PTX_TCGEN05.search(ptx):
        return MmaInfo(
            op_class="tcgen05",
            inst_mnk=(64, 64, 16),
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            acc_dtype=acc_dtype,
        )

    m_layout = _RE_TTGIR_MMA_LAYOUT.search(ttgir)
    if m_layout:
        inst_k = 16 if dtype_bits(a_dtype) <= 16 else 8
        return MmaInfo(
            op_class="mma.sync",
            inst_mnk=(int(m_layout.group(1)), int(m_layout.group(2)), inst_k),
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            acc_dtype=acc_dtype,
        )

    return None


def _extract_ptx_loop_sync_and_smem(
    ptx_loop: str,
    num_warps: int,
    warp_size: int,
) -> tuple[int, int, int]:
    """Count (bar.sync, st.shared bytes/CTA, ld.shared bytes/CTA) inside the PTX loop body."""
    if not ptx_loop:
        return 0, 0, 0
    bars = len(_RE_PTX_BAR_SYNC.findall(ptx_loop))
    threads = num_warps * warp_size
    st_bytes = 0
    for m in _RE_PTX_ST_SHARED.finditer(ptx_loop):
        vec = 4 if m.group(1) == "v4" else (2 if m.group(1) == "v2" else 1)
        st_bytes += vec * (int(m.group(2)) // 8) * threads
    ld_bytes = 0
    for m in _RE_PTX_LD_SHARED.finditer(ptx_loop):
        vec = 4 if m.group(1) == "v4" else (2 if m.group(1) == "v2" else 1)
        ld_bytes += vec * (int(m.group(2)) // 8) * threads
    return bars, st_bytes, ld_bytes


def _extract_all_dots(
    loop_body: str,
    ttgir: str,
    ptx: str,
    ptx_loop: str = "",
    num_warps: int = 4,
) -> tuple[list[MmaDotInfo], str | None]:
    """Extract all tt.dot operations inside the loop body with PTX-verified instruction counts.

    Returns (dots, override_diagnostic_or_none).
    """
    base_mma = _extract_mma(ttgir, ptx)
    if base_mma is None:
        return [], None
    im, in_, ik = base_mma.inst_mnk
    dots: list[MmaDotInfo] = []
    source = loop_body if loop_body else ttgir
    for m_dot in _RE_TT_DOT.finditer(source):
        tm = int(m_dot.group(1))
        tk = int(m_dot.group(2))
        a_dt = m_dot.group(3)
        tn = int(m_dot.group(5))
        b_dt = m_dot.group(6)
        acc_dt = m_dot.group(9)
        count = max(1, (tm // im) * (tn // in_) * (tk // ik))
        dots.append(
            MmaDotInfo(
                op_class=base_mma.op_class,
                tile_mnk=(tm, tn, tk),
                inst_mnk=base_mma.inst_mnk,
                a_dtype=a_dt,
                b_dtype=b_dt,
                acc_dtype=acc_dt,
                count=count,
            )
        )

    # Override analytical counts with actual PTX loop mma.sync counts * num_warps
    # so warp-replicated layouts (e.g. BM=64, 8 warps) reflect real hardware issue.
    override_diag: str | None = None
    if dots and ptx_loop and base_mma.op_class == "mma.sync":
        lines = ptx_loop.splitlines()
        mma_idxs = [i for i, ln in enumerate(lines) if "mma.sync.aligned" in ln]
        if mma_idxs:
            total_ptx = len(mma_idxs) * num_warps
            analytical_total = sum(d.count for d in dots)
            if len(dots) == 2:
                ex2_idxs = [i for i, ln in enumerate(lines) if "ex2.approx" in ln]
                if ex2_idxs:
                    split = ex2_idxs[0]
                    before = sum(1 for i in mma_idxs if i < split) * num_warps
                    after = sum(1 for i in mma_idxs if i > split) * num_warps
                    if before > 0 and after > 0:
                        if before != dots[0].count or after != dots[1].count:
                            override_diag = f"mma_count_ptx_override:ptx={before + after},tile={analytical_total}"
                        dots[0] = MmaDotInfo(
                            op_class=dots[0].op_class,
                            tile_mnk=dots[0].tile_mnk,
                            inst_mnk=dots[0].inst_mnk,
                            a_dtype=dots[0].a_dtype,
                            b_dtype=dots[0].b_dtype,
                            acc_dtype=dots[0].acc_dtype,
                            count=before,
                        )
                        dots[1] = MmaDotInfo(
                            op_class=dots[1].op_class,
                            tile_mnk=dots[1].tile_mnk,
                            inst_mnk=dots[1].inst_mnk,
                            a_dtype=dots[1].a_dtype,
                            b_dtype=dots[1].b_dtype,
                            acc_dtype=dots[1].acc_dtype,
                            count=after,
                        )
                        return dots, override_diag
            if len(dots) == 1:
                if total_ptx != analytical_total:
                    override_diag = (
                        f"mma_count_ptx_override:ptx={total_ptx},tile={analytical_total}"
                    )
                dots[0] = MmaDotInfo(
                    op_class=dots[0].op_class,
                    tile_mnk=dots[0].tile_mnk,
                    inst_mnk=dots[0].inst_mnk,
                    a_dtype=dots[0].a_dtype,
                    b_dtype=dots[0].b_dtype,
                    acc_dtype=dots[0].acc_dtype,
                    count=total_ptx,
                )
            elif total_ptx != analytical_total and analytical_total > 0:
                override_diag = f"mma_count_ptx_override:ptx={total_ptx},tile={analytical_total}"
                remaining = total_ptx
                for idx, d in enumerate(dots):
                    if idx == len(dots) - 1:
                        scaled = max(1, remaining)
                    else:
                        scaled = max(1, round(total_ptx * d.count / analytical_total))
                        remaining -= scaled
                    dots[idx] = MmaDotInfo(
                        op_class=d.op_class,
                        tile_mnk=d.tile_mnk,
                        inst_mnk=d.inst_mnk,
                        a_dtype=d.a_dtype,
                        b_dtype=d.b_dtype,
                        acc_dtype=d.acc_dtype,
                        count=scaled,
                    )
    return dots, override_diag


def parse_ttgir_summary(
    ttgir: str,
    ptx: str,
    requested_num_stages: int,
    element_bytes: int = 2,
) -> TtgirSummary:
    """Parse post-pipelining TTGIR and PTX into a structured summary."""
    diagnostics: list[str] = []
    unsupported: list[str] = []

    m_warps = _RE_NUM_WARPS.search(ttgir)
    num_warps = int(m_warps.group(1)) if m_warps else 4

    m_ws = _RE_WARP_SIZE.search(ttgir)
    warp_size = int(m_ws.group(1)) if m_ws else 32

    m_ctas = _RE_NUM_CTAS.search(ttgir)
    num_ctas = int(m_ctas.group(1)) if m_ctas else 1
    cluster_shape = (num_ctas, 1, 1)

    if "bCluster = true" in ttgir or num_ctas > 1:
        unsupported.append("cluster_smem")

    if "tt.atomic_cas" in ttgir or "tt.atomic_xchg" in ttgir:
        unsupported.append("atomic_work_stealing")

    # Detect actual async copy mechanism emitted after software pipelining.
    if "ttng.async_tma_copy_global_to_local" in ttgir:
        async_copy = "async_tensor"
    elif "ttg.async_copy_global_to_local" in ttgir:
        async_copy = "async_element"
    else:
        async_copy = "none"

    loop_bodies = _extract_loop_bodies(ttgir)
    loop_body = loop_bodies[0] if loop_bodies else ""
    num_top_level_loops = len(loop_bodies)

    # Extract materialized multi-buffer depth from 3D SMEM memdesc allocations
    # (excluding 1-element i64 mbarrier arrays).
    buf_depths = [
        int(m.group(1))
        for m in _RE_MULTIBUF_MEMDESC.finditer(ttgir)
        if not (int(m.group(2)) == 1 and m.group(4) == "i64")
    ]
    if async_copy != "none" and buf_depths:
        # Per KernelIR schema, pipeline_loop.num_stages is the actual number of
        # materialized multi-buffered SMEM pipeline buffers (not the Triton hint).
        buf_depth = max(buf_depths)
        num_stages = buf_depth
        if async_copy == "async_element":
            prefetch_distance, pf_diags, pf_unsup = _derive_cp_async_prefetch_distance(
                ptx, loop_body
            )
            diagnostics.extend(pf_diags)
            unsupported.extend(pf_unsup)
        else:
            prefetch_distance, pf_diags, pf_unsup = _derive_tma_prefetch_distance(
                ptx, ttgir, loop_body
            )
            diagnostics.extend(pf_diags)
            unsupported.extend(pf_unsup)
    else:
        prefetch_distance = 0
        num_stages = 1
        if requested_num_stages > 1:
            diagnostics.append("unpipelined_fallback")

    # Vectorized transfer width per thread (bytes), derived from PTX load/store instructions.
    ptx_loop = _extract_ptx_loop_body(ptx)
    ptx_load_width, ptx_store_width = _extract_ptx_transfer_widths(ptx, ptx_loop)
    contigs = [int(m.group(1)) for m in _RE_CONTIGUITY.finditer(ttgir)]
    if async_copy == "async_tensor":
        transfer_width_bytes = 128
    elif ptx_load_width is not None:
        transfer_width_bytes = ptx_load_width
    elif contigs:
        transfer_width_bytes = max(1, min(16, max(contigs) * element_bytes))
    else:
        transfer_width_bytes = 16
    store_transfer_width_bytes = (
        ptx_store_width if ptx_store_width is not None else transfer_width_bytes
    )

    smem_load_bytes = 0
    for m in _RE_LOCAL_LOAD.finditer(loop_body):
        rows, cols, dt = int(m.group(1)), int(m.group(2)), m.group(3)
        smem_load_bytes += (rows * cols * dtype_bits(dt)) // 8

    # Count prologue SMEM staging writes (e.g. Q staged via ttg.local_alloc %q before scf.for).
    first_for = ttgir.find("scf.for ")
    prologue_text = ttgir[:first_for] if first_for != -1 else ttgir
    prologue_smem_store_bytes = 0
    for m in _RE_PROLOGUE_LOCAL_ALLOC_VAL.finditer(prologue_text):
        rows, cols, dt = int(m.group(1)), int(m.group(2)), m.group(3)
        prologue_smem_store_bytes += (rows * cols * dtype_bits(dt)) // 8

    # Count synchronization points inside the pipelined loop.
    loop_barriers = (
        loop_body.count("ttg.async_wait")
        + loop_body.count("ttng.wait_barrier")
        + loop_body.count("gpu.barrier")
    )
    if loop_barriers == 0 and async_copy == "none":
        loop_barriers = 1

    mma = _extract_mma(ttgir, ptx)
    mma_dots, mma_override_diag = _extract_all_dots(
        loop_body, ttgir, ptx, ptx_loop=ptx_loop, num_warps=num_warps
    )
    if mma_override_diag is not None:
        diagnostics.append(mma_override_diag)
    loop_alu_per_warp, loop_sfu_per_warp, loop_shfl_per_warp = _count_ptx_loop_scalar_ops(ptx_loop)
    loop_ptx_barriers, loop_ptx_st_shared_bytes, loop_ptx_ld_shared_bytes = (
        _extract_ptx_loop_sync_and_smem(ptx_loop, num_warps, warp_size)
    )

    return TtgirSummary(
        num_warps=num_warps,
        warp_size=warp_size,
        cluster_shape=cluster_shape,
        async_copy=async_copy,
        num_stages=num_stages,
        prefetch_distance=prefetch_distance,
        transfer_width_bytes=transfer_width_bytes,
        mma=mma,
        smem_load_bytes_per_iter=smem_load_bytes,
        loop_barriers=loop_barriers,
        diagnostics=diagnostics,
        unsupported_features=unsupported,
        store_transfer_width_bytes=store_transfer_width_bytes,
        mma_dots=mma_dots,
        loop_alu_per_warp=loop_alu_per_warp,
        loop_sfu_per_warp=loop_sfu_per_warp,
        loop_shfl_per_warp=loop_shfl_per_warp,
        prologue_smem_store_bytes=prologue_smem_store_bytes,
        num_top_level_loops=num_top_level_loops,
        loop_ptx_barriers=loop_ptx_barriers,
        loop_ptx_st_shared_bytes=loop_ptx_st_shared_bytes,
        loop_ptx_ld_shared_bytes=loop_ptx_ld_shared_bytes,
    )
