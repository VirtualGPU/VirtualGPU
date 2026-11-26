"""Minimal TTGIR scalar evaluator for recovering swizzled per-CTA tile footprints."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PtrVal:
    """Symbolic pointer rooted at a kernel pointer argument with an element offset."""

    arg_index: int
    elem_offset: int = 0


@dataclass(frozen=True)
class ScalarEvalResult:
    """Per-CTA evaluated base element offsets for each pointer argument."""

    # Maps pointer arg index -> list of element offsets (length = grid_size)
    ptr_elem_offsets: dict[int, list[int]]
    # Optional 2D TMA coordinates: ptr arg index -> list of (coord0, coord1)
    tma_coords: dict[int, list[tuple[int, int]]]
    data_dependent: bool = False
    has_nested_loop: bool = False
    # Optional per-CTA total trip count summed across top-level scf.for loops
    loop_trip_counts: list[int] | None = None


_RE_FUNC_HEADER = re.compile(r"tt\.func\s+[^{]+\((.*)\)\s*(?:attributes\s*\{[^}]*\}\s*)?\{")
_RE_PARAM = re.compile(r"(%[A-Za-z0-9_]+)\s*:\s*([^,)]+)")
_RE_ASSIGN = re.compile(r"^\s*(%[A-Za-z0-9_]+)\s*=\s*(.+)$")
_RE_CONST_SCALAR = re.compile(r"^arith\.constant\s+(-?\d+|true|false)\s*:")
_RE_CONST_DENSE = re.compile(r"^arith\.constant\s+dense<(-?\d+|true|false)>\s*:")
_RE_GET_PID = re.compile(r"^tt\.get_program_id\s+(x|y|z)\b")
_RE_GET_NPROG = re.compile(r"^tt\.get_num_programs\s+(x|y|z)\b")
_RE_MAKE_RANGE = re.compile(r"^tt\.make_range\s*\{[^}]*start\s*=\s*(-?\d+)")
_RE_BINOP = re.compile(
    r"^arith\.(addi|subi|muli|divsi|divui|remsi|remui|minsi|maxsi|minui|maxui|andi|ori|xori|shli|shrsi|shrui)\s+(%[A-Za-z0-9_]+),\s*(%[A-Za-z0-9_]+)\b"
)
_RE_CMPI = re.compile(r"^arith\.cmpi\s+([a-z]+),\s*(%[A-Za-z0-9_]+),\s*(%[A-Za-z0-9_]+)\b")
_RE_SELECT = re.compile(
    r"^arith\.select\s+(%[A-Za-z0-9_]+),\s*(%[A-Za-z0-9_]+),\s*(%[A-Za-z0-9_]+)\b"
)
_RE_UNARY_PASS = re.compile(
    r"^(?:arith\.(?:extsi|extui|trunci|index_cast)|tt\.(?:splat|expand_dims|broadcast|bitcast|reshape)|ttg\.convert_layout)\s+(%[A-Za-z0-9_]+)\b"
)
_RE_ADDPTR = re.compile(r"^tt\.addptr\s+(%[A-Za-z0-9_]+),\s*(%[A-Za-z0-9_]+)\b")
_RE_TENSORMAP_CREATE = re.compile(r"ttng\.tensormap_create\s+(%[A-Za-z0-9_]+),\s*(%[A-Za-z0-9_]+),")
_RE_REINTERPRET_DESC = re.compile(r"^ttng\.reinterpret_tensor_descriptor\s+(%[A-Za-z0-9_]+)\b")
_RE_TMA_LOAD_STORE = re.compile(
    r"ttng\.async_tma_copy_(?:global_to_local|local_to_global)\s+(%[A-Za-z0-9_]+)\[(%[A-Za-z0-9_]+),\s*(%[A-Za-z0-9_]+)\]"
)
_RE_LOAD_USE = re.compile(r"(?:ttg\.async_copy_global_to_local|tt\.load)\s+(%[A-Za-z0-9_]+)\b")
_RE_STORE_USE = re.compile(r"tt\.store\s+(%[A-Za-z0-9_]+)\b")
_RE_SCF_FOR_IV = re.compile(r"scf\.for\s+(%[A-Za-z0-9_]+)\s*=\s*(%[A-Za-z0-9_]+)\b")
_RE_SCF_FOR_BOUNDS = re.compile(
    r"scf\.for\s+(%[A-Za-z0-9_]+)\s*=\s*(%[A-Za-z0-9_]+)\s+to\s+(%[A-Za-z0-9_]+)\s+step\s+(%[A-Za-z0-9_]+)\b"
)
_RE_SCF_FOR_ITER_ARGS = re.compile(r"iter_args\(([^)]*)\)")
_RE_ITER_BIND = re.compile(r"(%[A-Za-z0-9_]+)\s*=\s*(%[A-Za-z0-9_]+)")


def _c_div(a: int, b: int) -> int:
    if b == 0:
        return 0
    return int(a / b)


def _c_rem(a: int, b: int) -> int:
    if b == 0:
        return 0
    return a - int(a / b) * b


def _eval_cmpi(pred: str, a: int, b: int) -> int:
    if pred == "eq":
        return int(a == b)
    if pred == "ne":
        return int(a != b)
    if pred in ("slt", "ult"):
        return int(a < b)
    if pred in ("sle", "ule"):
        return int(a <= b)
    if pred in ("sgt", "ugt"):
        return int(a > b)
    if pred in ("sge", "uge"):
        return int(a >= b)
    return 0


def _parse_func_params(ttgir: str) -> list[tuple[str, bool]]:
    """Return list of (ssa_name, is_ptr) for the tt.func parameters."""
    m = _RE_FUNC_HEADER.search(ttgir)
    if not m:
        return []
    param_str = m.group(1)
    # Strip loc(...) annotations inside parameter list before splitting
    cleaned = re.sub(r'loc\("[^"]*"\([^)]*\)\)', "", param_str)
    cleaned = re.sub(r"loc\([^)]*\)", "", cleaned)
    cleaned = re.sub(r"\{[^}]*\}", "", cleaned)
    params: list[tuple[str, bool]] = []
    for part in cleaned.split(","):
        pm = _RE_PARAM.search(part.strip())
        if pm:
            ssa, ty = pm.group(1), pm.group(2).strip()
            params.append((ssa, "!tt.ptr<" in ty))
    return params


def _extract_top_level_lines(ttgir: str) -> tuple[list[str], list[str], list[str]]:
    """Split tt.func body into top-level lines (outside scf.for), inner-loop lines, and scf.for headers."""
    func_pos = ttgir.find("tt.func ")
    if func_pos == -1:
        return [], [], []
    body_start = ttgir.find("{", func_pos)
    if body_start == -1:
        return [], [], []

    top_lines: list[str] = []
    loop_lines: list[str] = []
    loop_headers: list[str] = []
    depth = 0
    in_loop = False
    for raw in ttgir[body_start + 1 :].splitlines():
        line = raw.strip()
        if not line:
            continue
        opens = line.count("{")
        closes = line.count("}")
        if depth == 0 and line.startswith("}"):
            break
        if depth == 0 and "scf.for " in line:
            in_loop = True
            loop_headers.append(line)
            depth += opens - closes
            continue
        if in_loop:
            depth += opens - closes
            if depth <= 0:
                in_loop = False
                depth = 0
            else:
                loop_lines.append(line)
        else:
            top_lines.append(line)
            depth += opens - closes
    return top_lines, loop_lines, loop_headers


def _parse_scf_for_bindings(loop_headers: list[str]) -> list[tuple[str, ...]]:
    """Bind each scf.for induction variable and iter_args to their iteration-0 values."""
    bindings: list[tuple[str, ...]] = []
    for header in loop_headers:
        m_iv = _RE_SCF_FOR_IV.search(header)
        if m_iv:
            bindings.append(("pass", m_iv.group(1), m_iv.group(2)))
        m_iter = _RE_SCF_FOR_ITER_ARGS.search(header)
        if m_iter:
            for dst, src in _RE_ITER_BIND.findall(m_iter.group(1)):
                bindings.append(("pass", dst, src))
    return bindings


def _parse_scf_for_bounds(loop_headers: list[str]) -> list[tuple[str, str, str]]:
    """Return (lb_ssa, ub_ssa, step_ssa) for each top-level scf.for header."""
    bounds: list[tuple[str, str, str]] = []
    for header in loop_headers:
        m = _RE_SCF_FOR_BOUNDS.search(header)
        if m:
            bounds.append((m.group(2), m.group(3), m.group(4)))
    return bounds


def _compile_instructions(
    lines: list[str],
) -> tuple[
    list[tuple[str, ...]],
    dict[str, str],
    list[tuple[str, str, str]],
    list[str],
]:
    """Parse SSA lines into compact evaluator instructions plus pointer/TMA sinks."""
    instrs: list[tuple[str, ...]] = []
    # desc_scratch_ssa -> ptr_ssa, desc_ssa -> scratch_ssa
    scratch_to_ptr: dict[str, str] = {}
    tma_uses: list[tuple[str, str, str]] = []  # (desc_ssa, c0_ssa, c1_ssa)
    ptr_sinks: list[str] = []

    for line in lines:
        m_tm = _RE_TENSORMAP_CREATE.search(line)
        if m_tm:
            scratch_to_ptr[m_tm.group(1)] = m_tm.group(2)
            continue

        m_tma = _RE_TMA_LOAD_STORE.search(line)
        if m_tma:
            tma_uses.append((m_tma.group(1), m_tma.group(2), m_tma.group(3)))
            continue

        m_ld = _RE_LOAD_USE.search(line)
        if m_ld:
            ptr_sinks.append(m_ld.group(1))
        m_st = _RE_STORE_USE.search(line)
        if m_st:
            ptr_sinks.append(m_st.group(1))

        ma = _RE_ASSIGN.match(line)
        if not ma:
            continue
        dst, rhs = ma.group(1), ma.group(2)

        if m := _RE_CONST_SCALAR.match(rhs):
            v = m.group(1)
            val = 1 if v == "true" else (0 if v == "false" else int(v))
            instrs.append(("const", dst, str(val)))
        elif m := _RE_CONST_DENSE.match(rhs):
            v = m.group(1)
            val = 1 if v == "true" else (0 if v == "false" else int(v))
            instrs.append(("const", dst, str(val)))
        elif m := _RE_GET_PID.match(rhs):
            instrs.append(("pid", dst, m.group(1)))
        elif m := _RE_GET_NPROG.match(rhs):
            instrs.append(("nprog", dst, m.group(1)))
        elif m := _RE_MAKE_RANGE.match(rhs):
            instrs.append(("const", dst, m.group(1)))
        elif m := _RE_BINOP.match(rhs):
            instrs.append(("binop", dst, m.group(1), m.group(2), m.group(3)))
        elif m := _RE_CMPI.match(rhs):
            instrs.append(("cmpi", dst, m.group(1), m.group(2), m.group(3)))
        elif m := _RE_SELECT.match(rhs):
            instrs.append(("select", dst, m.group(1), m.group(2), m.group(3)))
        elif m := _RE_ADDPTR.match(rhs):
            instrs.append(("addptr", dst, m.group(1), m.group(2)))
        elif m := _RE_REINTERPRET_DESC.match(rhs):
            if m.group(1) in scratch_to_ptr:
                scratch_to_ptr[dst] = scratch_to_ptr[m.group(1)]
        elif m := _RE_UNARY_PASS.match(rhs):
            instrs.append(("pass", dst, m.group(1)))
        elif rhs.startswith("tt.load "):
            instrs.append(("load_unknown", dst))

    return instrs, scratch_to_ptr, tma_uses, ptr_sinks


def evaluate_ttgir_cta_offsets(
    ttgir: str,
    grid: tuple[int, int, int],
    runtime_int_args: dict[str, int],
    param_int_values: list[int | None] | None = None,
) -> ScalarEvalResult:
    """Evaluate TTGIR address arithmetic for every CTA in grid.

    Recovers exact per-CTA base element offsets (and TMA 2D coordinates)
    for arbitrary swizzle expressions (e.g., GROUP_SIZE_M, diagonal, 2D grids)
    without pattern-matching kernel idioms.
    """
    params = _parse_func_params(ttgir)
    top_lines, loop_lines, loop_headers = _extract_top_level_lines(ttgir)
    has_nested_loop = any("scf.for " in ln for ln in loop_lines)
    if has_nested_loop:
        # Persistent kernels compute per-tile coordinates inside an outer scf.for;
        # do not silently return incomplete single-iteration offsets.
        return ScalarEvalResult(
            ptr_elem_offsets={},
            tma_coords={},
            data_dependent=False,
            has_nested_loop=True,
        )

    instrs, scratch_to_ptr, tma_uses, ptr_sinks = _compile_instructions(top_lines)

    # Evaluate iteration 0 of scf.for (binding IV -> lower bound and iter_args -> inits)
    # so unpipelined kernels (e.g. num_stages=1 where tt.addptr + tt.load stay inside
    # the loop body) recover exact tile offsets without falling back to linear maps.
    loop_instrs, loop_scratch_to_ptr, loop_tma, loop_sinks = _compile_instructions(loop_lines)
    if loop_instrs or loop_sinks or loop_tma:
        instrs = instrs + _parse_scf_for_bindings(loop_headers) + loop_instrs
        for k, v in loop_scratch_to_ptr.items():
            scratch_to_ptr.setdefault(k, v)
    all_ptr_sinks = ptr_sinks + [s for s in loop_sinks if s not in ptr_sinks]
    all_tma_uses = tma_uses + [u for u in loop_tma if u not in tma_uses]
    loop_bounds_ssa = _parse_scf_for_bounds(loop_headers)

    # Build compact integer-indexed slots instead of per-CTA dict copies.
    ssa_to_slot: dict[str, int] = {}

    def _slot(name: str) -> int:
        idx = ssa_to_slot.get(name)
        if idx is None:
            idx = len(ssa_to_slot)
            ssa_to_slot[name] = idx
        return idx

    # Seed parameter slots and track which slots are invariant across CTAs.
    ptr_arg_indices: dict[str, int] = {}
    ptr_counter = 0
    param_values: dict[int, Any] = {}
    for idx, (ssa, is_ptr) in enumerate(params):
        sl = _slot(ssa)
        if is_ptr:
            param_values[sl] = PtrVal(arg_index=ptr_counter, elem_offset=0)
            ptr_arg_indices[ssa] = ptr_counter
            ptr_counter += 1
        else:
            name = ssa.lstrip("%")
            if (
                param_int_values is not None
                and idx < len(param_int_values)
                and param_int_values[idx] is not None
            ):
                param_values[sl] = int(param_int_values[idx])  # type: ignore[arg-type]
            elif name in runtime_int_args:
                param_values[sl] = int(runtime_int_args[name])

    # Pre-assign slots for every SSA symbol referenced by instructions or sinks.
    for ins in instrs:
        op = ins[0]
        _slot(ins[1])
        if op == "pass":
            _slot(ins[2])
        elif op in ("binop", "cmpi", "select"):
            _slot(ins[3])
            _slot(ins[4])
            if op == "select":
                _slot(ins[2])
        elif op == "addptr":
            _slot(ins[2])
            _slot(ins[3])

    sink_slots: list[int] = [_slot(s) for s in all_ptr_sinks]
    loop_bound_slots: list[tuple[int, int, int]] = [
        (_slot(lb), _slot(ub), _slot(st)) for lb, ub, st in loop_bounds_ssa
    ]
    desc_to_ptr_idx: dict[str, int] = {}
    for desc_ssa, ptr_ssa in scratch_to_ptr.items():
        if ptr_ssa in ptr_arg_indices:
            desc_to_ptr_idx[desc_ssa] = ptr_arg_indices[ptr_ssa]
    tma_sink_slots: list[tuple[int, int, int]] = []
    for desc_ssa, c0_ssa, c1_ssa in all_tma_uses:
        p_idx = desc_to_ptr_idx.get(desc_ssa)
        if p_idx is not None:
            tma_sink_slots.append((p_idx, _slot(c0_ssa), _slot(c1_ssa)))

    gx, gy, gz = grid
    slots: list[Any] = [None] * len(ssa_to_slot)
    for sl, val in param_values.items():
        slots[sl] = val

    # Partition instructions into CTA-invariant (evaluated once) and PID-dependent
    # (evaluated per CTA in-place on `slots`, avoiding dict copies).
    pid_dep: set[int] = set()
    var_instrs: list[tuple[Any, ...]] = []
    data_dependent = False

    def _exec_one(ins_code: tuple[Any, ...], px: int, py: int, pz: int) -> None:
        nonlocal data_dependent
        op = ins_code[0]
        dst = ins_code[1]
        if op == "const":
            slots[dst] = ins_code[2]
        elif op == "pid":
            axis = ins_code[2]
            slots[dst] = px if axis == 0 else (py if axis == 1 else pz)
        elif op == "nprog":
            axis = ins_code[2]
            slots[dst] = gx if axis == 0 else (gy if axis == 1 else gz)
        elif op == "pass":
            val = slots[ins_code[2]]
            if val is not None:
                slots[dst] = val
        elif op == "binop":
            bop, a_sl, b_sl = ins_code[2], ins_code[3], ins_code[4]
            av, bv = slots[a_sl], slots[b_sl]
            if isinstance(av, int) and isinstance(bv, int):
                if bop == "addi":
                    slots[dst] = av + bv
                elif bop == "subi":
                    slots[dst] = av - bv
                elif bop == "muli":
                    slots[dst] = av * bv
                elif bop in ("divsi", "divui"):
                    slots[dst] = _c_div(av, bv)
                elif bop in ("remsi", "remui"):
                    slots[dst] = _c_rem(av, bv)
                elif bop in ("minsi", "minui"):
                    slots[dst] = av if av < bv else bv
                elif bop in ("maxsi", "maxui"):
                    slots[dst] = av if av > bv else bv
                elif bop == "andi":
                    slots[dst] = av & bv
                elif bop == "ori":
                    slots[dst] = av | bv
                elif bop == "xori":
                    slots[dst] = av ^ bv
                elif bop == "shli":
                    slots[dst] = av << bv
                elif bop in ("shrsi", "shrui"):
                    slots[dst] = av >> bv
            elif av == "_DATA_DEP" or bv == "_DATA_DEP":
                slots[dst] = "_DATA_DEP"
        elif op == "cmpi":
            pred, a_sl, b_sl = ins_code[2], ins_code[3], ins_code[4]
            av, bv = slots[a_sl], slots[b_sl]
            if isinstance(av, int) and isinstance(bv, int):
                slots[dst] = _eval_cmpi(pred, av, bv)
        elif op == "select":
            c_sl, t_sl, f_sl = ins_code[2], ins_code[3], ins_code[4]
            cv = slots[c_sl]
            if isinstance(cv, int):
                chosen = slots[t_sl] if cv else slots[f_sl]
                if chosen is not None:
                    slots[dst] = chosen
        elif op == "addptr":
            p_sl, o_sl = ins_code[2], ins_code[3]
            pv, ov = slots[p_sl], slots[o_sl]
            if isinstance(pv, PtrVal) and isinstance(ov, int):
                slots[dst] = PtrVal(pv.arg_index, pv.elem_offset + ov)
            elif ov == "_DATA_DEP":
                data_dependent = True
        elif op == "load_unknown":
            slots[dst] = "_DATA_DEP"

    axis_map = {"x": 0, "y": 1, "z": 2}
    for ins in instrs:
        op = ins[0]
        dst = ssa_to_slot[ins[1]]
        if op == "const":
            compiled_ins: tuple[Any, ...] = ("const", dst, int(ins[2]))
            depends = False
        elif op == "pid":
            compiled_ins = ("pid", dst, axis_map[ins[2]])
            depends = True
        elif op == "nprog":
            compiled_ins = ("nprog", dst, axis_map[ins[2]])
            depends = False
        elif op == "pass":
            src_sl = ssa_to_slot[ins[2]]
            compiled_ins = ("pass", dst, src_sl)
            depends = src_sl in pid_dep
        elif op == "binop":
            a_sl, b_sl = ssa_to_slot[ins[3]], ssa_to_slot[ins[4]]
            compiled_ins = ("binop", dst, ins[2], a_sl, b_sl)
            depends = (a_sl in pid_dep) or (b_sl in pid_dep)
        elif op == "cmpi":
            a_sl, b_sl = ssa_to_slot[ins[3]], ssa_to_slot[ins[4]]
            compiled_ins = ("cmpi", dst, ins[2], a_sl, b_sl)
            depends = (a_sl in pid_dep) or (b_sl in pid_dep)
        elif op == "select":
            c_sl, t_sl, f_sl = ssa_to_slot[ins[2]], ssa_to_slot[ins[3]], ssa_to_slot[ins[4]]
            compiled_ins = ("select", dst, c_sl, t_sl, f_sl)
            depends = (c_sl in pid_dep) or (t_sl in pid_dep) or (f_sl in pid_dep)
        elif op == "addptr":
            p_sl, o_sl = ssa_to_slot[ins[2]], ssa_to_slot[ins[3]]
            compiled_ins = ("addptr", dst, p_sl, o_sl)
            depends = (p_sl in pid_dep) or (o_sl in pid_dep)
        else:
            compiled_ins = ("load_unknown", dst)
            depends = False

        if depends:
            pid_dep.add(dst)
            var_instrs.append(compiled_ins)
        else:
            _exec_one(compiled_ins, 0, 0, 0)

    ptr_elem_offsets: dict[int, list[int]] = {}
    tma_coords: dict[int, list[tuple[int, int]]] = {}
    loop_trip_counts: list[int] | None = [] if loop_bound_slots else None

    for pz in range(gz):
        for py in range(gy):
            for px in range(gx):
                for ins_code in var_instrs:
                    _exec_one(ins_code, px, py, pz)

                seen_ptrs: set[int] = set()
                for sl in sink_slots:
                    val = slots[sl]
                    if isinstance(val, PtrVal) and val.arg_index not in seen_ptrs:
                        seen_ptrs.add(val.arg_index)
                        ptr_elem_offsets.setdefault(val.arg_index, []).append(val.elem_offset)

                if tma_sink_slots:
                    seen_tma: set[int] = set()
                    for p_idx, c0_sl, c1_sl in tma_sink_slots:
                        if p_idx in seen_tma:
                            continue
                        c0, c1 = slots[c0_sl], slots[c1_sl]
                        if isinstance(c0, int) and isinstance(c1, int):
                            seen_tma.add(p_idx)
                            tma_coords.setdefault(p_idx, []).append((c0, c1))

                if loop_trip_counts is not None:
                    cta_trips = 0
                    for lb_sl, ub_sl, st_sl in loop_bound_slots:
                        lb_v, ub_v, st_v = slots[lb_sl], slots[ub_sl], slots[st_sl]
                        if (
                            isinstance(lb_v, int)
                            and isinstance(ub_v, int)
                            and isinstance(st_v, int)
                            and st_v > 0
                        ):
                            diff = ub_v - lb_v
                            cta_trips += (diff + st_v - 1) // st_v if diff > 0 else 0
                        else:
                            loop_trip_counts = None
                            break
                    if loop_trip_counts is not None:
                        loop_trip_counts.append(cta_trips)

    return ScalarEvalResult(
        ptr_elem_offsets=ptr_elem_offsets,
        tma_coords=tma_coords,
        data_dependent=data_dependent,
        has_nested_loop=False,
        loop_trip_counts=loop_trip_counts,
    )
