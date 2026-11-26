"""Post-hoc ablation of the fused attention kernel's single-buffer round trip.

A copy of validation/attention.py's kernel with one compile-time switch. ABLATE:
0 = unchanged, 1 = no loads in the loop (K and V read once before it, with a per-tile
QK accumulator offset so the QK product is not hoisted out of the loop), 2 = the next
tile's load mask depends on the PV accumulator, 3 = the same reduction as 2 but kept
off the loads. Shapes are development shapes only.

  python -m microbench.attention_ablation check --target sm_121  # CPU: compile and count
  python -m microbench.attention_ablation run --out x.json  # GPU: warm timings
"""

import argparse
import json
import re
import statistics
import sys

import triton
import triton.language as tl

from validation.attention import launch_spec

CASE = {
    "case_id": "b2-q16-kv16-s2048-d128-full",
    "batch": 2,
    "query_heads": 16,
    "kv_heads": 16,
    "seq_len": 2048,
    "head_dim": 128,
    "causal": False,
}
CONFIGS = [
    {"config_id": cid, "block_m": bm, "block_n": bn, "num_warps": w, "num_stages": s}
    for cid, bm, bn, w, s in (
        ("m128n32-w8-s2", 128, 32, 8, 2),
        ("m128n32-w8-s3", 128, 32, 8, 3),
        ("m128n32-w4-s2", 128, 32, 4, 2),
        ("m128n32-w4-s3", 128, 32, 4, 3),
        ("m128n64-w8-s2", 128, 64, 8, 2),
        ("m128n64-w8-s3", 128, 64, 8, 3),
    )
]
VARIANTS = {0: "unchanged", 1: "noload", 2: "dep", 3: "dep_cost", 4: "timestamp"}


@triton.jit
def _tiles(
    q,
    k_base,
    v_base,
    rows,
    features,
    acc,
    normalizer,
    maximum,
    start,
    stop,
    stamps,
    SEQ: tl.constexpr,
    D: tl.constexpr,
    BN: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    ABLATE: tl.constexpr,
):
    offsets = tl.arange(0, BN)
    live = start >= 0
    if ABLATE == 1:
        first_cols = start + offsets
        first_addresses = first_cols[:, None] * D + features[None, :]
        k_fixed = tl.load(k_base + first_addresses, first_cols[:, None] < SEQ, other=0)
        v_fixed = tl.load(v_base + first_addresses, first_cols[:, None] < SEQ, other=0)
    for tile in tl.range(start, stop, BN):
        cols = tile + offsets
        addresses = cols[:, None] * D + features[None, :]
        inside = (cols[:, None] < SEQ) & live
        if ABLATE == 1:
            k = k_fixed
            v = v_fixed
        else:
            k = tl.load(k_base + addresses, inside, other=0)
            v = tl.load(v_base + addresses, inside, other=0)
        if ABLATE == 1:
            # A per-tile row offset keeps the loop-invariant QK product inside
            # the loop; softmax is unchanged by it.
            offset = tl.zeros((q.shape[0], BN), tl.float32) + tile.to(tl.float32)
            logits = tl.dot(q, tl.trans(k), offset) * SCALE_LOG2
        else:
            logits = tl.dot(q, tl.trans(k)) * SCALE_LOG2
        logits = tl.where(cols[None, :] < SEQ, logits, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(logits, 1))
        correction = tl.math.exp2(maximum - new_max)
        weights = tl.math.exp2(logits - new_max[:, None])
        acc = tl.dot(weights.to(tl.float16), v, acc * correction[:, None])
        normalizer = normalizer * correction + tl.sum(weights, 1)
        maximum = new_max
        if ABLATE == 4:
            # %globaltimer after the PV product, before the next tile's copies are issued.
            now = tl.inline_asm_elementwise(
                "mov.u64 $0, %globaltimer;", "=l,r", [tile], dtype=tl.int64, is_pure=False, pack=1
            )
            tl.store(stamps + (tile - start) // BN, now)
        if ABLATE == 2 or ABLATE == 3:
            probe = tl.max(tl.max(acc, 1), 0)
            if ABLATE == 2:
                live = probe != -12345.0
            else:
                normalizer = normalizer + tl.where(probe == -12345.0, 1.0, 0.0)
    return acc, normalizer, maximum


@triton.jit
def ablated_attention(
    Q,
    K,
    V,
    Out,
    LSE,
    TS,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    SEQ: tl.constexpr,
    D: tl.constexpr,
    CAUSAL: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    ABLATE: tl.constexpr,
):
    query_block = tl.program_id(0)
    head_batch = tl.program_id(1)
    batch = head_batch // HQ
    kv_head = (head_batch % HQ) // (HQ // HK)
    q_base = head_batch * SEQ * D
    kv_base = (batch * HK + kv_head) * SEQ * D
    rows = query_block * BM + tl.arange(0, BM)
    features = tl.arange(0, D)
    addresses = rows[:, None] * D + features[None, :]
    q = tl.load(Q + q_base + addresses, rows[:, None] < SEQ, other=0)
    acc = tl.full((BM, D), 0, tl.float32)
    normalizer = tl.full((BM,), 0, tl.float32)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    acc, normalizer, maximum = _tiles(
        q,
        K + kv_base,
        V + kv_base,
        rows,
        features,
        acc,
        normalizer,
        maximum,
        0,
        SEQ,
        TS + (head_batch * tl.num_programs(0) + query_block) * (SEQ // BN),
        SEQ,
        D,
        BN,
        SCALE_LOG2,
        ABLATE,
    )
    tl.store(
        Out + q_base + addresses, (acc / normalizer[:, None]).to(tl.float16), rows[:, None] < SEQ
    )
    tl.store(LSE + head_batch * SEQ + rows, maximum + tl.math.log2(normalizer), rows < SEQ)


def specialize(config, ablate):
    _, constants, options = launch_spec(CASE, config)
    return (*constants[:4], False, *constants[5:], ablate), options


PROFILES = {"sm_89": "profiles/rtx4070.json", "sm_121": "profiles/gb10.json"}


def ctas_per_sm(registers, smem, warps, target):
    """Resident CTAs per SM from the target's profile specs, for one variant."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root / PROFILES[target]).read_text())["specs"]
    unit = spec["register_alloc_unit_regs"]
    per_warp = -(-registers * spec["warp_size"] // unit) * unit
    gran = spec["shared_mem_alloc_granularity_bytes"]
    footprint = -(-smem // gran) * gran + spec["reserved_shared_mem_per_cta_bytes"]
    return min(
        spec["registers_per_sm"] // (per_warp * warps),
        spec["shared_mem_per_sm_bytes"] // footprint,
        spec["max_warps_per_sm"] // warps,
        spec["max_blocks_per_sm"],
    )


def loop_mma_count(ptx):
    """mma.sync instructions inside backward-branch loops of the PTX."""
    lines = ptx.splitlines()
    labels = {
        m.group(1): i for i, line in enumerate(lines) if (m := re.match(r"\s*(\$L__\w+):", line))
    }
    count = 0
    for i, line in enumerate(lines):
        m = re.search(r"bra(?:\.uni)?\s+(\$L__\w+);", line)
        if m and labels.get(m.group(1), i) < i:
            count += sum("mma.sync" in body for body in lines[labels[m.group(1)] : i])
    return count


def check(target):
    """Compile every variant offline and report resources and loop instruction counts."""
    from virtualgpu.frontend import MockTensor
    from virtualgpu.frontend.compiler import compile_triton_aot, specialize_triton_call

    from validation.attention_runner import _AOT_POINTERS

    pointers = (*_AOT_POINTERS, MockTensor("i64"))

    for config in CONFIGS:
        for ablate, name in VARIANTS.items():
            constants, options = specialize(config, ablate)
            parsed, sig, cx, attrs = specialize_triton_call(
                ablated_attention, target, (*pointers, *constants), options
            )
            art = compile_triton_aot(
                ablated_attention,
                sig,
                cx,
                target,
                options["num_warps"],
                options["num_stages"],
                attrs=attrs,
                options=parsed,
                use_cache=False,
            )
            ptx = art.ptx
            counts = {
                op: len(re.findall(op, ptx))
                for op in (
                    r"cp\.async\.c[ag]\.",
                    r"cp\.async\.wait_group",
                    r"mma\.sync",
                    r"bar\.sync",
                )
            }
            print(
                json.dumps(
                    {
                        "config": config["config_id"],
                        "variant": name,
                        "registers": art.ptxas.registers_per_thread,
                        "static_smem": art.static_smem_bytes,
                        "spill_bytes": art.ptxas.spill_stores_bytes + art.ptxas.spill_loads_bytes,
                        "ctas_per_sm": ctas_per_sm(
                            art.ptxas.registers_per_thread,
                            art.static_smem_bytes,
                            config["num_warps"],
                            target,
                        ),
                        "ptx_counts": counts,
                        "loop_mma": loop_mma_count(ptx),
                    }
                )
            )


def run(samples, out):
    import torch

    from validation.runner import timed_graph

    batch, hq, hk, seq, dim = (
        CASE[k] for k in ("batch", "query_heads", "kv_heads", "seq_len", "head_dim")
    )
    torch.manual_seed(7)
    q = torch.randn((batch, hq, seq, dim), device="cuda", dtype=torch.float16)
    k = torch.randn((batch, hk, seq, dim), device="cuda", dtype=torch.float16)
    v = torch.randn((batch, hk, seq, dim), device="cuda", dtype=torch.float16)
    o = torch.empty_like(q)
    lse = torch.empty((batch, hq, seq), device="cuda", dtype=torch.float32)
    smallest = min(c["block_m"] * c["block_n"] for c in CONFIGS)
    ts = torch.zeros(batch * hq * seq * seq // smallest, device="cuda", dtype=torch.int64)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    rows = []
    for config in CONFIGS:
        grid, _, _ = launch_spec(CASE, config)
        for ablate, name in VARIANTS.items():
            constants, options = specialize(config, ablate)

            def operation(constants=constants, options=options, grid=grid):
                ablated_attention[grid](q, k, v, o, lse, ts, *constants, **options)

            operation()
            torch.cuda.synchronize()
            elapsed = timed_graph(lambda: None, operation)
            times = [elapsed() for _ in range(samples)]
            rows.append(
                {
                    "config_id": config["config_id"],
                    "variant": name,
                    "grid": list(grid),
                    "sm_count": sms,
                    "num_warps": config["num_warps"],
                    "iterations_per_cta": seq // config["block_n"],
                    "kernel_ns_samples": times,
                    "kernel_ns_median": statistics.median(times),
                }
            )
            if name == "timestamp":
                # Last launch: one %globaltimer reading per CTA and iteration, program-major
                # (program = head_batch * grid[0] + query_block), relative to the earliest.
                stamps = ts[: grid[0] * grid[1] * (seq // config["block_n"])]
                rows[-1]["stamps_ns"] = (stamps - stamps.min()).tolist()
            print(
                f"{config['config_id']} {name}: {statistics.median(times) / 1e3:.1f} us",
                file=sys.stderr,
            )
    with open(out, "w") as handle:
        json.dump({"case": CASE, "rows": rows}, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check")
    c.add_argument("--target", default="sm_89", choices=sorted(PROFILES))
    r = sub.add_parser("run")
    r.add_argument("--samples", type=int, default=15)
    r.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "check":
        check(args.target)
    else:
        run(args.samples, args.out)


if __name__ == "__main__":
    main()
