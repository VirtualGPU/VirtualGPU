"""Post-hoc compute-base sweep (H7 b): the attention ablation's noload loop with extra softmax work,
and a GEMM whose loads always hit one small L2-resident tile.

  python -m microbench.attention_compute check --target sm_121
  python -m microbench.attention_compute cubins --target sm_121 --out DIR
  python -m microbench.attention_compute run --samples 15 --out x.json
"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import triton
import triton.language as tl

from microbench import attention_ablation as ablation

# 8 warps keep the ablation's 128 query rows; 4 warps use 64 so that no config spills. Every config
# runs one CTA per SM (m64n32-w4 would run two and is left out).
SHAPES = [
    {
        "config_id": f"m{bm}n{bn}-w{w}-s2",
        "block_m": bm,
        "block_n": bn,
        "num_warps": w,
        "num_stages": 2,
    }
    for bm, bn, w in ((128, 32, 8), (128, 64, 8), (64, 64, 4))
]
KNOBS = [
    (0, 0, 0),
    (2, 0, 0),
    (4, 0, 0),
    (0, 2, 0),
    (0, 4, 0),
    (0, 0, 2),
    (0, 0, 4),
]  # (EX2, ALU, RED)
GEMM = [
    {"config_id": f"m{bm}n{bn}k{bk}-w{w}-s3", "bm": bm, "bn": bn, "bk": bk, "num_warps": w}
    for bm, bn, bk in ((128, 128, 32), (128, 64, 32), (64, 64, 32))
    for w in (4, 8)
]
GEMM_STEPS = 8192  # at least 1 ms per launch


@triton.jit
def resident_gemm(
    A, B, C, STEPS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    """Each program multiplies the same BM x BK and BK x BN tiles STEPS times: every load is an L2 hit."""
    pid = tl.program_id(0)
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for step in tl.range(0, STEPS):
        a = tl.load(A + rm[:, None] * BK + rk[None, :] + (step % 2) * BM * BK)
        b = tl.load(B + rk[:, None] * BN + rn[None, :] + (step % 2) * BK * BN)
        acc = tl.dot(a, b, acc)
    tl.store(C + pid * BM * BN + rm[:, None] * BN + rn[None, :], acc.to(tl.float16))


def attention_points():
    for shape in SHAPES:
        for knobs in KNOBS:
            yield shape, knobs


def attention_spec(shape, knobs):
    constants, options = ablation.specialize(shape, 1)
    return (*constants, *knobs), options


def compile_point(kernel, target, pointers, constants, options):
    from virtualgpu.frontend.compiler import compile_triton_aot, specialize_triton_call

    parsed, sig, cx, attrs = specialize_triton_call(
        kernel, target, (*pointers, *constants), options
    )
    return compile_triton_aot(
        kernel,
        sig,
        cx,
        target,
        options["num_warps"],
        options["num_stages"],
        attrs=attrs,
        options=parsed,
        use_cache=False,
    )


def artifacts(target):
    from virtualgpu.frontend import MockTensor

    from validation.attention_runner import _AOT_POINTERS

    att_ptrs = (*_AOT_POINTERS, MockTensor("i64"))
    for shape, knobs in attention_points():
        constants, options = attention_spec(shape, knobs)
        art = compile_point(ablation.ablated_attention, target, att_ptrs, constants, options)
        yield "attention", shape, knobs, art
    gemm_ptrs = tuple(MockTensor(t) for t in ("f16", "f16", "f16"))
    for g in GEMM:
        options = {"num_warps": g["num_warps"], "num_stages": 3}
        art = compile_point(
            resident_gemm, target, gemm_ptrs, (GEMM_STEPS, g["bm"], g["bn"], g["bk"]), options
        )
        yield "gemm", g, None, art


def check(target):
    for family, shape, knobs, art in artifacts(target):
        print(
            json.dumps(
                {
                    "family": family,
                    "config": shape["config_id"],
                    "knobs": knobs,
                    "registers": art.ptxas.registers_per_thread,
                    "static_smem": art.static_smem_bytes,
                    "spill_bytes": art.ptxas.spill_stores_bytes + art.ptxas.spill_loads_bytes,
                    "ctas_per_sm": ablation.ctas_per_sm(
                        art.ptxas.registers_per_thread,
                        art.static_smem_bytes,
                        shape["num_warps"],
                        target,
                    ),
                }
            )
        )


def cubins(target, out):
    from virtualgpu.frontend.compiler import find_bundled_ptxas

    ptxas = find_bundled_ptxas(int(target.split("_")[1]))[1]
    out.mkdir(parents=True, exist_ok=True)
    for family, shape, knobs, art in artifacts(target):
        name = f"{family}-{shape['config_id']}" + (
            f"-e{knobs[0]}a{knobs[1]}r{knobs[2]}" if knobs else ""
        )
        ptx = out / f"{name}.ptx"
        ptx.write_text(art.ptx)
        subprocess.run(
            [
                ptxas,
                "-lineinfo",
                "--regAllocOptLevel=2",
                f"--gpu-name={target}a",
                str(ptx),
                "-o",
                str(out / f"{name}.cubin"),
            ],
            check=True,
        )
        print(name)


def run(samples, out):
    import torch

    from validation.attention import launch_spec
    from validation.runner import timed_graph

    case = ablation.CASE
    batch, hq, hk, seq, dim = (
        case[k] for k in ("batch", "query_heads", "kv_heads", "seq_len", "head_dim")
    )
    torch.manual_seed(7)
    q = torch.randn((batch, hq, seq, dim), device="cuda", dtype=torch.float16)
    k = torch.randn((batch, hk, seq, dim), device="cuda", dtype=torch.float16)
    v = torch.randn((batch, hk, seq, dim), device="cuda", dtype=torch.float16)
    o = torch.empty_like(q)
    lse = torch.empty((batch, hq, seq), device="cuda", dtype=torch.float32)
    ts = torch.zeros(1, device="cuda", dtype=torch.int64)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    rows = []

    def timed(name, operation, extra):
        operation()
        torch.cuda.synchronize()
        elapsed = timed_graph(lambda: None, operation)
        times = [elapsed() for _ in range(samples)]
        rows.append(
            {
                **extra,
                "sm_count": sms,
                "kernel_ns_samples": times,
                "kernel_ns_median": statistics.median(times),
            }
        )
        print(f"{name}: {statistics.median(times) / 1e3:.1f} us", file=sys.stderr)

    for shape, knobs in attention_points():
        grid, _, _ = launch_spec(case, shape)
        constants, options = attention_spec(shape, knobs)

        def op(constants=constants, options=options, grid=grid):
            ablation.ablated_attention[grid](q, k, v, o, lse, ts, *constants, **options)

        extra = {
            "family": "attention",
            "config_id": shape["config_id"],
            "knobs": knobs,
            "grid": list(grid),
        }
        timed(
            f"{shape['config_id']} {knobs}",
            op,
            extra | {"iterations_per_cta": seq // shape["block_n"]},
        )
    for g in GEMM:
        a = torch.randn((2 * g["bm"] * g["bk"],), device="cuda", dtype=torch.float16)
        b = torch.randn((2 * g["bk"] * g["bn"],), device="cuda", dtype=torch.float16)
        c = torch.empty((sms * g["bm"] * g["bn"],), device="cuda", dtype=torch.float16)

        def op(g=g, a=a, b=b, c=c):
            resident_gemm[(sms,)](
                a,
                b,
                c,
                GEMM_STEPS,
                g["bm"],
                g["bn"],
                g["bk"],
                num_warps=g["num_warps"],
                num_stages=3,
            )

        timed(
            g["config_id"],
            op,
            {
                "family": "gemm",
                "config_id": g["config_id"],
                "grid": [sms],
                "iterations_per_cta": GEMM_STEPS,
            },
        )
    Path(out).write_text(json.dumps({"case": case, "rows": rows}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "cubins"):
        c = sub.add_parser(name)
        c.add_argument("--target", default="sm_121", choices=sorted(ablation.PROFILES))
        if name == "cubins":
            c.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--samples", type=int, default=15)
    r.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "check":
        check(args.target)
    elif args.command == "cubins":
        cubins(args.target, args.out)
    else:
        run(args.samples, args.out)


if __name__ == "__main__":
    main()
