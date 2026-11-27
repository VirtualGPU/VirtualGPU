"""Post-hoc H7: the shared memory bytes ldmatrix reads, against the KernelIR's smem_load bytes.

`count` recompiles every v0.2 holdout config offline and counts, in the PTX loop body, the ldmatrix
bytes (128, 256 or 512 per .x1, .x2, .x4 warp instruction, times the warps) and the ld.shared bytes
per CTA per iteration. `counterfactual` replaces the archived KernelIR's smem_load bytes with that
sum and reruns the core (run it with a v0.2-final core and its profiles for a like-for-like check).

  python scripts/posthoc_ldmatrix_bytes.py count OUT.json
  python scripts/posthoc_ldmatrix_bytes.py counterfactual OUT.json IR_ARCHIVE V02_FINAL_CHECKOUT
"""

import argparse
import json
import os
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IR_ARCHIVE = "results/v02-formal/ir"
WIDTH = {"x1": 128, "x2": 256, "x4": 512}


def count(out, archive):
    import virtualgpu.frontend.triton_attention as frontend_attention
    import virtualgpu.frontend.triton_gemm as frontend_gemm
    from virtualgpu.frontend.ttgir import _extract_ptx_loop_body, _extract_ptx_loop_sync_and_smem

    from validation import v02_predict
    from validation.v02_inputs import COHORTS, TARGETS, case_inputs

    captured = {}

    def keep_ptx(compile_fn):
        def inner(*args, **kwargs):
            artifact = compile_fn(*args, **kwargs)
            captured["ptx"] = artifact.ptx
            return artifact

        return inner

    frontend_attention.compile_triton_aot = keep_ptx(frontend_attention.compile_triton_aot)
    frontend_gemm.compile_triton_aot = keep_ptx(frontend_gemm.compile_triton_aot)
    rows = {}
    for name in COHORTS:
        cases, configs, _, _ = case_inputs(name, "origin/main")
        family, target = name.split("-")[1], TARGETS[name.split("-")[0]]
        for case in cases:
            for path in sorted(os.listdir(f"{archive}/{name}/{case['case_id']}")):
                cid = path[:-5]
                ir = json.load(open(f"{archive}/{name}/{case['case_id']}/{path}"))
                captured.clear()
                v02_predict.frozen_ir(family, case, configs[cid], target)
                loop = _extract_ptx_loop_body(captured["ptx"])
                warps = ir["launch"]["num_warps"]
                ops = ir["body"]["pipeline_loop"]["ops"]
                # Scale an unrolled PTX body back to one iteration by its MMA count.
                unroll = len(re.findall(r"mma\.sync", loop)) / (
                    sum(o["count"] for o in ops if o["kind"] == "mma") / warps
                )
                ldmatrix = sum(
                    WIDTH[m] for m in re.findall(r"ldmatrix\.sync\.aligned\.m8n8\.(x[124])", loop)
                )
                rows[f"{name}/{case['case_id']}/{cid}"] = {
                    "ldmatrix_bytes_per_cta": ldmatrix * warps / unroll,
                    "ld_shared_bytes_per_cta": _extract_ptx_loop_sync_and_smem(loop, warps, 32)[2]
                    / unroll,
                    "ir_smem_load_bytes_per_cta": sum(
                        o.get("bytes_per_cta", 0) for o in ops if o["kind"] == "smem_load"
                    ),
                    "unroll": unroll,
                    "num_warps": warps,
                }
    Path(out).write_text(json.dumps(rows, indent=1))


def counterfactual(counts, archive, final):
    from virtualgpu import _core

    from validation.score import kendall_tau_b

    scores = json.load(open("validation/v02-holdout-scores.json"))
    profiles = {d: Path(f"{final}/profiles/{d}.json").read_text() for d in ("gb10", "rtx4070")}
    got = {}
    for key, row in sorted(json.load(open(counts)).items()):
        cohort, case, cid = key.split("/")
        ir = json.load(open(f"{archive}/{cohort}/{case}/{cid}.json"))
        smem = [o for o in ir["body"]["pipeline_loop"]["ops"] if o["kind"] == "smem_load"]
        if len(smem) != 1:
            continue
        profile = profiles[cohort.split("-")[0]]
        for cache in ("warm", "cold"):
            cfg = scores["cohorts"][cohort]["cases"][case]["cache"][cache]["core"]["configs"].get(
                cid
            )
            if not cfg or not cfg.get("nominal_latency_ns"):
                continue
            smem[0]["bytes_per_cta"] = row["ir_smem_load_bytes_per_cta"]
            base = _core.predict(json.dumps(ir), profile, l2_state=cache, draws=0, seed=17)[
                "time_ns"
            ]
            smem[0]["bytes_per_cta"] = int(
                round(row["ldmatrix_bytes_per_cta"] + row["ld_shared_bytes_per_cta"])
            )
            new = _core.predict(json.dumps(ir), profile, l2_state=cache, draws=0, seed=17)[
                "time_ns"
            ]
            if abs(base / cfg["nominal_latency_ns"] - 1) > 1e-6:
                raise SystemExit(f"{key} {cache}: the archive is not reproduced")
            got.setdefault((cohort, cache), {}).setdefault(case, {})[cid] = (
                cfg["measured_median_ns"],
                base,
                new,
            )
    for (cohort, cache), cases in sorted(got.items()):
        signed = {0: [], 1: []}
        regret = {0: [], 1: []}
        tau = {0: [], 1: []}
        for configs in cases.values():
            measured = {k: v[0] for k, v in configs.items()}
            oracle = min(measured.values())
            for i in (0, 1):
                pred = {k: v[1 + i] for k, v in configs.items()}
                signed[i] += [pred[k] / measured[k] - 1 for k in configs]
                shortlist = sorted(configs, key=pred.get)[:5]
                regret[i].append(min(measured[k] / oracle - 1 for k in shortlist))
                tau[i].append(kendall_tau_b(measured, pred))
        print(
            f"{cohort:18} {cache}: {len(signed[0])} configs; signed median {st.median(signed[0]) * 100:+.1f}% -> "
            f"{st.median(signed[1]) * 100:+.1f}%; median |err| {st.median(map(abs, signed[0])) * 100:.1f}% -> "
            f"{st.median(map(abs, signed[1])) * 100:.1f}%; mean top-5 regret {st.mean(regret[0]) * 100:.2f}% -> "
            f"{st.mean(regret[1]) * 100:.2f}%; mean tau {st.mean(tau[0]):.3f} -> {st.mean(tau[1]):.3f}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("count")
    c.add_argument("out")
    c.add_argument("--archive", default=IR_ARCHIVE)
    f = sub.add_parser("counterfactual")
    f.add_argument("counts")
    f.add_argument("archive")
    f.add_argument("final")
    args = ap.parse_args()
    if args.command == "count":
        count(args.out, args.archive)
    else:
        counterfactual(args.counts, args.archive, args.final)


if __name__ == "__main__":
    main()
