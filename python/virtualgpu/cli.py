"""Command-line interface for VirtualGPU offline extraction and prediction."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from virtualgpu.api import extract_gemm_configs, predict_gemm_configs, predict_ir_dir
from virtualgpu.autotune import autotune_offline


def _load_kernel(spec: str) -> Any:
    """Import a @triton.jit function from 'module.path:attr'."""
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(f"Kernel spec must be 'module:function', got {spec!r}")
    mod = importlib.import_module(module_name)
    return getattr(mod, attr)


def _load_suite_case(
    suite_path: Path,
    case_id: str,
    config_ids: Sequence[str] | None = None,
) -> tuple[tuple[int, int, int], list[dict[str, Any]]]:
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    cases = {c["case_id"]: c for c in suite["cases"]}
    configs_by_id = {c["config_id"]: c for c in suite["configs"]}
    if case_id not in cases:
        raise KeyError(f"Case {case_id!r} not found in {suite_path}")
    case = cases[case_id]
    shape = (int(case["shape"][0]), int(case["shape"][1]), int(case["shape"][2]))
    selected = list(config_ids) if config_ids else list(case["config_ids"])
    return shape, [configs_by_id[cid] for cid in selected]


def _resolve_configs_and_shape(
    args: argparse.Namespace,
) -> tuple[str, tuple[int, int, int], list[dict[str, Any]]]:
    if args.suite is not None:
        if not args.case:
            raise ValueError("--case is required when using --suite")
        shape, configs = _load_suite_case(args.suite, args.case, args.configs)
        return args.case, shape, configs
    if args.shape is None or args.configs_json is None:
        raise ValueError("Provide either --suite + --case, or --shape + --configs-json")
    configs = json.loads(args.configs_json.read_text(encoding="utf-8"))
    case_id = args.case or f"gemm_{args.shape[0]}x{args.shape[1]}x{args.shape[2]}"
    return case_id, (args.shape[0], args.shape[1], args.shape[2]), configs


def _cmd_extract(args: argparse.Namespace) -> int:
    fn = _load_kernel(args.kernel)
    _, shape, configs = _resolve_configs_and_shape(args)
    results = extract_gemm_configs(
        fn,
        shape=shape,
        configs=configs,
        target_arch=args.arch,
        dtype=args.dtype,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
        ir_out_dir=args.out_dir,
    )
    ok_count = 0
    for cid, item in results.items():
        if item["status"] == "ok":
            ok_count += 1
            r = item["ir"]["resources"]
            p = item["ir"]["body"]["pipeline_loop"]
            print(
                f"{cid}: ok regs={r['registers_per_thread']} smem={r['static_smem_bytes']} "
                f"spill={r['spill_stores_bytes']}/{r['spill_loads_bytes']} "
                f"stages={p['num_stages']}/{p['prefetch_distance']}"
            )
        else:
            print(f"{cid}: {item['status']} {item['reason']}", file=sys.stderr)
    return 0 if ok_count > 0 else 1


def _cmd_predict(args: argparse.Namespace) -> int:
    if args.kernel is not None:
        fn = _load_kernel(args.kernel)
        case_id, shape, configs = _resolve_configs_and_shape(args)
        records = predict_gemm_configs(
            fn,
            case_id=case_id,
            shape=shape,
            configs=configs,
            profile=args.profile,
            l2_state=args.cache,
            dtype=args.dtype,
            draws=args.draws,
            seed=args.seed,
            cache_dir=args.cache_dir,
            use_cache=not args.no_cache,
            ir_out_dir=args.ir_dir,
        )
    elif args.ir_dir is not None:
        if not args.case:
            raise ValueError("--case is required when predicting from --ir-dir")
        records = predict_ir_dir(
            args.ir_dir,
            case_id=args.case,
            profile=args.profile,
            l2_state=args.cache,
            draws=args.draws,
            seed=args.seed,
        )
    else:
        raise ValueError("Provide either --kernel (end-to-end) or --ir-dir (pre-extracted IRs)")

    output_text = json.dumps(records, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output_text, encoding="utf-8")
    else:
        print(output_text)
    return 0


def _cmd_autotune(args: argparse.Namespace) -> int:
    kernel = _load_kernel(args.kernel)
    configs = None
    if args.configs_json is not None:
        configs = json.loads(args.configs_json.read_text(encoding="utf-8"))
    report = autotune_offline(
        kernel,
        shape=(args.shape[0], args.shape[1], args.shape[2]),
        profile=args.profile,
        configs=configs,
        dtype=args.dtype,
        l2_state=args.cache,
        draws=args.draws,
        seed=args.seed,
        top_k=args.top_k,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
        strict=args.strict,
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report.to_json(indent=2), encoding="utf-8")
        print(report.summary())
    else:
        print(report.to_json(indent=2))
    return 0 if report.status == "ok" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="virtualgpu",
        description="VirtualGPU: offline Triton KernelIR extraction and GPU performance prediction.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ext = sub.add_parser(
        "extract", help="Compile Triton GEMM configs offline and emit KernelIR JSON files."
    )
    p_ext.add_argument("--kernel", required=True, help="Triton kernel import spec 'module:func'")
    p_ext.add_argument(
        "--suite", type=Path, help="Path to validation suite JSON (e.g. validation/suite.json)"
    )
    p_ext.add_argument("--case", help="Case ID in --suite")
    p_ext.add_argument("--configs", nargs="*", help="Optional subset of config IDs from --suite")
    p_ext.add_argument(
        "--shape",
        type=int,
        nargs=3,
        metavar=("M", "N", "K"),
        help="GEMM shape when not using --suite",
    )
    p_ext.add_argument(
        "--configs-json", type=Path, help="JSON list of config dicts when not using --suite"
    )
    p_ext.add_argument("--arch", required=True, help="Target SM architecture (e.g. sm_89, sm_121)")
    p_ext.add_argument("--dtype", default="f16", help="Input tensor dtype (default: f16)")
    p_ext.add_argument(
        "--out-dir", type=Path, required=True, help="Directory to write <config_id>.json KernelIRs"
    )
    p_ext.add_argument("--cache-dir", type=Path, help="AOT compilation cache directory")
    p_ext.add_argument("--no-cache", action="store_true", help="Disable AOT compilation cache")
    p_ext.set_defaults(func=_cmd_extract)

    p_pred = sub.add_parser(
        "predict", help="Predict kernel latency records in validation/score.py format."
    )
    p_pred.add_argument(
        "--kernel", help="Triton kernel import spec 'module:func' for end-to-end prediction"
    )
    p_pred.add_argument("--suite", type=Path, help="Path to validation suite JSON")
    p_pred.add_argument("--case", help="Case ID")
    p_pred.add_argument("--configs", nargs="*", help="Optional subset of config IDs from --suite")
    p_pred.add_argument(
        "--shape",
        type=int,
        nargs=3,
        metavar=("M", "N", "K"),
        help="GEMM shape when not using --suite",
    )
    p_pred.add_argument(
        "--configs-json", type=Path, help="JSON list of config dicts when not using --suite"
    )
    p_pred.add_argument(
        "--ir-dir",
        type=Path,
        help="Directory of KernelIR JSONs (input if no --kernel, or output dir if --kernel is set)",
    )
    p_pred.add_argument(
        "--profile",
        type=Path,
        required=True,
        help="GPU profile JSON path or bundled profile name, like rtx4070",
    )
    p_pred.add_argument(
        "--cache", choices=("cold", "warm"), default="cold", help="Initial L2 cache state"
    )
    p_pred.add_argument("--dtype", default="f16", help="Input tensor dtype (default: f16)")
    p_pred.add_argument(
        "--draws",
        type=int,
        default=32,
        help="Number of perturbation draws (default: 32; 0 = nominal only)",
    )
    p_pred.add_argument(
        "--seed", type=int, default=17, help="RNG seed for perturbation draws (default: 17)"
    )
    p_pred.add_argument("--cache-dir", type=Path, help="AOT compilation cache directory")
    p_pred.add_argument("--no-cache", action="store_true", help="Disable AOT compilation cache")
    p_pred.add_argument("--out", type=Path, help="Output JSON path (prints to stdout if omitted)")
    p_pred.set_defaults(func=_cmd_predict)

    p_auto = sub.add_parser(
        "autotune",
        help="Rank a @triton.autotune kernel offline on CPU and emit an AutotuneReport.",
    )
    p_auto.add_argument(
        "--kernel", required=True, help="Import spec 'module:kernel' (Autotuner or JITFunction)"
    )
    p_auto.add_argument(
        "--shape", type=int, nargs=3, required=True, metavar=("M", "N", "K"), help="GEMM shape"
    )
    p_auto.add_argument(
        "--configs-json",
        type=Path,
        help="Optional JSON list of configs (required for bare JITFunction)",
    )
    p_auto.add_argument(
        "--profile",
        type=Path,
        required=True,
        help="GPU profile JSON path or bundled profile name, like rtx4070",
    )
    p_auto.add_argument(
        "--cache", choices=("cold", "warm"), default="cold", help="Initial L2 cache state"
    )
    p_auto.add_argument("--dtype", default="f16", help="Input tensor dtype (default: f16)")
    p_auto.add_argument(
        "--draws", type=int, default=32, help="Number of perturbation draws (default: 32)"
    )
    p_auto.add_argument(
        "--seed", type=int, default=17, help="RNG seed for perturbation draws (default: 17)"
    )
    p_auto.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Shortlist size recommended for hardware verification (default: 5)",
    )
    p_auto.add_argument("--cache-dir", type=Path, help="AOT compilation cache directory")
    p_auto.add_argument("--no-cache", action="store_true", help="Disable AOT compilation cache")
    p_auto.add_argument(
        "--strict", action="store_true", help="Exit with error if no config can be modeled"
    )
    p_auto.add_argument(
        "--out", type=Path, help="Write AutotuneReport JSON to file (and print summary to stdout)"
    )
    p_auto.set_defaults(func=_cmd_autotune)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
