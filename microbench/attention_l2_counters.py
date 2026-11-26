"""Post-hoc L2 hit and miss counters for the fused attention ablation (Nsight Compute).

Profiles the unchanged variant of microbench/attention_ablation.py on its development shape,
after warm-up launches, and records the L2 read sector lookups that hit and miss. Targets the
Bifrost RTX 4070 (sm_89); a permission probe runs first, so ERR_NVGPUCTRPERM fails before any
measurement and nothing is published.

  python -m microbench.attention_l2_counters --nvcc /path/to/nvcc
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from microbench import spill_counters as spill
from microbench.build import build, find_cuda_root

ROOT = Path(__file__).resolve().parents[1]
POST_HOC = ROOT / "results/post-hoc"
CONFIGS = ("m128n32-w8-s2", "m128n32-w8-s3", "m128n64-w8-s2", "m128n64-w8-s3")
WARMUP = 3
HIT, MISS = (f"lts__t_sectors_srcunit_tex_op_read_lookup_{kind}.sum" for kind in ("hit", "miss"))
METRICS = (HIT, MISS, "launch__grid_size", "launch__block_size")


def target(config_id):
    # Run only under NCU: WARMUP launches, then the profiled one.
    import torch

    from microbench import attention_ablation as ablation
    from validation.attention import launch_spec

    case = ablation.CASE
    config = next(c for c in ablation.CONFIGS if c["config_id"] == config_id)
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
    grid, _, _ = launch_spec(case, config)
    constants, options = ablation.specialize(config, 0)
    for _ in range(WARMUP + 1):
        ablation.ablated_attention[grid](q, k, v, o, lse, ts, *constants, **options)
    torch.cuda.synchronize()
    print(json.dumps({"grid": list(grid), "num_warps": config["num_warps"]}), flush=True)


def row(config_id, action, launch):
    grid = launch["grid"][0] * launch["grid"][1] * launch["grid"][2]
    geometry = [spill.count(spill.require_metric(action, m)) for m in METRICS[2:]]
    if geometry != [grid, 32 * launch["num_warps"]]:
        raise ValueError(f"{config_id}: profiled launch geometry {geometry} differs")
    hit, miss = (spill.count(spill.require_metric(action, m)) for m in (HIT, MISS))
    if hit + miss == 0:
        raise ValueError(f"{config_id}: no L2 read sectors counted")
    return {
        "config_id": config_id,
        "grid": launch["grid"],
        "read_sectors_hit": hit,
        "read_sectors_miss": miss,
        "read_hit_fraction": hit / (hit + miss),
    }


def profile(ncu, api, scratch, env):
    rows = []
    for config_id in CONFIGS:
        program = [sys.executable, "-m", "microbench.attention_l2_counters", "--target", config_id]
        _, actions, log = spill.collect(
            ncu,
            api,
            scratch,
            config_id,
            "ablated_attention",
            METRICS,
            ["--launch-skip", str(WARMUP), "--launch-count", "1", *program],
            env,
        )
        launches = [json.loads(line) for line in log.splitlines() if line.startswith('{"grid":')]
        if len(actions) != 1 or len(launches) != 1:
            raise ValueError(f"{config_id}: expected one profiled launch")
        rows.append(row(config_id, actions[0], launches[0]))
    return rows


def measure(ncu, nvcc):
    env = dict(os.environ, LC_ALL="C")
    version = spill.run([ncu, "--version"], env).strip()
    gpu = spill.run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=uuid,name,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        env,
    )
    uuid, name, driver, capability = next(csv.reader(gpu.splitlines(), skipinitialspace=True))
    if capability != "8.9" or "RTX 4070" not in name:
        raise ValueError("This counter run targets the Bifrost RTX 4070")
    env["CUDA_VISIBLE_DEVICES"] = uuid
    revision = spill.run(["git", "rev-parse", "HEAD"], env).strip()
    if spill.run(["git", "status", "--porcelain"], env):
        raise ValueError("Run the counters from a clean committed checkout")
    if find_cuda_root(nvcc) is None:
        raise ValueError("nvcc not found in the virtual environment. Pass --nvcc /path/to/nvcc.")
    sys.path.insert(0, str(Path(ncu).resolve().parent / "extras/python"))
    import ncu_report as api

    from microbench.attention_ablation import CASE

    with tempfile.TemporaryDirectory(prefix="virtualgpu-l2-") as directory:
        scratch = Path(directory)
        binary = build("sm_89", scratch / "vgpu_bench", nvcc)
        # Permission probe: one counter on one small launch before anything is measured.
        _, actions, _ = spill.collect(
            ncu,
            api,
            scratch,
            "permission",
            "spill_reload_latency_kernel",
            (HIT,),
            [str(binary), "--mode", "spill-counter-probe"],
            env,
        )
        if len(actions) != 1:
            raise ValueError("Performance counter permission probe expected one CUDA launch")
        spill.count(spill.require_metric(actions[0], HIT))
        env.update(TRITON_CACHE_DIR=str(scratch / "triton"))
        rows = profile(ncu, api, scratch, env)
    return {
        "schema_version": 1,
        "result_class": "post-hoc",
        "provenance": {
            "utc": datetime.now(timezone.utc).isoformat(),
            "commit": revision,
            "gpu": name,
            "gpu_uuid": uuid,
            "driver": driver,
            "ncu": version,
            "metrics": list(METRICS),
            "target": "sm_89",
            "warmup_launches": WARMUP,
            "clock_control": "none",
            "cache_control": "none",
            "ablation_sha256": hashlib.sha256(
                (ROOT / "microbench/attention_ablation.py").read_bytes()
            ).hexdigest(),
        },
        "case": CASE,
        "configs": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--nvcc", type=Path)
    parser.add_argument("--target", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.target:
        target(args.target)
        return 0
    try:
        ncu = shutil.which(args.ncu)
        if ncu is None:
            raise ValueError("Nsight Compute CLI was not found. Pass --ncu /path/to/ncu.")
        data = measure(ncu, args.nvcc)
        stamp = datetime.fromisoformat(data["provenance"]["utc"]).strftime("%Y%m%dT%H%M%S%fZ")
        output = POST_HOC / f"attention-l2-counters-{stamp}.json"
        spill.publish(output, data)
    except (OSError, ImportError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
