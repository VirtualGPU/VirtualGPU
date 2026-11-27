"""Static local loads and stores per loop of causal attention, from offline SASS (S1).

For every config of the causal development cases of validation/attention-suite.json, this
compiles the kernel offline for a target (CPU only, needs Triton and its bundled ptxas and
nvdisasm), assembles the PTX with line info as the frontend does, disassembles it, and
counts `LDL` and `STL` by width in each of the two loops (docs/results/analysis/
v0.3-local-load-count-plan.md, item S1). It fits nothing and changes no model input.

An instruction belongs to the main or the diagonal loop when its inline stack has a frame
on one of the two `_attention_tiles` call sites of `fused_attention`; it is in the loop
body when it also lies inside a backward branch's address range, and peeled otherwise
(prologue or epilogue copies of a pipelined loop, or the helper's setup). Everything else
is outside both loops.

  python scripts/static_local_loads.py sm_89 OUT.json
"""

import argparse
import ast
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["TRITON_DISABLE_LINE_INFO"] = "0"

MANIFEST = ROOT / "validation/attention-suite.json"
LOCATION = re.compile(r'File "([^"]+)", line (\d+)')
INSTRUCTION = re.compile(r"/\*([0-9a-f]+)\*/\s+(.*?)\s*;")
LABEL = re.compile(r"^\s*(\.L_x_\d+):")
BRANCH = re.compile(r"\bBRA\b.*`\((\.L_x_\d+)\)")
LOCAL = re.compile(r"^(?:@!?U?P\w+\s+)?(LDL|STL)((?:\.[A-Z0-9]+)*)\s")
WIDTHS = {"U8": 1, "S8": 1, "U16": 2, "S16": 2, "64": 8, "128": 16}


def call_sites():
    """Line ranges of the main and diagonal `_attention_tiles` calls in fused_attention."""
    from validation import attention

    source, start = inspect.getsourcelines(attention.fused_attention.fn)
    tree = ast.parse("".join(source))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_attention_tiles"
    ]
    calls.sort(key=lambda node: node.lineno)
    offset = start - 1
    main, diagonal = calls[0], calls[1]
    return {
        "main": (main.lineno + offset, main.end_lineno + offset),
        "diagonal": (diagonal.lineno + offset, diagonal.end_lineno + offset),
    }, Path(inspect.getsourcefile(attention.fused_attention.fn)).resolve()


def width(suffixes):
    sizes = [WIDTHS[s] for s in suffixes.split(".") if s in WIDTHS]
    return sizes[0] if sizes else 4


def parse_sass(sass, sites, source_file):
    """Per instruction: address, local op and width, scope, and whether predicated."""
    rows, labels, branches = [], {}, []
    frames, pending, block = [], [], False
    for line in sass.splitlines():
        # A block of location lines holds until the next block.
        if line.lstrip().startswith("//##"):
            if not block:
                frames, block = [], True
            frames.extend((Path(f).resolve(), int(n)) for f, n in LOCATION.findall(line))
            continue
        block = False
        if m := LABEL.match(line):
            pending.append(m[1])
            continue
        if not (m := INSTRUCTION.search(line)):
            continue
        address, text = int(m[1], 16), m[2].strip()
        for label in pending:
            labels[label] = address
        pending = []
        if b := BRANCH.search(text):
            branches.append((address, b[1]))
        scope = "outside"
        for name, (lo, hi) in sites.items():
            if any(f == source_file and lo <= n <= hi for f, n in frames):
                scope = name
        local = LOCAL.match(text)
        rows.append(
            {
                "address": address,
                "op": local[1] if local else None,
                "bytes": width(local[2]) if local else 0,
                "predicated": text.startswith("@"),
                "scope": scope,
            }
        )
    loops = [(labels[t], a) for a, t in branches if t in labels and labels[t] <= a]
    for row in rows:
        row["in_loop"] = any(lo <= row["address"] <= hi for lo, hi in loops)
    return rows


def count(rows):
    out = {}
    for scope in ("main", "diagonal", "outside"):
        parts = {"loop_body": True, "peeled": False} if scope != "outside" else {"all": None}
        for part, inside in parts.items():
            sel = [
                r
                for r in rows
                if r["op"] and r["scope"] == scope and (inside is None or r["in_loop"] == inside)
            ]
            c = Counter()
            for r in sel:
                c[f"{r['op']}_count"] += 1
                c[f"{r['op']}_bytes"] += r["bytes"]
                c[f"{r['op']}_{r['bytes']}B"] += 1
                c[f"{r['op']}_predicated"] += r["predicated"]
            out[f"{scope}_{part}"] = dict(sorted(c.items()))
    return out


def trips(case, config):
    """Per loop: CTA iterations V and CTA loop entries E over the launch, as
    microbench.spill_counters counts them."""
    bm, bn, seq = config["block_m"], config["block_n"], case["seq_len"]
    copies = case["batch"] * case["query_heads"]
    starts = range(0, seq, bm)
    main = [-(-start // bn) for start in starts]
    diagonal = [-(-min(bm, seq - start) // bn) for start in starts]
    return {
        name: (copies * sum(t), copies * sum(1 for x in t if x))
        for name, t in (("main", main), ("diagonal", diagonal))
    }


def per_iteration(counts, case, config, op="LDL"):
    """Estimate per CTA iteration: the loop body on every iteration but the peeled one, the
    peeled copies once per loop entry, (body (V - E) + peeled E) / V."""
    out = {}
    for name, (visits, entries) in trips(case, config).items():
        body = counts[f"{name}_loop_body"].get(f"{op}_count", 0)
        peeled = counts[f"{name}_peeled"].get(f"{op}_count", 0)
        out[name] = (body * (visits - entries) + peeled * entries) / visits if visits else None
    return out


def compile_config(case, config, target):
    """PTX, the frontend's resources, and SASS with inline line info."""
    import triton
    import virtualgpu.frontend.triton_attention as frontend
    from virtualgpu.frontend.compiler import detect_ptx_target, find_bundled_ptxas

    import scripts.posthoc_spill_split as split

    caught = []
    original = frontend.compile_triton_aot

    def capture(**kwargs):
        caught.append(original(**kwargs))
        return caught[-1]

    frontend.compile_triton_aot = capture
    try:
        resources = split.frontend_resources(case, config, target)
    finally:
        frontend.compile_triton_aot = original
    ptx = caught[-1].ptx
    arch = int(re.sub(r"\D", "", target))
    _, ptxas = find_bundled_ptxas(arch)
    nvdisasm = Path(triton.__file__).parent / "backends/nvidia/bin/nvdisasm"
    with tempfile.TemporaryDirectory(prefix="vgpu-s1-") as scratch:
        ptx_file, cubin = Path(scratch) / "k.ptx", Path(scratch) / "k.cubin"
        ptx_file.write_text(ptx)
        flags = ["-lineinfo", "-v", "--regAllocOptLevel=2"]
        if config.get("enable_fp_fusion") is False:
            flags.append("--fmad=false")
        gpu = f"--gpu-name={detect_ptx_target(ptx, arch)}"
        subprocess.run(
            [ptxas, *flags, gpu, str(ptx_file), "-o", str(cubin)], check=True, capture_output=True
        )
        sass = subprocess.run(
            [str(nvdisasm), "-gi", "-c", str(cubin)], check=True, capture_output=True, text=True
        ).stdout
    return resources, sass


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", help="sm_89 or sm_121")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    import triton

    suite = json.loads(MANIFEST.read_text())
    configs = {c["config_id"]: c for c in suite["configs"]}
    cases = [c for c in suite["cases"] if c["split"] == "development" and c["causal"]]
    sites, source_file = call_sites()
    rows = []
    for case in cases:
        for config_id in case["config_ids"]:
            config = configs[config_id]
            resources, sass = compile_config(case, config, args.target)
            parsed = parse_sass(sass, sites, source_file)
            counts = count(parsed)
            ldl_bytes = sum(r["bytes"] for r in parsed if r["op"] == "LDL")
            stl_bytes = sum(r["bytes"] for r in parsed if r["op"] == "STL")
            loops = resources.get("spill_loop_count", 1)
            rows.append(
                {
                    "case_id": case["case_id"],
                    "config_id": config_id,
                    "registers_per_thread": resources["registers_per_thread"],
                    "spill_scope": resources["spill_scope"],
                    "spill_loop_count": loops,
                    "spill_loads_bytes": resources["spill_loads_bytes"],
                    "spill_stores_bytes": resources["spill_stores_bytes"],
                    "sass_ldl_bytes": ldl_bytes,
                    "sass_stl_bytes": stl_bytes,
                    "core_reloads_per_loop": resources["spill_loads_bytes"] / 4 / loops,
                    "counts": counts,
                    "trips": trips(case, config),
                    "ldl_per_iteration": per_iteration(counts, case, config),
                    "stl_per_iteration": per_iteration(counts, case, config, "STL"),
                }
            )
            print(
                f"{case['case_id']} {config_id}: spill loads {resources['spill_loads_bytes']} B,"
                f" SASS LDL {ldl_bytes} B",
                flush=True,
            )
    out = {
        "result_class": "post-hoc",
        "target": args.target,
        "triton_version": triton.__version__,
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip(),
        "configs": rows,
    }
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
