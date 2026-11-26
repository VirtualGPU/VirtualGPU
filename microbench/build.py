"""Build the CUDA C++ microbenchmark binary using the venv-installed nvcc."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sysconfig
from pathlib import Path

MICROBENCH_DIR = Path(__file__).resolve().parent
SRC_FILE = MICROBENCH_DIR / "src" / "vgpu_bench.cu"
DEFAULT_OUT = MICROBENCH_DIR / "build" / "vgpu_bench"


def find_cuda_root(nvcc_path: Path | None = None) -> Path | None:
    if nvcc_path is not None:
        resolved = nvcc_path.resolve()
        if not resolved.is_file():
            raise RuntimeError(f"Specified --nvcc path does not exist: {nvcc_path}")
        return resolved.parents[1]
    site_packages = Path(sysconfig.get_paths()["purelib"])
    nvidia_dir = site_packages / "nvidia"
    if nvidia_dir.is_dir():
        for candidate in sorted(nvidia_dir.glob("cu*"), reverse=True):
            if (candidate / "bin" / "nvcc").is_file():
                return candidate
        if (nvidia_dir / "cuda_nvcc" / "bin" / "nvcc").is_file():
            return nvidia_dir / "cuda_nvcc"
    nvcc_on_path = shutil.which("nvcc")
    if nvcc_on_path:
        return Path(nvcc_on_path).resolve().parents[1]
    return None


def build(arch: str, output: Path, nvcc_path: Path | None = None) -> Path:
    cuda_root = find_cuda_root(nvcc_path)
    if cuda_root is None:
        raise RuntimeError(
            "nvcc not found. Pass --nvcc /path/to/nvcc or install nvidia-cuda-nvcc in the project venv."
        )

    nvcc = cuda_root / "bin" / "nvcc"
    include_dir = cuda_root / "include"
    lib_dir = cuda_root / "lib"
    if not lib_dir.is_dir():
        lib_dir = cuda_root / "lib64"

    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(nvcc),
        "-O3",
        "-std=c++17",
        f"-arch={arch}",
        f"-I{include_dir}",
        f"-L{lib_dir}",
        f"-Xlinker=-rpath,{lib_dir}",
        str(SRC_FILE),
        "-o",
        str(output),
    ]

    env = os.environ.copy()
    env["PATH"] = f"{cuda_root / 'bin'}:{env.get('PATH', '')}"
    print("Building:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Build VirtualGPU CUDA microbenchmark binary.")
    parser.add_argument("--arch", default="sm_89", help="Target GPU arch, e.g. sm_89 or sm_121.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT, help="Output binary path.")
    parser.add_argument(
        "--nvcc", type=Path, default=None, help="Optional explicit path to nvcc binary."
    )
    args = parser.parse_args()
    out = build(args.arch, args.output, args.nvcc)
    print(f"Built {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
