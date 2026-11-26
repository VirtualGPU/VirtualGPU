"""Check a release tag against the package versions and print its CHANGELOG section."""

import re
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    if len(sys.argv) != 2 or not re.fullmatch(r"v\d+\.\d+\.\d+", sys.argv[1]):
        print("usage: release_notes.py vX.Y.Z", file=sys.stderr)
        return 2
    version = sys.argv[1][1:]
    py = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    rs = tomllib.loads((ROOT / "crates/vgpu-core/Cargo.toml").read_text())["package"]["version"]
    if py != version or rs != version:
        print(f"tag {version} but pyproject.toml has {py} and Cargo.toml has {rs}", file=sys.stderr)
        return 1
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    match = re.search(rf"^## {re.escape(version)}\n(.*?)(?=^## |\Z)", text, flags=re.S | re.M)
    if not match or not match.group(1).strip():
        print(f"CHANGELOG.md has no section for {version}", file=sys.stderr)
        return 1
    print(match.group(1).strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
