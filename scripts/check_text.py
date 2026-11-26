"""Fail on non-English text and on em or en dashes in tracked files."""

import re
import subprocess
import sys
from pathlib import Path

CJK = re.compile(r"[\u3000-\u9fff\uff00-\uffef]")
DASH = re.compile("[\u2013\u2014]|\u00e2\u20ac")
DASH_SUFFIXES = {".md", ".py", ".rs", ".cu", ".cuh", ".toml", ".yml", ".yaml", ".svg"}
# Sealed records and the sources they hash are never edited for style.
DASH_SKIP = ("validation/",)


def main() -> int:
    files = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    failures = []
    for rel in files:
        try:
            text = Path(rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        check_dash = Path(rel).suffix in DASH_SUFFIXES and not rel.startswith(DASH_SKIP)
        for lineno, line in enumerate(text.splitlines(), 1):
            if CJK.search(line):
                failures.append(f"{rel}:{lineno}: non-English text")
            elif check_dash and DASH.search(line):
                failures.append(f"{rel}:{lineno}: em or en dash")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"Checked {len(files)} tracked files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
