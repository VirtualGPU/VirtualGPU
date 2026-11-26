"""Fail on relative Markdown links that point at a missing file or heading."""

import re
import subprocess
import sys
from pathlib import Path

LINK = re.compile(r"\]\(([^)\s]+)\)")
HEADING = re.compile(r"^#+\s+(.*)$")
ANCHORS = {}


def slug(text):
    text = re.sub(r"[`*_]", "", text.strip().lower())
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def anchors(path):
    if path not in ANCHORS:
        found = set()
        in_code = False
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("```"):
                in_code = not in_code
            elif not in_code and (m := HEADING.match(line)):
                found.add(slug(m.group(1)))
        ANCHORS[path] = found
    return ANCHORS[path]


def main() -> int:
    files = subprocess.check_output(["git", "ls-files", "*.md"], text=True).splitlines()
    failures = []
    for rel in files:
        path = Path(rel)
        text = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
        text = re.sub(r"`[^`\n]*`", "", text)
        for target in LINK.findall(text):
            if re.match(r"^[a-z]+:", target):
                continue
            file_part, _, frag = target.partition("#")
            dest = (path.parent / file_part).resolve() if file_part else path.resolve()
            if not dest.exists():
                failures.append(f"{rel}: missing {target}")
            elif frag and dest.suffix == ".md" and slug(frag) not in anchors(dest):
                failures.append(f"{rel}: no heading for {target}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"Checked links in {len(files)} Markdown files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
