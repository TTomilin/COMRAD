"""Strip machine- and user-specific details from result artifacts before they are committed.

Absolute paths into a repository checkout become repo-relative, the W&B login name and SLURM node names are masked.
Files are rewritten in place; any personal path the rules cannot rewrite is reported and makes the script exit 1.

    python scripts/anonymize_results.py results/<study>
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv", ".err", ".out", ".log", ".txt", ".md", ".tex", ".yaml", ".yml"}
REPO_DIR_NAMES = ("COMRAD", "ViZDoom")

_PATH_START = r"(?<![\w/:.~])"
_CHECKOUT_PATH = re.compile(_PATH_START + r"(?:/[\w.@~-]+)+?/(?:" + "|".join(REPO_DIR_NAMES) + r")(?:/|(?![\w.-]))")
_WANDB_LOGIN = re.compile(r"(logged in as: )(?!<user>)\S+")
_SLURM_NODE = re.compile(r"(\*\*\* JOB \d+ ON )(?!<node>)\S+")
_PERSONAL = re.compile(_PATH_START + r"/(?:home\d*|Users|gpfs|scratch[\w-]*)/(?![$%<{])[^\s\"',;)]+|" + _WANDB_LOGIN.pattern)


def is_text_artifact(path: str | Path) -> bool:
    return Path(path).suffix.lower() in TEXT_SUFFIXES


def anonymize(text: str) -> str:
    text = _CHECKOUT_PATH.sub(lambda match: "" if match.group(0).endswith("/") else ".", text)
    text = _WANDB_LOGIN.sub(r"\1<user>", text)
    return _SLURM_NODE.sub(r"\1<node>", text)


def find_personal(text: str) -> list[str]:
    return sorted({match.group(0) for match in _PERSONAL.finditer(text)})


def _artifacts(roots: list[Path]):
    for root in roots:
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        yield from (path for path in files if is_text_artifact(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("results")])
    args = parser.parse_args()

    rewritten, leaks = 0, {}
    for path in _artifacts(args.paths):
        text = path.read_text(encoding="utf-8", errors="surrogateescape")
        clean = anonymize(text)
        if clean != text:
            path.write_text(clean, encoding="utf-8", errors="surrogateescape")
            rewritten += 1
        remaining = find_personal(clean)
        if remaining:
            leaks[path] = remaining

    print(f"anonymized {rewritten} file(s)")
    for path, remaining in leaks.items():
        print(f"{path}: personal data left: {', '.join(remaining[:3])}", file=sys.stderr)
    return 1 if leaks else 0


if __name__ == "__main__":
    sys.exit(main())
