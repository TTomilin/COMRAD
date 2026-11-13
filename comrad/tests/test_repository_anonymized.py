from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "anonymize_results.py"
SPEC = importlib.util.spec_from_file_location("anonymize_results", MODULE_PATH)
anonymize_results = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(anonymize_results)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('"checkpoint": "/gpfs/home1/user/ViZDoom/train_dir/run/a.pth"', '"checkpoint": "train_dir/run/a.pth"'),
        ("x,/Users/me/code/COMRAD/train_dir/run,1", "x,train_dir/run,1"),
        ('"repo": "/scratch-shared/user/ViZDoom"', '"repo": "."'),
        ("wandb: Currently logged in as: someone (comrad) to", "wandb: Currently logged in as: <user> (comrad) to"),
        ("error: *** JOB 123 ON gcn7 CANCELLED", "error: *** JOB 123 ON <node> CANCELLED"),
        ("https://github.com/TTomilin/COMRAD/blob/main", "https://github.com/TTomilin/COMRAD/blob/main"),
        ("https://wandb.ai/comrad/COMRAD/runs/x", "https://wandb.ai/comrad/COMRAD/runs/x"),
        ("./train_dir/COMRAD_run/cfg.json", "./train_dir/COMRAD_run/cfg.json"),
    ],
)
def test_anonymize(raw, expected):
    assert anonymize_results.anonymize(raw) == expected
    assert anonymize_results.anonymize(expected) == expected
    assert anonymize_results.find_personal(expected) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('"dir": "/home/user/data/run"', ["/home/user/data/run"]),
        ('ROOT="${ROOT:-/home/user/ViZDoom}"', ["/home/user/ViZDoom}"]),
        ("#SBATCH --output=/scratch-shared/%u/logs/%j.out", []),
        ('ROOT="/scratch-shared/$USER/train_dir"', []),
        ("rsync <hpc_host>:/home/<user>/ViZDoom/train_dir .", []),
    ],
)
def test_find_personal(text, expected):
    assert anonymize_results.find_personal(text) == expected


def _tracked_text_files() -> list[tuple[str, str]]:
    try:
        listing = subprocess.run(["git", "ls-files", "-z"], cwd=REPO_ROOT, check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("tracked files are only checked inside a git checkout")
    files = []
    for name in listing.decode().split("\0"):
        path = REPO_ROOT / name
        if not name or not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" not in data[:8192]:
            files.append((name, data.decode("utf-8", errors="surrogateescape")))
    return files


def test_tracked_files_are_anonymized():
    dirty = []
    for name, text in _tracked_text_files():
        is_result = name.startswith("results/") and anonymize_results.is_text_artifact(name)
        if anonymize_results.find_personal(text) or (is_result and anonymize_results.anonymize(text) != text):
            dirty.append(name)
    assert not dirty, (
        "personal paths, logins or node names in tracked files; run `python scripts/anonymize_results.py results` "
        f"for results and use repo-relative paths or $USER/%u/<user> placeholders elsewhere: {dirty[:5]}"
    )
