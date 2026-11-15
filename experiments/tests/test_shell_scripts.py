import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXCLUDED_DIRS = (REPO_ROOT / "scripts" / "wad2image",)
SHELL_SCRIPTS = sorted(
    path
    for folder in ("experiments", "scripts")
    for path in (REPO_ROOT / folder).rglob("*")
    if path.suffix in {".sh", ".sbatch"} and not any(path.is_relative_to(excluded) for excluded in EXCLUDED_DIRS)
)
ARRAY_TASK_RANK = REPO_ROOT / "experiments" / "iclr2027_rebuttal" / "array_task_rank.sh"
ROOT_INDEPENDENT = {ARRAY_TASK_RANK}

STRICT_MODE = re.compile(r"^set -E?euo pipefail$", re.M)
BANNED_PARTITION = re.compile(r"(?:--partition[= ]\s*|(?<![\w-])-p\s*)(gpu_a100|gpu_h100)\b")
SBATCH_DIRECTIVE = re.compile(r"^#SBATCH ", re.M)
SCRIPT_DIR_FROM_SOURCE = re.compile(r'^\s*SCRIPT_DIR="\$\(cd "\$\(dirname "\$\{BASH_SOURCE\[0\]\}"\)" && pwd\)"$', re.M)
SCRIPT_RELATIVE_ROOT = re.compile(r'(\$SCRIPT_DIR|\$\(dirname "(?:\$0|\$\{BASH_SOURCE\[0\]\})"\))/((?:\.\./?)+)"')
SUBMIT_DIR_ROOT = re.compile(r"\$\{SLURM_SUBMIT_DIR:-\$PWD\}")
SLURM_VARIABLE = re.compile(r"(\$\{)?\b(SLURM_[A-Z_]+)\b(:[-?])?")


def script_id(path):
    return str(path.relative_to(REPO_ROOT))


def code_lines(script):
    return "\n".join(line for line in script.read_text().splitlines() if not line.lstrip().startswith("#"))


def test_scripts_are_collected():
    collected = {script_id(path) for path in SHELL_SCRIPTS}
    assert {
        "experiments/iclr2027_rebuttal/kd6j_heldout_eval_array.sh",
        "experiments/very_difficult/very_difficult_heldout_eval.sh",
        "scripts/record_demo_all_checkpoints.sbatch",
    } <= collected
    assert not any(name.startswith("scripts/wad2image/") for name in collected)
    assert ROOT_INDEPENDENT <= set(SHELL_SCRIPTS)


@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=script_id)
def test_script_parses(script):
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=script_id)
def test_script_uses_strict_mode(script):
    assert STRICT_MODE.search(script.read_text()), "add `set -euo pipefail` so failures propagate to SLURM"


@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=script_id)
def test_script_does_not_request_full_gpus(script):
    banned = BANNED_PARTITION.findall(script.read_text())
    assert not banned, f"use --partition=gpu_mig (9 CPUs, 60G) instead of {banned}"


@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=script_id)
def test_script_resolves_repo_root(script):
    text = script.read_text()
    relative_roots = SCRIPT_RELATIVE_ROOT.findall(text)
    for anchor, relative in relative_roots:
        if anchor == "$SCRIPT_DIR":
            assert SCRIPT_DIR_FROM_SOURCE.search(text), 'SCRIPT_DIR must be "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"'
        assert (script.parent / relative).resolve() == REPO_ROOT, f"{relative!r} does not lead to the repository root"
    from_submit_dir = bool(SBATCH_DIRECTIVE.search(text) and SUBMIT_DIR_ROOT.search(text))
    if script in ROOT_INDEPENDENT:
        assert not relative_roots and not from_submit_dir, "listed as root-independent but resolves the repo root"
    else:
        assert relative_roots or from_submit_dir, (
            'resolve the repo root via "$(dirname "${BASH_SOURCE[0]}")/.." or, in sbatch jobs (the script runs from the '
            "SLURM spool dir), via ${SLURM_SUBMIT_DIR:-$PWD}"
        )


@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=script_id)
def test_slurm_variables_have_defaults(script):
    for brace, name, default in SLURM_VARIABLE.findall(code_lines(script)):
        assert brace and default, f"{name} is unset outside SLURM; use ${{{name}:-...}} under set -u"


@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=script_id)
def test_array_split_uses_task_rank(script):
    if script != ARRAY_TASK_RANK:
        assert "SLURM_ARRAY_TASK_COUNT" not in code_lines(script), (
            "split array work by the rank printed by experiments/iclr2027_rebuttal/array_task_rank.sh, "
            "so resubmitted tasks (e.g. --array=2) still cover the manifest"
        )


def array_task_rank(**slurm):
    env = {key: value for key, value in os.environ.items() if not key.startswith("SLURM_ARRAY_")}
    env.update({f"SLURM_ARRAY_TASK_{key.upper()}": str(value) for key, value in slurm.items()})
    return subprocess.run(["bash", str(ARRAY_TASK_RANK)], capture_output=True, text=True, env=env)


@pytest.mark.parametrize(
    ("slurm", "expected"),
    [
        ({}, "0 1"),
        ({"id": 0, "min": 0, "max": 3, "count": 4}, "0 4"),
        ({"id": 3, "min": 0, "max": 3, "count": 4}, "3 4"),
        ({"id": 2, "min": 2, "max": 2, "count": 1}, "0 1"),
        ({"id": 7, "min": 5, "max": 7, "count": 3}, "2 3"),
    ],
)
def test_array_task_rank(slurm, expected):
    result = array_task_rank(**slurm)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


@pytest.mark.parametrize("slurm", [{"id": 3, "min": 1, "max": 3, "count": 2}, {"id": 4, "min": 0, "max": 4, "count": 3}])
def test_array_task_rank_rejects_sparse_arrays(slurm):
    result = array_task_rank(**slurm)
    assert result.returncode != 0
    assert "contiguous" in result.stderr
