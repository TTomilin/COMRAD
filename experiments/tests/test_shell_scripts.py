import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_RELATIVE_ROOT = re.compile(r'(?:\$SCRIPT_DIR|\$\(dirname "(?:\$0|\$\{BASH_SOURCE\[0\]\})"\))/((?:\.\./?)+)"')
SHELL_SCRIPTS = sorted(
    path for folder in ("experiments", "scripts") for path in (REPO_ROOT / folder).rglob("*") if path.suffix in {".sh", ".sbatch"}
)


@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_script_relative_root_is_repo_root(script):
    for relative in SCRIPT_RELATIVE_ROOT.findall(script.read_text()):
        assert (script.parent / relative).resolve() == REPO_ROOT, f"{relative!r} does not lead to the repository root"


def test_root_lookups_are_found():
    assert sum(len(SCRIPT_RELATIVE_ROOT.findall(path.read_text())) for path in SHELL_SCRIPTS) >= 6
