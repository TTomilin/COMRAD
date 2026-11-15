import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_MODULES = sorted(
    ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)
    for path in (REPO_ROOT / "experiments").rglob("*.py")
    if "tests" not in path.parts and path.name != "__init__.py"
)
SCRIPT_MODULES = sorted(path.stem for path in (REPO_ROOT / "scripts").glob("*.py") if path.name != "__init__.py")
FILE_INVOCATION = re.compile(r"python3?\s+experiments/\S+\.py")
SYS_PATH_EDIT = re.compile(r"sys\.path\.(insert|append|extend)\(")
LAUNCH_FILES = [
    path
    for folder in ("experiments", "scripts", "docs")
    for path in (REPO_ROOT / folder).rglob("*")
    if path.suffix in {".sh", ".sbatch", ".md", ".py"} and "wad2image" not in path.parts
] + [REPO_ROOT / "README.md", REPO_ROOT / "results" / "README.md"]

_PROBE = """
import importlib, json, sys, traceback
failures = {}
for name in sys.argv[1:]:
    try:
        importlib.import_module(name)
    except BaseException:
        failures[name] = traceback.format_exc(limit=-1)
print(json.dumps(failures))
"""


def _import_failures(modules: list[str], sys_path_0: Path) -> dict[str, str]:
    probe = f"import sys; sys.path[0] = {str(sys_path_0)!r}; exec({_PROBE!r})"
    result = subprocess.run(
        [sys.executable, "-c", probe, *modules],
        cwd=REPO_ROOT,
        env={**os.environ, "MPLBACKEND": "Agg", "COMRAD_EXPERIMENT_PROFILE": "full_obs_clean"},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_experiment_modules_import_as_launched_with_python_dash_m() -> None:
    assert EXPERIMENT_MODULES
    assert _import_failures(EXPERIMENT_MODULES, REPO_ROOT) == {}


def test_scripts_import_as_launched_from_their_file() -> None:
    assert SCRIPT_MODULES
    assert _import_failures(SCRIPT_MODULES, REPO_ROOT / "scripts") == {}


def test_experiments_do_not_edit_sys_path() -> None:
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "experiments").rglob("*.py")
        if SYS_PATH_EDIT.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], "import through the `experiments` / `scripts` packages instead of editing sys.path"


def test_experiment_modules_are_launched_with_python_dash_m() -> None:
    offenders = [
        f"{path.relative_to(REPO_ROOT)}: {match}"
        for path in LAUNCH_FILES
        if path.is_file()
        for match in FILE_INVOCATION.findall(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], "launch experiment modules as `python -m experiments.<package>.<module>` from the repo root"
