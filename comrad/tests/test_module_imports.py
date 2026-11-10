import importlib
import pkgutil

import pytest

import comrad

MODULES = sorted(
    module.name
    for module in pkgutil.walk_packages(comrad.__path__, "comrad.")
    if not module.name.startswith(("comrad.tests", "comrad.scenarios"))
)


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)
