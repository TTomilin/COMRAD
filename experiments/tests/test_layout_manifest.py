import json
import sys
from pathlib import Path

import pytest

from comrad.envs.wad_catalog import WadBatch
from experiments.iclr2027_rebuttal import generate_layout_training_batch as layout_batch
from experiments.iclr2027_rebuttal.profiles import layout_batch_dir
from scripts import heldout_wad_evaluation as heldout

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def fake_coop_puzzle(monkeypatch):
    class FakeScenario:
        def __init__(self, config, name=None):
            self.config = {"seed": None, **config}

        def generate(self, path):
            Path(path).write_bytes(repr(sorted(self.config.items())).encode())

    module_name, class_name = heldout.SCENARIO_CLASSES["coop_puzzle"]
    fake_module = type(sys)("fake_coop_puzzle")
    setattr(fake_module, class_name, FakeScenario)
    monkeypatch.setitem(sys.modules, module_name, fake_module)


def test_training_batch_defaults_to_the_directory_training_reads():
    assert layout_batch.default_batch_dir("coop_puzzle") == REPO_ROOT / layout_batch_dir("coop_puzzle")


@pytest.mark.usefixtures("fake_coop_puzzle")
@pytest.mark.parametrize("split, generate", [
    ("train", lambda out: layout_batch.generate_training_batch("coop_puzzle", out, layout_batch.training_seeds(2))),
    ("heldout", lambda out: heldout.generate_heldout_batch("coop_puzzle", out, [1009, 2027])),
])
def test_layout_manifest_is_readable_by_the_evaluator_and_by_training(tmp_path, split, generate):
    entries = generate(tmp_path)
    assert all(e.wad_id.startswith(f"coop_puzzle_{split}_seed_") for e in entries)
    assert heldout.load_manifest(tmp_path) == entries
    batch = WadBatch.from_dir(str(tmp_path))
    assert len(batch) == len(entries)


@pytest.mark.usefixtures("fake_coop_puzzle")
def test_manifest_with_inconsistent_id_is_rejected(tmp_path):
    heldout.generate_heldout_batch("coop_puzzle", tmp_path, [1009])
    manifest = tmp_path / "batch_registry.json"
    payload = json.loads(manifest.read_text())
    payload[0]["id"] = "something_else"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="id"):
        heldout.load_manifest(tmp_path)


@pytest.mark.parametrize("batch_dir", sorted((REPO_ROOT / "experiments" / "iclr2027_rebuttal" / "layouts").glob("batch_*")))
def test_committed_training_layout_batches_load(batch_dir):
    assert heldout.load_manifest(batch_dir)
