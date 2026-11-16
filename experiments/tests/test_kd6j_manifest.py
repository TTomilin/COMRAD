import json
from pathlib import Path

import pytest

from experiments.iclr2027_rebuttal.kd6j_manifest import (
    EVAL_SETS,
    SUPERSEDED_RUNS,
    ManifestError,
    build_manifest,
    load_superseded,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ARRAY_SCRIPT = REPO_ROOT / "experiments" / "iclr2027_rebuttal" / "kd6j_heldout_eval_array.sh"
INVENTORY = REPO_ROOT / "results" / "rebuttal_kd6j" / "checkpoint_inventory.json"
PROFILE_OF_CELL = {
    "pixel_full_clean": "full_obs_clean",
    "pixel_full_global": "full_obs_global",
    "pixel_full_global_v2": "full_obs_global_v2",
    "state_entity": "state_entity",
    "blind": "blind_ablation",
}


def make_run(root: Path, profile: str, scenario: str, algo: str, cell: str, seed: int, steps: int,
             stamp: str = "20261109_000000", config: bool = True) -> Path:
    run = root / f"mode3_comrad_{profile}_{stamp}" / profile / f"00_{algo}_{cell}_env_{scenario}_see_{seed}"
    (run / "checkpoint_p0").mkdir(parents=True)
    if steps:
        (run / "checkpoint_p0" / f"checkpoint_000001000_{steps}.pth").write_bytes(b"")
    if config:
        (run / "config.json").write_text("{}")
    return run


def profile_for(eval_set, cell: tuple[str, str, str], seed: int) -> str:
    if len(eval_set.profiles) == 1:
        return eval_set.profiles[0]
    scenario, _, tag = cell
    if tag in PROFILE_OF_CELL:
        return PROFILE_OF_CELL[tag]
    if scenario in ("rhythm_sync_dense", "armory_siege", "ammo_carrier", "stealth_labyrinth") and seed in (42, 68):
        return "obs_modality_2x2"
    return "visibility_telemetry" if seed == 42 else "full_obs_clean"


def populate(root: Path, eval_set) -> int:
    n = 0
    for cell, seeds in eval_set.expected().items():
        for seed in seeds:
            make_run(root, profile_for(eval_set, cell, seed), *cell, seed, eval_set.target_steps + 20_352)
            n += 1
    return n


@pytest.mark.parametrize("name", sorted(EVAL_SETS))
def test_manifest_covers_exactly_the_expected_cells(tmp_path, name):
    eval_set = EVAL_SETS[name]
    n = populate(tmp_path, eval_set)
    rows = build_manifest(eval_set, tmp_path, set(), tmp_path / "out")
    assert len(rows) == n
    assert {((r.scenario, r.algorithm, r.cell_tag), r.train_seed) for r in rows} == {
        (cell, s) for cell, seeds in eval_set.expected().items() for s in seeds
    }
    assert all(r.output_json.parent == tmp_path / "out" / eval_set.out_dir for r in rows)
    assert all(r.run_interventions == (eval_set.run_interventions and r.cell_tag == "pixel_partial") for r in rows)
    assert [r.run_interventions for r in rows] == sorted((r.run_interventions for r in rows), reverse=True)


def test_set_sizes_match_the_committed_result_directories():
    sizes = {name: sum(len(s) for s in e.expected().values()) for name, e in EVAL_SETS.items()}
    assert sizes == {"kd6j_50m": 103, "full_obs_global_v2": 15, "state_entity": 20}
    assert len(list((REPO_ROOT / "results" / "rebuttal_kd6j" / "heldout_runs").glob("*.json"))) == 118
    assert len(list((REPO_ROOT / "results" / "rebuttal_kd6j" / "heldout_runs_state_entity").glob("*.json"))) == 20


def test_unfinished_and_configless_launches_drop_out(tmp_path):
    eval_set = EVAL_SETS["state_entity"]
    populate(tmp_path, eval_set)
    make_run(tmp_path, "state_entity", "rhythm_sync_dense", "MAPPO", "state_entity", 42, 15_286_272, "20261115_221356")
    make_run(tmp_path, "state_entity", "rhythm_sync_dense", "MAPPO", "state_entity", 42, 0, "20261115_230735", config=False)
    make_run(tmp_path, "state_entity", "armory_siege", "QMIX", "state_entity", 42, 50_020_352, "20261115_200641", config=False)
    assert len(build_manifest(eval_set, tmp_path, set(), tmp_path / "out")) == 20


def test_two_completed_runs_for_one_cell_raise_and_name_both(tmp_path):
    eval_set = EVAL_SETS["state_entity"]
    populate(tmp_path, eval_set)
    dup = make_run(tmp_path, "state_entity", "ammo_carrier", "MAPPO", "state_entity", 68, 50_020_352, "20261116_095126")
    with pytest.raises(ManifestError, match="duplicate runs for ammo_carrier/MAPPO/state_entity seed 68") as err:
        build_manifest(eval_set, tmp_path, set(), tmp_path / "out")
    assert str(dup) in str(err.value)
    rows = build_manifest(eval_set, tmp_path, {dup.relative_to(tmp_path).as_posix()}, tmp_path / "out")
    assert dup not in {r.run_dir for r in rows} and len(rows) == 20


def test_missing_seed_raises_before_submission(tmp_path):
    eval_set = EVAL_SETS["full_obs_global_v2"]
    populate(tmp_path, eval_set)
    victim = next(tmp_path.rglob("*_env_coop_puzzle_see_154"))
    (next((victim / "checkpoint_p0").iterdir())).rename(victim / "checkpoint_p0" / "checkpoint_000000500_60000000.pth")
    with pytest.raises(ManifestError, match=r"coop_puzzle/MAPPO/pixel_full_global_v2 seeds \[154\]"):
        build_manifest(eval_set, tmp_path, set(), tmp_path / "out")


def test_unexpected_run_raises(tmp_path):
    eval_set = EVAL_SETS["kd6j_50m"]
    populate(tmp_path, eval_set)
    make_run(tmp_path, "obs_modality_2x2", "ammo_carrier", "MAPPO", "pixel_partial", 81, 50_020_352)
    with pytest.raises(ManifestError, match="unexpected run ammo_carrier/MAPPO/pixel_partial seed 81"):
        build_manifest(eval_set, tmp_path, set(), tmp_path / "out")


def test_profile_prefix_does_not_capture_v2(tmp_path):
    eval_set = EVAL_SETS["kd6j_50m"]
    populate(tmp_path, eval_set)
    make_run(tmp_path, "full_obs_global_v2", "coop_puzzle", "MAPPO", "pixel_full_global", 42, 50_020_352)
    assert len(build_manifest(eval_set, tmp_path, set(), tmp_path / "out")) == 103


def test_superseded_entries_need_a_reason(tmp_path):
    path = tmp_path / "superseded.json"
    path.write_text(json.dumps([{"run_dir": "a/b"}]))
    with pytest.raises(ManifestError, match="reason"):
        load_superseded(path)
    assert isinstance(load_superseded(SUPERSEDED_RUNS), set)


def test_committed_inventory_resolves_every_set_without_superseded_entries():
    inventory = json.loads(INVENTORY.read_text())["runs"]
    for eval_set in EVAL_SETS.values():
        complete = {}
        for run in inventory:
            if run["profile"].removeprefix("comrad_") in eval_set.profiles and run["has_config"] \
                    and run["checkpoint_steps"] >= eval_set.min_steps:
                key = ((run["scenario"], run["algorithm"], run["cell_tag"]), run["train_seed"])
                complete.setdefault(key, []).append(run["run_dir"])
        expected = {(cell, s) for cell, seeds in eval_set.expected().items() for s in seeds}
        assert set(complete) == expected, eval_set.name
        assert all(len(dirs) == 1 for dirs in complete.values()), eval_set.name


def test_array_script_selects_runs_only_through_the_manifest_module():
    text = ARRAY_SCRIPT.read_text()
    assert "kd6j_manifest --eval-set \"$EVAL_SET\"" in text
    assert "<<'EOF'" not in text and "KD6J_PROFILES" not in text and "KD6J_SEEDS" not in text
