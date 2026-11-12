import importlib
import re
from pathlib import Path

import pytest

from comrad import train_all
from experiments.common import CANONICAL_STEALTH_QMIX_REPLAY_BUFFER
from experiments.iclr2027_rebuttal import profiles as rebuttal
from experiments.very_difficult import profiles as very_difficult


def _algo_from_cmd(cmd: str) -> str:
    match = re.search(r"--algo=([^\s]+)", cmd)
    assert match is not None, cmd
    return match.group(1)


def _env_from_cmd(cmd: str) -> str:
    match = re.search(r"--env=([^\s]+)", cmd)
    assert match is not None, cmd
    return match.group(1)


@pytest.fixture
def load_runs(monkeypatch):
    def load(profile: str):
        monkeypatch.setenv("COMRAD_EXPERIMENT_PROFILE", profile)
        import experiments.runs as runs

        return importlib.reload(runs)

    return load


def test_runs_rejects_unknown_profile(load_runs):
    with pytest.raises(ValueError, match="COMRAD_EXPERIMENT_PROFILE"):
        load_runs("benchmark")


@pytest.mark.parametrize("profile", sorted({**very_difficult.PROFILES, **rebuttal.PROFILES}))
def test_every_profile_generates_runs(load_runs, profile):
    runs = load_runs(profile)
    generated = list(runs.RUN_DESCRIPTION.generate_experiments("/tmp/experiments", makedirs=False))
    assert runs.ACTIVE_PROFILE == profile
    assert len(generated) == runs.TOTAL_RUNS > 0
    assert all(cmd.startswith("python -m comrad.train ") for cmd, *_ in generated)


def test_very_difficult_profiles_keep_benchmark_protocol():
    scripted = very_difficult.very_difficult_scripted_experiments()
    assert {_env_from_cmd(e.cmd) for e in scripted} == {"stealth_labyrinth_scripted_scout", "lava_maze_scripted_observer"}
    for exp in scripted:
        assert "--train_for_env_steps=50000000" in exp.cmd
        assert ("--epsilon_decay_steps=10000000" in exp.cmd) is (_algo_from_cmd(exp.cmd) == "IDQN")
        assert sorted(p["seed"] for p in exp.params) == very_difficult.VERY_DIFFICULT_SEEDS
    oracle = very_difficult.very_difficult_oracle_experiments()
    assert len(oracle) == 4 * 2
    for exp in oracle:
        assert "--train_for_env_steps=100000000" in exp.cmd
        shared_vision_qmix = "stealth_labyrinth_shared_vision" in exp.cmd and _algo_from_cmd(exp.cmd) == "QMIX"
        assert (f"--replay_buffer_size={CANONICAL_STEALTH_QMIX_REPLAY_BUFFER}" in exp.cmd) is shared_vision_qmix


def test_rebuttal_kd6j_profiles_generate_expected_commands():
    # 1. Full-vs-Partial profile on the 9 scenarios outside the 2x2 subset (MAPPO & QMIX, both cells at 100M)
    full_obs_exps = rebuttal.full_observability_experiments()
    remaining = len(train_all.BENCHMARK_SCENARIOS) - len(rebuttal.REPRESENTATIVE_2X2_ENV_NAMES)
    assert remaining == 9
    assert len(full_obs_exps) == remaining * 2 * 2
    assert not {_env_from_cmd(e.cmd) for e in full_obs_exps} & set(rebuttal.REPRESENTATIVE_2X2_ENV_NAMES)
    assert sum("--full_observability=True" in e.cmd for e in full_obs_exps) == remaining * 2
    for exp in full_obs_exps:
        assert "--obs_modality=pixel" in exp.cmd
        assert f"--train_for_env_steps={rebuttal.REBUTTAL_KD6J_TRAIN_FOR_ENV_STEPS}" in exp.cmd
        if "--algo=QMIX" in exp.cmd:
            assert f"--replay_buffer_size={CANONICAL_STEALTH_QMIX_REPLAY_BUFFER}" in exp.cmd

    # 2. 2x2 State vs. Pixel x Partial vs. Full Observability profile (4 scenarios x 2 algos x 4 cells = 32 experiments)
    mod_2x2_exps = rebuttal.obs_modality_2x2_experiments()
    assert len(mod_2x2_exps) == 4 * 2 * 4
    state_exps = [e for e in mod_2x2_exps if "--obs_modality=state" in e.cmd]
    assert len(state_exps) == 4 * 2 * 2
    for exp in state_exps:
        assert "--obs_scale=1.0" in exp.cmd
        if "--algo=QMIX" in exp.cmd:
            assert f"--replay_buffer_size={CANONICAL_STEALTH_QMIX_REPLAY_BUFFER}" in exp.cmd

    # 3. Blind ablation profile (4 scenarios x MAPPO = 4 experiments)
    blind_exps = rebuttal.blind_ablation_experiments()
    assert len(blind_exps) == 4
    for exp in blind_exps:
        assert "--algo=MAPPO" in exp.cmd
        assert "--obs_modality=blind" in exp.cmd
        assert "--full_observability=False" in exp.cmd

    # 4. Visibility telemetry: 1 MAPPO Pixel-Partial seed on each of the 9 non-2x2 scenarios
    tel_exps = rebuttal.visibility_telemetry_experiments()
    assert len(tel_exps) == remaining
    for exp in tel_exps:
        assert "--algo=MAPPO" in exp.cmd
        assert "--full_observability=False" in exp.cmd
        assert len(exp.params) == 1 and exp.params[0]["seed"] == 42


def test_rebuttal_kd6j_minimal_design_has_81_runs_at_shared_budget():
    runs = 0
    for exps in (
        rebuttal.obs_modality_2x2_experiments(),
        rebuttal.blind_ablation_experiments(),
        rebuttal.visibility_telemetry_experiments(),
    ):
        for exp in exps:
            assert f"--train_for_env_steps={rebuttal.REBUTTAL_KD6J_TRAIN_FOR_ENV_STEPS}" in exp.cmd
            runs += len(exp.params)
    if rebuttal.REBUTTAL_KD6J_SEEDS == [42, 68]:
        assert runs == 64 + 8 + 9


def test_full_obs_clean_profile_trains_only_missing_mappo_seeds():
    exps = rebuttal.full_obs_clean_experiments()
    seeds_by_cell = {}
    for exp in exps:
        assert "--algo=MAPPO" in exp.cmd
        assert "--obs_modality=pixel" in exp.cmd
        assert "--use_additional_input=False" in exp.cmd
        assert f"--train_for_env_steps={rebuttal.REBUTTAL_KD6J_TRAIN_FOR_ENV_STEPS}" in exp.cmd
        assert _env_from_cmd(exp.cmd) in rebuttal.FULL_OBS_CLEAN_ENV_NAMES
        assert ("--full_observability=True" in exp.cmd) == ("pixel_full_clean" in exp.base_name)
        seeds_by_cell[exp.base_name] = sorted(p["seed"] for p in exp.params)
    if rebuttal.FULL_OBS_CLEAN_SEEDS == [42, 68, 81]:
        assert seeds_by_cell["MAPPO_pixel_partial_env_rhythm_sync_dense"] == [81]
        assert seeds_by_cell["MAPPO_pixel_partial_env_armory_siege"] == [81]
        assert seeds_by_cell["MAPPO_pixel_partial_env_coop_puzzle"] == [68, 81]
        assert sum(len(s) for s in seeds_by_cell.values()) == 3 * 3 + 4


def test_full_obs_global_profile_trains_only_missing_seeds():
    exps = rebuttal.full_obs_global_experiments()
    seeds_by_env = {_env_from_cmd(e.cmd): sorted(p["seed"] for p in e.params) for e in exps}
    # Rhythm Sync (all seeds) and Armory Siege s81 already exist at 50M frames.
    assert seeds_by_env == {"armory_siege": [42, 68], "coop_puzzle": [42, 68, 81]}
    for exp in exps:
        assert exp.base_name == f"MAPPO_pixel_full_global_env_{_env_from_cmd(exp.cmd)}"
        assert exp.root_dir_name == rebuttal.FULL_OBS_GLOBAL_ROOT_DIR == "full_obs_global"
        assert "--algo=MAPPO" in exp.cmd
        assert "--obs_modality=pixel" in exp.cmd
        assert "--full_observability=True" in exp.cmd
        assert "--full_obs_mode=global_state" in exp.cmd
        assert "--use_additional_input=False" in exp.cmd
        assert f"--train_for_env_steps={rebuttal.REBUTTAL_KD6J_TRAIN_FOR_ENV_STEPS}" in exp.cmd


def test_full_obs_global_v2_profile_matches_table3_protocol_on_layout_pools():
    exps = rebuttal.full_obs_global_v2_experiments(envs=rebuttal.FULL_OBS_CLEAN_ENV_NAMES)
    assert {_env_from_cmd(e.cmd) for e in exps} == set(rebuttal.FULL_OBS_CLEAN_ENV_NAMES)
    repo = Path(__file__).resolve().parents[2]
    for exp in exps:
        env = _env_from_cmd(exp.cmd)
        assert exp.base_name == f"MAPPO_pixel_full_global_v2_env_{env}"
        assert exp.root_dir_name == "full_obs_global_v2"
        assert sorted(p["seed"] for p in exp.params) == [42, 68, 81, 97, 154]
        layout_dir = rebuttal.layout_batch_dir(env)
        for flag in ("--algo=MAPPO", "--full_observability=True", "--full_obs_mode=global_state",
                     "--global_state_version=2", "--normalize_input_keys obs", "--use_additional_input=False",
                     "--train_for_env_steps=100000000", f"--wad_batch=experiments/iclr2027_rebuttal/layouts/batch_{env}_layouts",
                     "--curriculum=uniform"):
            assert flag in exp.cmd
        assert (repo / layout_dir / "batch_registry.json").is_file()


def test_state_entity_profile_yields_15_state_and_5_pixel_runs():
    from sample_factory.launcher.run_description import RunDescription

    exps = rebuttal.state_entity_experiments()
    cells = {("armory_siege", "MAPPO"), ("armory_siege", "QMIX"), ("ammo_carrier", "MAPPO"), ("ammo_carrier", "QMIX"),
             ("rhythm_sync_dense", "MAPPO")}
    state = [e for e in exps if "state_entity" in e.base_name]
    pixel = [e for e in exps if "pixel_partial" in e.base_name]
    assert len(state) + len(pixel) == len(exps) == 10
    # state cells are submitted first, Armory Siege first
    assert exps.index(pixel[0]) == len(state)
    assert _env_from_cmd(exps[0].cmd) == "armory_siege"
    for group, modality in ((state, "state"), (pixel, "pixel")):
        assert {(_env_from_cmd(e.cmd), _algo_from_cmd(e.cmd)) for e in group} == cells
        for exp in group:
            assert exp.root_dir_name == rebuttal.STATE_ENTITY_ROOT_DIR == "state_entity"
            assert f"--obs_modality={modality}" in exp.cmd
            assert "--full_observability=False" in exp.cmd
            assert f"--train_for_env_steps={rebuttal.REBUTTAL_KD6J_TRAIN_FOR_ENV_STEPS}" in exp.cmd
            assert ("--state_encoding=entity" in exp.cmd) is (modality == "state")
            assert ("--obs_scale=1.0" in exp.cmd) is (modality == "state")
    if rebuttal.STATE_ENTITY_SEEDS == [42, 68, 81]:
        assert all(sorted(p["seed"] for p in e.params) == [42, 68, 81] for e in state)
        assert all(sorted(p["seed"] for p in e.params) == [81] for e in pixel)
        assert sum(len(e.params) for e in exps) == 20

    # pixel seed-81 runs use exactly the obs_modality_2x2 Pixel-Partial command
    reference = {
        e.base_name: e.cmd for e in rebuttal.obs_modality_2x2_experiments() if "pixel_partial" in e.base_name
    }
    for exp in pixel:
        assert exp.cmd == reference[exp.base_name]

    # run_algo.sh gives each seed its own UDP port slot when the command mentions the profile
    runs = list(RunDescription("x", experiments=exps).generate_experiments("/tmp/train_all", makedirs=False))
    assert len(runs) == 20
    assert all("state_entity" in cmd for cmd, _, _, _ in runs)


def test_state_entity_profile_subset_filters(monkeypatch):
    exps = rebuttal.state_entity_experiments(envs={"armory_siege"}, modalities={"state"})
    assert [e.base_name for e in exps] == ["MAPPO_state_entity_env_armory_siege", "QMIX_state_entity_env_armory_siege"]
    monkeypatch.setenv("COMRAD_STATE_ENTITY_ENVS", "armory_siege")
    monkeypatch.setenv("COMRAD_STATE_ENTITY_MODALITIES", "state")
    assert [e.base_name for e in rebuttal.state_entity_experiments()] == [e.base_name for e in exps]
    monkeypatch.setenv("COMRAD_STATE_ENTITY_ENVS", "")
    monkeypatch.setenv("COMRAD_STATE_ENTITY_MODALITIES", "")
    assert len(rebuttal.state_entity_experiments()) == 10
    monkeypatch.setenv("COMRAD_STATE_ENTITY_ALGOS", "QMIX")
    assert {e.base_name.split("_")[0] for e in rebuttal.state_entity_experiments()} == {"QMIX"}
    assert len(rebuttal.state_entity_experiments()) == 4
