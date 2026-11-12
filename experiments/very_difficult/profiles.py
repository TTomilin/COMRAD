"""Training profiles of the Very Difficult learnability references (paper App. very_difficult).

Launched through `experiments.runs` (see experiments/README.md).
"""
from sample_factory.launcher.run_description import Experiment, ParamGrid

from comrad.train_all import BenchmarkScenario, _command_for_scenario, _experiment_name_for_scenario
from experiments.common import CANONICAL_STEALTH_QMIX_REPLAY_BUFFER, algorithm_by_name, with_train_steps

# Algorithms, hyper-parameters and seeds follow the benchmark protocol exactly (comrad.train_all.ALGORITHMS).
# Scripted-partner envs expose a single learner to SF, so IPPO/IDQN run their single-agent paths; the learner gets the
# benchmark reward (team reward for IPPO via SharedRewardWrapper before the partner is hidden, individual for IDQN).
VERY_DIFFICULT_SCRIPTED_NAME = "comrad_very_difficult_scripted"
VERY_DIFFICULT_ORACLE_NAME = "comrad_very_difficult_oracle"
VERY_DIFFICULT_SCRIPTED_ROOT_DIR = "very_difficult_scripted"
VERY_DIFFICULT_ORACLE_ROOT_DIR = "very_difficult_oracle"
VERY_DIFFICULT_SEEDS = [42, 68, 81, 97, 154]
VERY_DIFFICULT_SCRIPTED_SCENARIOS = [
    BenchmarkScenario("Stealth Labyrinth (scripted scout)", "stealth_labyrinth_scripted_scout", True),
    BenchmarkScenario("Lava Maze (scripted observer)", "lava_maze_scripted_observer", True),
]
VERY_DIFFICULT_SCRIPTED_ALGORITHMS = ["IPPO", "IDQN"]
# Converse direction: the leader / sender role is learned against a fixed follower / receiver (scripted gunner,
# scripted navigator). Same algorithms, seeds and single-learner budget as the scripted-partner runs above.
VERY_DIFFICULT_CONVERSE_NAME = "comrad_very_difficult_converse"
VERY_DIFFICULT_CONVERSE_ROOT_DIR = "very_difficult_converse"
VERY_DIFFICULT_CONVERSE_SCENARIOS = [
    BenchmarkScenario("Stealth Labyrinth (scripted gunner)", "stealth_labyrinth_scripted_gunner", True),
    BenchmarkScenario("Lava Maze (scripted navigator)", "lava_maze_scripted_navigator", True),
]
VERY_DIFFICULT_CONVERSE_ALGORITHMS = VERY_DIFFICULT_SCRIPTED_ALGORITHMS
# Table 2 evaluates the checkpoint at 100M counted env steps. SF keeps only the last `keep_checkpoints=2` periodic
# checkpoints, so training to 125M would leave no 100M snapshot; these runs therefore stop exactly at the evaluation
# frontier (no algorithm used here has a schedule tied to train_for_env_steps; only HAPPO's linear LR decay is).
VERY_DIFFICULT_ORACLE_TRAIN_FOR_ENV_STEPS = 100_000_000
# SF's env-step counter (also driving the DQN/QMIX epsilon schedule) sums frames over learners. With N=2 learners the
# 100M frontier gives each agent 50M of its own frames (12.5M joint steps). A single learner behind a scripted partner
# reaches the same per-agent experience AND the same joint game time at half the counter value, so its budget and the
# IDQN epsilon horizon (benchmark default 20M counted = 10M per agent) are halved.
VERY_DIFFICULT_SCRIPTED_TRAIN_FOR_ENV_STEPS = VERY_DIFFICULT_ORACLE_TRAIN_FOR_ENV_STEPS // 2
BENCHMARK_EPSILON_DECAY_STEPS = 20_000_000
VERY_DIFFICULT_ORACLE_SCENARIOS = [
    BenchmarkScenario("Lava Maze (oracle message head)", "lava_maze_oracle_msg", True),
    BenchmarkScenario("Lava Maze (oracle direction hint)", "lava_maze_oracle_hint", True),
    BenchmarkScenario("Stealth Labyrinth (always lit)", "stealth_labyrinth_lit", True),
    BenchmarkScenario("Stealth Labyrinth (shared vision)", "stealth_labyrinth_shared_vision", True),
]
VERY_DIFFICULT_ORACLE_ALGORITHMS = ["MAPPO", "QMIX"]
VERY_DIFFICULT_EXTRA_ARGS = {
    ("stealth_labyrinth_shared_vision", "QMIX"): f"--replay_buffer_size={CANONICAL_STEALTH_QMIX_REPLAY_BUFFER}",
    ("stealth_labyrinth_scripted_scout", "IDQN"): f"--epsilon_decay_steps={BENCHMARK_EPSILON_DECAY_STEPS // 2}",
    ("lava_maze_scripted_observer", "IDQN"): f"--epsilon_decay_steps={BENCHMARK_EPSILON_DECAY_STEPS // 2}",
    ("stealth_labyrinth_scripted_gunner", "IDQN"): f"--epsilon_decay_steps={BENCHMARK_EPSILON_DECAY_STEPS // 2}",
    ("lava_maze_scripted_navigator", "IDQN"): f"--epsilon_decay_steps={BENCHMARK_EPSILON_DECAY_STEPS // 2}",
}


def _very_difficult_experiments(scenarios, algorithm_names, root_dir_name, train_for_env_steps):
    experiments = []
    for scenario in scenarios:
        for algorithm in (algorithm_by_name(name) for name in algorithm_names):
            cmd = with_train_steps(_command_for_scenario(scenario, algorithm), train_for_env_steps)
            extra_args = VERY_DIFFICULT_EXTRA_ARGS.get((scenario.env, algorithm.name))
            if extra_args:
                cmd = f"{cmd} {extra_args}"
            if algorithm.uses_shared_reward:
                shared_reward_alpha = 1.0 if scenario.actor_critic_shared_reward else 0.0
                params = ParamGrid([("shared_reward_alpha", [shared_reward_alpha]), ("seed", VERY_DIFFICULT_SEEDS)])
            else:
                params = ParamGrid([("seed", VERY_DIFFICULT_SEEDS)])
            experiments.append(
                Experiment(
                    _experiment_name_for_scenario(scenario, algorithm),
                    cmd,
                    params.generate_params(randomize=False),
                    root_dir_name=root_dir_name,
                )
            )
    return experiments


def very_difficult_scripted_experiments():
    return _very_difficult_experiments(
        VERY_DIFFICULT_SCRIPTED_SCENARIOS,
        VERY_DIFFICULT_SCRIPTED_ALGORITHMS,
        VERY_DIFFICULT_SCRIPTED_ROOT_DIR,
        VERY_DIFFICULT_SCRIPTED_TRAIN_FOR_ENV_STEPS,
    )


def very_difficult_converse_experiments():
    return _very_difficult_experiments(
        VERY_DIFFICULT_CONVERSE_SCENARIOS,
        VERY_DIFFICULT_CONVERSE_ALGORITHMS,
        VERY_DIFFICULT_CONVERSE_ROOT_DIR,
        VERY_DIFFICULT_SCRIPTED_TRAIN_FOR_ENV_STEPS,
    )


def very_difficult_oracle_experiments():
    return _very_difficult_experiments(
        VERY_DIFFICULT_ORACLE_SCENARIOS,
        VERY_DIFFICULT_ORACLE_ALGORITHMS,
        VERY_DIFFICULT_ORACLE_ROOT_DIR,
        VERY_DIFFICULT_ORACLE_TRAIN_FOR_ENV_STEPS,
    )


PROFILES = {
    "very_difficult_scripted": (VERY_DIFFICULT_SCRIPTED_NAME, very_difficult_scripted_experiments),
    "very_difficult_oracle": (VERY_DIFFICULT_ORACLE_NAME, very_difficult_oracle_experiments),
    "very_difficult_converse": (VERY_DIFFICULT_CONVERSE_NAME, very_difficult_converse_experiments),
}
