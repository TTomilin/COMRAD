from comrad.train_all import ALGORITHMS, TRAIN_FOR_ENV_STEPS, AlgorithmSpec

# preprocess_cfg caps the default QMIX replay buffer to 1 GB / (obs_bytes * 2 * num_agents). Inputs wider than the
# canonical 3-channel Stealth Labyrinth frame (shared vision, state vectors) would change the capacity, so QMIX runs
# that are compared against the benchmark pin it to this value.
CANONICAL_STEALTH_QMIX_REPLAY_BUFFER = 1_000_000_000 // (3 * 72 * 128 * 2 * 2)


def algorithm_by_name(name: str) -> AlgorithmSpec:
    for algorithm in ALGORITHMS:
        if algorithm.name == name:
            return algorithm
    raise KeyError(name)


def with_train_steps(cmd: str, train_for_env_steps: int) -> str:
    """Benchmark command with its --train_for_env_steps budget replaced."""
    benchmark_flag = f"--train_for_env_steps={TRAIN_FOR_ENV_STEPS}"
    if benchmark_flag not in cmd:
        raise ValueError(f"{benchmark_flag} not found in {cmd!r}")
    return cmd.replace(benchmark_flag, f"--train_for_env_steps={train_for_env_steps}")
