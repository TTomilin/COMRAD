"""
Print the exact benchmark command (from comrad/train_all.py ALGORITHMS) for a benchmark or very-difficult scenario,
optionally overriding resource/budget flags. Algorithm hyper-parameters are never touched unless explicitly passed
via --set.

Example:
  python -m experiments.very_difficult.very_difficult_cmd --env lava_maze_oracle_msg --algo QMIX \
      --set num_workers=1 --set num_envs_per_worker=2 --set device=cpu
"""
import argparse
import shlex

from comrad.train_all import ALGORITHMS, BENCHMARK_SCENARIOS
from experiments.very_difficult.profiles import (
    VERY_DIFFICULT_CONVERSE_SCENARIOS,
    VERY_DIFFICULT_EXTRA_ARGS,
    VERY_DIFFICULT_ORACLE_SCENARIOS,
    VERY_DIFFICULT_SCRIPTED_SCENARIOS,
)


def benchmark_tokens(algo: str, env: str | None = None):
    """Benchmark command tokens plus the per-experiment params the train_all launcher adds for this env."""
    for spec in ALGORITHMS:
        if spec.name == algo:
            tokens = shlex.split(spec.cmd)
            break
    else:
        raise SystemExit(f"Unknown algorithm {algo!r}; known: {[s.name for s in ALGORITHMS]}")
    if env is not None:
        scenarios = (
            BENCHMARK_SCENARIOS
            + VERY_DIFFICULT_SCRIPTED_SCENARIOS
            + VERY_DIFFICULT_ORACLE_SCENARIOS
            + VERY_DIFFICULT_CONVERSE_SCENARIOS
        )
        scenario = next((s for s in scenarios if s.env == env), None)
        if spec.uses_shared_reward and scenario is not None:
            tokens.append(f"--shared_reward_alpha={1.0 if scenario.actor_critic_shared_reward else 0.0}")
        tokens += shlex.split(VERY_DIFFICULT_EXTRA_ARGS.get((env, algo), ""))
    return tokens


def apply_overrides(tokens, overrides):
    out = list(tokens)
    for key, value in overrides.items():
        flag = f"--{key}="
        out = [t for t in out if not t.startswith(flag)]
        out.append(f"{flag}{value}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--algo", required=True)
    p.add_argument("--set", action="append", default=[], help="key=value flag override")
    args = p.parse_args()

    overrides = {"env": args.env}
    for item in args.set:
        key, _, value = item.partition("=")
        overrides[key] = value
    print(" ".join(apply_overrides(benchmark_tokens(args.algo, args.env), overrides)))


if __name__ == "__main__":
    main()
