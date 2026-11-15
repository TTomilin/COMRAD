"""
Evaluates every policy on every generated WAD (bypass letting normal batch sampler choose layouts independently) for fair comparison
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DOOMGEN_ROOT = REPO_ROOT / "DoomGen"
for _path in (DOOMGEN_ROOT, DOOMGEN_ROOT / "src", DOOMGEN_ROOT / "examples" / "benchmark"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from comrad.train import (  # noqa: E402
    configure_batch_env_and_agents,
    parse_args as parse_comrad_args,
    register_model_factory,
    register_vizdoom_components,
)
from comrad.utils.confidence import ci95_half_width  # noqa: E402
from comrad.utils.doom_utils import resolve_archived_use_additional_input  # noqa: E402
from comrad.utils.recording_actions import reshape_deterministic_actions  # noqa: E402
from sample_factory.algo.learning.learner import Learner  # noqa: E402
from sample_factory.algo.sampling.batched_sampling import preprocess_actions  # noqa: E402
from sample_factory.algo.utils.action_distributions import argmax_actions  # noqa: E402
from sample_factory.algo.utils.env_info import extract_env_info  # noqa: E402
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs  # noqa: E402
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor  # noqa: E402
from sample_factory.cfg.arguments import load_from_checkpoint  # noqa: E402
from sample_factory.enjoy import make_env as make_eval_env  # noqa: E402
from sample_factory.model.actor_critic import create_actor_critic  # noqa: E402
from sample_factory.model.model_utils import get_rnn_size  # noqa: E402


HELDOUT_SEEDS = (1009, 2027, 3037, 4051, 5099)
DEFAULT_ARCHIVE_ROOT = "train_dir/mode3_comrad_benchmark_20260503_134649"
DEFAULT_SCENARIOS = ("ammo_carrier", "armory_siege", "coop_puzzle")
DEFAULT_ALGORITHMS = ("IPPO", "MAPPO", "HAPPO", "QMIX", "QPLEX-D")
DEFAULT_TARGET_STEPS = 100_000_000

POLICY_DIRECTORY_TOKENS = {
    "IPPO": "_IPPO_",
    "MAPPO": "_MAPPO_",
    "HAPPO": "_HAPPO_",
    "QMIX": "_QMIX_",
    "QPLEX-D": "_QPLEX_dmaq_",
    "IDQN": "_IDQN_",
}
# directory-name algorithm token used by comrad/train_all.py experiment names ("{ALGO}_env_{env}")
POLICY_NAME_TOKENS = {"QPLEX-D": "QPLEX_dmaq"}
SCENARIO_CLASSES = {
    "stag_hunt_arena": ("examples.benchmark.stag_hunt_arena", "StagHuntArenaScenario"),
    "rhythm_sync": ("examples.benchmark.rhythm_sync", "RhythmSyncScenario"),
    "foraging_commons": ("examples.benchmark.foraging_commons", "ForagingCommonsScenario"),
    "coop_puzzle": ("examples.benchmark.coop_puzzle", "CoopPuzzleScenario"),
    "platform_chain": ("examples.benchmark.platform_chain", "PlatformChainScenario"),
    "armory_siege": ("examples.benchmark.armory_siege", "ArmorySiegeScenario"),
    "coop_health_gathering": ("examples.benchmark.coop_health_gathering", "CoopHealthGatheringScenario"),
    "lavapit": ("examples.benchmark.lavapit", "LavaPitScenario"),
    "smart_enemies": ("examples.benchmark.smart_enemies", "SmartEnemiesScenario"),
    "dumb_enemies": ("examples.benchmark.dumb_enemies", "DumbEnemiesScenario"),
    "stealth_labyrinth": ("examples.benchmark.stealth_labyrinth", "StealthLabyrinthScenario"),
    "ammo_carrier": ("examples.benchmark.ammo_carrier", "AmmoCarrierScenario"),
    "lava_maze": ("examples.benchmark.lava_maze", "LavaMazeScenario"),
}
# Learnability-reference variants (App. very_difficult): env name -> (generator scenario, generator overrides).
# Held-out layouts only change the seed; the variant's ACS/lighting change is kept.
SCENARIO_VARIANTS = {
    "rhythm_sync_dense": ("rhythm_sync", {}),
    "stealth_labyrinth_scripted_scout": ("stealth_labyrinth", {"scripted_scout": True}),
    "stealth_labyrinth_scripted_gunner": ("stealth_labyrinth", {"scripted_gunner": True}),
    # fully scripted reference (experiments/very_difficult/fully_scripted_reference_eval.py): both ACS partners, evaluated in the
    # stealth_labyrinth_scripted_gunner env with a no-op scout seat
    "stealth_labyrinth_scripted_both": ("stealth_labyrinth", {"scripted_scout": True, "scripted_gunner": True}),
    "stealth_labyrinth_lit": ("stealth_labyrinth", {"always_lit": True}),
    "stealth_labyrinth_shared_vision": ("stealth_labyrinth", {}),
    "lava_maze_scripted_observer": ("lava_maze", {}),
    "lava_maze_scripted_navigator": ("lava_maze", {}),
    "lava_maze_oracle_msg": ("lava_maze", {}),
    "lava_maze_oracle_hint": ("lava_maze", {}),
}
CHECKPOINT_RE = re.compile(r"^(?:best|checkpoint)_\d+_(?P<steps>\d+)(?:_|\.)")


@dataclass(frozen=True)
class PolicySpec:
    scenario: str
    algorithm: str
    run_dir: str
    checkpoint: str
    checkpoint_steps: int
    config_sha256: str
    checkpoint_sha256: str
    train_seed: int | None = None


@dataclass(frozen=True)
class WadManifestEntry:
    scenario: str
    wad_id: str
    seed: int
    filename: str
    sha256: str
    config: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_csv_argument(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    return values


def checkpoint_steps(path: Path) -> int:
    match = CHECKPOINT_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unrecognized checkpoint filename: {path.name}")
    return int(match.group("steps"))


def select_checkpoint(run_dir: Path, target_steps: int) -> tuple[Path, int]:
    candidates = []
    for path in (run_dir / "checkpoint_p0").glob("*.pth"):
        try:
            steps = checkpoint_steps(path)
        except ValueError:
            continue
        candidates.append((path, steps))
    if not candidates:
        raise FileNotFoundError(f"No parseable checkpoint files in {run_dir / 'checkpoint_p0'}")
    # Prefer a non-retrospective periodic checkpoint at equal distance
    # The archive does not always retain an exact 100M snapshot, so this selection is recorded in the result metadata instead of silently substituting best
    candidates.sort(key=lambda item: (abs(item[1] - target_steps), item[0].name.startswith("best_"), -item[1]))
    return candidates[0]


def discover_policy_specs(
    archive_root: Path,
    scenarios: Iterable[str],
    algorithms: Iterable[str],
    target_steps: int,
) -> list[PolicySpec]:
    policies: list[PolicySpec] = []
    for scenario in scenarios:
        scenario_root = archive_root / scenario
        if not scenario_root.is_dir():
            raise FileNotFoundError(f"Missing archive scenario directory: {scenario_root}")
        for algorithm in algorithms:
            if algorithm not in POLICY_DIRECTORY_TOKENS:
                raise ValueError(f"Unsupported algorithm {algorithm!r}; choices: {sorted(POLICY_DIRECTORY_TOKENS)}")
            token = POLICY_DIRECTORY_TOKENS[algorithm]
            matches = sorted(path for path in scenario_root.iterdir() if path.is_dir() and token in path.name)
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected exactly one {algorithm} run for {scenario} under {scenario_root}, found {matches}"
                )
            run_dir = matches[0]
            config_path = run_dir / "config.json"
            if not config_path.is_file():
                raise FileNotFoundError(f"Missing archived config: {config_path}")
            checkpoint, steps = select_checkpoint(run_dir, target_steps)
            policies.append(
                PolicySpec(
                    scenario=scenario,
                    algorithm=algorithm,
                    run_dir=str(run_dir),
                    checkpoint=str(checkpoint),
                    checkpoint_steps=steps,
                    config_sha256=sha256_file(config_path),
                    checkpoint_sha256=sha256_file(checkpoint),
                )
            )
    return policies


def seeded_run_pattern(scenario: str, algorithm: str, train_seed: int, experiment_tag: str = "") -> re.Pattern[str]:
    algo_token = re.escape(POLICY_NAME_TOKENS.get(algorithm, algorithm))
    tag_part = f"_{re.escape(experiment_tag)}" if experiment_tag else "(?:_(?:full_obs|pixel_partial|pixel_full|state_partial|state_full|blind))?"
    return re.compile(
        rf"(?:^|_){algo_token}{tag_part}_env_{re.escape(scenario)}_(?:s\.r\.alp_[0-9.]+_)?see_{int(train_seed)}(?:_|$)"
    )


def discover_seeded_policy_specs(
    archive_root: Path,
    scenarios: Iterable[str],
    algorithms: Iterable[str],
    train_seeds: Iterable[int],
    target_steps: int,
    experiment_tag: str = "",
) -> list[PolicySpec]:
    """Multi-seed discovery for train_all.py profile archives (one run per scenario/algorithm/seed anywhere under
    archive_root). `failed_*` directories are ignored so later successful reruns are picked up."""
    run_dirs = sorted(
        path.parent
        for path in archive_root.rglob("config.json")
        if not any(part.startswith("failed_") for part in path.relative_to(archive_root).parts)
    )
    policies: list[PolicySpec] = []
    for scenario in scenarios:
        for algorithm in algorithms:
            if algorithm not in POLICY_DIRECTORY_TOKENS:
                raise ValueError(f"Unsupported algorithm {algorithm!r}; choices: {sorted(POLICY_DIRECTORY_TOKENS)}")
            for train_seed in train_seeds:
                pattern = seeded_run_pattern(scenario, algorithm, train_seed, experiment_tag=experiment_tag)
                matches = [run_dir for run_dir in run_dirs if pattern.search(run_dir.name)]
                if len(matches) != 1:
                    raise FileNotFoundError(
                        f"Expected exactly one {algorithm} seed-{train_seed} run for {scenario} under {archive_root}, "
                        f"found {matches}"
                    )
                run_dir = matches[0]
                checkpoint, steps = select_checkpoint(run_dir, target_steps)
                policies.append(
                    PolicySpec(
                        scenario=scenario,
                        algorithm=algorithm,
                        run_dir=str(run_dir),
                        checkpoint=str(checkpoint),
                        checkpoint_steps=steps,
                        config_sha256=sha256_file(run_dir / "config.json"),
                        checkpoint_sha256=sha256_file(checkpoint),
                        train_seed=int(train_seed),
                    )
                )
    return policies


def resolve_generator(scenario: str) -> tuple[str, dict[str, Any]]:
    generator, overrides = SCENARIO_VARIANTS.get(scenario, (scenario, {}))
    if generator not in SCENARIO_CLASSES:
        raise ValueError(
            f"Unsupported scenario {scenario!r}; choices: {sorted(set(SCENARIO_CLASSES) | set(SCENARIO_VARIANTS))}"
        )
    return generator, dict(overrides)


def generate_heldout_batch(scenario: str, output_dir: Path, seeds: Iterable[int]) -> list[WadManifestEntry]:
    return generate_layout_batch(scenario, output_dir, seeds, split="heldout")


def generate_layout_batch(scenario: str, output_dir: Path, seeds: Iterable[int], split: str) -> list[WadManifestEntry]:
    """Generate only layout changes: each WAD uses the scenario's default config (plus variant overrides) and a new seed."""
    generator, overrides = resolve_generator(scenario)
    module_name, class_name = SCENARIO_CLASSES[generator]
    module = __import__(module_name, fromlist=[class_name])
    scenario_cls = getattr(module, class_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries: list[WadManifestEntry] = []
    for seed in seeds:
        wad_id = f"{scenario}_{split}_seed_{seed}"
        filename = f"{wad_id}.wad"
        wad_path = output_dir / filename
        generated = scenario_cls({**overrides, "seed": int(seed)}, name=wad_id)
        generated.generate(str(wad_path))
        entries.append(
            WadManifestEntry(
                scenario=scenario,
                wad_id=wad_id,
                seed=int(seed),
                filename=filename,
                sha256=sha256_file(wad_path),
                config=dict(generated.config),
            )
        )

    write_manifest(output_dir, entries)
    return entries


def write_manifest(batch_dir: Path, entries: Iterable[WadManifestEntry]) -> None:
    """Write a manifest readable by both `load_manifest` and `comrad.envs.wad_catalog.WadBatch` (which keys on `id`)."""
    payload = [{"id": entry.wad_id, **asdict(entry)} for entry in entries]
    (batch_dir / "batch_registry.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _manifest_entry(item: dict[str, Any]) -> WadManifestEntry:
    fields = dict(item)
    wad_id = fields.pop("id", fields.get("wad_id"))
    if wad_id != fields.get("wad_id"):
        raise ValueError(f"manifest entry id {wad_id!r} != wad_id {fields.get('wad_id')!r}")
    return WadManifestEntry(**fields)


def load_manifest(batch_dir: Path) -> list[WadManifestEntry]:
    manifest_path = batch_dir / "batch_registry.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [_manifest_entry(item) for item in payload]
    for entry in entries:
        wad_path = batch_dir / entry.filename
        actual_sha = sha256_file(wad_path)
        if actual_sha != entry.sha256:
            raise RuntimeError(f"WAD hash mismatch for {wad_path}: {actual_sha} != {entry.sha256}")
    return entries


def singleton_batch_dir(batch_dir: Path, entry: WadManifestEntry) -> tempfile.TemporaryDirectory[str]:
    """Expose one absolute WAD through the standard batch-env interface."""
    temp_dir = tempfile.TemporaryDirectory(prefix="comrad_heldout_single_")
    payload = [{"id": entry.wad_id, "filename": str((batch_dir / entry.filename).resolve()), "seed": entry.seed}]
    Path(temp_dir.name, "batch_registry.json").write_text(json.dumps(payload), encoding="utf-8")
    return temp_dir


VALUE_BASED_ALGOS = ("IDQN", "QMIX", "VDN", "QPLEX")
# Lava Maze variants: USER22 (levels counter) is incremented by ACS every tic the navigator stands on the goal tile
# until GenerateMaze relocates the goal, so we additionally count actual goal relocations (USER13/USER18 changes).
GOAL_TRACKED_SCENARIOS = ("lava_maze",)


def archived_algo(run_dir: Path) -> str:
    """Algorithm stored in the archived config.json.

    It must be passed explicitly: sample_factory's load_from_checkpoint keeps every argument given on the command line
    over the archived value, so a placeholder like --algo=APPO would silently rebuild the wrong model (e.g. MAPPO
    checkpoints fail to load, QMIX/IDQN would be evaluated with actor-critic action sampling).
    """
    algo = json.loads((run_dir / "config.json").read_text(encoding="utf-8")).get("algo")
    if not algo:
        raise ValueError(f"Archived config without 'algo': {run_dir / 'config.json'}")
    return str(algo)


def action_head_sizes(action_space: Any) -> list[int]:
    if hasattr(action_space, "n"):
        return [int(action_space.n)]
    if hasattr(action_space, "spaces"):
        return [int(space.n) for space in action_space.spaces if hasattr(space, "n")]
    raise ValueError(f"Unsupported action space for greedy evaluation: {action_space}")


def greedy_q_actions(action_logits: torch.Tensor, head_sizes: list[int]) -> torch.Tensor:
    """Greedy per-head argmax over Q-values, identical to the inference worker's value-based action rule (epsilon=0)."""
    if len(head_sizes) == 1:
        return action_logits.argmax(dim=1)
    return torch.stack([q.argmax(dim=1) for q in torch.split(action_logits, head_sizes, dim=1)], dim=1)


class GoalRelocationCounter:
    """Counts goal relocations (valid USER13/USER18 goal -> different valid goal) within an episode."""

    def __init__(self) -> None:
        self.prev: tuple[int, int] | None = None
        self.count = 0

    @staticmethod
    def _goal(infos: Any) -> tuple[int, int] | None:
        info = infos[0] if isinstance(infos, (list, tuple)) and infos else infos
        if not isinstance(info, dict) or "USER13" not in info or "USER18" not in info:
            return None
        try:
            goal = (int(info["USER13"]), int(info["USER18"]))
        except (TypeError, ValueError):
            return None
        return goal if goal[0] >= 0 and goal[1] >= 0 else None

    def observe(self, infos: Any) -> None:
        goal = self._goal(infos)
        if goal is None:
            return
        if self.prev is not None and goal != self.prev:
            self.count += 1
        self.prev = goal

    def reset(self) -> None:
        self.prev = None
        self.count = 0


def fresh_episode_infos(infos: Any) -> Any:
    """Infos describing the first frame of the episode that the env has just auto-reset into.

    On the step that ends an episode, MultiAgentEnv returns the old episode's terminal infos and stores the new
    episode's initial infos under ``reset_info``; counters seeded after the reset must read the latter.
    """
    if isinstance(infos, (list, tuple)) and infos and all(isinstance(i, dict) for i in infos):
        if any("reset_info" in i for i in infos):
            return [i["reset_info"] if isinstance(i.get("reset_info"), dict) else {} for i in infos]
    return infos


def build_eval_cfg(policy: PolicySpec, wad_batch: Path, device: str, deterministic: bool) -> Any:
    run_dir = Path(policy.run_dir)
    archived_payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    argv = [
        f"--algo={archived_algo(run_dir)}",
        f"--train_dir={run_dir.parent}",
        f"--experiment={run_dir.name}",
        f"--env={policy.scenario}",
        f"--device={device}",
        "--with_wandb=False",
        "--wandb_record_every=0",
        "--no_render",
    ]
    if "full_observability" in archived_payload:
        argv.append(f"--full_observability={bool(archived_payload['full_observability'])}")
    if "full_obs_mode" in archived_payload:
        argv.append(f"--full_obs_mode={archived_payload['full_obs_mode']}")
    # Global-state runs from before --global_state_version existed used encoding version 1
    global_state_version = int(archived_payload.get("global_state_version") or 1)
    argv.append(f"--global_state_version={global_state_version}")
    if "obs_modality" in archived_payload:
        argv.append(f"--obs_modality={archived_payload['obs_modality']}")
    if "obs_scale" in archived_payload:
        argv.append(f"--obs_scale={float(archived_payload['obs_scale'])}")
    # Archived state runs from before --state_encoding existed used the `ego` vector
    state_encoding = str(archived_payload.get("state_encoding") or "ego")
    argv.append(f"--state_encoding={state_encoding}")
    if "state_sight_range" in archived_payload:
        argv.append(f"--state_sight_range={float(archived_payload['state_sight_range'])}")
    use_additional_input = resolve_archived_use_additional_input(archived_payload)
    if use_additional_input is not None:
        argv.append(f"--use_additional_input={use_additional_input}")
    if deterministic:
        argv.append("--eval_deterministic=True")
    cfg = parse_comrad_args(argv=argv, evaluation=True)
    cfg = load_from_checkpoint(cfg)
    cfg.train_dir = str(run_dir.parent)
    cfg.experiment = run_dir.name
    cfg.env = policy.scenario
    cfg.wad_batch = str(wad_batch)
    cfg.curriculum = "uniform"
    cfg.no_render = True
    cfg.save_video = False
    cfg.with_wandb = False
    cfg.device = device
    cfg.eval_deterministic = deterministic
    cfg.policy_index = 0
    cfg.num_envs = 1
    if "full_observability" in archived_payload:
        cfg.full_observability = bool(archived_payload["full_observability"])
    if "full_obs_mode" in archived_payload:
        cfg.full_obs_mode = str(archived_payload["full_obs_mode"])
    cfg.global_state_version = global_state_version
    if "obs_modality" in archived_payload:
        cfg.obs_modality = str(archived_payload["obs_modality"])
    if "obs_scale" in archived_payload:
        cfg.obs_scale = float(archived_payload["obs_scale"])
    cfg.state_encoding = state_encoding
    if "state_sight_range" in archived_payload:
        cfg.state_sight_range = float(archived_payload["state_sight_range"])
    if use_additional_input is not None:
        cfg.use_additional_input = use_additional_input
    # Unlike normal training, this script loads an archived config first
    # The held-out batch must be registered after that load so the evaluator does not accidentally use the training WAD
    configure_batch_env_and_agents(cfg)
    register_model_factory(cfg)
    return cfg


def joint_true_objective(infos: Any) -> float:
    """Benchmark score J of the episode that just ended: every scenario wrapper writes `true_objective` into the
    terminal infos, so its absence means the episode boundary or the wrapper stack is wrong."""
    if isinstance(infos, dict) and "true_objective" in infos:
        return float(infos["true_objective"])
    if isinstance(infos, (list, tuple)):
        values = [float(info["true_objective"]) for info in infos if isinstance(info, dict) and "true_objective" in info]
        if values:
            return float(np.mean(values))
    raise KeyError("terminal infos carry no 'true_objective'")


def evaluate_policy_on_wad(policy: PolicySpec, batch_dir: Path, entry: WadManifestEntry, episodes: int, device: str, deterministic: bool) -> list[dict[str, Any]]:
    with singleton_batch_dir(batch_dir, entry) as temp_batch:
        cfg = build_eval_cfg(policy, Path(temp_batch), device, deterministic)
        torch_device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
        env = make_eval_env(cfg, render_mode=None)
        try:
            env_info = extract_env_info(env, cfg)
            actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
            actor_critic.eval()
            actor_critic.model_to_device(torch_device)
            checkpoint_dict = Learner.load_checkpoint([policy.checkpoint], torch_device)
            if checkpoint_dict is None:
                raise RuntimeError(f"Could not load checkpoint: {policy.checkpoint}")
            actor_critic.load_state_dict(checkpoint_dict["model"])

            value_based = str(cfg.algo).upper() in VALUE_BASED_ALGOS
            head_sizes = action_head_sizes(env_info.action_space) if value_based else []
            track_goals = any(policy.scenario.startswith(prefix) for prefix in GOAL_TRACKED_SCENARIOS)
            goals = GoalRelocationCounter()

            obs, infos = env.reset()
            goals.observe(infos)
            action_mask = obs.pop("action_mask").to(torch_device) if "action_mask" in obs else None
            rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=torch_device)
            episode_records: list[dict[str, Any]] = []
            with torch.no_grad():
                while len(episode_records) < episodes:
                    normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
                    policy_outputs = actor_critic(
                        normalized_obs, rnn_states, action_mask=action_mask, sample_actions=not value_based
                    )
                    if value_based:
                        actions = greedy_q_actions(policy_outputs["action_logits"], head_sizes)
                    else:
                        actions = policy_outputs["actions"]
                        if cfg.eval_deterministic:
                            actions = reshape_deterministic_actions(
                                actions, argmax_actions(actor_critic.action_distribution())
                            )
                    if actions.ndim == 1:
                        actions = unsqueeze_tensor(actions, dim=-1)
                    actions = preprocess_actions(env_info, actions)
                    rnn_states = policy_outputs["new_rnn_states"]
                    obs, rewards, terminated, truncated, infos = env.step(actions)
                    action_mask = obs.pop("action_mask").to(torch_device) if "action_mask" in obs else None
                    dones = make_dones(terminated, truncated).cpu().numpy()
                    for agent_idx, done in enumerate(dones):
                        if done:
                            rnn_states[agent_idx].zero_()
                    if np.all(dones):
                        # terminal info comes from the last live frame; the returned obs already belong to the next episode
                        goals.observe(infos)
                        record = {
                            "scenario": policy.scenario,
                            "algorithm": policy.algorithm,
                            "train_seed": policy.train_seed,
                            "checkpoint": policy.checkpoint,
                            "wad_id": entry.wad_id,
                            "wad_seed": entry.seed,
                            "wad_sha256": entry.sha256,
                            "episode_index": len(episode_records),
                            "true_objective": joint_true_objective(infos),
                        }
                        if track_goals:
                            record["goals_reached"] = goals.count
                        episode_records.append(record)
                        goals.reset()
                        goals.observe(fresh_episode_infos(infos))
                    else:
                        goals.observe(infos)
            return episode_records
        finally:
            env.close()


def summarize_policy(records: list[dict[str, Any]], policy: PolicySpec) -> dict[str, Any]:
    scores = np.asarray([float(record["true_objective"]) for record in records], dtype=float)
    layout_scores: dict[str, list[float]] = {}
    for record in records:
        layout_scores.setdefault(str(record["wad_id"]), []).append(float(record["true_objective"]))
    layout_means = {wad_id: float(np.mean(values)) for wad_id, values in sorted(layout_scores.items())}
    layout_values = np.asarray(list(layout_means.values()), dtype=float)
    ci95 = ci95_half_width(layout_values)
    summary = {
        **asdict(policy),
        "episodes": int(len(records)),
        "mean_true_objective": float(np.mean(scores)),
        "std_true_objective": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "layout_means": layout_means,
        "layout_mean_ci95": ci95,
    }
    if records and all("goals_reached" in record for record in records):
        goals = np.asarray([float(record["goals_reached"]) for record in records], dtype=float)
        summary["mean_goals_reached"] = float(np.mean(goals))
        summary["std_goals_reached"] = float(np.std(goals, ddof=1)) if len(goals) > 1 else 0.0
    return summary


def render_table(summary: dict[str, Any]) -> str:
    policies = summary["policy_summaries"]
    scenarios = list(dict.fromkeys(item["scenario"] for item in policies))
    algorithms = list(dict.fromkeys(item["algorithm"] for item in policies))
    by_key = {(item["scenario"], item["algorithm"]): item for item in policies}
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\scriptsize",
        "  \\caption{Frozen-policy performance on five held-out DoomGen layouts. Entries are mean benchmark true objective over all evaluation episodes; the uncertainty is the 95\\% CI over the five layout means.}",
        "  \\label{tab:heldout_wad_evaluation}",
        "  \\begin{tabular}{l" + "c" * len(algorithms) + "}",
        "    \\toprule",
        "    Scenario & " + " & ".join(algorithms) + " \\\\",
        "    \\midrule",
    ]
    for scenario in scenarios:
        values = []
        for algorithm in algorithms:
            item = by_key[(scenario, algorithm)]
            values.append(f"{item['mean_true_objective']:.2f} $\\pm$ {item['layout_mean_ci95']:.2f}")
        lines.append(f"    {scenario.replace('_', ' ').title()} & " + " & ".join(values) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def write_raw_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError("Cannot write an empty raw-results CSV")
    fieldnames = list(dict.fromkeys(key for rec in records for key in rec.keys()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--batch-root", default="results/heldout_wad_batches")
    parser.add_argument("--output-dir", default="results/heldout_wad_evaluation")
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--algorithms", default=",".join(DEFAULT_ALGORITHMS))
    parser.add_argument("--seeds", default=",".join(map(str, HELDOUT_SEEDS)))
    parser.add_argument("--episodes-per-wad", type=int, default=20)
    parser.add_argument("--target-steps", type=int, default=DEFAULT_TARGET_STEPS)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--overwrite-batches", action="store_true")
    parser.add_argument(
        "--train-seeds",
        default="",
        help="Comma-separated training seeds for multi-seed train_all.py archives (e.g. 42,68,81,97,154); "
        "aggregate with experiments/very_difficult/very_difficult_reference_table.py",
    )
    parser.add_argument(
        "--experiment-tag",
        default="",
        help="Optional experiment tag prefix in train_all.py run names (e.g. full_obs, pixel_full, state_partial, state_full, blind).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_cli()
    scenarios = parse_csv_argument(args.scenarios)
    algorithms = parse_csv_argument(args.algorithms)
    seeds = tuple(int(value) for value in parse_csv_argument(args.seeds))
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("Held-out protocol requires exactly five distinct layout seeds")
    if int(args.episodes_per_wad) <= 0:
        raise ValueError("--episodes-per-wad must be positive")

    batch_root = Path(args.batch_root)
    output_dir = Path(args.output_dir)
    manifests: dict[str, list[WadManifestEntry]] = {}
    for scenario in scenarios:
        batch_dir = batch_root / scenario
        if args.overwrite_batches and batch_dir.exists():
            shutil.rmtree(batch_dir)
        if not (batch_dir / "batch_registry.json").is_file():
            manifests[scenario] = generate_heldout_batch(scenario, batch_dir, seeds)
        else:
            manifests[scenario] = load_manifest(batch_dir)
        actual_seeds = tuple(entry.seed for entry in manifests[scenario])
        if actual_seeds != seeds:
            raise RuntimeError(f"Batch {batch_dir} has seeds {actual_seeds}; expected {seeds}")

    manifest_metadata = {
        scenario: {
            "path": str((batch_root / scenario / "batch_registry.json").resolve()),
            "sha256": sha256_file(batch_root / scenario / "batch_registry.json"),
            "entries": [asdict(entry) for entry in entries],
        }
        for scenario, entries in manifests.items()
    }
    if args.generate_only:
        print(json.dumps({"heldout_seeds": seeds, "batches": manifest_metadata}, indent=2))
        raise SystemExit(0)

    archive_root = Path(args.archive_root)
    train_seeds = tuple(int(value) for value in parse_csv_argument(args.train_seeds)) if args.train_seeds else ()
    if train_seeds:
        policies = discover_seeded_policy_specs(
            archive_root,
            scenarios,
            algorithms,
            train_seeds,
            int(args.target_steps),
            experiment_tag=str(args.experiment_tag or ""),
        )
    else:
        policies = discover_policy_specs(archive_root, scenarios, algorithms, int(args.target_steps))
    register_vizdoom_components()
    raw_records: list[dict[str, Any]] = []
    policy_summaries: list[dict[str, Any]] = []
    for policy in policies:
        records: list[dict[str, Any]] = []
        for entry in manifests[policy.scenario]:
            records.extend(
                evaluate_policy_on_wad(
                    policy, batch_root / policy.scenario, entry, int(args.episodes_per_wad), args.device, bool(args.deterministic)
                )
            )
        raw_records.extend(records)
        policy_summaries.append(summarize_policy(records, policy))

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = output_dir / "raw_episode_scores.csv"
    summary_json = output_dir / "summary.json"
    table_tex = output_dir / "heldout_wad_results_table.tex"
    write_raw_csv(raw_csv, raw_records)
    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_root": str(archive_root.resolve()),
        "archive_root_path": str(archive_root),
        "target_checkpoint_steps": int(args.target_steps),
        "heldout_layout_seeds": list(seeds),
        "episodes_per_wad": int(args.episodes_per_wad),
        "episodes_per_policy": int(args.episodes_per_wad) * len(seeds),
        "eval_deterministic": bool(args.deterministic),
        "device": args.device,
        "batch_manifests": manifest_metadata,
        "policy_summaries": policy_summaries,
        "raw_episode_scores_csv": str(raw_csv.resolve()),
        "raw_episode_scores_sha256": sha256_file(raw_csv),
        "protocol_sha256": canonical_json_sha256(
            {
                "scenarios": scenarios,
                "algorithms": algorithms,
                "seeds": seeds,
                "episodes_per_wad": int(args.episodes_per_wad),
                "target_steps": int(args.target_steps),
                "deterministic": bool(args.deterministic),
            }
        ),
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if train_seeds:
        # one row per (scenario, algorithm) would hide the seed dimension; aggregate with very_difficult_reference_table.py
        summary["train_seeds"] = list(train_seeds)
        summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        table_tex.write_text("% multi-seed run: see experiments/very_difficult/very_difficult_reference_table.py\n", encoding="utf-8")
    else:
        table_tex.write_text(render_table(summary), encoding="utf-8")
    print(f"batch {batch_root}")
    print(f"score {raw_csv}")
    print(f"{summary_json}")
    print(f"{table_tex}")


if __name__ == "__main__":
    main()
