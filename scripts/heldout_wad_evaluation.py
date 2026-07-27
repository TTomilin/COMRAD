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
DEFAULT_ARCHIVE_ROOT = "/home/knguyen2/ViZDoom/train_dir/mode3_comrad_benchmark_20260503_134649"
DEFAULT_SCENARIOS = ("ammo_carrier", "armory_siege", "coop_puzzle")
DEFAULT_ALGORITHMS = ("IPPO", "MAPPO", "HAPPO", "QMIX", "QPLEX-D")
DEFAULT_TARGET_STEPS = 100_000_000

POLICY_DIRECTORY_TOKENS = {
    "IPPO": "_IPPO_",
    "MAPPO": "_MAPPO_",
    "HAPPO": "_HAPPO_",
    "QMIX": "_QMIX_",
    "QPLEX-D": "_QPLEX_dmaq_",
}
SCENARIO_CLASSES = {
    "ammo_carrier": ("examples.benchmark.ammo_carrier", "AmmoCarrierScenario"),
    "armory_siege": ("examples.benchmark.armory_siege", "ArmorySiegeScenario"),
    "coop_puzzle": ("examples.benchmark.coop_puzzle", "CoopPuzzleScenario"),
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


def generate_heldout_batch(scenario: str, output_dir: Path, seeds: Iterable[int]) -> list[WadManifestEntry]:
    """Generate only layout changes: each WAD uses the scenario's default config plus a new seed."""
    if scenario not in SCENARIO_CLASSES:
        raise ValueError(f"Unsupported scenario {scenario!r}; choices: {sorted(SCENARIO_CLASSES)}")
    module_name, class_name = SCENARIO_CLASSES[scenario]
    module = __import__(module_name, fromlist=[class_name])
    scenario_cls = getattr(module, class_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries: list[WadManifestEntry] = []
    for index, seed in enumerate(seeds):
        wad_id = f"{scenario}_heldout_seed_{seed}"
        filename = f"{wad_id}.wad"
        wad_path = output_dir / filename
        generated = scenario_cls({"seed": int(seed)}, name=wad_id)
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

    manifest_path = output_dir / "batch_registry.json"
    manifest_payload = [asdict(entry) for entry in entries]
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return entries


def load_manifest(batch_dir: Path) -> list[WadManifestEntry]:
    manifest_path = batch_dir / "batch_registry.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [WadManifestEntry(**item) for item in payload]
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


def build_eval_cfg(policy: PolicySpec, wad_batch: Path, device: str, deterministic: bool) -> Any:
    run_dir = Path(policy.run_dir)
    argv = [
        "--algo=APPO",  # overwritten by the archived config below
        f"--train_dir={run_dir.parent}",
        f"--experiment={run_dir.name}",
        f"--env={policy.scenario}",
        f"--device={device}",
        "--with_wandb=False",
        "--wandb_record_every=0",
        "--no_render",
    ]
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
    # Unlike normal training, this script loads an archived config first
    # The held-out batch must be registered after that load so the evaluator does not accidentally use the training WAD
    configure_batch_env_and_agents(cfg)
    register_model_factory(cfg)
    return cfg


def joint_true_objective(infos: Any, rewards: torch.Tensor) -> float:
    if isinstance(infos, dict) and "true_objective" in infos:
        return float(infos["true_objective"])
    if isinstance(infos, (list, tuple)):
        values = [float(info["true_objective"]) for info in infos if isinstance(info, dict) and "true_objective" in info]
        if values:
            return float(np.mean(values))
    return float(torch.mean(rewards).item())


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

            obs, _infos = env.reset()
            action_mask = obs.pop("action_mask").to(torch_device) if "action_mask" in obs else None
            rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=torch_device)
            episode_records: list[dict[str, Any]] = []
            with torch.no_grad():
                while len(episode_records) < episodes:
                    normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
                    policy_outputs = actor_critic(normalized_obs, rnn_states, action_mask=action_mask)
                    actions = policy_outputs["actions"]
                    if cfg.eval_deterministic:
                        actions = reshape_deterministic_actions(actions, argmax_actions(actor_critic.action_distribution()))
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
                        episode_records.append(
                            {
                                "scenario": policy.scenario,
                                "algorithm": policy.algorithm,
                                "checkpoint": policy.checkpoint,
                                "wad_id": entry.wad_id,
                                "wad_seed": entry.seed,
                                "wad_sha256": entry.sha256,
                                "episode_index": len(episode_records),
                                "true_objective": joint_true_objective(infos, rewards),
                            }
                        )
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
    ci95 = float(1.96 * np.std(layout_values, ddof=1) / np.sqrt(len(layout_values))) if len(layout_values) > 1 else 0.0
    return {
        **asdict(policy),
        "episodes": int(len(records)),
        "mean_true_objective": float(np.mean(scores)),
        "std_true_objective": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "layout_means": layout_means,
        "layout_mean_ci95": ci95,
    }


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
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
    return parser.parse_args()


if __name__ == "__main__":
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
        return

    archive_root = Path(args.archive_root)
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
    table_tex.write_text(render_table(summary), encoding="utf-8")
    print(f"batch {batch_root}")
    print(f"score {raw_csv}")
    print(f"{summary_json}")
    print(f"{table_tex}")
