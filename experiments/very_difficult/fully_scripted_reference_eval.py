"""
Fully scripted upper references for the Very Difficult tier (App. very_difficult), on the same five held-out layouts
and episode count as the learned-role evaluation (scripts/heldout_wad_evaluation.py):

* stealth_labyrinth: ACS scout (player 0) + ACS gunner (player 1). Held-out WADs are generated with
  {"scripted_scout": True, "scripted_gunner": True} (same seeds/layouts as the scripted-gunner batch; only scripts differ)
  and run in the `stealth_labyrinth_scripted_gunner` env, whose single learner seat (the scout) sends no-ops; the scout
  is frozen and moved by ACS, so the learner seat has no influence.
* lava_maze: Python BFS observer (comrad.scripted.lava_maze_observer) in the learner seat of
  `lava_maze_scripted_navigator` (Python colour-convention navigator), on the scripted-navigator held-out WADs.

Run from the repo root:
  python -m experiments.very_difficult.fully_scripted_reference_eval --scenario lava_maze --wad-index 0
  python -m experiments.very_difficult.fully_scripted_reference_eval --summarize
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

from comrad.envs.doom_params import default_doom_cfg
from comrad.scripted.lava_maze_observer import LavaMazeObserverPolicy
from comrad.train import configure_batch_env_and_agents
from comrad.utils.confidence import ci95_half_width
from sample_factory.envs.create_env import create_env
from sample_factory.utils.attr_dict import AttrDict
from scripts.heldout_wad_evaluation import (
    HELDOUT_SEEDS,
    GoalRelocationCounter,
    generate_heldout_batch,
    load_manifest,
    singleton_batch_dir,
)

BATCH_ROOT = REPO_ROOT / "results" / "heldout_wad_batches"
OUT_ROOT = REPO_ROOT / "results" / "heldout_very_difficult" / "fully_scripted"
SETUPS = {
    # scenario -> (held-out batch name, env the WADs are run in)
    "stealth_labyrinth": ("stealth_labyrinth_scripted_both", "stealth_labyrinth_scripted_gunner"),
    "lava_maze": ("lava_maze_scripted_navigator", "lava_maze_scripted_navigator"),
}


def ensure_batch(scenario: str) -> Path:
    batch_name, _ = SETUPS[scenario]
    batch_dir = BATCH_ROOT / batch_name
    if not (batch_dir / "batch_registry.json").exists():
        if scenario != "stealth_labyrinth":
            raise FileNotFoundError(f"Missing held-out batch {batch_dir}")
        generate_heldout_batch(batch_name, batch_dir, HELDOUT_SEEDS)
    return batch_dir


def make_env(env_name: str, temp_batch: str):
    cfg = default_doom_cfg(env=env_name)
    cfg.algo = "IPPO"
    cfg.num_agents = -1
    cfg.num_bots = 0
    cfg.num_humans = 0
    cfg.env_frameskip = 4
    cfg.wandb_record_every = 0
    cfg.wad_batch = temp_batch
    cfg.curriculum = "uniform"
    configure_batch_env_and_agents(cfg)
    return create_env(cfg.env, cfg=cfg, env_config=AttrDict({"worker_index": 0, "vector_index": 0, "env_id": 0}))


def run_wad(scenario: str, wad_index: int, episodes: int) -> list[dict]:
    batch_dir = ensure_batch(scenario)
    entry = load_manifest(batch_dir)[wad_index]
    _, env_name = SETUPS[scenario]
    records: list[dict] = []
    with singleton_batch_dir(batch_dir, entry) as temp_batch:
        env = make_env(env_name, temp_batch)
        try:
            assert env.num_agents == 1, env.num_agents
            observer = LavaMazeObserverPolicy() if scenario == "lava_maze" else None
            goals = GoalRelocationCounter()
            noop = np.zeros(len(env.action_space.spaces), dtype=np.int64)
            obs, infos = env.reset()
            info = infos[0]
            goals.observe(infos)
            steps = 0
            while len(records) < episodes:
                action = np.array(observer.act(obs[0], info)) if observer is not None else noop
                obs, rew, term, trunc, infos = env.step([action])
                info = infos[0]
                steps += 1
                if term[0] or trunc[0]:
                    goals.observe(infos)
                    record = {
                        "scenario": scenario,
                        "wad_id": entry.wad_id,
                        "wad_seed": entry.seed,
                        "wad_sha256": entry.sha256,
                        "episode_index": len(records),
                        "episode_steps": steps,
                        "true_objective": float(info.get("true_objective", 0.0) or 0.0),
                    }
                    if scenario == "lava_maze":
                        record["goals_reached"] = goals.count
                    else:
                        # end-of-episode telemetry: scout/gunner alive, turrets left, team HP, rooms seen
                        for key, name in (("USER41", "scout_alive"), ("USER42", "gunner_alive"),
                                          ("USER44", "enemies_left"), ("USER50", "team_hp"), ("USER48", "rooms_seen")):
                            record[name] = info.get(key)
                        record["truncated"] = bool(trunc[0])
                    records.append(record)
                    print(json.dumps(record), flush=True)
                    goals.reset()
                    steps = 0
                    if observer is not None:
                        observer.reset()
                    info = info.get("reset_info", info)
                else:
                    goals.observe(infos)
        finally:
            env.close()
    return records


def summarize() -> dict:
    summary = {}
    for scenario in SETUPS:
        records = []
        for path in sorted((OUT_ROOT / scenario).glob("wad_*.json")):
            records.extend(json.loads(path.read_text()))
        if not records:
            continue
        key = "goals_reached" if scenario == "lava_maze" else "true_objective"
        by_wad: dict[str, list[float]] = {}
        for r in records:
            by_wad.setdefault(r["wad_id"], []).append(float(r[key]))
        layout_means = {k: float(np.mean(v)) for k, v in sorted(by_wad.items())}
        vals = np.asarray(list(layout_means.values()))
        summary[scenario] = {
            "metric": key,
            "episodes": len(records),
            "layouts": len(layout_means),
            "mean": float(np.mean([float(r[key]) for r in records])),
            "layout_mean_ci95": ci95_half_width(vals),
            "layout_means": layout_means,
            "mean_true_objective": float(np.mean([r["true_objective"] for r in records])),
        }
    out = REPO_ROOT / "results" / "very_difficult_fully_scripted_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=sorted(SETUPS))
    p.add_argument("--wad-index", type=int, default=0)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--generate-only", action="store_true")
    p.add_argument("--summarize", action="store_true")
    args = p.parse_args()
    if args.summarize:
        summarize()
        return
    if args.generate_only:
        print(ensure_batch(args.scenario))
        return
    records = run_wad(args.scenario, args.wad_index, args.episodes)
    out_dir = OUT_ROOT / args.scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"wad_{args.wad_index}.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
