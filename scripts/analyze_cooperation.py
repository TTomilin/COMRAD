"""Runs a trained policy and computes cooperation metrics."""
import csv
import os
import numpy as np
from sample_factory.enjoy import enjoy_with_data
from comrad.train import parse_args, register_vizdoom_components, register_model_factory


class CooperationTracker:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.coop = [0.0] * num_agents
        self.defect = [0.0] * num_agents
        self.episodes = []

    def on_step(self, infos):
        for i in range(min(self.num_agents, len(infos))):
            self.coop[i] += float(infos[i].get("coop_step_signal", 0.0))
            self.defect[i] += float(infos[i].get("defect_step_signal", 0.0))

    def on_episode_end(self, record):
        if not any(self.coop) and not any(self.defect):
            return

        for i in range(self.num_agents):
            total = self.coop[i] + self.defect[i]
            record[f"agent{i}_coop_rate"] = self.coop[i] / max(total, 1.0)
            record[f"agent{i}_defect_rate"] = self.defect[i] / max(total, 1.0)
            record[f"agent{i}_coop_steps"] = self.coop[i]
            record[f"agent{i}_defect_steps"] = self.defect[i]
        self.episodes.append(record)
        self.coop = [0.0] * self.num_agents
        self.defect = [0.0] * self.num_agents


def main():
    register_vizdoom_components()
    cfg = parse_args(evaluation=True)

    if cfg.num_agents < 1:
        from comrad.utils.doom_utils import get_num_agents
        cfg.num_agents = get_num_agents(cfg, cfg.env)

    register_model_factory(cfg)

    tracker = CooperationTracker(cfg.num_agents)
    enjoy_with_data(
        cfg,
        step_callback=tracker.on_step,
        episode_callback=tracker.on_episode_end,
    )

    if not tracker.episodes:
        print("No episodes completed.")
        return

    summary = {}
    for col in ["cooperation_index", "defector_index",
                 "agent0_coop_rate", "agent0_defect_rate",
                 "agent1_coop_rate", "agent1_defect_rate"]:
        vals = [d[col] for d in tracker.episodes if col in d]
        if vals:
            summary[f"{col}_mean"] = float(np.mean(vals))
            summary[f"{col}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0

    print(f"\nCooperation: {cfg.env} ({len(tracker.episodes)} episodes)")
    for k, v in sorted(summary.items()):
        if k.endswith("_mean"):
            print(f"  {k}: {v:.4f}")

    csv_path = os.path.join(cfg.train_dir, cfg.experiment, "cooperation_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(tracker.episodes[0].keys()))
        writer.writeheader()
        writer.writerows(tracker.episodes)
    print(f"\nPer-episode results saved to: {csv_path}")

    return summary


if __name__ == "__main__":
    main()
