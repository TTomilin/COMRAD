import csv
import os
import numpy as np

outt = "results/schelling"

scenarios = [
    "ammo_carrier", "armory_siege", "coop_health_gathering", "coop_puzzle",
    "dumb_enemies", "foraging_commons", "lava_maze", "lavapit",
    "platform_chain", "rhythm_sync_dense", "smart_enemies",
    "stag_hunt_arena", "stealth_labyrinth",
]
algos = ["IPPO", "MAPPO", "HAPPO"]

rows = []
# For each (scenario, algo) pair:
# x = mean of per agent coop rates
# y = true_objective per agent
# y_reward = per agent reward, per row
for scenario in scenarios:
    for algo in algos:
        path = os.path.join(outt, f"{scenario}_{algo}_cooperation.csv")
        if not os.path.exists(path):
            continue
        x_vals = []
        y_vals = []
        r_vals = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                agent = int(row.get("agent", 0))
                a0c = row.get(f"agent{agent}_coop_rate", "0").strip()
                if a0c:
                    x_vals.append(float(a0c))
                y_vals.append(float(row["true_objective"]))
                r_vals.append(float(row["reward"]))
        x_mean = np.mean(x_vals)
        x_std = np.std(x_vals)
        y_mean = np.mean(y_vals)
        y_std = np.std(y_vals)
        r_mean = np.mean(r_vals)
        r_std = np.std(r_vals)
        rows.append({
            "scenario": scenario,
            "algo": algo,
            "coop_rate_mean": f"{x_mean:.4f}",
            "coop_rate_std": f"{x_std:.4f}",
            "true_obj_mean": f"{y_mean:.4f}",
            "true_obj_std": f"{y_std:.4f}",
            "reward_mean": f"{r_mean:.4f}",
            "reward_std": f"{r_std:.4f}",
        })

out_path = os.path.join(outt, "algo_markers.csv")
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["scenario", "algo", "coop_rate_mean", "coop_rate_std", "true_obj_mean", "true_obj_std", "reward_mean", "reward_std"])
    writer.writeheader()
    writer.writerows(rows)

for r in rows:
    print(f"{r['scenario']} / {r['algo']}: x={r['coop_rate_mean']}±{r['coop_rate_std']}, "
          f"true_obj={r['true_obj_mean']}±{r['true_obj_std']}, "
          f"reward={r['reward_mean']}±{r['reward_std']}")
