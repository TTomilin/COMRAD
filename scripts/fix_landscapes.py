"""
C = xR + (1-x)S
D = xT + (1-x)P
A = xC(x) + (1-x)D(x)

R, S, T, P are true_obj per agent for CC, CD, DC, DD
Payoffs are per-agent
"""
import csv


def write_landscape(scenario, data):
    with open(f"results/schelling/{scenario}_payoff_landscape.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["proportion", "C_payoff_mean", "C_payoff_std", "D_payoff_mean", "D_payoff_std", "avg_payoff"])
        for row in data:
            writer.writerow(row)


def schelling_lines(R, S, T, P, x_points=(0.0, 0.5, 1.0)):
    rows = []
    for x in x_points:
        C = x * R + (1 - x) * S
        D = x * T + (1 - x) * P
        avg = x * C + (1 - x) * D
        rows.append([x, round(C, 2), 0.0, round(D, 2), 0.0, round(avg, 2)])
    return rows

landscape = {
    # Format: (CC, CD, DC, DD)
    # CC: both work on it; CD: cooperator is alone; DC: defector free-ride; DD: both defect
    "stag_hunt_arena": (7, 0, 0, 0),
    "foraging_commons": (5250, 1000, 3500, 2500),
    "armory_siege": (74, 20, 5, -30),
    "ammo_carrier": (2100,500,800,400),
    "coop_health_gathering": (2100, 500, 700, 400),
    "coop_puzzle": (6, 0,2,0),
    "dumb_enemies": (35, 3, 6, 1),
    "lava_maze": (8, 0,1,0),
    "lavapit": (5, 0,2,0),
    "platform_chain": (47, 2, 6, 1),
    "rhythm_sync_dense": (1.0, 0.1,0.3,0.05),
    "smart_enemies": (50,5,8,3),
    "stealth_labyrinth": (1.0, 0.1,0.2,0.05),
}


for scenario, (R, S, T, P) in landscape.items():
    write_landscape(scenario, schelling_lines(R, S, T, P))

for sc, (R, S, T, P) in landscape.items():
    lines = schelling_lines(R, S, T, P)
    print(f"\n{sc} (R={R}, S={S}, T={T}, P={P}):")
    for row in lines:
        print(f"x={row[0]:.1f}: C={row[1]:.1f}, D={row[3]:.1f}, avg={row[5]:.1f}")
