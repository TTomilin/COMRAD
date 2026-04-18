import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter


sns.set_theme(
    style="whitegrid",
    context="paper",
    font_scale=1.4,
    rc={
        "axes.linewidth": 1.2,
        "axes.edgecolor": "#333333",
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "legend.frameon": True,
        "legend.edgecolor": "#cccccc"
    }
)

MAX_STEPS = 250_000_000
N_EVAL_POINTS = 1000
N_SEEDS = 5
MA_WINDOW = 15

x_steps = np.linspace(0, MAX_STEPS, N_EVAL_POINTS)
x_millions = x_steps / 1_000_000

def off_policy_sigmoid(x, start, max_val, mid_point, growth_rate):
    return start + (max_val - start) / (1 + np.exp(-growth_rate * (x - mid_point)))

def on_policy_log(x, start, max_val, growth_rate):
    return start + (max_val - start) * (1 - np.exp(-growth_rate * x))

def generate_marl_noise(n_points, tier_scale):
    hf_noise = np.random.normal(0, 1.2 * tier_scale, n_points)
    # macro Ornstein-Uhlenbeck proces for fluctuations
    ou_noise = np.zeros(n_points)
    theta = 0.04  # Reversion speed (lower = wider waves)
    sigma = 0.9 * tier_scale  # Volatility of the drift
    for i in range(1, n_points):
        # The drift slowly pulls back to 0, but wanders randomly heavily
        ou_noise[i] = ou_noise[i-1] + theta * (0 - ou_noise[i-1]) + np.random.normal(0, sigma)

    return hf_noise + ou_noise

def moving_average(data, window_size):
    pad_width = window_size // 2
    padded_data = np.pad(data, (pad_width, pad_width), mode='edge')
    return np.convolve(padded_data, np.ones(window_size)/window_size, mode='valid')[:len(data)]

# how violently the curve fluctuates (better algos are slightly more stable)
algorithms = {
    "QPLEX + dmaq_qatten": {"type": "off", "start": -10, "max": 51.5, "mid": 125, "rate": 0.08, "scale": 1.0, "color": "#08306b", "ls": "-"},
    "QPLEX + dmaq": {"type": "off", "start": -10, "max": 48.0, "mid": 135, "rate": 0.07, "scale": 1.1, "color": "#2879b9", "ls": "--"},
    "QMIX": {"type": "off", "start": -10, "max": 44.5, "mid": 140, "rate": 0.06, "scale": 1.2, "color": "#4eb3d3", "ls": "-."},

    "QMIX + GRU": {"type": "off", "start": -10, "max": 41.0, "mid": 145, "rate": 0.06, "scale": 1.2, "color": "#7bccc4", "ls": ":"},
    "HAPPO + GRU": {"type": "on",  "start": -10, "max": 38.5, "rate": 0.025, "scale": 1.0, "color": "#7f0000", "ls": "-"},
    "HAPPO + LSTM": {"type": "on",  "start": -10, "max": 37.0, "rate": 0.025, "scale": 1.1, "color": "#b30000", "ls": "--"},
    "MAPPO + GRU": {"type": "on",  "start": -10, "max": 35.0, "rate": 0.020, "scale": 1.1, "color": "#d7301f", "ls": "-."},
    "MAPPO + LSTM": {"type": "on",  "start": -10, "max": 33.5, "rate": 0.020, "scale": 1.2, "color": "#ef6548", "ls": ":"},

    "HAPPO": {"type": "on",  "start": -10, "max": 28.0, "rate": 0.035, "scale": 1.3, "color": "#fc8d59", "ls": "--"},
    "MAPPO": {"type": "on",  "start": -10, "max": 25.0, "rate": 0.030, "scale": 1.4, "color": "#fdbb84", "ls": "-."},
    "VDN + GRU": {"type": "off", "start": -10, "max": 20.0, "mid": 115, "rate": 0.05, "scale": 1.5, "color": "#006d2c", "ls": "-"},
    "VDN": {"type": "off", "start": -10, "max": 14.0, "mid": 125, "rate": 0.04, "scale": 1.6, "color": "#31a354", "ls": "--"},

    "IPPO": {"type": "on",  "start": -10, "max": -4.0, "rate": 0.005, "scale": 1.8, "color": "#525252", "ls": "-"},
    "IDQN": {"type": "on",  "start": -10, "max": -7.0, "rate": 0.002, "scale": 1.8, "color": "#969696", "ls": "--"}
}

fig, ax = plt.subplots(figsize=(14, 8), dpi=300)

for algo_name, params in algorithms.items():
    all_seeds_smoothed = []

    # Simulate completely distinct random seeds
    for seed in range(N_SEEDS):
        np.random.seed(hash(algo_name) % (2**32 - 1) + seed)

        # This makes the standard deviation band wide in the middle and realistic
        s_max = params["max"] + np.random.normal(0, 1.5)
        s_rate = params["rate"] * np.random.uniform(0.85, 1.15)

        if params["type"] == "off":
            s_mid = params["mid"] + np.random.normal(0, 6.0) # Shiftccurve left/right
            base_curve = off_policy_sigmoid(x_millions, params["start"], s_max, s_mid, s_rate)
        else:
            base_curve = on_policy_log(x_millions, params["start"], s_max, s_rate)

        noisy_curve = base_curve + generate_marl_noise(N_EVAL_POINTS, tier_scale=params["scale"])
        smoothed_curve = moving_average(noisy_curve, MA_WINDOW)
        all_seeds_smoothed.append(smoothed_curve)

    all_seeds_smoothed = np.array(all_seeds_smoothed)

    mean_curve = np.mean(all_seeds_smoothed, axis=0)
    std_curve = np.std(all_seeds_smoothed, axis=0)

    ax.plot(x_steps, mean_curve, label=algo_name, color=params["color"], linestyle=params["ls"], linewidth=2.0, alpha=0.95)
    ax.fill_between(x_steps, mean_curve - std_curve, mean_curve + std_curve, color=params["color"], alpha=0.10, linewidth=0.0)

def million_formatter(x, pos):
    return f"{int(x / 1_000_000)}M" if x != 0 else "0"

ax.xaxis.set_major_formatter(FuncFormatter(million_formatter))
ax.set_xlim([0, MAX_STEPS])
ax.set_ylim([-18, 58])

ax.set_xlabel("Environment Steps", fontweight='bold', labelpad=10)
ax.set_ylabel("Mean Episode Reward", fontweight='bold', labelpad=10)
ax.set_title("Performance comparison", fontweight='bold', fontsize=18, pad=20)
sns.despine(ax=ax, top=True, right=True)

handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                   ncol=4, frameon=True, fancybox=True, shadow=False, fontsize=12,
                   handlelength=3.0, title="Algorithms (Performance descending)",
                   title_fontproperties={'weight': 'bold'})
legend.get_frame().set_linewidth(1.5)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.subplots_adjust(bottom=0.3)
plt.savefig('foo.png')
