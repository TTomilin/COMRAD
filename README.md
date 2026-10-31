<!-- <p align="center">
  <img src="https://raw.githubusercontent.com/TTomilin/COMRAD/main/docs/assets/comrad_logo.svg" alt="COMRAD" width="440">
</p> -->

<h1 align="center"><b>COMRAD: Cooperative multi-agent reinforcement learning from pixels.</b></h1>

<p align="center">
  <a href="https://ttomilin.github.io/COMRAD/"><img src="https://img.shields.io/badge/%F0%9F%93%96%20Docs-f2801a?style=for-the-badge" alt="Docs" height="39"></a> &nbsp;
  <a href="https://lmbored.github.io/comrad_site/"><img src="https://img.shields.io/badge/%F0%9F%8E%AE%20Project%20Page-6b74ff?style=for-the-badge" alt="Project page" height="39"></a>
</p>

<p align="center">
  <a href="https://pypi.org/project/comrad-bench/"><img src="https://img.shields.io/pypi/v/comrad-bench" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.11%20%7C%203.12-blue" alt="Python 3.11 | 3.12">
  <a href="https://github.com/TTomilin/COMRAD/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/TTomilin/COMRAD/main/docs/assets/comrad_demo.gif" alt="COMRAD agents playing" width="720">
</p>

COMRAD is a benchmark of **13 first-person cooperative scenarios** built on [ViZDoom](https://github.com/Farama-Foundation/ViZDoom). Each one isolates a coordination skill: committing to a joint plan, synchronising in time, sharing a commons, swapping roles, or communicating without words. Agents see only their own first-person view.

## Scenarios

> "Cooperation is the thorough conviction that nobody can get there unless everybody gets there." - Virginia Burden

<table>
  <tr>
    <td align="center"><a href="https://ttomilin.github.io/COMRAD/scenarios/stag_hunt_arena/"><img src="https://raw.githubusercontent.com/TTomilin/COMRAD/main/docs/assets/scenarios/stag_hunt_arena.jpg" width="240"><br><b>Stag Hunt Arena</b></a><br>2–4 players</td>
    <td align="center"><a href="https://ttomilin.github.io/COMRAD/scenarios/rhythm_sync/"><img src="https://raw.githubusercontent.com/TTomilin/COMRAD/main/docs/assets/scenarios/rhythm_sync.jpg" width="240"><br><b>Rhythm Sync</b></a><br>2 players</td>
    <td align="center"><a href="https://ttomilin.github.io/COMRAD/scenarios/foraging_commons/"><img src="https://raw.githubusercontent.com/TTomilin/COMRAD/main/docs/assets/scenarios/foraging_commons.jpg" width="240"><br><b>Foraging Commons</b></a><br>2–4 players</td>
    <td align="center"><a href="https://ttomilin.github.io/COMRAD/scenarios/coop_puzzle/"><img src="https://raw.githubusercontent.com/TTomilin/COMRAD/main/docs/assets/scenarios/coop_puzzle.jpg" width="240"><br><b>Co-op Puzzle</b></a><br>2 players</td>
  </tr>
  <tr>
    <td align="center"><a href="https://ttomilin.github.io/COMRAD/scenarios/platform_chain/"><img src="https://raw.githubusercontent.com/TTomilin/COMRAD/main/docs/assets/scenarios/platform_chain.jpg" width="240"><br><b>Platform Chain</b></a><br>2 players</td>
    <td align="center"><a href="https://ttomilin.github.io/COMRAD/scenarios/armory_siege/"><img src="https://raw.githubusercontent.com/TTomilin/COMRAD/main/docs/assets/scenarios/armory_siege.jpg" width="240"><br><b>Armory Siege</b></a><br>2–8 players</td>
    <td align="center"><a href="https://ttomilin.github.io/COMRAD/scenarios/coop_health_gathering/"><img src="https://raw.githubusercontent.com/TTomilin/COMRAD/main/docs/assets/scenarios/coop_health_gathering.jpg" width="240"><br><b>Co-op Health Gathering</b></a><br>2–8 players</td>
    <td align="center"><a href="https://ttomilin.github.io/COMRAD/scenarios/lavapit/"><img src="https://raw.githubusercontent.com/TTomilin/COMRAD/main/docs/assets/scenarios/lavapit.jpg" width="240"><br><b>Lava Pit</b></a><br>2 players</td>
  </tr>
  <tr>
    <td align="center"><a href="https://ttomilin.github.io/COMRAD/scenarios/smart_enemies/"><img src="https://raw.githubusercontent.com/TTomilin/COMRAD/main/docs/assets/scenarios/smart_dumb_enemies.jpg" width="240"><br><b>Smart / Dumb Enemies</b></a><br>2–4 players</td>
    <td align="center"><a href="https://ttomilin.github.io/COMRAD/scenarios/stealth_labyrinth/"><img src="https://raw.githubusercontent.com/TTomilin/COMRAD/main/docs/assets/scenarios/stealth_labyrinth.jpg" width="240"><br><b>Stealth Labyrinth</b></a><br>2 players</td>
    <td align="center"><a href="https://ttomilin.github.io/COMRAD/scenarios/ammo_carrier/"><img src="https://raw.githubusercontent.com/TTomilin/COMRAD/main/docs/assets/scenarios/ammo_carrier.jpg" width="240"><br><b>Ammo Carrier</b></a><br>2 players</td>
    <td align="center"><a href="https://ttomilin.github.io/COMRAD/scenarios/lava_maze/"><img src="https://raw.githubusercontent.com/TTomilin/COMRAD/main/docs/assets/scenarios/lava_maze.jpg" width="240"><br><b>Lava Maze</b></a><br>2 players</td>
  </tr>
</table>

## Install

```bash
pip install comrad-bench
```

Requires Python 3.11 or 3.12 on Linux or macOS. The ViZDoom engine comes with the `vizdoom` wheel.

## Quickstart

COMRAD scenarios are standard [PettingZoo](https://pettingzoo.farama.org/) and [Gymnasium](https://gymnasium.farama.org/) environments, so you can train them with any MARL library.

```python
import comrad

env = comrad.parallel_env("stag_hunt_arena")
observations, infos = env.reset(seed=42)

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)

print("score:", {agent: info.get("true_objective") for agent, info in infos.items()})
env.close()
```

Gymnasium, with per-agent lists:

```python
import gymnasium as gym
import comrad  # registers comrad/<scenario>-v0

env = gym.make("comrad/stag_hunt_arena-v0", num_agents=2)
observations, infos = env.reset(seed=42)
```

See the [quickstart](https://ttomilin.github.io/COMRAD/getting-started/quickstart/) and [API reference](https://ttomilin.github.io/COMRAD/api/) for options such as `num_agents`, `shared_reward` and `render_mode`.

## Baselines

<p align="center">
  <img src="https://raw.githubusercontent.com/TTomilin/COMRAD/main/docs/assets/figures/baseline_results_heatmap.png" alt="Baseline results" width="720">
</p>

The heatmap shows seven MARL algorithms on every scenario. [More results →](https://ttomilin.github.io/COMRAD/baselines/)

## Train with the bundled algorithms

The package includes a modified [Sample Factory](https://github.com/alex-petrenko/sample-factory) with **IPPO, MAPPO, HAPPO, IDQN, VDN, QMIX and QPLEX**:

```bash
python -m comrad.train --env=stag_hunt_arena --algo=MAPPO --train_for_env_steps=10000000
```

See the [per-algorithm commands](https://ttomilin.github.io/COMRAD/training/sample-factory/) and [how to reproduce the paper](https://ttomilin.github.io/COMRAD/training/reproducing/).

## Documentation

**https://ttomilin.github.io/COMRAD/**: scenarios, API, integrations, training, and contributing.

**https://lmbored.github.io/comrad_site/**: Project page. Visit for a playable demo, scoreboard, and paper overview.

## Citation

```bibtex
@misc{comrad,
  title  = {COMRAD: A Benchmark for Embodied Multi-Agent Reinforcement Learning},
  author = {Anonymized Authors},
  year   = {2026},
  url    = {https://github.com/TTomilin/COMRAD}
}
```

## License and acknowledgements

COMRAD is released under the [MIT license](https://github.com/TTomilin/COMRAD/blob/main/LICENSE). It builds on [ViZDoom](https://github.com/Farama-Foundation/ViZDoom) and [Sample Factory](https://github.com/alex-petrenko/sample-factory). HAPPO follows [HARL](https://github.com/PKU-MARL/HARL), IDQN/VDN/QMIX follow [PyMARL](https://github.com/oxwhirl/pymarl), and QPLEX follows the [official implementation](https://github.com/wjh720/QPLEX). Experiments were tracked with [Weights & Biases](https://wandb.ai/) and run on the TU/e Supercomputing Center. We thank everyone who contributed to this work.
