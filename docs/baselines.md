# Baselines

We trained seven MARL algorithms on every scenario with the bundled [Sample Factory trainer](training/sample-factory.md): IPPO, MAPPO, HAPPO, IDQN, VDN, QMIX and QPLEX. Scores are each scenario's `true_objective`. See the paper for the exact protocol.

![Baseline results heatmap](assets/figures/baseline_results_heatmap.png)

## Learning curves

![Baseline learning curves](assets/figures/baseline_learning_curves.png)

## Scaling the team

Armory Siege with 2 to 8 agents (QMIX):

![Agent scaling: learning](assets/figures/agent_scaling_learning.png)

![Agent scaling: throughput](assets/figures/agent_scaling_throughput.png)

To rerun these experiments, see [Reproducing the paper](training/reproducing.md).
