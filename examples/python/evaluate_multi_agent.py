import os
import sys
import pickle
from dataclasses import MISSING
from pathlib import Path
from benchmarl.experiment import Experiment
import torch
from examples.python.train_multi_agent_extensive import VizdoomTask

def checkpoint(path: Path, out: Path) -> Path:
    state = torch.load(path, map_location="cpu")
    true = False

    for k, v in list(state.items()):
        if not k.startswith("loss"):
            continue
        if not isinstance(v, (dict, torch.nn.ModuleDict)):
            continue
        loss_dict = v
        if "entropy_coeff" in loss_dict and "entropy_coef" not in loss_dict:
            loss_dict["entropy_coef"] = loss_dict.pop("entropy_coeff")
            true = True
        if "critic_coeff" in loss_dict and "critic_coef" not in loss_dict:
            loss_dict["critic_coef"] = loss_dict.pop("critic_coeff")
            true = True

    if not true:
        return path

    p = out / f"{path.stem}_eval.pt"
    torch.save(state, p)
    return p


def load_exp(p: Path) -> Experiment:
    pp = p.parent.parent.resolve()
    cnf = pp / "config.pkl"

    with open(cnf, "rb") as f:
        task = pickle.load(f)
        task_config = pickle.load(f)
        algorithm_config = pickle.load(f)
        model_config = pickle.load(f)
        seed = pickle.load(f)
        experiment_config = pickle.load(f)
        critic_model_config = pickle.load(f)
        callbacks = pickle.load(f)

    task.config = task_config
    task.config["render_mode"] = "rgb_array"
    task.config["enable_video"] = False # To prevent wandb.log() to run, setting to True is also fine but will get some warning

    save_folder = Path.cwd() / "checkpoints/eval_runs"
    save_folder.mkdir(parents=True, exist_ok=True)
    new_cp = checkpoint(p, save_folder)

    # Ref: .venv/lib/python3.12/site-packages/benchmarl/conf/experiment/base_experiment.yaml
    experiment_config.restore_file = str(new_cp)
    experiment_config.loggers = ["csv"] # or wandb
    experiment_config.save_folder = str(save_folder)
    experiment_config.evaluation_episodes = 3 # To average the episodes

    return Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=seed,
        config=experiment_config,
        callbacks=callbacks,
    )


def main():
    path = Path("checkpoints/mappo_doom_cnn__5fa405e9_25_10_15-16_50_27/checkpoints/checkpoint_1536.pt").resolve()
    experiment = load_exp(path)

    try:
        print(f"Prev mean return: {experiment.mean_return}")
        runs = int(os.environ.get("RUN", "3"))
        for i in range(runs):
            print(f"\nRun {i+1}/{runs}")
            experiment.evaluate()
    finally:
        experiment.close()


if __name__ == "__main__":
    main()