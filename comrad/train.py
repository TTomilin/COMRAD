import functools
import os
import sys
import datetime
import wandb
from typing import Optional

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import make_runner

from comrad.models.doom_model import make_vizdoom_encoder
from comrad.envs.doom_params import add_doom_env_args, add_doom_env_eval_args, doom_override_defaults, add_wandb_args
from comrad.utils.doom_utils import (
    DOOM_ENVS,
    DoomBatchSpec,
    make_doom_env_from_spec,
    make_doom_env_from_batch,
    doom_env_by_name,
    get_num_agents,
)
from comrad.utils.video_uploader import upload_video
from comrad.models.mappo_model import make_mappo_actor_critic
from comrad.models.qmix_model import make_qmix_actor_critic


def register_vizdoom_envs():
    for env_spec in DOOM_ENVS:
        make_env_func = functools.partial(make_doom_env_from_spec, env_spec)
        register_env(env_spec.name, make_env_func)


def register_batch_env(cfg) -> str:
    from comrad.utils.curriculum import BatchCurriculum
    from comrad.envs.wad_catalog import WadBatch

    base_spec = doom_env_by_name(cfg.env)
    batch_spec = DoomBatchSpec(base=base_spec, batch_dir=cfg.wad_batch, swap_every=getattr(cfg, "wad_swap_every", 5),)
    env_name = f"{cfg.env}_batch"

    batch = WadBatch.from_dir(batch_spec.batch_dir)
    strategy = getattr(cfg, "wad_curriculum", "uniform")
    interestingness = None
    interestingness_graph_path = getattr(cfg, "interestingness_graph_path", None)
    if strategy == "omni" and interestingness_graph_path is not None:
        interestingness = get_intrestingness_graph(interestingness_graph_path)
    curriculum = BatchCurriculum(
        len(batch.entries),
        strategy=strategy,
        interestingness=interestingness if strategy == "omni" else None,
        replay_schedule=getattr(cfg, "plr_replay_schedule", "proportionate"),
        replay_prob=getattr(cfg, "plr_replay_prob", 0.5),
        rho=getattr(cfg, "plr_rho", 1.0),
        staleness_coef=getattr(cfg, "plr_staleness_coef", 0.1),
        score_transform=getattr(cfg, "plr_score_transform", "rank"),
        temperature=getattr(cfg, "plr_temperature", 0.1),
        alpha=getattr(cfg, "plr_alpha", 1.0),
        staleness_transform=getattr(cfg, "plr_staleness_transform", "power"),
        staleness_temperature=getattr(cfg, "plr_staleness_temperature", 1.0),
        plr_score_key=getattr(cfg, "plr_score_key", "mean_value_l1"),
        max_score_coef=getattr(cfg, "plr_max_score_coef", 0.0),
    )
    make_env_func = functools.partial(make_doom_env_from_batch, batch_spec, curriculum)
    register_env(env_name, make_env_func)
    return env_name

def get_intrestingness_graph(interestingness_graph_path: Optional[str]) -> Optional[dict]:
    import json
    if interestingness_graph_path is None:
        return None
    with open(interestingness_graph_path, "r") as f:
        raw = json.load(f)
    return {
        int(k): {int(kk): bool(vv) for kk, vv in v.items()}
        for k, v in raw.items()
    }

def register_vizdoom_models():
    global_model_factory().register_encoder_factory(make_vizdoom_encoder)


def register_vizdoom_components():
    register_vizdoom_envs()
    register_vizdoom_models()


def configure_batch_env_and_agents(cfg):
    # When --wad_batch is given, override cfg.env with the pool name and
    # register the pool environment. Existing DOOM_ENVS are unaffected.
    if getattr(cfg, "wad_batch", None):
        cfg.env = register_batch_env(cfg)

    if cfg.num_agents < 1:
        # Strip "_batch" to lookup standard env base properties.
        cfg.num_agents = get_num_agents(cfg, cfg.env.replace("_batch", ""))


def register_model_factory(cfg):
    """
    model registration facotry
    """
    if getattr(cfg, 'num_agents', 1) > 1:
        if str(getattr(cfg, 'algo', 'APPO')).upper() in ('QMIX', 'VDN', 'QPLEX'):
            global_model_factory().register_actor_critic_factory(make_qmix_actor_critic)
        elif str(getattr(cfg, 'algo', 'APPO')).upper() == 'HAPPO':
            from comrad.models.happo_model import make_happo_actor_critic
            global_model_factory().register_actor_critic_factory(make_happo_actor_critic)
        else:
            # MAPPO, IPPO
            global_model_factory().register_actor_critic_factory(make_mappo_actor_critic)


def parse_args(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_doom_env_args(parser)

    # This is the record_to param, it saves as pngs with an action.json
    # Use ffmpeg to transform into mp4 vid
    # ffmpeg -i %05d.png -c:v libx264 -pix_fmt yuv420p -movflags +faststart -f mp4 vid.mp4
    # It's possible to use this only and run ffmpeg after each episode then upload to wandb but that's more overhead
    # especially for many envs
    add_doom_env_eval_args(parser)

    # Log videos to wandb
    add_wandb_args(parser)

    doom_override_defaults(parser)
    final_cfg = parse_full_cfg(parser, argv)

    # auto add experiment name if not provided
    # Only rename experiment for training script, to avoid conflict for enjoy script
    # But currently train.py is hardcoded into if statement
    if not (any('--experiment=' in i for i in sys.argv)) and any('train.py' in i for i in sys.argv):
        if str(getattr(final_cfg, 'algo', 'APPO')).upper() in ('QMIX', 'VDN', 'QPLEX'):
            mixer = getattr(final_cfg, 'mixer', 'qmix')
            if mixer == 'vdn' or getattr(final_cfg, 'algo', 'APPO').upper() == 'VDN':
                algo_name = "VDN"
            elif mixer in ('dmaq', 'dmaq_qatten'):
                algo_name = f"QPLEX_{mixer}"
            else:
                algo_name = "QMIX"
        elif str(getattr(final_cfg, 'algo', 'APPO')).upper() == 'HAPPO':
            algo_name = "HAPPO"
        elif str(getattr(final_cfg, 'algo', 'APPO')).upper() == 'MAPPO':
            algo_name = "MAPPO"
        else:
            algo_name = "IPPO"
        slurm_id = os.environ.get('SLURM_JOB_ID')
        jid = f"_j{slurm_id}" if slurm_id and os.environ.get('SLURM_JOB_NAME') else ""
        final_cfg.experiment = f"{final_cfg.env}_{algo_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}{jid}"

    return final_cfg


def main():
    register_vizdoom_components()
    cfg = parse_args()
    configure_batch_env_and_agents(cfg)

    if cfg.num_agents > 1:
        register_model_factory(cfg)

    cfg, runner = make_runner(cfg)

    if not (not getattr(cfg, "with_wandb", False) or getattr(cfg, "wandb_record_every", 0) <= 0 or getattr(wandb, "run", None) is None):
        upload_video(runner, cfg)

    status = runner.init()
    if status == ExperimentStatus.SUCCESS:
        status = runner.run()

    # status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
