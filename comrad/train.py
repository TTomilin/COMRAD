import functools
import os
import sys
import datetime
import wandb

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import make_runner

from comrad.models.doom_model import make_vizdoom_encoder
from comrad.envs.doom_params import add_doom_env_args, add_doom_env_eval_args, doom_override_defaults, add_wandb_args
from comrad.utils.doom_utils import DOOM_ENVS, make_doom_env_from_spec
from comrad.utils.video_uploader import upload_video
from comrad.models.mappo_model import make_mappo_actor_critic
from comrad.models.qmix_model import make_qmix_actor_critic


def register_vizdoom_envs():
    for env_spec in DOOM_ENVS:
        make_env_func = functools.partial(make_doom_env_from_spec, env_spec)
        register_env(env_spec.name, make_env_func)


def register_vizdoom_models():
    global_model_factory().register_encoder_factory(make_vizdoom_encoder)


def register_vizdoom_components():
    register_vizdoom_envs()
    register_vizdoom_models()


def register_model_factory(cfg):
    """
    model registration facotry
    """
    if getattr(cfg, 'num_agents', 1) > 1:
        if str(getattr(cfg, 'algo', 'APPO')).upper() in ('QMIX', 'VDN'):
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
        if str(getattr(final_cfg, 'algo', 'APPO')).upper() in ('QMIX', 'VDN'):
            algo_name = "VDN" if getattr(final_cfg, 'mixer', 'qmix') == 'vdn' else "QMIX"
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

    if cfg.num_agents < 1:
        from comrad.utils.doom_utils import get_num_agents
        n_agents = get_num_agents(cfg, cfg.env)
        cfg.num_agents = n_agents

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

# import sys
# sys.argv = sys.argv[:1]

# import json
# from sample_factory.utils.attr_dict import AttrDict
# from comrad.utils.doom_utils import make_doom_env

# cfg_dict=json.load(open('train_dir/pitfall_399c/config.json'))
# cfg=AttrDict(cfg_dict)
# env_config=AttrDict({'worker_index':0, 'vector_index':0, 'safe_init':False})

# env=make_doom_env('doom_pitfall', cfg, env_config)

# obs, infos=env.reset(seed=42)
# print('Initial reset obs:', [type(o) for o in obs], [o.shape for o in obs])

# num_agents = env.unwrapped.num_agents
# for i in range(5):
#     actions = [env.action_space.sample() for _ in range(num_agents)]
#     obs, rewards, terms, truncs, infos = env.step(actions)
#     print(f'Step {i}: obs: {[o.shape for o in obs]}, rewards: {rewards}, terms: {terms}, truncs: {truncs}', 'infos:', infos)

# obs2, infos2 = env.reset()
# print('Manual reset without seed obs:', [type(o) for o in obs], [o.shape for o in obs2])
