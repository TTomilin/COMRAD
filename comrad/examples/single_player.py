"""
Testing using doom_gym play_human_mode(), single player only. Kinda broken. Dont use.
"""
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.utils.attr_dict import AttrDict
from comrad.envs.doom_params import add_doom_env_args, doom_override_defaults
from comrad.utils.doom_utils import make_doom_env
from comrad.envs.doom_gym import VizdoomEnv
def main():
    argv = ["--env=doom_pitfall", "--num_agents=1", "--res_w=1920", "--res_h=1080"]
    parser, _ = parse_sf_args(argv=argv, evaluation=False)
    add_doom_env_args(parser)
    doom_override_defaults(parser)
    cfg = parse_full_cfg(parser, argv)
    env_config = AttrDict({"worker_index": 0, "vector_index": 0})
    env = make_doom_env(cfg.env, cfg, env_config, render_mode="human")
    print(f"Starting: {cfg.env}")
    VizdoomEnv.play_human_mode(env, num_episodes=10, num_actions=14)
    env.close()
if __name__ == "__main__":
    main()
