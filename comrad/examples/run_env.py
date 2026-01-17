import cv2
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.utils.attr_dict import AttrDict
from sf.doom.doom_params import add_doom_env_args, doom_override_defaults
from sf.doom.doom_utils import make_doom_env

def main():
    argv = ["--env=safe_ground2", "--num_agents=2", "--res_w=1920", "--res_h=1080"]
    parser, _ = parse_sf_args(argv=argv, evaluation=False)
    add_doom_env_args(parser)
    doom_override_defaults(parser)
    cfg = parse_full_cfg(parser, argv)
    env_config = AttrDict({"worker_index": 0, "vector_index": 0})
    env = make_doom_env(cfg.env, cfg, env_config, render_mode="human")
    obs, info = env.reset()
    steps = total_cost = total_reward = 0
    num_agents = getattr(env, "num_agents", 1)
    while True:
        if num_agents > 1:
            action = [env.action_space.sample() for _ in range(num_agents)]
        else:
            action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if isinstance(info, list):
            cv2.waitKey(1)
        steps += 1

        if isinstance(info, list):
            total_cost += sum(i.get("cost", 0) for i in info)
            total_reward += sum(reward)
            terminated = all(terminated) if isinstance(terminated, list) else terminated
            truncated = all(truncated) if isinstance(truncated, list) else truncated
        else:
            total_cost += info.get("cost", 0)
            total_reward += reward

        if terminated or truncated:
            break
    print(f"Episode finished in {steps} steps. Reward: {total_reward:.2f}. Cost: {total_cost:.2f}")
    stats = info[0].get("env_stats", {}) if isinstance(info, list) else info.get("env_stats", {})
    print("Final Statistics:", stats)

    env.close()
    if isinstance(info, list):
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
