import sys
import json
from sample_factory.utils.attr_dict import AttrDict
from comrad.utils.doom_utils import make_doom_env

# import vizdoom as vzd
# game = vzd.DoomGame()
# game.load_config("comrad/doom/scenarios/prot_beta_long.cfg")
# game.init()

sys.argv = sys.argv[:1]
cfg_dict=json.load(open('train_dir/pitfall_399c/config.json'))
cfg=AttrDict(cfg_dict)
env_config=AttrDict({'worker_index':0, 'vector_index':0, 'safe_init':False})

env=make_doom_env('doom_pitfall', cfg, env_config)

obs, infos=env.reset(seed=42)
print('Initial reset obs:', [type(o) for o in obs], [o.shape for o in obs])

num_agents = env.unwrapped.num_agents
for i in range(5):
    actions = [env.action_space.sample() for _ in range(num_agents)]
    obs, rewards, terms, truncs, infos = env.step(actions)
    print(f'Step {i}: obs: {[o.shape for o in obs]}, rewards: {rewards}, terms: {terms}, truncs: {truncs}', 'infos:', infos)

obs2, infos2 = env.reset()
print('Manual reset without seed obs:', [type(o) for o in obs], [o.shape for o in obs2])
