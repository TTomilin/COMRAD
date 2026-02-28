import cv2
import time
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.utils.attr_dict import AttrDict
from comrad.envs.doom_params import add_doom_env_args, doom_override_defaults
from comrad.utils.doom_utils import make_doom_env

class you:
    def __init__(self, env):
        """
        WASD: Move/Strafe
        Arrow LR: Turn
        Space: Jump
        C: Attack shoot
        E: Use
        1-5: Select weapon
        ESC: exit
        """

        self.env = env
        self.actions = {
            "turn": 0,
            "move": 0,
            "strafe": 0,
            "attack": 0,
            "use": 0,
            "jump": 0,
            "weapon": 0,
        }
        self.running = True
        self.listener = None

    def start(self):
        from pynput.keyboard import Key, Listener

        def on_press(key):
            try:
                k = key.char
            except Exception:
                k = key.name if hasattr(key, "name") else key

            if k == "w":
                self.actions["move"] = 1
            elif k == "s":
                self.actions["move"] = 2
            elif k == "a":
                self.actions["strafe"] = 1
            elif k == "d":
                self.actions["strafe"] = 2
            elif key == Key.left:
                self.actions["turn"] = 1
            elif key == Key.right:
                self.actions["turn"] = 2
            elif key == Key.space:
                self.actions["jump"] = 1
            elif k == "c":
                self.actions["attack"] = 1
            elif k == "e":
                self.actions["use"] = 1
            elif k in ["1", "2", "3", "4", "5"]:
                self.actions["weapon"] = int(k)
            elif key == Key.esc:
                self.running = False

        def on_release(key):
            try:
                k = key.char
            except Exception:
                k = key.name if hasattr(key, "name") else key

            if k == "w" and self.actions["move"] == 1:
                self.actions["move"] = 0
            elif k == "s" and self.actions["move"] == 2:
                self.actions["move"] = 0
            elif k == "a" and self.actions["strafe"] == 1:
                self.actions["strafe"] = 0
            elif k == "d" and self.actions["strafe"] == 2:
                self.actions["strafe"] = 0
            elif key == Key.left and self.actions["turn"] == 1:
                self.actions["turn"] = 0
            elif key == Key.right and self.actions["turn"] == 2:
                self.actions["turn"] = 0
            elif key == Key.space:
                self.actions["jump"] = 0
            elif k == "c":
                self.actions["attack"] = 0
            elif k == "e":
                self.actions["use"] = 0
            elif k in ["1", "2", "3", "4", "5"]:
                if self.actions["weapon"] == int(k):
                    self.actions["weapon"] = 0

        self.listener = Listener(on_press=on_press, on_release=on_release)
        self.listener.start()

    def get_action(self):
        # return (self.actions["turn"], 1 if self.actions["move"] == 1 else 0) #parallel
        return (self.actions["turn"], 1 if self.actions["move"] == 1 else 0, self.actions["jump"])  # pitfall
        # return (self.actions["turn"], self.actions["move"], self.actions["strafe"], self.actions["attack"]) #safeground

        # #armory
        # wp = self.actions["weapon"]
        # w = {2: 1, 3: 2, 5: 3}.get(wp, 0)
        # return (self.actions["move"], self.actions["strafe"], self.actions["turn"], self.actions["attack"], w)

    def is_running(self):
        return self.running

def main():
    argv = ["--env=doom_pitfall", "--num_agents=2", "--res_w=1920", "--res_h=1080", "--env_frameskip=1"]
    parser, _ = parse_sf_args(argv=argv, evaluation=False)
    add_doom_env_args(parser)
    doom_override_defaults(parser)
    cfg = parse_full_cfg(parser, argv)
    env_config = AttrDict({"worker_index": 0, "vector_index": 0})
    env = make_doom_env(cfg.env, cfg, env_config, render_mode="human")
    obs, info = env.reset()
    steps = total_cost = total_reward = 0
    num_agents = getattr(env, "num_agents", 1)

    human = you(cfg.env)
    human.start()
    pid = 0 # Specify you want to be player 0 or 1

    last_time = time.time()
    fps = 35.0
    ftime = 1.0 / fps

    try:
        while human.is_running():
            if num_agents > 1:
                actions = []
                human_action = human.get_action()
                for i in range(num_agents):
                    if i == pid:
                        actions.append(human_action)
                    else:
                        actions.append(env.action_space.sample())
            else:
                actions = human.get_action()

            obs, reward, terminated, truncated, info = env.step(actions)
            env.render()

            now = time.time()
            space = now - last_time
            wait = ftime - space

            if isinstance(info, list):
                ms = int(wait * 1000)
                if ms < 1:
                    ms = 1
                key = cv2.waitKey(ms) & 0xFF
                if key == 27: # ESC
                    break
            else:
                if wait > 0:
                    time.sleep(wait)
            last_time = time.time()
            steps += 1
            if isinstance(info, list):
                for i in info:
                    if isinstance(i, dict):
                        total_cost += float(i.get("cost", 0))
                if isinstance(reward, (list, tuple)):
                    total_reward += sum(float(r) for r in reward)
                else:
                    total_reward += float(reward)
                terminated = all(terminated) if isinstance(terminated, list) else terminated
                truncated = all(truncated) if isinstance(truncated, list) else truncated
            else:
                if isinstance(info, dict):
                    total_cost += float(info.get("cost", 0))
                if isinstance(reward, (list, tuple)):
                    total_reward += float(reward[0])
                else:
                    total_reward += float(reward)
            if terminated or truncated:
                print(f"Episode finished in {steps} steps. Reward: {total_reward:.2f}. Cost: {total_cost:.2f}")
                obs, info = env.reset()
                steps = total_cost = total_reward = 0
    except KeyboardInterrupt:
        pass
    except Exception:
        pass
    finally:
        env.close()
        if isinstance(info, list):
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
