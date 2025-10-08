#!/usr/bin/env python3

# test if creating env might casue deadlock

import sys
import time
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pettingzoo_wrapper import make

def clean():
    try:
        subprocess.run(['pkill', '-9', 'vizdoom'], capture_output=True, timeout=5)
        time.sleep(0.5)
    except Exception:
        pass

def test_make_env(num_agents=2, timeout=120):
    start_time = time.time()
    try:
        print("Creating env")
        env = make(
            scenario="pitfall",
            num_agents=num_agents,
            resolution="160x120",
            skip_frames=4,
            async_mode=True,
            host_address="127.0.0.1",
            port=5029,
            netmode=1,
            ticrate=35,
            use_multi_binary_action_space=False,
            seed=42,
            enable_video=False,
        )
        elapsed = time.time() - start_time
        print(f"{elapsed:.2f}")
        
        print("\nResetting env")
        reset_start = time.time()
        obs, infos = env.reset(seed=42)
        elapsed = time.time() - reset_start
        print(f"{elapsed:.2f}")
        print(env.agents)
        print(list(obs.keys()))
        
        print("\nDo random")
        step_times = []
        for i in range(5):
            step_start = time.time()
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            obs, rewards, term, trunc, infos = env.step(actions)
            step_time = time.time() - step_start
            step_times.append(step_time)
            print(f"Step {i+1}, {step_time:.3f}, rewards {dict(rewards)}")
            
            if any(term.values()) or any(trunc.values()):
                break
        
        print("\nClose env")
        close_start = time.time()
        env.close()
        elapsed = time.time() - close_start
        print(f"{elapsed:.2f}")
        
        total_time = time.time() - start_time
        print(f"{total_time:.2f}s")
        return True
        
    except TimeoutError as e:
        elapsed = time.time() - start_time
        print(f"\ntimeout {elapsed:.2f}s")
        return False
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\nerror {elapsed:.2f}s")
        print(e)
        import traceback
        traceback.print_exc()
        return False


def test_make_envs():
    num_tests = 3
    for i in range(num_tests):
        print(f"Test {i+1}/{num_tests}")
        
        if i > 0: # skip first iter
            clean()
            time.sleep(1)
        
        success = test_make_env(num_agents=2, timeout=60)
        if not success:
            print(f"\nfailed test {i+1}")
            return False
        
        if i < num_tests - 1:
            time.sleep(2) # wait till next test
    return True


def test_env_rety(num_agents=2, max_attempts=2): # test with retry
    for attempt in range(max_attempts):
        try:
            if attempt > 0:
                clean()
                time.sleep(2)
            return test_make_env(num_agents=num_agents, timeout=120)
            
        except TimeoutError as e:
            print(f"\n{attempt+1} timed out")
            return False
        except Exception as e:
            print(e)
            return False
    return False


if __name__ == "__main__":
    # cleanup
    clean()
    time.sleep(1)
    
    success = test_env_rety(num_agents=2, max_attempts=2)
    if success:
        success = test_make_envs()
