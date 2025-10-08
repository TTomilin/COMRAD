#!/usr/bin/env python3
# Without benchmarl

import sys
from pathlib import Path
import torch
import wandb
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pettingzoo_wrapper import make

def test():
    if wandb.api.api_key is None:
        print("wandb offline")
        wandb_mode = "offline"
    else:
        print("wandb online")
        wandb_mode = "online"
        
    wandb_mode = "offline" # not using wandb
    
    run = wandb.init(
        project="test-no-benchmarl",
        mode=wandb_mode,
        config={
            "scenario": "pitfall",
            "num_agents": 2,
            "episodes": 3,
            "max_steps_per_episode": 100
        }
    )
    
    try:
        print("Creating env")
        # no visual render
        env = make(
            scenario="pitfall",
            num_agents=2,
            resolution="160x120",
            skip_frames=4,
            async_mode=True,
            host_address="127.0.0.1",
            port=5050,
            netmode=1,
            ticrate=35,
            seed=42,
            enable_video=False,
            record_every=100,
            video_fps=35,
        )
        
        for episode in range(3):
            print(f"\n--- Episode {episode + 1}/3 ---")
    
            obs, infos = env.reset()
            episode_rewards = {agent: 0.0 for agent in env.agents}
            steps = 0
            done = False
            
            while not done and steps < 100:
                # Random actions
                actions = {agent: env.action_space(agent).sample() for agent in env.agents}
                
                # Take step
                obs, rewards, terminations, truncations, infos = env.step(actions)
                
                # Update rewards
                for agent in env.agents:
                    episode_rewards[agent] += rewards[agent]
                
                steps += 1
                done = any(terminations.values()) or any(truncations.values())
            
            # Log whole ep
            total_reward = sum(episode_rewards.values())
            avg_reward = total_reward / len(env.agents)
            
            wandb.log({
                "episode": episode + 1,
                "episode_steps": steps,
                "total_reward": total_reward,
                "avg_reward": avg_reward,
                "agent_0_reward": episode_rewards.get("agent_0", 0),
                "agent_1_reward": episode_rewards.get("agent_1", 0),
            })
            
            print(f"Steps: {steps}, Total reward: {total_reward:.2f}, Avg: {avg_reward:.2f}")
                
        visual_render() # run 5 sec with random actions
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        env.close()
        wandb.finish()
    
    return True

def visual_render():
    try:
        print("Starting vis render")
        env = make(
            scenario="pitfall",
            num_agents=2,
            resolution="800x600",
            render_mode="human",
            seed=42,
            netmode=1,
            skip_frames=1,
            async_mode=True,
            ticrate=20,
        )
        print("Done vis render")
        
        env.reset()
        
        for step in range(100):  # 5sec, 20 ticrate
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            obs, rewards, terminations, truncations, infos = env.step(actions)
            
            try:
                env.render()
            except Exception as e:
                print(f"Error: {e}")
            
            time.sleep(0.05)
            
            if any(terminations.values()) or any(truncations.values()):
                env.reset()
        
        env.close()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    yes = test()
    print("1" if yes else "0")

if __name__ == "__main__":
    main()