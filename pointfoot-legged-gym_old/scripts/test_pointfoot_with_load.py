import os
import torch
import numpy as np
from legged_gym.envs.pointfoot.PF.pointfoot_with_load import PointFootWithLoad
from legged_gym.envs.pointfoot.PF.pointfoot_with_load_config import PointFootWithLoadCfg
from legged_gym.utils.task_registry import task_registry

def test_pointfoot_with_load():
    # Environment configuration
    env_cfg = PointFootWithLoadCfg()
    # Modify any config parameters if needed
    env_cfg.env.num_envs = 4  # Use fewer envs for testing
    
    # Create environment
    env = task_registry.make_env(name="pointfoot_with_load",
                               cfg=env_cfg,
                               sim_device="cuda:0",
                               graphics_device_id=0,
                               headless=False)
    
    # Test parameters
    num_episodes = 3
    max_steps = 1000
    
    for episode in range(num_episodes):
        print(f"\nStarting Episode {episode + 1}")
        
        # Reset environment
        obs = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Random actions for testing
            actions = torch.randn(env_cfg.env.num_envs, env_cfg.env.num_actions, device=env.device)
            
            # Step the environment
            obs, rewards, dones, info = env.step(actions)
            episode_reward += rewards.mean()
            
            # Print some debug info every 100 steps
            if step % 100 == 0:
                print(f"Step {step}")
                print(f"Load Position: {env.load_pos[0].cpu().numpy()}")
                print(f"Load Velocity: {env.load_lin_vel[0].cpu().numpy()}")
                print(f"Robot Base Position: {env.root_states[0, 0:3].cpu().numpy()}")
                print(f"Current Reward: {rewards.mean().item():.3f}")
            
            # Check if episode is done
            if dones.any():
                print(f"Episode finished after {step + 1} steps")
                break
        
        print(f"Episode {episode + 1} finished with total reward: {episode_reward:.3f}")

if __name__ == "__main__":
    test_pointfoot_with_load() 