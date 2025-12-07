from pathlib import Path

import gymnasium as gym
import racecar_gym.envs.gym_api
import torch
import torch.nn as nn
import numpy as np
from dqn_wrapper import DQNWrapper
from agents.dqn_agent import DQNNetwork, RacecarReplayBuffer
from train_dqn_circle import (
    num_bins_motor, num_bins_steering, use_pose, use_velocity,
use_acceleration, use_lidar
)

best_model_path = Path("best_dqn_model/final_dqn_model.pth")



if __name__ == "__main__":
    track = 'circle_cw'
    scenario = 'config/scenarios/' + track + '.yml'
    render_mode = "human"
    env = gym.make(
        id='SingleAgentRaceEnv-v0',
        scenario=scenario,
        vehicle_config_path='config/vehicles/og_racecar.yml',
        render_mode=render_mode
    )
    env = DQNWrapper(
        env,
        num_bins_motor=num_bins_motor,
        num_bins_steering=num_bins_steering,
        use_pose=use_pose,
        use_velocity=use_velocity,
        use_acceleration=use_acceleration,
        use_lidar=use_lidar
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    state_dim = env.observation_space_dim
    num_actions = len(env.allowed_actions)
    payload = torch.load(best_model_path, weights_only=False, map_location=device)
    online_net = DQNNetwork(state_dim, num_actions).to(device)
    online_net.load_state_dict(payload['online_net_state_dict'])
    state, info = env.reset()
    episode_reward = 0
    done = False
    step_count = 0
    episode_q_values = []
    episode_actions = []

    while not done:
        # Select action
        action, q_values = online_net.select_action(torch.FloatTensor(state).to(device), deterministic=True)
        print(f"Step {step_count}: Action={action}, Q-values={q_values.cpu().detach().numpy()}")

        # Store for analysis
        episode_q_values.append(q_values)
        episode_actions.append(action)

        # Execute
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        episode_reward += reward
        state = next_state
        step_count += 1

        #print(f"  Step {step_count}: Reward={episode_reward:.2f}, "
        #      f"Max Q={q_values.max():.2f}, Min Q={q_values.min():.2f}")

    print(f"Episode finished: {step_count} steps, reward={episode_reward:.2f}")

