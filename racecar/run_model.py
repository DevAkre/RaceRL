"""Run a trained Q-learning agent using the saved Q-table in models/q_table.pkl

This script mirrors the environment setup used in training and runs a small
number of episodes with render_mode='human' so you can watch the agent.
"""
import argparse
from time import sleep

import gymnasium as gym

import agents
import wrapper
import racecar_gym.envs.gym_api  
import numpy as np
import logging
import cv2


def display_frame(frame):
    """Display a frame using OpenCV."""
    arr = np.asarray(frame)
    # Clip and convert to uint8 (handles int64, float, etc.)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
   
    # Convert RGB to BGR for OpenCV
    bgr_frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imshow('Racecar', bgr_frame)
    cv2.waitKey(1)

def main(model_path: str, n_episodes: int = 3, track: str = 'circle'):
    # Create a template environment with the same action wrapper used during
    # training so the agent's q-table lines up with the action space.
    
    # Training hyperparameters
    n_episodes = 500        # Number of episodes to practice

    # render_mode = 'human'  # Set to 'human' to visualize training
    render_mode = 'rgb_array_follow'
    # render_mode = 'rgb_array_birds_eye'

    
    # Template environment used only for agent initialization (not used for stepping)
    env = gym.make(
        id='SingleAgentRaceEnv-v0', 
        scenario='config/scenarios/' + track + '.yml',
        vehicle_config_path='config/vehicles/racecar.yml',
        render_mode=render_mode,
        render_options=dict(width=320, height=240)
    )
    env.metadata['render_fps'] = 1
    env = wrapper.DiscreteActionWrapper(env, num_bins_motor=3, num_bins_steering=5)
    env = wrapper.DiscretizeObservationWrapper(env, lidar_bins=[0, 2.5, 5.0], num_velocity_bins=3, num_acceleration_bins=3)
    #Episode Statictics 
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # Get State and Action spaces
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    agent = agents.QLearningRacecarAgent(env=env)

    # Print q-table load info
    print(f"Loading Q-table from {model_path}")
    agent.load(model_path)

    # Print Q-table stats
    avg_action = np.mean([len(q_vals) for q_vals in agent.q_values.values()])
    print(f"Q-table has {len(agent.q_values.keys())} state entries, each with {avg_action} actions on average")

    rgb_frame = None

    for ep in range(n_episodes):
        obs, info = env.reset(options=dict(mode='grid'))
        done = False
        t = 0
        steer = 0
        while not done:
            lidar = obs['lidar']
            # print(lidar)
            if lidar[2] < 2.0:
                if lidar[1] < lidar[3]:
                    if lidar[2] == 0:
                        steer = 4
                    elif lidar[2] == 1:
                        steer = 3
                elif lidar[3] < lidar[1]:
                    if lidar[2] == 0:
                        steer = 0
                    elif lidar[2] == 1:
                        steer = 1
            else:
                steer = 2
            action = {'motor': np.int64(2), 'steering': np.int64(steer)}
            # action = agent.get_action(obs, explore=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            obs = next_obs

            # Render occasionally 
            if render_mode != None and render_mode != 'human':
                if t % 1 == 0:
                    rgb_frame = env.render()
                    display_frame(rgb_frame)
            
            if render_mode == 'human':
                print(f"Action taken: {action}, Reward: {reward}, Done: {done}")
                sleep(1.0 / 60)

            t += 1
        if "episode" in info:
            episode_data = info["episode"]
            logging.info(f"Episode {ep}: "
                        f"reward={episode_data['r']:.1f}, "
                        f"length={episode_data['l']}, "
                        f"time={episode_data['t']:.2f}s")
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='models/q_table.pkl', help='Path to saved q-table')
    parser.add_argument('--track', '-t', default='circle', help='Track to run the model on (e.g., austria, berlin, circle)')
    parser.add_argument('--episodes', '-e', type=int, default=3, help='Number of episodes to run')
    args = parser.parse_args()
    main(args.model, args.episodes, args.track)