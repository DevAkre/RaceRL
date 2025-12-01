"""Run a trained Q-learning agent using the saved Q-table in models/q_table.pkl

This script mirrors the environment setup used in training and runs a small
number of episodes with render_mode='human' so you can watch the agent.
"""
import argparse
from time import sleep

import gymnasium as gym
from typing import Optional
import agents
import wrapper
import numpy as np
import logging
import cv2

import display_util as display

def main(model_path: str, n_episodes: int = 3, track: str = 'circle', render_mode: Optional[str] = 'rgb_array_follow', output_video: Optional[str] = None):
    # Create a template environment with the same action wrapper used during
    # training so the agent's q-table lines up with the action space.
    
    # Training hyperparameters
    n_episodes = 1        # Number of episodes to practice
    num_bins_motor=3
    num_bins_steering=5

    # Template environment used only for agent initialization (not used for stepping)
    env = gym.make(
        id='SingleAgentRaceEnv-v0', 
        scenario='config/scenarios/' + track + '.yml',
        vehicle_config_path='config/vehicles/racecar.yml',
        render_mode=render_mode,
        render_options=dict(width=320, height=240)
    )
    env = wrapper.DiscreteAction(env, num_bins_motor=num_bins_motor, num_bins_steering=num_bins_steering)
    env = wrapper.FlattenAction(env)
    env = wrapper.DiscretizeObservation(env, lidar_bins=[0, 2.5, 5.0], num_velocity_bins=3, num_acceleration_bins=0)
    env = wrapper.FlattenObservation(env)
    #Episode Statictics 
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
    env.reset()
    # Get State and Action spaces
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    agent = agents.QLearningRacecarAgent(
        env
    )

    # Load the saved q-table
    print(f"Loading agent from {model_path}")
    agent.load(model_path)

    # Print Q-table stats
    avg_action = np.mean([len(q_vals) for q_vals in agent.q_values.values()])
    print(f"Q-table has {len(agent.q_values.keys())} state entries, each with {avg_action} actions on average")

    rgb_frame = None
    recorded_frames = []  # Store frames for video recording
    for ep in range(n_episodes):
        obs, info = env.reset(options=dict(mode='grid'))
        done = False
        t = 0
        episode_frames = []  # Frames for this episode
        while not done:
            action = agent.get_action(obs, explore=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            obs = next_obs

            # Render occasionally 
            if render_mode != None and render_mode != 'human':
                if t % 1 == 0:
                    rgb_frame = env.render()
                    # Get motor and steering values
                    motor = action // num_bins_steering
                    steer = action % num_bins_steering
                    frame = display.prepare_frame(rgb_frame, float(np.linalg.norm(info['velocity'][3:])), info['time'], motor, num_bins_motor, steer, num_bins_steering)
                    # Store frames for first episode if output_video is specified
                    if output_video and ep == 0:
                        episode_frames.append(frame)
                    display.display_frame(frame)
            
            if render_mode == 'human':
                print(f"Action taken: {action}, Reward: {reward}, Done: {done}")
                sleep(1.0 / 30)

            t += 1
        
        # Save frames from first episode
        if ep == 0 and output_video and episode_frames:
            recorded_frames = episode_frames
            
        if "episode" in info:
            episode_data = info["episode"]
            logging.info(f"Episode {ep}: "
                        f"reward={episode_data['r']:.1f}, "
                        f"length={episode_data['l']}, "
                        f"time={episode_data['t']:.2f}s")
    env.close()
    
    # Save video if output path was specified
    if output_video and recorded_frames:
        display.save_frames_to_video(recorded_frames, output_video, fps=100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='models/q_table.pkl', help='Path to saved q-table')
    parser.add_argument('--track', '-t', default='circle', help='Track to run the model on (e.g., austria, berlin, circle)')
    parser.add_argument('--episodes', '-e', type=int, default=3, help='Number of episodes to run')
    parser.add_argument('--render_mode', '-r', type=str, default='rgb_array_follow', help='Render mode for the environment (e.g., human, rgb_array_follow, rgb_array_birds_eye, None)')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output path for video file (MP4) of first episode')
    args = parser.parse_args()
    main(args.model, args.episodes, args.track, args.render_mode, args.output)
