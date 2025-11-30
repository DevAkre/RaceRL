"""Run a trained Q-learning agent using the saved Q-table in models/q_table.pkl

This script mirrors the environment setup used in training and runs a small
number of episodes with render_mode='human' so you can watch the agent.
"""
import argparse
from time import sleep

import gymnasium as gym
from typing import Optional, List
import agents
import wrapper
import racecar_gym.envs.gym_api  
import numpy as np
import logging
import cv2

def prepare_frame(frame, velocity: float = 0.0, time: float = 0.0, motor: int = 0, motor_bins: int = 3, steer: int = 2, steer_bins:int = 5) -> np.ndarray:
    arr = np.asarray(frame)
    # Clip and convert to uint8 (handles int64, float, etc.)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
   
    # Convert RGB to BGR for OpenCV
    bgr_frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    frame_height, frame_width, _ = bgr_frame.shape
    # Overlay velocity text bottom right corner
    cv2.putText(bgr_frame, f"{velocity:.2f} m/s", (frame_width - 100, frame_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    # Overlay time text bottom left corner
    cv2.putText(bgr_frame, f"Time: {time:.1f}s", (10, frame_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    # Show motor as a bar at the bottom center
    motor_bar_length = frame_width // 4
    motor_bar_height = frame_height // 10
    motor_x = int((frame_width - motor_bar_length) / 2)
    motor_y = frame_height - motor_bar_height - 10
    cv2.rectangle(bgr_frame, (motor_x, motor_y), (motor_x + motor_bar_length, motor_y + motor_bar_height), (200, 200, 200), 1)
    motor_fill_length = int((motor / (motor_bins-1)) * motor_bar_length) 
    cv2.rectangle(bgr_frame, (motor_x, motor_y), (motor_x + motor_fill_length, motor_y + motor_bar_height), (0, 255, 0), -1)
    # Label Brake/Neutral/Forward
    cv2.putText(bgr_frame, "B", (motor_x, motor_y + motor_bar_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(bgr_frame, "N", (motor_x + motor_bar_length // 2 - 5, motor_y + motor_bar_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(bgr_frame, "F", (motor_x + motor_bar_length - 7, motor_y + motor_bar_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    # Show steering as a bar above the motor bar
    steer_bar_length = frame_width // 4
    steer_bar_height = frame_height // 10
    steer_x = int((frame_width - steer_bar_length) / 2)
    steer_y = motor_y - steer_bar_height
    cv2.rectangle(bgr_frame, (steer_x, steer_y), (steer_x + steer_bar_length, steer_y + steer_bar_height), (200, 200, 200), 1)
    steer_fill_length = int((steer / (steer_bins-1)) * steer_bar_length) 
    cv2.rectangle(bgr_frame, (steer_x, steer_y), (steer_x + steer_fill_length, steer_y + steer_bar_height), (255, 123, 123), -1)
    # Label Left/Right
    cv2.putText(bgr_frame, "L", (steer_x, steer_y + steer_bar_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(bgr_frame, "R", (steer_x + steer_bar_length - 7, steer_y + steer_bar_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return bgr_frame

def display_frame(frame, velocity: float = 0.0, time: float = 0.0):
    """Display a frame using OpenCV."""
    # Overlay steering and motor values top left corner
    cv2.imshow('Racecar', frame)
    cv2.waitKey(1)

def save_frames_to_video(frames: List[np.ndarray], output_path: str, fps: int = 30):
    """Save a list of frames to an MP4 video using OpenCV VideoWriter.
    
    Args:
        frames: List of RGB frames as numpy arrays
        output_path: Path to save the output video file
        fps: Frames per second for the output video
    """
    if not frames:
        print("No frames to save!")
        return
    
    print(f"Creating video with OpenCV at {output_path}...")
    
    # Get frame dimensions from first frame
    first_frame =  frames[0]
    height, width = first_frame.shape[:2]
    
    # Define the codec and create VideoWriter object
    # Use mp4v codec for MP4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # pyright: ignore[reportAttributeAccessIssue]
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        return
    
    # Write each frame to the video
    print(f"Writing {len(frames)} frames to video...")
    for i, frame in enumerate(frames):
        arr = np.asarray(frame)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        # Convert RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
    
    # Release the VideoWriter
    out.release()
    print(f"Video saved successfully to {output_path}")

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
        render_options=dict(width=320, height=160)
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
                    frame = prepare_frame(rgb_frame, float(np.linalg.norm(info['velocity'][3:])), info['time'], motor, num_bins_motor, steer, num_bins_steering)
                    # Store frames for first episode if output_video is specified
                    if output_video and ep == 0:
                        episode_frames.append(frame)
                    display_frame(frame)
            
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
        save_frames_to_video(recorded_frames, output_video, fps=100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='models/q_table.pkl', help='Path to saved q-table')
    parser.add_argument('--track', '-t', default='circle', help='Track to run the model on (e.g., austria, berlin, circle)')
    parser.add_argument('--episodes', '-e', type=int, default=3, help='Number of episodes to run')
    parser.add_argument('--render_mode', '-r', type=str, default='rgb_array_follow', help='Render mode for the environment (e.g., human, rgb_array_follow, rgb_array_birds_eye, None)')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output path for video file (MP4) of first episode')
    args = parser.parse_args()
    main(args.model, args.episodes, args.track, args.render_mode, args.output)
