import gymnasium as gym
import numpy as np
from gymnasium import Wrapper
from stable_baselines3.common.monitor import Monitor
import racecar_gym.envs.gym_api
from collections import OrderedDict

class RacecarRewardWrapper(Wrapper):
    """
    Custom reward shaping for racecar environment.
    The base environment returns 0 reward, so we create meaningful rewards to teach the agent to drive effectively.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.prev_progress = 0
        self.prev_checkpoint = 0
        self.episode_start_time = 0
        self.total_reward = 0
        self.step_count = 0
        
    def reset(self, **kwargs):
        """Reset tracking variables"""
        self.prev_progress = 0
        self.prev_checkpoint = 0
        self.episode_start_time = 0
        self.total_reward = 0
        self.step_count = 0
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def step(self, action):
        """Execute action and compute custom reward"""
        obs, _, terminated, truncated, info = self.env.step(action)
        
        # Extract useful info
        progress = info['progress']
        velocity = info['velocity']
        wall_collision = info['wall_collision']
        wrong_way = info['wrong_way']
        checkpoint = info['checkpoint']
        progress_delta = progress - self.prev_progress
        
        # Handle lap completion (progress resets to 0)
        if progress_delta < -0.5:  # Lap completed
            progress_delta = (1.0 - self.prev_progress) + progress
            progress_reward = progress_delta * 2000  # HUGE reward for lap completion
        else:
            progress_reward = progress_delta * 500  
        
        #Speed reward (encourage faster driving)
        forward_velocity = velocity[0]  # vx component
        speed_reward = max(0, forward_velocity) * 1.0       

        
        collision_penalty = -50.0 if wall_collision else 0.0  
        
        # Wrong way penalty
        wrong_way_penalty = -20.0 if wrong_way else 0.0  
        

        time_penalty = -0.001         
        # Encourage going faster
        velocity_magnitude = np.linalg.norm(velocity[:3])
        if velocity_magnitude < 0.1:
            standing_penalty = -5.0  
        else:
            standing_penalty = 0.0
        
        #Alive bonus (reward for staying on track)
        alive_bonus = 0.1  
        
        # TOTAL REWARD
        reward = (
            progress_reward +
            speed_reward +
            collision_penalty +
            wrong_way_penalty +
            time_penalty +
            standing_penalty +
            alive_bonus
        )
        
        # Update tracking variables
        self.prev_progress = progress
        self.prev_checkpoint = checkpoint
        self.total_reward += reward
        self.step_count += 1
        
        # Terminal conditions
        if wall_collision:
            terminated = True
            reward -= 50  # Extra penalty for crashing
        
        # Lap completed
        if progress > 0.99 and self.step_count > 100:
            truncated = True
            reward += 500  # Big bonus for finishing
        
        return obs, reward, terminated, truncated, info


class StatePreprocessor(Wrapper):
    """
    Flatten and normalize observations for RL agent.
    Converts dict observation to flat numpy array.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Calculate flattened observation size
        # lidar: 1080, velocity: 6, acceleration: 6, pose: 6, time: 1
        self.obs_dim = 1080 + 6 + 6 + 6 + 1  # = 1099
        
        # Define new observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flatten_obs(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._flatten_obs(obs), reward, terminated, truncated, info
    
    def _flatten_obs(self, obs_dict):
        """Convert dict observation to flat array"""
        lidar = obs_dict['lidar'].astype(np.float32)
        velocity = obs_dict['velocity'].astype(np.float32)
        acceleration = obs_dict['acceleration'].astype(np.float32)
        pose = obs_dict['pose'].astype(np.float32)
        time = np.array([obs_dict['time']], dtype=np.float32)
        
        # Concatenate all observations
        flat_obs = np.concatenate([
            lidar,
            velocity,
            acceleration,
            pose,
            time
        ])
        
        return flat_obs


class ActionWrapper(Wrapper):
    """
    Convert flat action array to dict format expected by environment.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Define action space as flat Box
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
    
    def step(self, action):
        """Convert flat action to dict"""

        
        # action is [motor, steering]
        action_dict = OrderedDict([
            ('motor', np.array([action[0]], dtype=np.float32)),
            ('steering', np.array([action[1]], dtype=np.float32))
        ])
        
        return self.env.step(action_dict)


def make_env(render_mode=None, track='columbia'):
    """
    Create and wrap the racecar environment
    It returns a wrapped gym environment ready for RL training
    """
    print("Track being used:", track)
    # Track selection
    track_map = {
        'austria': 'SingleAgentAustria-v0',
        'columbia': 'SingleAgentColumbia-v0',
        'montreal': 'SingleAgentMontreal-v0',
        'circle': 'SingleAgentCircle_cw-v0'
    }
    
    if track not in track_map:
        raise ValueError(f"Unknown track: {track}. Choose from {list(track_map.keys())}")
    
    # Create base environment
    env = gym.make(track_map[track], render_mode=render_mode)
    
    # Apply wrappers
    env = RacecarRewardWrapper(env)
    env = StatePreprocessor(env)
    env = ActionWrapper(env)
    env = Monitor(env)  
    
    return env