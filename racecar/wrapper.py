import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Dict, Box, MultiDiscrete, Tuple
from typing import Optional, SupportsFloat


class DiscreteAction(gym.ActionWrapper):
    """Wrap a continuous action space and present a discrete action space.
    """

    def __init__(self, env, num_bins_motor: int = 3, num_bins_steering: int = 5):
        super().__init__(env)
        self.num_bins_motor = num_bins_motor
        self.num_bins_steering = num_bins_steering
        allowed_motor_values = np.linspace(-1.0, 1.0, num_bins_motor)
        allowed_steering_values = np.linspace(-1.0, 1.0, num_bins_steering)
        # print(f"DiscreteActionWrapper: allowed motor values: {allowed_motor_values}")
        # print(f"DiscreteActionWrapper: allowed steering values: {allowed_steering_values}")
        self._bin_size_motor = allowed_motor_values[1] - allowed_motor_values[0]
        self._bin_size_steering = allowed_steering_values[1] - allowed_steering_values[0]
        # Present actions as a Dict with separate Discrete entries for motor and steering
        self.action_space = Dict({
            'motor': Discrete(num_bins_motor),
            'steering': Discrete(num_bins_steering),
        })

    def action(self, action) -> dict:
        """Convert a discrete action (dict with 'motor' and 'steering' indices) to
        the environment's continuous action dict.
        """
        # Expect mapping-like input with keys 'motor' and 'steering'
        try:
            motor_index = int(action['motor'])
            steering_index = int(action['steering'])
        except Exception:
            # Fallback: if action is a sequence like (motor_idx, steering_idx)
            try:
                motor_index, steering_index = map(int, action)
            except Exception:
                raise ValueError("Action must be int, mapping with 'motor'/'steering', or a 2-tuple")

        # Clamp indices to valid ranges
        motor_index = max(0, min(motor_index, self.num_bins_motor - 1))
        steering_index = max(0, min(steering_index, self.num_bins_steering - 1))

        # Convert indices to continuous values
        motor_value = -1.0 + motor_index * self._bin_size_motor
        steering_value = -1.0 + steering_index * self._bin_size_steering

        return {
            'motor': np.array(motor_value, dtype=np.float64),
            'steering': np.array(steering_value, dtype=np.float64),
        }
    
class FlattenAction(gym.ActionWrapper):
    """Wrap a discrete action space and present a flattened discrete action space.
    """

    def __init__(self, env):
        super().__init__(env)
        if not isinstance(env.action_space, Dict):
            raise ValueError("Expected action space to be of type gym.spaces.Dict")
        motor_space = env.action_space.spaces.get('motor', None)
        steering_space = env.action_space.spaces.get('steering', None)
        if not isinstance(motor_space, Discrete) or not isinstance(steering_space, Discrete):
            raise ValueError("Expected 'motor' and 'steering' action spaces to be of type gym.spaces.Discrete")
        self.num_bins_motor = motor_space.n
        self.num_bins_steering = steering_space.n
        self.action_space = Discrete(self.num_bins_motor * self.num_bins_steering)

    def action(self, action) -> dict:
        """Convert a flattened discrete action index to the environment's discrete action dict.
        """
        action_index = int(action)
        motor_index = action_index // self.num_bins_steering
        steering_index = action_index % self.num_bins_steering
        return {
            'motor': np.int64(motor_index),
            'steering': np.int64(steering_index),
        }
    
class DiscretizeObservation(gym.ObservationWrapper):
    """
    Wrap a continuous observation space and present a discrete observation space.
    """

    def __init__(self, env, lidar_bins: list = [0.0, 2.5, 5.0], num_velocity_bins: int = 10, num_acceleration_bins: Optional[int] = 3):
        """Initialize the discretization wrapper.
        Args:
            env: The environment to wrap.
            num_lidar_bins: Number of discrete bins for lidar distance readings.
            lidar_threshold: Maximum distance for lidar readings.
        """
        super().__init__(env)
        acceleration__min, acceleration_max = 0, 100.0
        velocity_min, velocity_max = 0, 2.0
        if num_acceleration_bins is None:
            self.acceleration_bins = None
        else:
            self.acceleration_bins = np.linspace(acceleration__min, acceleration_max, num_acceleration_bins + 1)
        self.velocity_bins = np.linspace(velocity_min, velocity_max, num_velocity_bins + 1)
        self.lidar_bins = np.array(lidar_bins)
        old_obs_space = self.env.observation_space
        if not isinstance(old_obs_space, Dict):
            raise ValueError("Expected observation space to be of type gym.spaces.Dict")
        new_obs_space = {}
        for key, space in old_obs_space.spaces.items():
            if space is None or space.shape is None:
                continue
            if key == 'lidar':
                num_lidar_rays = space.shape[0]
                new_obs_space[key] = MultiDiscrete([len(self.lidar_bins)] * num_lidar_rays)
            elif key == 'velocity':
                new_obs_space[key] = Discrete(len(self.velocity_bins))  # One value representing discretized velocity
            elif key == 'acceleration' and self.acceleration_bins is not None:
                new_obs_space[key] = Discrete(len(self.acceleration_bins))  # One value representing discretized acceleration
            else:
                # new_obs_space[key] = space
                continue # Do not include other observation components
        self.observation_space = Dict(new_obs_space)

    def observation(self, observation: dict) -> dict:
        new_obs = {}
        for key, value in observation.items():
            if key == 'lidar':
                discretized_lidar = np.digitize(value, self.lidar_bins) - 1
                new_obs[key] = discretized_lidar
            elif key == 'velocity':
                # get the vector norm
                target_velocity = np.linalg.norm(value[3:]) # use only translational velocity components
                discretized_velocity = np.digitize([target_velocity], self.velocity_bins) - 1
                new_obs[key] = discretized_velocity[0]
            elif key == 'acceleration' and self.acceleration_bins is not None:
                target_accel = value[3:] # use only translational velocity components
                accel_magnitude = np.linalg.norm(target_accel)
                discretized_acceleration = np.digitize([accel_magnitude], self.acceleration_bins) - 1
                new_obs[key] = discretized_acceleration[0]
            else:
                # new_obs[key] = value
                continue # Do not include other observation components
        return new_obs
    
class FlattenObservation(gym.ObservationWrapper):
    """Wrap a discrete observation space and present a flattened discrete observation space.
    """

    def __init__(self, env):
        super().__init__(env)
        if not isinstance(env.observation_space, Dict):
            raise ValueError("Expected observation space to be of type gym.spaces.Dict")
        self.obs_spaces = env.observation_space.spaces
        self.obs_sizes = []
        for key, space in self.obs_spaces.items():
            if isinstance(space, Discrete):
                self.obs_sizes.append(space.n)
            elif isinstance(space, MultiDiscrete):
                self.obs_sizes.append(int(np.prod(space.nvec)))
            else:
                raise ValueError("Expected observation subspaces to be of type Discrete or MultiDiscrete")
        self.total_size = int(np.prod(self.obs_sizes))
        self.observation_space = Discrete(self.total_size)

    def observation(self, observation: dict) -> int:
        """Convert a discrete observation dict to a flattened discrete observation index.
        """
        flat_index = 0
        multiplier = 1
        for i, (key, space) in enumerate(self.obs_spaces.items()):
            if isinstance(space, Discrete):
                value = int(observation[key])
                flat_index += value * multiplier
                multiplier *= space.n
            elif isinstance(space, MultiDiscrete):
                value = observation[key]
                # Convert multi-discrete value to flat index
                sub_index = 0
                sub_multiplier = 1
                for j in range(len(space.nvec)):
                    sub_index += value[j] * sub_multiplier
                    sub_multiplier *= space.nvec[j]
                flat_index += sub_index * multiplier
                multiplier *= int(np.prod(space.nvec))
        return flat_index
    
class FrameSkipping(gym.Wrapper):
    """Wrapper to skip frames in the environment.
    Makes the agent act only every 'skip' steps.
    Actions are "sticky" for the skipped frames, i.e., the last action is repeated.

    Makes the agent act more slowly
    """

    def __init__(self, env, skip: int = 4):
        super().__init__(env)
        self.skip = skip
        self.last_action = None

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        self.last_action = action
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            # Add info 
            if done:
                break
        return obs, total_reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        self.last_action = None
        return self.env.reset(**kwargs)
    