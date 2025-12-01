import gymnasium as gym
from typing import Optional, SupportsFloat
from itertools import product

import numpy as np
from gymnasium.spaces import Discrete, Dict, Box, MultiDiscrete, Tuple


class DiscreteActionWrapper(gym.ActionWrapper):
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


class DQNActionWrapper(gym.ActionWrapper):
    """Wrap a continuous action space and present a discrete action space.
    """

    def __init__(self, env, num_bins_motor: int = 3, num_bins_steering: int = 5):
        super().__init__(env)
        self.num_bins_motor = num_bins_motor
        self.num_bins_steering = num_bins_steering
        allowed_motor_values = np.linspace(-1.0, 1.0, num_bins_motor)
        allowed_steering_values = np.linspace(-1.0, 1.0, num_bins_steering)
        self.allowed_actions = [
           (m, s) for m,s in product(allowed_motor_values,allowed_steering_values)
        ]
        # print(f"DiscreteActionWrapper: allowed motor values: {allowed_motor_values}")
        # print(f"DiscreteActionWrapper: allowed steering values: {allowed_steering_values}")
        self._bin_size_motor = allowed_motor_values[1] - allowed_motor_values[0]
        self._bin_size_steering = allowed_steering_values[1] - allowed_steering_values[0]
        # Present actions as a Dict with separate Discrete entries for motor and steering
        self.action_space = Dict({
            'motor': Discrete(num_bins_motor),
            'steering': Discrete(num_bins_steering),
        })


    def action(self, action:int) -> dict:
        """
        Convert a discrete action (dict with 'motor' and 'steering' indices) to
        the environment's continuous action dict.
        """
        if action < 0:
            action = 0
        if action >= len(self.allowed_actions):
            action = len(self.allowed_actions) - 1


        motor_value, steering_value = self.allowed_actions[action]
        return {
            'motor': np.array(motor_value, dtype=np.float64),
            'steering': np.array(steering_value, dtype=np.float64),
        }

class VectorizedDiscreteActionWrapper(gym.ActionWrapper):
    """Wrap a continuous action space and present a discrete action space in a vectorized environment.
        Uses DiscreteActionWrapper for each individual environment.
    """
    def __init__(self, envs, num_bins_motor: int = 3, num_bins_steering: int = 5):
        super().__init__(envs)
        self.env = envs
        self.num_envs = envs.num_envs
        self.num_bins_motor = num_bins_motor
        self.num_bins_steering = num_bins_steering
        self.wrapper = DiscreteActionWrapper(self.env.single_env, num_bins_motor, num_bins_steering)
        self.single_env = self.wrapper
        # Action space consisting of all actions from individual wrappers
        self.action_space = Tuple([self.wrapper.action_space for _ in range(self.num_envs)])

    def action(self, action) -> Tuple:
        """Convert vectorized discrete actions to continuous actions for each environment.
        """
        continuous_actions = []
        for i in range(self.num_envs):
            continuous_action = self.wrapper.action(action[i])
            continuous_actions.append(continuous_action)
        return Tuple(continuous_actions)

    def step(self, action):
        return self.env.step(action)

class DiscretizeObservationWrapper(gym.ObservationWrapper):
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
        velocity_min, velocity_max = 0, 5.0
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
                new_obs_space[key] = MultiDiscrete([len(self.lidar_bins) - 1] * num_lidar_rays)
            elif key == 'velocity':
                new_obs_space[key] = Discrete(1) # One value representing discretized velocity
            elif key == 'acceleration' and self.acceleration_bins is not None:
                new_obs_space[key] = Discrete(1) # One value representing discretized acceleration
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


class DQNObservationWrapper(gym.ObservationWrapper):
    """
    Wrap the observation to convert it to a flat numpy array for DQN.
    """

    def __init__(self, env, use_lidar=True, use_pose=True, use_velocity=True,
                 use_acceleration=True, normalize=True):
        """
        Args:
            use_lidar: Whether to include lidar data
            use_pose: Whether to include pose data
            use_velocity: Whether to include velocity data
            use_acceleration: Whether to include acceleration data
            normalize: Whether to normalize the features
        """
        super().__init__(env)
        self.use_lidar = use_lidar
        self.use_pose = use_pose
        self.use_velocity = use_velocity
        self.use_acceleration = use_acceleration
        self.normalize = normalize

        self.feature_stats = {}
        self.initialized = False

        old_obs_space = self.env.observation_space
        if not isinstance(old_obs_space, Dict):
            raise ValueError("Expected observation space to be of type gym.spaces.Dict")
        new_obs_space = {}
        for key, space in old_obs_space.spaces.items():
            if use_lidar and key == 'lidar':
                new_obs_space[key] = space
            elif use_pose and key == 'pose':
                new_obs_space[key] = space
            elif use_velocity and key == 'velocity':
                new_obs_space[key] = space
            elif use_acceleration and key == 'acceleration':
                new_obs_space[key] = space
        self.observation_space = Dict(new_obs_space)
        self.observation_space_dim = self._get_observation_space_dim()

    def _get_observation_space_dim(self):
        return sum(
            [val.shape[0] for val in self.observation_space.values()]
        )

    def observation(self, state):
        features = []

        if self.use_pose:
            features.append(state['pose'])

        if self.use_acceleration:
            features.append(state['acceleration'])

        if self.use_velocity:
            features.append(state['velocity'])

        if self.use_lidar:
            features.append(state['lidar'])

        features = np.concatenate(features, axis=0).astype(np.float32)

        return features




class VectorizedDiscretizeObservationWrapper(gym.ObservationWrapper):
    """
    Wrap a continuous observation space and present a discrete observation space in a vectorized environment.
    Uses DiscretizeObservationWrapper for each individual environment.
    """

    def __init__(self, envs, lidar_bins: list = [0.0, 2.5, 5.0], num_velocity_bins: int = 10, num_acceleration_bins: Optional[int] = 3):
        """Initialize the discretization wrapper.
        Args:
            envs: The vectorized environment to wrap.
            num_lidar_bins: Number of discrete bins for lidar distance readings.
            lidar_threshold: Maximum distance for lidar readings.
        """
        super().__init__(envs)
        self.envs = envs
        self.num_envs = envs.num_envs
        self.wrapper = DiscretizeObservationWrapper(self.env.single_env, lidar_bins, num_velocity_bins, num_acceleration_bins)
        self.single_env = self.wrapper
        self.observation_space = Tuple([self.wrapper.observation_space for _ in range(self.num_envs)])

    def observation(self, observation: Tuple) -> Tuple:
        """Convert vectorized observations to discrete observations for each environment.
        """
        discrete_observations = []
        for i in range(self.num_envs):
            discrete_obs = self.wrapper.observation(observation[i])
            discrete_observations.append(discrete_obs)
        return Tuple(discrete_observations)

    def step(self, action):
        return self.envs.step(action)

    def reset(self, **kwargs):
        return self.envs.reset(**kwargs)



class VectorizedRecordEpisodeStatistics(gym.wrappers.RecordEpisodeStatistics):
    """
    Vectorized version of RecordEpisodeStatistics wrapper.
    Records episode statistics (return, length) for each environment in a vectorized env.
    """

    def __init__(self, env, buffer_length: int = 100):
        super().__init__(env, buffer_length)
        self.num_envs = env.num_envs
        self.single_env = env.single_env
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

    def step(self, action):
        observations, rewards, terminations, truncations, infos = self.env.step(action)
        self.episode_returns += rewards
        self.episode_lengths += 1

        for i in range(self.num_envs):
            if terminations[i] or truncations[i]:
                info = infos[i]
                info['episode'] = {
                    'r': self.episode_returns[i],
                    'l': self.episode_lengths[i],
                }
                self.returns.append(self.episode_returns[i])
                self.lengths.append(self.episode_lengths[i])
                self.episode_returns[i] = 0.0
                self.episode_lengths[i] = 0

        return observations, rewards, terminations, truncations, infos

class RewardWrapper(gym.RewardWrapper):
    """
    Wrap the reward to modify it.
    """

    def __init__(self, env, crash_penalty: float = -100.0, living_penalty: float = -1.0):
        """Initialize the reward wrapper.
        Args:
            env: The environment to wrap.
            crash_penalty: Penalty to apply on crash (termination).
        """
        super().__init__(env)
        self.crash_penalty = crash_penalty
        self.living_penalty = living_penalty

    def reward(self, reward: SupportsFloat) -> float:
        return float(reward)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Ensure reward is a concrete float before arithmetic to satisfy type checkers
        reward = float(reward)
        if terminated:
            reward = reward + float(self.crash_penalty)
        reward = reward + float(self.living_penalty)
        return obs, reward, terminated, truncated, info
