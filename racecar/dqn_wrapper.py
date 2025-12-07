from itertools import product

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Dict


class DQNWrapper(gym.Wrapper):
    """
    A wrapper that applies DQN-specific action, observation, and reward shaping.
    """

    def __init__(
        self,
        env,
        use_lidar=True,
        use_pose=True,
        use_velocity=True,
        use_acceleration=True,
        normalize=True,
        num_bins_motor: int = 3,
        num_bins_steering: int = 5
    ):
        super().__init__(env)
        self.env = DQNActionWrapper(env, num_bins_motor, num_bins_steering)
        self.env = DQNObservationWrapper(
            self.env,
            use_lidar=use_lidar,
            use_pose=use_pose,
            use_velocity=use_velocity,
            use_acceleration=use_acceleration,
            normalize=normalize
        )
        self.env = DQNStepRacingWrapper(self.env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)


class DQNStepRacingWrapper(gym.Wrapper):
    """
    Balances speed, control, and crash avoidance.

    Philosophy:
    - Reward going fast (encourages movement)
    - Penalize crashes (but not so much agent freezes)
    - Penalize being stuck (forces action)
    - Penalize loss of control (teaches smooth driving)
    """

    def __init__(self, env):
        super().__init__(env)
        self.prev_velocity = None
        self.stuck_counter = 0
        self.prev_steering = 0

    def reset(self, **kwargs):
        self.prev_velocity = None
        self.stuck_counter = 0
        self.prev_steering = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Start with base reward
        shaped_reward = reward

        velocity_feature_indices = slice(3, 9)

        # 0-2 -> pose
        # 3-9 -> velocity
        # 10-1089 -> lidar

        velocity = observation[velocity_feature_indices]
        forward_velocity = velocity[0]
        lateral_velocity = velocity[1]

        motor, steering = self.env.allowed_actions[action]

        # 1. Moderate speed reward (not too aggressive)
        if forward_velocity > 0:
            shaped_reward += forward_velocity * 0.1  # Was 0.2, now 0.1

        # 2. Speed bonuses (conservative)
        if forward_velocity > 8.0:
            shaped_reward += 1.0  # Was 2.0
        elif forward_velocity > 4.0:
            shaped_reward += 0.3  # Was 0.8

        # 3. NEW: Crash penalty amplification
        if terminated and reward <= -3.0:
            shaped_reward -= 5.0  # Extra penalty for crashing

        # 4. NEW: Control penalty (sliding)
        if abs(lateral_velocity) > 5.0:
            shaped_reward -= 0.5

        # 5. NEW: Extreme steering penalty at speed
        if abs(steering) > 0.7 and forward_velocity > 6.0:
            shaped_reward -= 0.8  # Punish reckless turns

        # 6. Throttle bonus (keep this)
        if motor > 0.5 and forward_velocity < 8.0:
            shaped_reward += 0.3  # Was 0.5

        # Update history
        self.prev_velocity = velocity.copy()
        self.prev_steering = steering
        info['original_reward'] = reward
        info['shaped_reward'] = shaped_reward
        info['forward_velocity'] = forward_velocity
        info['stuck_counter'] = self.stuck_counter
        return observation, shaped_reward, terminated, truncated, info


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
        space_dim = 0
        for key, value in self.observation_space.spaces.items():
            if key == 'pose':
                space_dim += 3
            else:
                space_dim += value.shape[0]
        return space_dim

    def observation(self, state):
        features = []

        if self.use_pose:
            features.append(state['pose'][3:6])  # Use only x, y, yaw

        if self.use_acceleration:
            features.append(state['acceleration'])

        if self.use_velocity:
            features.append(state['velocity'])

        if self.use_lidar:
            features.append(state['lidar'])

        features = np.concatenate(features, axis=0).astype(np.float32)

        return features

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
