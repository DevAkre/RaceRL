import torch
import numpy as np

class RacecarDQNWrapper:
    """Converts racecar gym dictionary states into neural network inputs"""

    def __init__(self, use_lidar=True, use_pose=True, use_velocity=True,
                 use_acceleration=True, normalize=True):
        """
        Args:
            use_lidar: Whether to include lidar data
            use_pose: Whether to include pose data
            use_velocity: Whether to include velocity data
            use_acceleration: Whether to include acceleration data
            normalize: Whether to normalize the features
        """
        self.use_lidar = use_lidar
        self.use_pose = use_pose
        self.use_velocity = use_velocity
        self.use_acceleration = use_acceleration
        self.normalize = normalize

        self.feature_stats = {}
        self.initialized = False

    def preprocess(self, state):
        """
        Convert a single racecar gym state dictionary to a flat numpy array.

        Args:
            state: Dictionary with keys 'pose', 'lidar', 'velocity', 'acceleration', 'time'

        Returns:
            features: 1D numpy array of concatenated features
        """
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

    def get_state_dim(self, sample_state):
        """
        Calculate the total dimensionality of preprocessed state.

        Args:
            sample_state: A sample state dictionary from the environment

        Returns:
            state_dim: Integer representing total feature dimension
        """
        preprocessed = self.preprocess(sample_state)
        return len(preprocessed)

    def preprocess_batch(self, states):
        """
        Convert a list of state dictionaries to a batched tensor.

        Args:
            states: List of state dictionaries

        Returns:
            batch_tensor: Tensor of shape (batch_size, state_dim)
        """
        # Preprocess each state individually
        preprocessed_states = [self.preprocess(state) for state in states]

        # Stack into a batch
        batch_array = np.stack(preprocessed_states, axis=0)

        # Convert to tensor
        batch_tensor = torch.FloatTensor(batch_array)

        return batch_tensor

