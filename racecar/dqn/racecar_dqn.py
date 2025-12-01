import torch
import random
from collections import deque

import numpy as np
from torch import nn
from torch.nn.functional import relu


class RacecarDQN(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_sizes=[512, 256]):
        """
        DQN for racecar gym with vector state inputs.

        Args:
            state_dim: Dimension of preprocessed state vector
            num_actions: Number of discrete actions
            hidden_sizes: List of hidden layer sizes
        """
        super(RacecarDQN, self).__init__()

        self.state_dim = state_dim
        self.num_actions = num_actions

        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, num_actions))

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        """
        Forward pass.

        Args:
            state: Tensor of shape (batch_size, state_dim)
        Returns:
            q_values: Tensor of shape (batch_size, num_actions)
        """
        return self.network(state)

    def select_action(self, state, epsilon=0.0):
        """
        Select action using epsilon-greedy policy.
        Args:
            state: Preprocessed state array or tensor
            epsilon: Exploration rate
        Returns:
            action: Integer action index
        """
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.num_actions)
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state)
            if state.dim() == 1:
                state = state.unsqueeze(0)  # Add batch dimension
            q_values = self.forward(state)
            action = q_values.argmax(dim=1).item()

        return action


class RacecarReplayBuffer:
    """Replay buffer that handles dictionary states"""

    def __init__(self, capacity):
        """
        Args:
            capacity: Maximum buffer size
            state_preprocessor: RacecarStatePreprocessor instance
        """
        self.buffer = deque(maxlen=capacity)
        #self.preprocessor = state_preprocessor

    def add(self, state, action, reward, next_state, done):
        """
        Add transition to buffer.

        Args:
            state: Dictionary state from environment
            action: Integer action
            reward: Float reward
            next_state: Dictionary next state
            done: Boolean terminal flag
        """
        # Store raw dictionary states
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a batch and preprocess states.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        # Sample transitions
        batch = random.sample(self.buffer, batch_size)

        # Separate components
        states, actions, rewards, next_states, dones = zip(*batch)

        # Preprocess state dictionaries into batched tensors
        states_tensor = torch.FloatTensor(np.stack(states, axis=0))
        next_states_tensor = torch.FloatTensor(np.stack(next_states, axis=0))

        #states_tensor = self.preprocessor.preprocess_batch(states)
        #next_states_tensor = self.preprocessor.preprocess_batch(next_states)

        # Convert other components to tensors
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        dones_tensor = torch.FloatTensor(dones)

        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor

    def __len__(self):
        return len(self.buffer)
