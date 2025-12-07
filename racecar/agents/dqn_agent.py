import torch
import random
from collections import deque
from pathlib import Path

import numpy as np
import gymnasium as gym
from torch import nn
from torch.nn.functional import relu
from .base_agent import RacecarAgent


from dqn_wrapper import DQNActionWrapper, DQNObservationWrapper


class DQNNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_sizes=[512, 256]):
        """
        DQN for racecar gym with vector state inputs.

        Args:
            state_dim: Dimension of preprocessed state vector
            num_actions: Number of discrete actions
            hidden_sizes: List of hidden layer sizes
        """
        super(DQNNetwork, self).__init__()

        self.state_dim = state_dim
        self.num_actions = num_actions

        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, self.num_actions))

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

    def select_action(self, state, epsilon=0.0, deterministic=False):
        """
        Select action using epsilon-greedy policy.
        Args:
            state: Preprocessed state array or tensor
            epsilon: Exploration rate
        Returns:
            action: Integer action index
        """
        if not deterministic:
            if np.random.rand() < epsilon:
                return np.random.randint(0, self.num_actions), []
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state)
            if state.dim() == 1:
                state = state.unsqueeze(0)  # Add batch dimension
            q_values = self.forward(state)
            action = q_values.argmax(dim=1).item()

        return action, q_values.squeeze()


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


class DQNRaceCarAgent(RacecarAgent):
    """Tabular Double Q-learning agent for environments with discrete actions.

    This agent uses two defaultdicts to hold Q-values and supports epsilon-greedy
    action selection. Supports both single and batch updates.
    """

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.1,
        initial_epsilon: float = 1.0,
        epsilon_decay: float = 1e-3,
        final_epsilon: float = 0.1,
        discount_factor: float = 0.95,
        batch_size: int = 64,
        num_bins_motor:int=5,
        num_bins_steering:int=5,
        num_episodes:int=1000,
        target_update_freq:int=10,
        best_model_dir_path: Path = Path("./best_dqn_model")
    ) -> None:
        super().__init__(
            env,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
            batch_size=batch_size
        )
        self.env = DQNActionWrapper(DQNObservationWrapper(env), num_bins_motor=num_bins_motor,
                                    num_bins_steering=num_bins_steering)
        self.num_episodes = num_episodes
        self.target_update_freq = target_update_freq
        self.total_steps = 0
        self.episode_rewards = []
        self.best_model_dir_path = best_model_dir_path
        self.best_model_dir_path.mkdir(parents=True, exist_ok=True)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.initialize_networks()

    def initialize_networks(self):
        state_dim = self.env.observation_space_dim
        num_actions = len(self.env.allowed_actions)
        self.online_net = DQNNetwork(state_dim, num_actions).to(self.device)
        self.target_net = DQNNetwork(state_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.replay_buffer = RacecarReplayBuffer(capacity=10000)
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()


    def train(self):
        best_reward = -float('inf')
        for episode in range(self.num_episodes):
            reward = self.train_one_episode()
            if reward > best_reward:
                print(f"New best model with reward {reward} at episode {episode}")
                best_reward = reward
                self.save(self.best_model_dir_path/"best_dqn_model.pth")
            self.episode_rewards.append(reward)
            self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

    def train_one_episode(self):
        state, info = self.env.reset()
        done = False
        episode_reward = 0
        while not done:
            # Select action
            action, _ = self.online_net.select_action(torch.FloatTensor(state).to(self.device), self.epsilon)

            # Execute action (mock for demonstration)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            reward = np.random.randn()
            done = np.random.rand() < 0.01  # Random termination

            # Store transition
            self.replay_buffer.add(state, action, reward, next_state, done)

            # Training step
            if len(self.replay_buffer) >= self.batch_size:
                # Sample batch
                states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = \
                    self.replay_buffer.sample(self.batch_size)

                # Move to device
                states_batch = states_batch.to(self.device)
                actions_batch = actions_batch.to(self.device)
                rewards_batch = rewards_batch.to(self.device)
                next_states_batch = next_states_batch.to(self.device)
                dones_batch = dones_batch.to(self.device)

                # Compute current Q-values
                current_q_values = self.online_net(states_batch)
                current_q = current_q_values.gather(1, actions_batch.unsqueeze(1)).squeeze()

                # Compute target Q-values
                with torch.no_grad():
                    next_q_values = self.target_net(next_states_batch)
                    max_next_q = next_q_values.max(dim=1)[0]
                    target_q = rewards_batch + self.discount_factor * max_next_q * (1 - dones_batch)

                # Compute loss and update
                loss = self.loss_fn(current_q, target_q)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10)
                self.optimizer.step()

                    # Update target network
            if self.total_steps % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())
                print(f"Target network updated at step {self.total_steps}")

            state = next_state
            episode_reward += reward
            self.total_steps += 1
            return episode_reward

    def save(self, path: Path) -> None:
        """Save Q-tables and agent parameters to disk."""
        payload = {
            'online_net_state_dict': self.online_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            "lr": self.lr,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "final_epsilon": self.final_epsilon,
            'episode_rewards': self.episode_rewards,
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        """Load Q-tables and parameters from disk."""
        payload = torch.load(path, map_location='cpu')
        self.online_net.load_state_dict(payload['online_net_state_dict'])
        self.online_net.cpu().eval()
        self.target_net.load_state_dict(payload['target_net_state_dict'])
        self.target_net.eval()
        # Load optimizer state
        #self.optimizer.load_state_dict(payload['optimizer_state_dict'])
        self.lr = payload.get("lr", self.lr)
        self.discount_factor = payload.get("discount_factor", self.discount_factor)
        self.epsilon = payload.get("epsilon", self.epsilon)
        self.epsilon_decay = payload.get("epsilon_decay", self.epsilon_decay)
        self.final_epsilon = payload.get("final_epsilon", self.final_epsilon)
        self.training_error = payload.get("training_error", [])

    def run_episode(self, render=False, verbose=True):
        """
        Run one complete episode.
        Args:
            env: Gymnasium environment
            render: Whether to render
            verbose: Whether to print step info
        Returns:
            Dictionary with episode statistics
        """
        state, info = self.env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        episode_q_values = []
        episode_actions = []

        while not done:
            if render:
                self.env.render()

            # Select action
            action, q_values = self.online_net.select_action(state, deterministic=True)

            # Store for analysis
            episode_q_values.append(q_values)
            episode_actions.append(action)

            # Execute
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            episode_reward += reward
            state = next_state
            step_count += 1

            if verbose and step_count % 100 == 0:
                print(f"  Step {step_count}: Reward={episode_reward:.2f}, "
                      f"Max Q={q_values.max():.2f}, Min Q={q_values.min():.2f}")

        if verbose:
            print(f"Episode finished: {step_count} steps, reward={episode_reward:.2f}")

        return {
            'reward': episode_reward,
            'steps': step_count,
            'q_values': episode_q_values,
            'actions': episode_actions,
            'avg_q': np.mean([q.mean() for q in episode_q_values]),
            'max_q': np.max([q.max() for q in episode_q_values]),
        }
