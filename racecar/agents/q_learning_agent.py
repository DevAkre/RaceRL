import math
from collections import defaultdict
from typing import Any, Union

import pickle
import gymnasium as gym
import numpy as np

from .base_agent import RacecarAgent

class QLearningRacecarAgent(RacecarAgent):
    """Tabular Q-learning agent for environments with discrete actions.

    This agent uses a defaultdict to hold Q-values and supports epsilon-greedy
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
        batch_size: int = 1
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
        self.batch_size = batch_size
        action_n = self._flat_action_n
        if action_n is None:
            raise ValueError("QLearningRacecarAgent requires a discrete action space or Dict of Discrete subspaces")
        # Use captured action_n inside the defaultdict to avoid static type issues
        self.q_values = defaultdict(lambda: np.zeros(action_n))
        self.training_error = []
        # precompute multipliers for flattening indices if dict action
        if self._action_is_dict:
            # multipliers[i] = product of sizes for subsequent dims
            multipliers = []
            for i in range(len(self._action_sizes)):
                if i + 1 < len(self._action_sizes):
                    multipliers.append(int(math.prod(self._action_sizes[i + 1 :])))
                else:
                    multipliers.append(1)
            self._action_multipliers = multipliers

    def get_action(self, obs: Any, explore: bool = True) -> Any:
        state = self.discretizer(obs)

        # Epsilon-greedy
        if explore and np.random.random() < self.epsilon:
            sample = self.env.action_space.sample()
            # sample may be dict for Dict action space; return as-is
            return sample

        flat = int(np.argmax(self.q_values[state]))
        # Convert flat index into environment action format
        return self._flat_index_to_action(flat)

    def update(
        self, obs: Any, action, reward: float, terminated: bool, next_obs: Any
    ) -> None:
        """Update Q-values from a transition or batch of transitions.

        For batch_size=1 (default): expects single values for obs, action, reward, terminated, next_obs.
        For batch_size>1: expects obs and next_obs as tuples/lists of observations (dicts),
                         action as tuple/list of actions (dicts),
                         reward as tuple/list of floats,
                         terminated as tuple/list of bools.
        """
        if self.batch_size == 1:
            # Single transition update
            state = self.discretizer(obs)
            next_state = self.discretizer(next_obs)

            future_q_value = (not terminated) * np.max(self.q_values[next_state])
            flat_action = self._action_to_flat_index(action)
            target = reward + self.discount_factor * future_q_value
            td_error = target - self.q_values[state][flat_action]

            self.q_values[state][flat_action] = (
                self.q_values[state][flat_action] + self.lr * td_error
            )
            self.training_error.append(td_error)
        else:
            # Batch update: expect tuples/lists of length batch_size
            try:
                obs_batch = list(obs) if not isinstance(obs, list) else obs
                next_obs_batch = list(next_obs) if not isinstance(next_obs, list) else next_obs
                action_batch = list(action) if not isinstance(action, list) else action
                # For reward and terminated, handle scalar or iterable
                if isinstance(reward, (list, tuple)):
                    reward_batch = list(reward)
                else:
                    # single scalar passed; replicate for batch
                    reward_batch = [float(reward)] * self.batch_size
                if isinstance(terminated, (list, tuple)):
                    terminated_batch = list(terminated)
                else:
                    terminated_batch = [bool(terminated)] * self.batch_size
            except Exception as e:
                raise ValueError("For batch updates, obs/action/next_obs/reward/terminated must be tuples or lists") from e

            # Process each transition in the batch
            for o, a, r, term, no in zip(obs_batch, action_batch, reward_batch, terminated_batch, next_obs_batch):
                state = self.discretizer(o)
                next_state = self.discretizer(no)

                future_q_value = (not term) * np.max(self.q_values[next_state])
                flat_action = self._action_to_flat_index(a)
                target = float(r) + self.discount_factor * future_q_value
                td_error = target - self.q_values[state][flat_action]

                self.q_values[state][flat_action] = (
                    self.q_values[state][flat_action] + self.lr * td_error
                )
                self.training_error.append(td_error)

    def save(self, path: str) -> None:
        """Save Q-table and agent parameters to disk."""
        payload = {
            "q_values": dict(self.q_values),
            "lr": self.lr,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "final_epsilon": self.final_epsilon
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path: str) -> None:
        """Load Q-table and parameters from disk."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        # Replace q_values with a defaultdict again, capture action count safely
        qdict = payload.get("q_values", {})
        # Recompute action mapping from current env action space using helper
        try:
            # attempt to (re)setup mapping; this will raise if the action space is not supported
            self._setup_action_space()
            if self._flat_action_n is None:
                raise ValueError()
        except Exception:
            raise ValueError("Environment action space must be discrete or Dict of Discrete to load Q-values")

        # self._flat_action_n is guaranteed to be set by _setup_action_space above
        assert self._flat_action_n is not None
        flat_n = int(self._flat_action_n)
        self.q_values = defaultdict(lambda: np.zeros(flat_n))
        # copy saved entries
        for k, v in qdict.items():
            self.q_values[k] = np.array(v, dtype=float)

        # restore basic params
        self.lr = payload.get("lr", self.lr)
        self.discount_factor = payload.get("discount_factor", self.discount_factor)
        self.epsilon = payload.get("epsilon", self.epsilon)
        self.epsilon_decay = payload.get("epsilon_decay", self.epsilon_decay)
        self.final_epsilon = payload.get("final_epsilon", self.final_epsilon)
        self.training_error = payload.get("training_error", [])
