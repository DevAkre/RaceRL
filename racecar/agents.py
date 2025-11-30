from collections import defaultdict
import gymnasium as gym
import numpy as np
import pickle
from typing import Any, Callable, Hashable, Optional, Tuple, Union
from typing import Dict as TypingDict, List
import math


class RacecarAgent:
    """Base class for racecar agents.

    This class holds shared configuration and provides an interface that
    concrete agents should implement.
    """

    def __init__(
        self,
        env: Union[gym.Env, gym.vector.VectorEnv],
        learning_rate: float = 0.1,
        initial_epsilon: float = 1.0,
        epsilon_decay: float = 1e-3,
        final_epsilon: float = 0.1,
        discount_factor: float = 0.95,
        batch_size: int = 1
    ) -> None:
        self.env = env
        self.batch_size = batch_size
        # Basic hyperparameters commonly used by RL agents
        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Validate if action space is flat and Discrete
        if not self.set_up_action_space():
            raise ValueError("Environment action space must be discrete")
        self.num_states = None  # To be set by concrete agents if needed
    
    def set_up_action_space(self) -> bool:
        """Check if the environment's action space is supported (Discrete or Dict of Discrete)."""
        action_space = self.env.action_space
        if self.batch_size == 1:
            if isinstance(action_space, gym.spaces.Discrete):
                self.num_actions = action_space.n
        elif self.batch_size > 1:
            if isinstance(action_space, gym.spaces.MultiDiscrete):
                self.num_actions = max(action_space.nvec)
        if self.num_actions is not None:
            return True
        return False

    def get_action(self, obs: Any) -> Any:
        """Return an action given an observation. Must be implemented by subclasses."""
        raise NotImplementedError()

    def update(
        self, obs: Any, action: Any, reward: float, terminated: bool, next_obs: Any
    ) -> None:
        """Update agent internals from a transition. Optional for on-policy agents."""
        raise NotImplementedError()

    def decay_epsilon(self) -> None:
        """Decay exploration parameter after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save(self, path: str) -> None:
        """Save agent to disk. Subclasses may override to include more fields."""
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, path: str) -> None:
        """Load agent state from disk. Subclasses overriding should call super()."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        # update internal dict with saved values
        self.__dict__.update(data)


class QLearningRacecarAgent(RacecarAgent):
    """Tabular Q-learning agent for environments with discrete actions.

    This agent uses a defaultdict to hold Q-values and supports epsilon-greedy
    action selection. Supports both single and batch updates.
    """

    def __init__(
        self,
        env: Union[gym.Env, gym.vector.VectorEnv],
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
        if self.num_actions is None:
            raise ValueError("QLearningRacecarAgent requires a discrete action space or Dict of Discrete subspaces")
        # Use captured action_n inside the defaultdict to avoid static type issues
        self.q_values = defaultdict(lambda: np.zeros(self.num_actions))
        self.action_space = gym.spaces.Discrete(self.num_actions)
    
    def get_action(self, obs: Any, explore: bool = True) -> Any:
        """Return an action for the given observation using epsilon-greedy policy.

        Args:
            obs: The current observation (state).
            explore: If True, use epsilon-greedy exploration; otherwise, exploit.
        Returns:
            The selected action.
        """
        if self.batch_size == 1:
            return self._get_action_single(obs, explore)
        elif self.batch_size > 1:
            actions = np.empty(self.batch_size, dtype=object)
            for i in range(self.batch_size):
                actions[i] = self._get_action_single(obs[i], explore)
            return actions
    
    def _get_action_single(self, obs: Any, explore: bool = True) -> Any:
        if explore and np.random.rand() < self.epsilon:
            # Explore: random action
            action = self.action_space.sample()
        else:
            # Exploit: best known action
            q_vals = self.q_values[obs]
            action = np.argmax(q_vals)
        return action

    def update(
        self, obs, action, reward, terminated, next_obs
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
            self._update_single(obs, action, reward, terminated, next_obs)
        elif self.batch_size > 1:
            # Batch update
            for i in range(self.batch_size):
                self._update_single(
                    obs[i], action[i], reward[i], terminated[i], next_obs[i]
                )

    def _update_single(
        self, obs: Any, action: Any, reward: float, terminated: bool, next_obs: Any
    ) -> None:
        """Update Q-values from a single transition."""
        state = obs
        next_state = next_obs

        # Current Q-value
        current_q = self.q_values[state][action]

        # Compute TD target
        if terminated:
            td_target = reward
        else:
            td_target = reward + self.discount_factor * np.max(self.q_values[next_state])

        # TD error
        td_error = td_target - current_q

        # Q-value update
        self.q_values[state][action] = current_q + self.lr * td_error

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
        self.q_values = defaultdict(lambda: np.zeros(self.num_actions))
        # copy saved entries
        for k, v in qdict.items():
            self.q_values[k] = np.array(v, dtype=float)

        # restore basic params
        self.lr = payload.get("lr", self.lr)
        self.discount_factor = payload.get("discount_factor", self.discount_factor)
        self.epsilon = payload.get("epsilon", self.epsilon)
        self.epsilon_decay = payload.get("epsilon_decay", self.epsilon_decay)
        self.final_epsilon = payload.get("final_epsilon", self.final_epsilon)
