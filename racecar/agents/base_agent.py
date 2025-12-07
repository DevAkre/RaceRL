import math
from enum import Enum
from typing import (
    Any, Optional, Union, Callable, Hashable, List, Tuple
)

import gymnasium as gym
import pickle
import numpy as np


class RaceCarAgentType(Enum):
    QLearningRacecarAgent = 'q_learning'
    DoubleQLearningRaceCarAgent = 'double_q_learning'


class RacecarAgent:
    """Base class for racecar agents.

    This class holds shared configuration and provides an interface that
    concrete agents should implement. It also provides a very small
    discretizer helper suitable for turning continuous observations into
    hashable states for tabular methods.
    """

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.1,
        initial_epsilon: float = 1.0,
        epsilon_decay: float = 1e-3,
        final_epsilon: float = 0.1,
        discount_factor: float = 0.95,
        discretizer: Optional[Callable[[Any], Hashable]] = None,
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

        # Optional discretizer: maps raw obs -> hashable state key
        # If not provided, a default coarse discretizer will be used.
        self.discretizer = discretizer or self.default_discretizer

        # Action-space helpers (support Dict of Discrete subspaces)
        self._action_is_dict = False
        self._action_keys: List[str] = []
        self._action_sizes: List[int] = []
        self._action_multipliers: List[int] = []
        self._flat_action_n: Optional[int] = None
        # Setup action mapping from the environment's action_space
        try:
            self._setup_action_space()
        except Exception:
            # If action space isn't discrete or is unknown, leave as None and let
            # concrete agents validate if they require discrete actions.
            self._flat_action_n = None
        self.num_actions = self._flat_action_n
        self.num_states = None  # To be set by concrete agents if needed


    # Helper to set up internal flat action count and mapping multipliers
    def _setup_from_env_space(self, space) -> int:
        # Single Discrete space
        n = getattr(space, "n", None)
        if n is not None:
            return int(n)

        # Dict space: expect subspaces to be Discrete
        if hasattr(space, "spaces") and isinstance(space.spaces, dict):
            self._action_is_dict = True
            self._action_keys = list(space.spaces.keys())
            sizes = []
            for k in self._action_keys:
                sub = space.spaces[k]
                sub_n = getattr(sub, "n", None)
                if sub_n is None:
                    raise ValueError("RLAgent requires Discrete subspaces in Dict action space")
                sizes.append(int(sub_n))
            self._action_sizes = sizes
            # flat action count is product of subspace sizes
            return int(math.prod(sizes))

        raise ValueError("RLAgent requires a discrete action space or Dict of Discrete subspaces")

    def _setup_action_space(self) -> None:
        """Inspect self.env.action_space and prepare mapping helpers.

        Supports a single Discrete action space or a gym.spaces.Dict of Discrete
        subspaces. Computes flat action count and multipliers for index math.
        """
        space = getattr(self.env, "action_space", None)
        if space is None:
            raise ValueError("Environment has no action_space")

        if self.batch_size == 1:
            self._setup_single_action_space(space)
            return
        else:
            # For batch_size > 1, expect Tuple of action spaces
            if not (hasattr(space, "spaces") and isinstance(space.spaces, (list, tuple, Tuple))):
                raise ValueError("For batch_size > 1, environment action_space must be a Tuple of action spaces")
            # Setup based on the single_env's action space
            single_env = getattr(self.env, "single_env", None)
            single_space = getattr(single_env, "action_space", None)
            print(single_space)
            if single_space is None:
                raise ValueError("Vectorized environment has no single_env.action_space")
            self._setup_single_action_space(single_space)

    def _setup_single_action_space(self, space) -> None:
        # Single Discrete
        n = getattr(space, "n", None)
        if n is not None:
            self._action_is_dict = False
            self._flat_action_n = int(n)
            self._action_keys = []
            self._action_sizes = []
            self._action_multipliers = []
            return

        # Dict of Discrete subspaces
        if hasattr(space, "spaces") and isinstance(getattr(space, "spaces"), dict):
            self._action_is_dict = True
            self._action_keys = list(space.spaces.keys())
            sizes: List[int] = []
            for k in self._action_keys:
                sub = space.spaces[k]
                sub_n = getattr(sub, "n", None)
                if sub_n is None:
                    raise ValueError("QLearningRacecarAgent requires Discrete subspaces in Dict action space")
                sizes.append(int(sub_n))
            self._action_sizes = sizes
            # flat action count is product of subspace sizes
            self._flat_action_n = int(math.prod(self._action_sizes))

            # multipliers: product of sizes for subsequent dims
            multipliers: List[int] = []
            for i in range(len(self._action_sizes)):
                if i + 1 < len(self._action_sizes):
                    multipliers.append(int(math.prod(self._action_sizes[i + 1 :])))
                else:
                    multipliers.append(1)
            self._action_multipliers = multipliers
            return

        raise ValueError("Environment action space must be Discrete or Dict of Discrete subspaces")

    def _flat_index_to_action(self, index: int):
        """Convert a flat integer index into an action accepted by the env.

        Returns either an int (for single Discrete) or a dict mapping subspace
        names to indices for Dict action spaces.
        """
        if self._flat_action_n is None:
            raise ValueError("Action mapping not set up for this agent")

        if not self._action_is_dict:
            return int(index)

        idx = int(index)
        result = {}
        for key, size, mult in zip(self._action_keys, self._action_sizes, self._action_multipliers):
            comp = idx // mult
            result[key] = int(comp % size)
            idx = idx - comp * mult
        return result

    def _action_to_flat_index(self, action) -> int:
        """Convert an env-format action (int, dict, or sequence) into a flat index."""
        if self._flat_action_n is None:
            raise ValueError("Action mapping not set up for this agent")

        if not self._action_is_dict:
            return int(action)

        try:
            if isinstance(action, dict):
                indices = [int(action[k]) for k in self._action_keys]
            else:
                indices = [int(x) for x in action]
        except Exception as e:
            raise ValueError(f"Action must be a mapping with keys: {self._action_keys}") from e

        flat = 0
        for ind, mult, size in zip(indices, self._action_multipliers, self._action_sizes):
            clamped = max(0, min(ind, size - 1))
            flat += int(clamped) * int(mult)
        return int(flat)

    def default_discretizer(self, obs: Any) -> Hashable:
        """Default discretizations:
        1. Round numeric observations to 2 decimals
        2. Convert list/tuple/ndarray observations to tuple of rounded values
        3. Convert dict observations to tuple of (key, rounded value) pairs
        """
        if isinstance(obs, (int, float, np.number)):
            return round(float(obs), 2)

        if isinstance(obs, (list, tuple, np.ndarray)):
            arr = np.asarray(obs, dtype=float)
            return tuple(np.round(arr, 2).tolist())

        if isinstance(obs, dict):
            items = []
            for k in sorted(obs.keys()):
                v = obs[k]
                if isinstance(v, (int, float, np.number)):
                    items.append((k, round(float(v), 2)))
                elif isinstance(v, (list, tuple, np.ndarray)):
                    arr = np.asarray(v, dtype=float)
                    items.append((k, tuple(np.round(arr, 2).tolist())))
                else:
                    items.append((k, v))
            return tuple(items)

        # Fallback to using the observation as-is if it's already hashable
        return obs

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
