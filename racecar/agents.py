from collections import defaultdict
import gymnasium as gym
import numpy as np
import pickle
from typing import Any, Callable, Hashable, Optional, Tuple
from typing import Dict as TypingDict, List
import math
from enum import Enum

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


class DoubleQLearningRaceCarAgent(RacecarAgent):
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
            raise ValueError("DoubleQLearningRacecarAgent requires a discrete action space or Dict of Discrete subspaces")
        # Use captured action_n inside the defaultdict to avoid static type issues
        self.q1_values = defaultdict(lambda: np.zeros(action_n))
        self.q2_values = defaultdict(lambda: np.zeros(action_n))
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

        q_sum = self.q1_values[state] + self.q2_values[state]
        flat = int(np.argmax(q_sum))
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

            if np.random.random() < 0.5:
                # Update Q1
                best_next_action = int(np.argmax(self.q1_values[next_state]))
                future_q_value = (not terminated) * self.q2_values[next_state][best_next_action]
                flat_action = self._action_to_flat_index(action)
                target = reward + self.discount_factor * future_q_value
                td_error = target - self.q1_values[state][flat_action]

                self.q1_values[state][flat_action] += self.lr * td_error
            else:
                # Update Q2
                best_next_action = int(np.argmax(self.q2_values[next_state]))
                future_q_value = (not terminated) * self.q1_values[next_state][best_next_action]
                flat_action = self._action_to_flat_index(action)
                target = reward + self.discount_factor * future_q_value
                td_error = target - self.q2_values[state][flat_action]

                self.q2_values[state][flat_action] += self.lr * td_error

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

                if np.random.random() < 0.5:
                    # Update Q1
                    best_next_action = int(np.argmax(self.q1_values[next_state]))
                    future_q_value = (not term) * self.q2_values[next_state][best_next_action]
                    flat_action = self._action_to_flat_index(a)
                    target = float(r) + self.discount_factor * future_q_value
                    td_error = target - self.q1_values[state][flat_action]

                    self.q1_values[state][flat_action] += self.lr * td_error
                else:
                    # Update Q2
                    best_next_action = int(np.argmax(self.q2_values[next_state]))
                    future_q_value = (not term) * self.q1_values[next_state][best_next_action]
                    flat_action = self._action_to_flat_index(a)
                    target = float(r) + self.discount_factor * future_q_value
                    td_error = target - self.q2_values[state][flat_action]

                    self.q2_values[state][flat_action] += self.lr * td_error

                self.training_error.append(td_error)

    def save(self, path: str) -> None:
        """Save Q-tables and agent parameters to disk."""
        payload = {
            "q1_values": dict(self.q1_values),
            "q2_values": dict(self.q2_values),
            "lr": self.lr,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "final_epsilon": self.final_epsilon
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path: str) -> None:
        """Load Q-tables and parameters from disk."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        # Replace q_values with defaultdicts again, capture action count safely
        q1dict = payload.get("q1_values", {})
        q2dict = payload.get("q2_values", {})
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
        self.q1_values = defaultdict(lambda: np.zeros(flat_n))
        self.q2_values = defaultdict(lambda: np.zeros(flat_n))
        # copy saved entries
        for k, v in q1dict.items():
            self.q1_values[k] = np.array(v, dtype=float)
        for k, v in q2dict.items():
            self.q2_values[k] = np.array(v, dtype=float)
        # restore basic params
        self.lr = payload.get("lr", self.lr)
        self.discount_factor = payload.get("discount_factor", self.discount_factor)
        self.epsilon = payload.get("epsilon", self.epsilon)
        self.epsilon_decay = payload.get("epsilon_decay", self.epsilon_decay)
        self.final_epsilon = payload.get("final_epsilon", self.final_epsilon)
        self.training_error = payload.get("training_error", [])




