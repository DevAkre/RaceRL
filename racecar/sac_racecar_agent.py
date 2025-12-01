import os
import sys
import argparse
import numpy as np
import gymnasium as gym
import racecar_gym.envs.gym_api

# ------------------- CUSTOM ENV IMPORT -------------------
# Add RaceRL repo to path (adjust to your local clone)
sys.path.append("C:/Users/poibo/Documents/RaceRL")

# Import the module that registers all RaceRL environments
# From the repo, the environments are in 'racecar/envs'
try:
    import racecar_gym.envs
except ImportError as e:
    print("Failed to import RaceRL envs. Make sure the path is correct.")
    raise e

# ---------------- Verify environment registration ----------------
from gymnasium.envs.registration import registry

def check_env_exists(env_id):
    if env_id not in registry:
        print(f"ERROR: Environment '{env_id}' not found!")
        race_rl_envs = [e for e in registry if "SingleAgent" in e or "MultiAgent" in e]
        print(f"Available RaceRL environments ({len(race_rl_envs)}):")
        for e in race_rl_envs:
            print(" -", e)
        sys.exit(1)
    else:
        print(f"Environment '{env_id}' found in registry.")

# ---------------- Wrappers ----------------
from gymnasium import ObservationWrapper, ActionWrapper, spaces

class FlattenObservation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Dict):
            raise ValueError("FlattenObservation expects a Dict observation_space")
        self.obs_keys = list(env.observation_space.spaces.keys())
        total_dim = int(sum(np.prod(env.observation_space.spaces[k].shape) for k in self.obs_keys))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32)

    def observation(self, obs):
        return np.concatenate([np.ravel(np.array(obs[k], dtype=np.float32)) for k in self.obs_keys])

class DictToBox(ActionWrapper):
    def __init__(self, env, low=None, high=None):
        super().__init__(env)
        if isinstance(env.action_space, spaces.Dict):
            n_actions = sum(int(np.prod(sp.shape)) for sp in env.action_space.spaces.values())
            if low is None or high is None:
                low = -np.ones(n_actions, dtype=np.float32)
                high = np.ones(n_actions, dtype=np.float32)
            self._map_keys = list(env.action_space.spaces.keys())
        else:
            raise ValueError("DictToBox expects a Dict action_space")
        self.action_space = spaces.Box(low=np.array(low, dtype=np.float32),
                                       high=np.array(high, dtype=np.float32),
                                       dtype=np.float32)

    def action(self, action):
        out = {}
        pos = 0
        for k in self._map_keys:
            sp = self.env.action_space.spaces[k]
            size = int(np.prod(sp.shape))
            chunk = np.array(action[pos:pos+size], dtype=np.float32).reshape(sp.shape)
            out[k] = chunk
            pos += size
        return out

# -------------- Helper to adapt env --------------
from stable_baselines3.common.monitor import Monitor

def make_adapted_env(env_id="SingleAgentCircle_cw-v0", render_mode=None):
    check_env_exists(env_id)
    kwargs = {}
    if render_mode is not None:
        kwargs['render_mode'] = render_mode
    env = gym.make(env_id, **kwargs)

    if isinstance(env.observation_space, spaces.Dict):
        env = FlattenObservation(env)

    if isinstance(env.action_space, spaces.Dict):
        lows, highs = [], []
        for k, sp in env.action_space.spaces.items():
            if hasattr(sp, 'low') and hasattr(sp, 'high'):
                lows.extend(np.ravel(sp.low).tolist())
                highs.extend(np.ravel(sp.high).tolist())
            else:
                lows.extend([-1.0]*int(np.prod(sp.shape)))
                highs.extend([1.0]*int(np.prod(sp.shape)))
        env = DictToBox(env, low=np.array(lows, dtype=np.float32), high=np.array(highs, dtype=np.float32))

    env = Monitor(env)
    return env

# -------------- Training / Playing --------------
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

def train(env_id, total_timesteps=30000, model_path="models/racecar_sac_model"):
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    env = make_adapted_env(env_id=env_id, render_mode=None)
    vec_env = DummyVecEnv([lambda: env])

    model = SAC("MlpPolicy", vec_env, verbose=1)
    ckpt_cb = CheckpointCallback(save_freq=10_000, save_path=os.path.dirname(model_path) or "./",
                                 name_prefix=os.path.basename(model_path))

    model.learn(total_timesteps=total_timesteps, callback=ckpt_cb)
    model.save(model_path)
    print("Saved model to:", model_path)

def play(env_id, model_path="models/racecar_sac_model"):
    env = make_adapted_env(env_id=env_id, render_mode="human")
    vec_env = DummyVecEnv([lambda: env])

    print("Loading model from", model_path)
    model = SAC.load(model_path, env=vec_env)
    obs, _ = vec_env.reset()
    done = False
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = vec_env.step(action)
        vec_env.render()
        if isinstance(terminated, (list, tuple, np.ndarray)):
            if any(terminated) or any(truncated):
                break
        else:
            if terminated or truncated:
                break
    print("Episode finished.")

# -------------- CLI --------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "play"], help="train or play")
    parser.add_argument("--env", default="SingleAgentCircle_cw-v0", help="Gym env id")
    parser.add_argument("--timesteps", type=int, default=30000, help="Total training timesteps")
    parser.add_argument("--model-path", default="models/racecar_sac_model", help="Where to save/load model")
    args = parser.parse_args()

    if args.mode == "train":
        train(env_id=args.env, total_timesteps=args.timesteps, model_path=args.model_path)
    else:
        play(env_id=args.env, model_path=args.model_path)
