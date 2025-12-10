import os
import sys
import argparse
import numpy as np
import gymnasium as gym
import racecar_gym.envs
import pybullet as p
import imageio

# Add RaceRL repo to path
sys.path.append("C:/Users/poibo/Documents/RaceRL")

# Verify RaceRL registrations
from gymnasium.envs.registration import registry

def check_env_exists(env_id):
    if env_id not in registry:
        print(f"ERROR: Environment '{env_id}' not found!")
        race_rl_envs = [e for e in registry if "SingleAgent" in e or "MultiAgent" in e]
        for e in race_rl_envs:
            print(" -", e)
        sys.exit(1)
    else:
        print(f"Environment '{env_id}' found in registry.")

# Wrappers
from gymnasium import ObservationWrapper, ActionWrapper, spaces

class FlattenObservation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Dict):
            raise ValueError("FlattenObservation expects Dict obs space")
        self.obs_keys = list(env.observation_space.spaces.keys())
        total_dim = int(sum(np.prod(env.observation_space.spaces[k].shape) for k in self.obs_keys))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32
        )

    def observation(self, obs):
        return np.concatenate([np.ravel(np.array(obs[k], dtype=np.float32)) for k in self.obs_keys])


class DictToBox(ActionWrapper):
    def __init__(self, env, low=None, high=None):
        super().__init__(env)
        if isinstance(env.action_space, spaces.Dict):
            n_actions = sum(int(np.prod(sp.shape)) for sp in env.action_space.spaces.values())

            if low is None or high is None:
                low = -np.ones(n_actions)
                high = np.ones(n_actions)

            self._map_keys = list(env.action_space.spaces.keys())
        else:
            raise ValueError("DictToBox expects Dict action space")

        self.action_space = spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            dtype=np.float32
        )

    def action(self, action):
        out = {}
        pos = 0
        for k in self._map_keys:
            sp = self.env.action_space.spaces[k]
            size = int(np.prod(sp.shape))
            out[k] = np.array(action[pos:pos+size]).reshape(sp.shape)
            pos += size
        return out


# Monitor and chase camera
from stable_baselines3.common.monitor import Monitor

def setup_chase_camera(env, distance=6.0, yaw=50, pitch=-30):
    env = env.unwrapped
    if not hasattr(env, "vehicle_id"):
        return lambda: None

    vehicle_id = env.vehicle_id
    def update_camera():
        pos, _ = p.getBasePositionAndOrientation(vehicle_id)
        p.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=pos,
        )
    return update_camera


def make_adapted_env(env_id="SingleAgentCircle_cw-v0", render_mode=None):
    check_env_exists(env_id)

    default_scenario = r"C:/Users/poibo/Documents/RaceRL/racecar/config/scenarios/circle_cw.yml"

    env = gym.make(
        env_id,
        scenario=default_scenario,
        render_mode=render_mode
    )

    if isinstance(env.observation_space, spaces.Dict):
        env = FlattenObservation(env)

    if isinstance(env.action_space, spaces.Dict):
        lows, highs = [], []
        for k, sp in env.action_space.spaces.items():
            if hasattr(sp, "low") and hasattr(sp, "high"):
                lows.extend(np.ravel(sp.low))
                highs.extend(np.ravel(sp.high))
            else:
                size = int(np.prod(sp.shape))
                lows.extend([-1] * size)
                highs.extend([1] * size)
        env = DictToBox(env, lows, highs)

    env = Monitor(env, "logs/", allow_early_resets=True)

    update_camera = None
    if render_mode == "human":
        update_camera = setup_chase_camera(env)

    return env, update_camera


from stable_baselines3.common.callbacks import BaseCallback

class RaceRLVideoCallback(BaseCallback):
    def __init__(self, vec_env, render_env, video_folder="videos/", freq=5000, length=100000):
        super().__init__()
        self.vec_env = vec_env
        self.render_env = render_env
        self.video_folder = video_folder
        self.freq = freq
        self.length = length
        os.makedirs(video_folder, exist_ok=True)

    def _on_step(self):
        if self.num_timesteps % self.freq == 0:
            frames = []
            obs, _ = self.render_env.reset()

            for t in range(self.length):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.render_env.step(action)

                frame = self.render_env.render()
                frames.append(frame)

                if done or truncated:
                    obs, _ = self.render_env.reset()

            video_path = os.path.join(self.video_folder, f"train_step_{self.num_timesteps}.mp4")
            imageio.mimsave(video_path, frames, fps=30)
            print(f"[Video] Saved training video at step {self.num_timesteps}")

        return True


from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

def train(env_id, total_timesteps=30000, model_path="models/racecar_sac_model"):
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    # Training uses rgb_array_follow for frame capture
    train_env, _ = make_adapted_env(env_id=env_id, render_mode="rgb_array_follow")
    vec_env = DummyVecEnv([lambda: train_env])

    # A second env for recording only (avoids interfering with training)
    render_env, _ = make_adapted_env(env_id=env_id, render_mode="rgb_array_follow")

    model = SAC("MlpPolicy", vec_env, verbose=1)

    ckpt_cb = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.dirname(model_path) or "./",
        name_prefix=os.path.basename(model_path),
    )

    video_cb = RaceRLVideoCallback(
        vec_env=vec_env,
        render_env=render_env,
        video_folder="videos/",
        freq=5000,
        length=800
    )

    model.learn(total_timesteps=total_timesteps, callback=[ckpt_cb, video_cb])
    model.save(model_path)
    
    print("Saved model to:", model_path)


def play(env_id, model_path="models/racecar_sac_model"):
    env, update_camera = make_adapted_env(env_id=env_id, render_mode="human")
    vec_env = DummyVecEnv([lambda: env])

    print("Loading:", model_path)
    model = SAC.load(model_path, env=vec_env)

    obs, _ = vec_env.reset()

    while True:
        if update_camera:
            update_camera()

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = vec_env.step(action)
        print(info[0])
        vec_env.render()

        if terminated or truncated:
            obs = vec_env.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "play"])
    parser.add_argument("--env", default="SingleAgentCircle_cw-v0")
    parser.add_argument("--timesteps", type=int, default=30000)
    parser.add_argument("--model-path", default="models/racecar_sac_model")
    args = parser.parse_args()

    if args.mode == "train":
        train(env_id=args.env, total_timesteps=args.timesteps, model_path=args.model_path)
    else:
        play(env_id=args.env, model_path=args.model_path)
