import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy

# setup
log_dir = "logs/"
window_size = 50         # oving average window for smoothing
env_name = "SingleAgentCircle_cw-v0"  # title

# check if results exist
if not os.path.exists(log_dir):
    raise FileNotFoundError(f"Log directory '{log_dir}' does not exist. Make sure your env used Monitor logging.")

results = load_results(log_dir)
timesteps, rewards = ts2xy(results, "timesteps")  # extract timesteps and episode rewards

# create a moving average to smooth and minimize noise
def moving_average(x, window=50):
    return np.convolve(x, np.ones(window)/window, mode='valid')

smoothed_rewards = moving_average(rewards, window=window_size)
smoothed_timesteps = timesteps[len(timesteps) - len(smoothed_rewards):]

# plotting
plt.figure(figsize=(10,5))
plt.plot(smoothed_timesteps, smoothed_rewards, color='blue', label='Episode Reward (smoothed)')
plt.xlabel("Timesteps")
plt.ylabel("Reward")
plt.title(f"Training Progress: {env_name}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()