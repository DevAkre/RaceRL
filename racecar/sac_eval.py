import os 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3.common.results_plotter import ts2xy

# change for user
log_path = "C:/Users/poibo/Documents/RaceRL/racecar/logs/.monitor.csv"
env_name = "SingleAgentCircle_cw-v0"

results = pd.read_csv(log_path, comment='#')

# extract timesteps and rewards
timesteps, rewards = ts2xy(results, "timesteps")

print("Timesteps:", timesteps)
print("Rewards:", rewards)

plt.figure(figsize=(10,5))
plt.plot(timesteps, rewards, marker='o', label='Episode Reward')
plt.xlabel("Timesteps")
plt.ylabel("Reward")
plt.title(f"Training Progress: {env_name}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

