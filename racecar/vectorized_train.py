import gymnasium as gym
import agents
from racecar_gym.envs import gym_api  # type: ignore
from typing import Optional
import wrapper
from tqdm import tqdm
import os
from time import sleep
import logging
import numpy as np

def train(n_instances = 3, n_steps: int = 500, render_mode: Optional[str] = None, track: str = 'circle'): 
    # Training hyperparameters
    learning_rate = 0.01        # How fast to learn (higher = faster but less stable)
    milestone_num = 100       # Print milestone stats every N episodes
    start_epsilon = 1.0        # Start with full exploration        
    epsilon_decay = start_epsilon/(n_steps/2)  # Reduce exploration over time
    final_epsilon = 0.05         # Always keep some exploration
    gamma = 0.99                 # Discount factor for future rewards

    scenario = 'config/scenarios/' + track + '.yml'
    
    wrappers = [
        wrapper.DiscreteAction,
        wrapper.FlattenAction,
        wrapper.DiscretizeObservation,
        wrapper.FlattenObservation
    ]
    # Init vectorized environment
    envs = gym.make_vec(
        id='SingleAgentRaceEnv-v0',
        vectorization_mode='async',
        num_envs=n_instances,
        wrappers=wrappers,
        scenario=scenario,
        vehicle_config_path='config/vehicles/racecar.yml',
        render_mode=render_mode
    )

    envs.reset()

    # Get State and Action spaces
    print(f"Observation space: {envs.observation_space}")
    print(f"Action space: {envs.action_space}")    

    agent = agents.QLearningRacecarAgent(
        envs,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=gamma,
        batch_size=n_instances
    )

    pbar = tqdm(total=n_steps)

    print(f"Starting training for {n_steps} steps with {n_instances} parallel instances ")

    for ep in range(n_steps):
        done = [False] * n_instances
        obs, info = envs.reset(options=dict(mode='grid'))

        actions = agent.get_action(obs)
        next_obs, rewards, terminated, truncated, info = envs.step(actions)
        pbar.update(1)
        done = [t or tr for t, tr in zip(terminated, truncated)]
        agent.update(obs, actions, rewards, terminated, next_obs)
        obs = next_obs
        agent.decay_epsilon()
        pbar.update(1)


    return envs, agent

# Average reward graph
def plot_avg_rewards(env):
    import matplotlib.pyplot as plt

    # Plot learning progress
    episodes = range(len(env.return_queue))
    rewards = list(env.return_queue)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, alpha=0.3, label='Episode Rewards')
    n_episodes = len(rewards)
    # Add moving average for clearer trend
    if n_episodes >= 10:
        window = n_episodes // 10
    else:
        window = 1
    if len(rewards) > window:
        moving_avg = [sum(rewards[i:i+window])/window
                    for i in range(len(rewards)-window+1)]
        plt.plot(range(window-1, len(rewards)), moving_avg,
                label=f'{window}-Episode Moving Average', linewidth=2)

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Learning Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def save_agent(agent, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Check if extension is .pkl
    if not path.endswith('.pkl'):
        path += '.pkl'
    # Check if model already exists
    if os.path.exists(path):
        print(f"Warning: Overwriting existing model at {path}")
        r = np.random.randint(10000)
        path = path.replace('.pkl', f'_{r}.pkl')
        print(f"Renamed existing model to {path}")
    agent.save(path)
    print(f"Saved trained agent to {path}")

def main(num_instances: int = 3, n_steps: int = 100_000, render_mode: Optional[str] = None, track: str = 'circle', save_path: str = 'models/q_table.pkl'):
    env, agent = train(n_instances=num_instances, n_steps=n_steps, render_mode=render_mode, track=track)
    save_agent(agent, save_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', '-t', default='circle', help='Track to run the model on (e.g., austria, berlin, circle)')
    parser.add_argument('--steps', '-e', type=int, default=100_000, help='Number of steps to run')
    parser.add_argument('--save-path', '-s', type=str, default='models/q_table.pkl', help='Path to save the trained model')
    parser.add_argument('--num-instances', '-n', type=int, default=3, help='Number of parallel environment instances to use')
    parser.add_argument('--render-mode', '-r', type=str, default=None, help='Render mode for the environment (e.g., None, human, rgb_array_follow, rgb_array_birds_eye)')
    args = parser.parse_args()
    main(n_steps=args.steps, track=args.track, save_path= args.save_path, render_mode=args.render_mode, num_instances=args.num_instances)