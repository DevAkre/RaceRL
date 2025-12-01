import gymnasium as gym
import agents
import racecar_gym.envs.gym_api  # type: ignore
from typing import Optional
import wrapper
from tqdm import tqdm
import os
from time import sleep
import logging

def train(n_episodes: int = 500, render_mode: Optional[str] = None, track: str = 'circle'): 
    # Training hyperparameters
    learning_rate = 0.01        # How fast to learn (higher = faster but less stable)
    milestone_num = 10       # Print milestone stats every N episodes
    start_epsilon = 1        # Start with full exploration        
    epsilon_decay = start_epsilon/(n_episodes/2)  # Reduce exploration over time
    final_epsilon = 0.05         # Always keep some exploration
    gamma = 0.99                 # Discount factor for future rewards
    lidar_bins = [0.0, 2.5, 5.0]
    num_velocity_bins = 3
    num_acceleration_bins = 0  # No acceleration bins
    env_data={'lidar_bins': lidar_bins, 'num_velocity_bins': num_velocity_bins, 'num_acceleration_bins': num_acceleration_bins}

    scenario = 'config/scenarios/' + track + '.yml'
    # Init single environment
    env = gym.make(
        id='SingleAgentRaceEnv-v0', 
        scenario=scenario,
        vehicle_config_path='config/vehicles/racecar.yml',
        render_mode='human'
    )
    env = wrapper.DiscreteAction(env, num_bins_motor=3, num_bins_steering=5)
    env = wrapper.FlattenAction(env)
    env = wrapper.DiscretizeObservation(env, lidar_bins, num_velocity_bins, num_acceleration_bins)
    env = wrapper.FlattenObservation(env)
    #Episode Statictics 
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
    env.reset()
    # Get State and Action spaces
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    agent = agents.QLearningRacecarAgent(
        env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=gamma
    )

    pbar = tqdm(total=n_episodes)

    print(f"Starting training for {n_episodes} episodes")
    progress = []
    time_to_finish = []
    for ep in range(n_episodes):
        done = False

        obs, info = env.reset(options=dict(mode='grid'))
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                if info['lap'] > 1:
                    progress.append(1.0)
                    time_to_finish.append(info['time'])
                else:
                    progress.append(info['progress'])
            agent.update(obs, action, float(reward), done, next_obs)
            obs = next_obs
            # Log episode statistics (available in info after episode ends)
        if "episode" in info:
                episode_data = info["episode"]
                logging.info(f"Episode {ep}: "
                            f"reward={episode_data['r']:.1f}, "
                            f"length={episode_data['l']}, "
                            f"time={episode_data['t']:.2f}s")

                # Additional analysis for milestone episodes
                if ep % milestone_num == 0 and ep > 0:
                    # Look at recent performance 
                    recent_rewards = list(env.return_queue)[-milestone_num:]
                    if recent_rewards:
                        avg_recent = sum(recent_rewards) / len(recent_rewards)
                        print(f"  -> Average reward over last 100 episodes: {avg_recent:.1f}")
                        print(f"  -> Epsilon: {agent.epsilon:.3f}")
                    recent_progress = progress[-milestone_num:]
                    if recent_progress:
                        avg_progress = sum(recent_progress) / len(recent_progress)
                        print(f"  -> Average progress over last {milestone_num} episodes: {avg_progress:.4f}")
                    if time_to_finish:
                        recent_times = time_to_finish[-milestone_num:]
                        if recent_times:
                            avg_time = sum(recent_times) / len(recent_times)
                            print(f"  -> Average time to finish over last {len(recent_times)} completed episodes: {avg_time:.2f}s")
        # Decay epsilon
        agent.decay_epsilon()
        pbar.update(1)
    
    pbar.close()
    return env, agent, env_data

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

def save_agent(agent, path: str, env_data: Optional[dict] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Check if extension is .pkl
    if not path.endswith('.pkl'):
        path += '.pkl'
    # Rename if file exists
    if os.path.exists(path):
        base, ext = os.path.splitext(path)
        count = 1
        new_path = f"{base}_v{count}{ext}"
        while os.path.exists(new_path):
            count += 1
            new_path = f"{base}_v{count}{ext}"
        path = new_path
    agent.save(path)
    env_data_path = path.replace('.pkl', '_env_data.pkl')
    if env_data is not None:
        with open(env_data_path, "wb") as f:
            import pickle
            pickle.dump(env_data, f)
    print(f"Saved trained agent to {path}")
    print(f"Saved environment data to {env_data_path}")

def main(n_episodes: int = 500, render_mode: Optional[str] = None, track: str = 'circle', save_path: str = 'models/q_table.pkl'):
    env, agent, env_data = train(n_episodes=n_episodes, render_mode=render_mode, track=track)
    save_agent(agent, save_path, env_data)
    plot_avg_rewards(env)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', '-t', default='circle', help='Track to run the model on (e.g., austria, berlin, circle)')
    parser.add_argument('--episodes', '-e', type=int, default=500, help='Number of episodes to run')
    parser.add_argument('--save-path', '-s', type=str, default='models/q_table.pkl', help='Path to save the trained model')
    parser.add_argument('--render-mode', '-r', type=str, default=None, help='Render mode for the environment (e.g., None, human, rgb_array_follow, rgb_array_birds_eye)')
    args = parser.parse_args()
    main(n_episodes=args.episodes, track=args.track, save_path= args.save_path)