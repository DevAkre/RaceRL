import gymnasium as gym
import agents
from racecar_gym.envs import gym_api  # type: ignore
from typing import Optional
import wrapper
from tqdm import tqdm
import os
from time import sleep
import logging

def train(n_instances = 3, n_episodes: int = 500, render_mode: Optional[str] = None, track: str = 'circle'): 
    # Training hyperparameters
    learning_rate = 0.01        # How fast to learn (higher = faster but less stable)
    milestone_num = 100       # Print milestone stats every N episodes
    start_epsilon = 1.0        # Start with full exploration        
    epsilon_decay = start_epsilon/(n_episodes/2)  # Reduce exploration over time
    final_epsilon = 0.05         # Always keep some exploration
    gamma = 0.95                 # Discount factor for future rewards

    scenarios = ['config/scenarios/circle.yml' for _ in range(n_instances)]

    # Template environment used only for agent initialization (not used for stepping)
    envs = gym_api.VectorizedSingleAgentRaceEnv(
        scenarios=scenarios,
        vehicle_config_path='config/vehicles/racecar.yml',
        render_mode=render_mode
    )
    print(envs.reset())

    envs = wrapper.VectorizedDiscreteActionWrapper(envs, num_bins_motor=3, num_bins_steering=5)

    envs = wrapper.VectorizedDiscretizeObservationWrapper(envs, lidar_bins=[0, 2.5, 5.0], num_velocity_bins=3, num_acceleration_bins=3)
    #Episode Statictics 
    # env = wrapper.VectorizedRecordEpisodeStatistics(envs, buffer_length=n_episodes)

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

    pbar = tqdm(total=n_episodes)


    print(f"Starting training for {n_episodes} episodes")

    for iter in range(n_episodes // n_instances):
        done = False
        obs = envs.reset()

        while not done:
            action = agent.get_action(obs)
            next_obs, rewards, terminates, truncates, infos = envs.step(action)
            done = any([e for e in (terminates or truncates)])
            agent.update(obs, action, float(rewards), done, next_obs)
            obs = next_obs
            if render_mode != None and render_mode == 'human':
                envs.render()
                print(f"Observation: {obs}, Action: {action}, Reward: {reward}, Done: {done}")
                sleep(0.01)  # Small delay to make rendering visible
            # Log episode statistics (available in info after episode ends)
        
        # if "episode" in info:
        #         episode_data = info["episode"]
        #         logging.info(f"Episode {ep}: "
        #                     f"reward={episode_data['r']:.1f}, "
        #                     f"length={episode_data['l']}, "
        #                     f"time={episode_data['t']:.2f}s")

        #         # Additional analysis for milestone episodes
        #         if ep % milestone_num == 0 and ep > 0:
        #             # Look at recent performance 
        #             recent_rewards = list(env.return_queue)[-milestone_num:]
        #             if recent_rewards:
        #                 avg_recent = sum(recent_rewards) / len(recent_rewards)
        #                 print(f"  -> Average reward over last 100 episodes: {avg_recent:.1f}")
        #                 print(f"  -> Epsilon: {agent.epsilon:.3f}")
        # Decay epsilon
        agent.decay_epsilon()
        pbar.update(1)
    
    pbar.close()
    return env, agent

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
    agent.save(path)
    print(f"Saved trained agent to {path}")

def main(num_instances: int = 3, n_episodes: int = 500, render_mode: Optional[str] = None, track: str = 'circle', save_path: str = 'models/q_table.pkl'):
    env, agent = train(n_instances=num_instances, n_episodes=n_episodes, render_mode=render_mode, track=track)
    save_agent(agent, save_path)
    plot_avg_rewards(env)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', '-t', default='circle', help='Track to run the model on (e.g., austria, berlin, circle)')
    parser.add_argument('--episodes', '-e', type=int, default=500, help='Number of episodes to run')
    parser.add_argument('--save-path', '-s', type=str, default='models/q_table.pkl', help='Path to save the trained model')
    parser.add_argument('--num-instances', '-n', type=int, default=3, help='Number of parallel environment instances to use')
    parser.add_argument('--render-mode', '-r', type=str, default=None, help='Render mode for the environment (e.g., None, human, rgb_array_follow, rgb_array_birds_eye)')
    args = parser.parse_args()
    main(n_episodes=args.episodes, track=args.track, save_path= args.save_path, render_mode=args.render_mode, num_instances=args.num_instances)