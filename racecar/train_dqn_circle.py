from pathlib import Path
import gymnasium as gym
import racecar_gym.envs.gym_api
import torch
import torch.nn as nn
import numpy as np
from dqn_wrapper import DQNWrapper
from tqdm import tqdm


from agents.dqn_agent import DQNNetwork, RacecarReplayBuffer

learning_rate = 0.0001       # Keep
discount_factor = 0.97       # Slightly lower
initial_epsilon = 1.0       # Keep
epsilon_decay = 0.995        # Faster decay
final_epsilon = 0.05         # More exploitation
batch_size = 128             # Keep
target_update_freq = 1000    # Less frequent updates
num_episodes = 500           # Enough to learn
target_update_freq = 500
best_model_dir_path = Path("./best_dqn_model")

num_bins_motor = 5
num_bins_steering = 5
use_pose = True
use_velocity = True
use_acceleration = False
use_lidar = True

total_steps = 0
episode_rewards = []
episode_steps_all = []

def save_model(path: Path) -> None:
    """Save Q-tables and agent parameters to disk."""
    payload = {
        'online_net_state_dict': online_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        "lr": learning_rate,
        "discount_factor": discount_factor,
        "epsilon": epsilon,
        "epsilon_decay": epsilon_decay,
        "final_epsilon": final_epsilon,
        'episode_rewards': episode_rewards,
        'episode_steps_all': episode_steps_all,
    }
    torch.save(payload, path)

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


if __name__ == "__main__":
    track = 'circle_cw'
    scenario = 'config/scenarios/' + track + '.yml'
    render_mode = "rgb_array"  # Set to None to disable rendering
    env = gym.make(
        id='SingleAgentRaceEnv-v0',
        scenario=scenario,
        vehicle_config_path='config/vehicles/og_racecar.yml',
        render_mode=render_mode
    )
    env = DQNWrapper(
        env,
        num_bins_motor=num_bins_motor,
        num_bins_steering=num_bins_steering,
        use_pose=use_pose,
        use_velocity=use_velocity,
        use_acceleration=use_acceleration,
        use_lidar=use_lidar
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    state_dim = env.observation_space_dim
    num_actions = len(env.allowed_actions)
    online_net = DQNNetwork(state_dim, num_actions).to(device)
    target_net = DQNNetwork(state_dim, num_actions).to(device)
    target_net.load_state_dict(online_net.state_dict())
    replay_buffer = RacecarReplayBuffer(capacity=100_000)  # Was 10000

    optimizer = torch.optim.Adam(online_net.parameters(), lr=learning_rate)
    loss_fn = nn.SmoothL1Loss()
    epsilon = initial_epsilon
    best_model_dir_path.mkdir(parents=True, exist_ok=True)

    best_reward = -float('inf')
    total_steps = 0  # Move this here for clarity

    for episode in tqdm(range(num_episodes), total=num_episodes):
        episode_steps = 0
        state, info = env.reset()
        done = False
        episode_reward = 0  # Track episode total

        while not done:
            # Select action
            action, _ = online_net.select_action(
                torch.FloatTensor(state).to(device),
                epsilon
            )

            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_steps += 1
            done = terminated or truncated

            # Store transition
            replay_buffer.add(state, action, reward, next_state, done)

            # Training step
            if len(replay_buffer) >= batch_size:
                # Sample batch
                states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = \
                    replay_buffer.sample(batch_size)

                # Move to device
                states_batch = states_batch.to(device)
                actions_batch = actions_batch.to(device)
                rewards_batch = rewards_batch.to(device)
                next_states_batch = next_states_batch.to(device)
                dones_batch = dones_batch.to(device)

                # Compute current Q-values
                current_q_values = online_net(states_batch)
                current_q = current_q_values.gather(1, actions_batch.unsqueeze(1)).squeeze()

                # Compute target Q-values
                with torch.no_grad():
                    next_q_values = target_net(next_states_batch)
                    max_next_q = next_q_values.max(dim=1)[0]
                    target_q = rewards_batch + discount_factor * max_next_q * (1 - dones_batch)

                # Compute loss and update
                loss = loss_fn(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10)
                optimizer.step()

            # FIXED: Update target network (with proper condition)
            if total_steps % target_update_freq == 0 and total_steps > 0:
                target_net.load_state_dict(online_net.state_dict())

            state = next_state
            episode_reward += reward  # Accumulate episode reward
            total_steps += 1

        episode_steps_all.append(episode_steps)

        # FIXED: Compare and save based on episode_reward, not last step reward!
        if episode_reward > best_reward:
            print_str = (f"New best model with reward {episode_reward:.2f}"
                         + f"at episode {episode} with {episode_steps} steps.")
            print(print_str)
            best_reward = episode_reward
            best_reward_episode = episode
            best_reward_steps = episode_steps
            save_model(best_model_dir_path/"best_dqn_model.pth")

        # FIXED: Append episode total, not last step reward!
        episode_rewards.append(episode_reward)

        # Decay epsilon
        epsilon = max(final_epsilon, epsilon * epsilon_decay)

        # Add monitoring every 10 episodes
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_reward
            print(f"Episode {episode}: Reward={episode_reward:.2f}, Avg(10)={avg_reward:.2f}, Epsilon={epsilon:.3f}")
            avg_steps = np.mean(episode_steps_all[-10:]) if len(episode_steps_all) >= 10 else episode_steps


            # Sample Q-values for monitoring
            with torch.no_grad():
                test_state = torch.FloatTensor(env.reset()[0]).unsqueeze(0).to(device)
                test_q = online_net(test_state)

            print(f"\nEpisode {episode}:")
            print(f"  Episode Reward: {episode_reward:.2f}")
            print(f"  Avg Reward (10): {avg_reward:.2f}")
            print(f"  Avg Steps (10): {avg_steps:.2f}")
            print(f"  Best Reward: {best_reward:.2f}")
            print(f"  Best Reward episode: {best_reward_episode}")
            print(f"  Best Reward steps: {best_reward_steps}")
            print(f"  Episode Steps: {episode_steps}")
            print(f"  Epsilon: {epsilon:.4f}")
            print(f"  Q-values: mean={test_q.mean():.2f}, max={test_q.max():.2f}")
            print(f"  Buffer Size: {len(replay_buffer)}")

            # Sanity check
            if test_q.mean() > 100:
                print("  ⚠️  Q-values getting high!")


    save_model(best_model_dir_path/"final_dqn_model.pth")
    plot_avg_rewards(env)
